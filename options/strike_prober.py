"""
Strike Prober for delta-targeted option selection.

Temporarily subscribes to a band of strikes around ATM, computes actual
deltas via QuantLib, and returns ranked strikes by proximity to target.
All temporary subscriptions are cleaned up before returning.

Key principles:
- Brief probes only: subscribe, compute, cleanup
- Never leave orphan subscriptions
- Respect subscription cap even during probes
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from ib_insync import IB, Contract, FuturesOption, Ticker

from .greeks_engine import GreeksEngine, Greeks
from .subscription_manager import SubscriptionManager, ContractType
from .quality_enums import DeltaSource, QuoteState, QualityFlag
from .quote_validator import QuoteValidator

logger = logging.getLogger(__name__)


# Re-export enums for backward compatibility
__all__ = ['DeltaSource', 'QuoteState', 'QualityFlag', 'StrikeWithDelta', 'ProbeResult', 'StrikeProber']


@dataclass
class StrikeWithDelta:
    """A strike with its computed delta, quality metadata, and option contract."""
    strike: float
    right: str  # 'C' or 'P'
    delta: float
    iv: float
    option_contract: Contract

    # Quality metadata (institutional-grade tracking)
    delta_source: DeltaSource = DeltaSource.TIER_3_IBKR
    quote_state: QuoteState = QuoteState.MISSING

    # Quote data
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    spread_pct: Optional[float] = None  # (ask-bid)/mid for diagnostics
    quote_timestamp: Optional[datetime] = None

    # Quote sizes (from IBKR ticker)
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    last_price: Optional[float] = None
    last_size: Optional[int] = None

    # Full model Greeks (our QuantLib calculation)
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

    # Full broker Greeks (IBKR modelGreeks)
    broker_delta: Optional[float] = None
    broker_iv: Optional[float] = None
    broker_gamma: Optional[float] = None
    broker_theta: Optional[float] = None
    broker_vega: Optional[float] = None

    # Reproducibility inputs (what we used for calculation)
    underlying_price_used: Optional[float] = None
    rate_used: Optional[float] = None
    dte_used: Optional[float] = None

    def distance_to_target(self, target_delta: float) -> float:
        """Absolute distance from target delta."""
        # For puts, delta is negative, so compare absolute values
        actual = abs(self.delta)
        return abs(actual - target_delta)

    @property
    def is_liquid(self) -> bool:
        """Check if this strike has valid market quotes (TIER_1 eligible)."""
        return self.quote_state in (QuoteState.LIVE_TRADABLE, QuoteState.INDICATIVE)


@dataclass
class ProbeResult:
    """Result of a strike probe operation."""
    underlying: Contract
    expiry: str
    strikes_probed: int
    strikes_with_data: int
    calls: list[StrikeWithDelta]
    puts: list[StrikeWithDelta]
    probe_duration_ms: float

    def best_for_delta(self, target_delta: float, right: str) -> Optional[StrikeWithDelta]:
        """Find strike closest to target delta."""
        candidates = self.calls if right == 'C' else self.puts
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.distance_to_target(target_delta))


class StrikeProber:
    """
    Probes strikes to find best matches for target deltas.

    Usage:
        prober = StrikeProber(ib, sub_manager, greeks_engine)

        result = await prober.probe_strikes(
            underlying=cl_future,
            underlying_price=75.50,
            expiry='20250315',
            trading_class='LO',
            band_width=10
        )

        # Find best 25-delta call
        best_25d_call = result.best_for_delta(0.25, 'C')
    """

    # How long to wait for quotes after subscribing
    # 1.5s needed for modelGreeks to populate (IBKR takes ~1s to compute)
    # This is critical for illiquid options that only have modelGreeks, no quotes
    QUOTE_WAIT_SECONDS = 1.5

    # Minimum bid/ask to consider valid
    MIN_VALID_PRICE = 0.01

    def __init__(
        self,
        ib: IB,
        sub_manager: SubscriptionManager,
        greeks_engine: Optional[GreeksEngine] = None,
        quote_validator: Optional[QuoteValidator] = None
    ):
        """
        Initialize strike prober.

        Args:
            ib: Connected IB instance
            sub_manager: Subscription manager for cap enforcement
            greeks_engine: Greeks calculator (created if not provided)
            quote_validator: Quote validator for quality classification
        """
        self.ib = ib
        self.sub_manager = sub_manager
        self.greeks = greeks_engine or GreeksEngine()
        self.validator = quote_validator or QuoteValidator()

        # Lock for single-strike probing (rarely used)
        self._single_probe_lock = asyncio.Lock()
        self._single_probe_subs: list[Contract] = []

    async def probe_strikes(
        self,
        underlying: Contract,
        underlying_price: float,
        expiry: str,
        trading_class: str,
        band_width: int = 10,
        strike_increment: Optional[float] = None
    ) -> ProbeResult:
        """
        Probe strikes around ATM to compute deltas.

        This method is concurrency-safe - multiple probes can run in parallel.

        Args:
            underlying: Underlying futures contract
            underlying_price: Current futures price
            expiry: Option expiry in YYYYMMDD format
            trading_class: IBKR trading class (LO, LNE, OB, OH)
            band_width: Number of strikes above/below ATM to probe
            strike_increment: Strike spacing (auto-detected if not provided)

        Returns:
            ProbeResult with ranked strikes for calls and puts
        """
        start_time = datetime.now()
        temp_subscriptions: list[Contract] = []  # Local to this probe

        try:
            # Get available strikes from chain
            strikes = await self._get_strike_range(
                underlying, expiry, trading_class, underlying_price, band_width, strike_increment
            )

            if not strikes:
                logger.warning(f"No strikes found for {underlying.localSymbol} {expiry}")
                return self._empty_result(underlying, expiry, start_time)

            logger.info(
                f"Probing {len(strikes)} strikes for {underlying.localSymbol} "
                f"{expiry} (ATM ~{underlying_price:.2f})"
            )

            # Build option contracts for all strikes (calls and puts)
            option_contracts = self._build_option_contracts(
                underlying, expiry, trading_class, strikes
            )

            # Check subscription headroom
            needed = len(option_contracts)
            available = self.sub_manager.headroom()

            if needed > available:
                # Reduce probe scope to fit
                logger.warning(
                    f"Reducing probe from {needed} to {available} contracts (cap limit)"
                )
                # Prioritize strikes closest to ATM
                option_contracts = self._prioritize_atm(
                    option_contracts, underlying_price, available
                )

            # Subscribe to all probe contracts (returns list of subscribed)
            temp_subscriptions = await self._subscribe_probe_contracts(option_contracts)

            # Wait for quotes to arrive
            await asyncio.sleep(self.QUOTE_WAIT_SECONDS)

            # Compute deltas for all contracts with valid quotes
            calls, puts = await self._compute_deltas(
                temp_subscriptions, underlying_price, expiry
            )

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            return ProbeResult(
                underlying=underlying,
                expiry=expiry,
                strikes_probed=len(option_contracts),
                strikes_with_data=len(calls) + len(puts),
                calls=sorted(calls, key=lambda x: x.strike),
                puts=sorted(puts, key=lambda x: x.strike),
                probe_duration_ms=duration_ms
            )

        finally:
            # ALWAYS cleanup temporary subscriptions
            await self._cleanup_subscriptions(temp_subscriptions)

    async def probe_single_strike(
        self,
        underlying: Contract,
        underlying_price: float,
        expiry: str,
        trading_class: str,
        strike: float,
        right: str
    ) -> Optional[StrikeWithDelta]:
        """
        Probe a single strike to get its current delta.

        Useful for checking if a specific option has drifted.
        Uses a lock since single probes are infrequent.
        """
        async with self._single_probe_lock:
            temp_subs: list[Contract] = []

            try:
                contract = self._build_single_option(
                    underlying, expiry, trading_class, strike, right
                )

                # Qualify the contract
                qualified = await self._qualify_contract(contract)
                if not qualified:
                    return None

                # Subscribe temporarily
                if not self.sub_manager.subscribe(
                    qualified, ContractType.OPTION, reason="probe:single"
                ):
                    logger.warning("Cannot probe: at subscription cap")
                    return None

                temp_subs.append(qualified)

                # Wait for quote
                await asyncio.sleep(self.QUOTE_WAIT_SECONDS)

                # Get ticker and compute delta
                ticker = self.sub_manager.get_ticker(qualified)
                if not ticker:
                    return None

                return self._compute_single_delta(
                    qualified, ticker, underlying_price, expiry, right
                )

            finally:
                await self._cleanup_subscriptions(temp_subs)

    async def _get_strike_range(
        self,
        underlying: Contract,
        expiry: str,
        trading_class: str,
        atm_price: float,
        band_width: int,
        strike_increment: Optional[float]
    ) -> list[float]:
        """Get list of strikes to probe."""
        # Request option chain parameters
        # For futures options, futFopExchange must be specified
        exchange = underlying.exchange or "NYMEX"
        chains = await self.ib.reqSecDefOptParamsAsync(
            underlying.symbol,
            exchange,
            underlying.secType,
            underlying.conId
        )

        if not chains:
            logger.warning(f"No option chains found for {underlying.localSymbol}")
            return []

        # Find matching chain
        chain = None
        for c in chains:
            if c.tradingClass == trading_class:
                chain = c
                break

        if not chain:
            logger.warning(f"Trading class {trading_class} not found in chains")
            return []

        # Filter to requested expiry and get strikes
        if expiry not in chain.expirations:
            logger.warning(f"Expiry {expiry} not in chain expirations")
            return []

        all_strikes = sorted(chain.strikes)

        # Find ATM strike index
        atm_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - atm_price))

        # Extract band around ATM
        start_idx = max(0, atm_idx - band_width)
        end_idx = min(len(all_strikes), atm_idx + band_width + 1)

        return all_strikes[start_idx:end_idx]

    def _build_option_contracts(
        self,
        underlying: Contract,
        expiry: str,
        trading_class: str,
        strikes: list[float]
    ) -> list[Contract]:
        """Build option contracts for all strikes (calls and puts)."""
        contracts = []

        for strike in strikes:
            for right in ['C', 'P']:
                opt = FuturesOption(
                    symbol=underlying.symbol,
                    lastTradeDateOrContractMonth=expiry,
                    strike=strike,
                    right=right,
                    exchange=underlying.exchange or 'NYMEX',
                    currency=underlying.currency or 'USD',
                    tradingClass=trading_class
                )
                contracts.append(opt)

        return contracts

    def _build_single_option(
        self,
        underlying: Contract,
        expiry: str,
        trading_class: str,
        strike: float,
        right: str
    ) -> Contract:
        """Build a single option contract."""
        return FuturesOption(
            symbol=underlying.symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            exchange=underlying.exchange or 'NYMEX',
            currency=underlying.currency or 'USD',
            tradingClass=trading_class
        )

    def _prioritize_atm(
        self,
        contracts: list[Contract],
        atm_price: float,
        limit: int
    ) -> list[Contract]:
        """Keep contracts closest to ATM when over cap."""
        # Sort by distance from ATM
        sorted_contracts = sorted(
            contracts,
            key=lambda c: abs(c.strike - atm_price)
        )
        return sorted_contracts[:limit]

    async def _qualify_contract(self, contract: Contract) -> Optional[Contract]:
        """Qualify contract to get conId."""
        try:
            qualified = await self.ib.qualifyContractsAsync(contract)
            if qualified:
                return qualified[0]
        except Exception as e:
            logger.debug(f"Failed to qualify {contract}: {e}")
        return None

    async def _subscribe_probe_contracts(self, contracts: list[Contract]) -> list[Contract]:
        """Subscribe to probe contracts, qualifying first. Returns subscribed contracts."""
        subscribed: list[Contract] = []

        # Qualify all contracts in batch
        try:
            qualified = await self.ib.qualifyContractsAsync(*contracts)
        except Exception as e:
            logger.error(f"Failed to qualify probe contracts: {e}")
            return subscribed

        # Subscribe to qualified contracts
        for contract in qualified:
            if contract.conId:
                if self.sub_manager.subscribe(
                    contract, ContractType.OPTION, reason="probe:band"
                ):
                    subscribed.append(contract)

        logger.debug(f"Subscribed to {len(subscribed)} probe contracts")
        return subscribed

    async def _cleanup_subscriptions(self, subscriptions: list[Contract]):
        """Unsubscribe a list of temporary subscriptions."""
        if not subscriptions:
            return

        for contract in subscriptions:
            self.sub_manager.unsubscribe(contract)

        logger.debug(f"Cleaned up {len(subscriptions)} probe subscriptions")

    async def _compute_deltas(
        self,
        contracts: list[Contract],
        underlying_price: float,
        expiry: str
    ) -> tuple[list[StrikeWithDelta], list[StrikeWithDelta]]:
        """Compute deltas for all probed contracts."""
        calls = []
        puts = []

        expiry_date = date(int(expiry[:4]), int(expiry[4:6]), int(expiry[6:8]))

        for contract in contracts:
            ticker = self.sub_manager.get_ticker(contract)
            if not ticker:
                continue

            result = self._compute_single_delta(
                contract, ticker, underlying_price, expiry, contract.right
            )

            if result:
                if contract.right == 'C':
                    calls.append(result)
                else:
                    puts.append(result)

        return calls, puts

    def _compute_single_delta(
        self,
        contract: Contract,
        ticker: Ticker,
        underlying_price: float,
        expiry: str,
        right: str
    ) -> Optional[StrikeWithDelta]:
        """Compute delta for a single contract from its ticker with quality metadata."""
        bid = ticker.bid if ticker.bid and ticker.bid > 0 else None
        ask = ticker.ask if ticker.ask and ticker.ask > 0 else None

        # Get timestamp from ticker (may be None)
        quote_ts = getattr(ticker, 'time', None)

        # Need valid bid/ask for mid price
        if not bid or not ask:
            # Try last price as fallback
            if ticker.last and ticker.last > self.MIN_VALID_PRICE:
                mid = ticker.last
            else:
                # Fallback: use IBKR's modelGreeks for illiquid options
                # IBKR computes these even when no live quotes available
                if ticker.modelGreeks and ticker.modelGreeks.delta is not None:
                    mg = ticker.modelGreeks
                    logger.debug(
                        f"Using modelGreeks (TIER_3) for {contract.localSymbol}: "
                        f"delta={mg.delta:.3f}"
                    )
                    return StrikeWithDelta(
                        strike=contract.strike,
                        right=right,
                        delta=mg.delta,
                        iv=mg.impliedVol or 0.0,
                        option_contract=contract,
                        # Quality metadata: TIER_3 model-only
                        delta_source=DeltaSource.TIER_3_IBKR,
                        quote_state=QuoteState.MODEL_ONLY,
                        bid=None,
                        ask=None,
                        mid=mg.optPrice,
                        spread_pct=None,
                        quote_timestamp=quote_ts,
                        # Quote sizes (not available when MODEL_ONLY)
                        bid_size=None,
                        ask_size=None,
                        last_price=ticker.last if ticker.last and ticker.last > 0 else None,
                        last_size=ticker.lastSize if hasattr(ticker, 'lastSize') else None,
                        # No model Greeks (we didn't compute - broker only)
                        gamma=None,
                        theta=None,
                        vega=None,
                        # Full broker Greeks
                        broker_delta=mg.delta,
                        broker_iv=mg.impliedVol,
                        broker_gamma=mg.gamma,
                        broker_theta=mg.theta,
                        broker_vega=mg.vega,
                        # Reproducibility (broker used their own inputs)
                        underlying_price_used=mg.undPrice,
                        rate_used=None,  # Not available from IBKR
                        dte_used=None,   # Not available from IBKR
                    )
                return None
        else:
            mid = (bid + ask) / 2

        if mid < self.MIN_VALID_PRICE:
            return None

        # Compute spread percentage for diagnostics
        spread_pct = (ask - bid) / mid if mid > 0 else None

        # Parse expiry date
        expiry_date = date(int(expiry[:4]), int(expiry[4:6]), int(expiry[6:8]))

        # Compute Greeks via QuantLib (more accurate than IBKR's model)
        greeks = self.greeks.calc_greeks(
            option_price=mid,
            underlying_price=underlying_price,
            strike=contract.strike,
            expiry=expiry_date,
            right=right
        )

        if not greeks:
            return None

        # Use QuoteValidator for proper quote state classification
        # This checks spread thresholds, quote age, halted, crossed, and all hard rules
        quote_state = self.validator.compute_quote_state(ticker)

        # Capture quote sizes from ticker
        bid_size = ticker.bidSize if hasattr(ticker, 'bidSize') and ticker.bidSize else None
        ask_size = ticker.askSize if hasattr(ticker, 'askSize') and ticker.askSize else None
        last_price = ticker.last if ticker.last and ticker.last > 0 else None
        last_size = ticker.lastSize if hasattr(ticker, 'lastSize') and ticker.lastSize else None

        # Capture broker Greeks for comparison (even when we computed our own)
        mg = ticker.modelGreeks
        broker_delta = mg.delta if mg else None
        broker_iv = mg.impliedVol if mg else None
        broker_gamma = mg.gamma if mg else None
        broker_theta = mg.theta if mg else None
        broker_vega = mg.vega if mg else None

        # Compute DTE for reproducibility
        dte_used = (expiry_date - date.today()).days / 365.0

        return StrikeWithDelta(
            strike=contract.strike,
            right=right,
            delta=greeks.delta,
            iv=greeks.iv,
            option_contract=contract,
            # Quality metadata: TIER_1 market-derived
            delta_source=DeltaSource.TIER_1_MARKET,
            quote_state=quote_state,
            bid=bid,
            ask=ask,
            mid=mid,
            spread_pct=spread_pct,
            quote_timestamp=quote_ts,
            # Quote sizes
            bid_size=bid_size,
            ask_size=ask_size,
            last_price=last_price,
            last_size=last_size,
            # Full model Greeks (our calculation)
            gamma=greeks.gamma,
            theta=greeks.theta,
            vega=greeks.vega,
            # Full broker Greeks (for comparison)
            broker_delta=broker_delta,
            broker_iv=broker_iv,
            broker_gamma=broker_gamma,
            broker_theta=broker_theta,
            broker_vega=broker_vega,
            # Reproducibility inputs
            underlying_price_used=underlying_price,
            rate_used=self.greeks.risk_free_rate,
            dte_used=dte_used,
        )

    def _empty_result(
        self,
        underlying: Contract,
        expiry: str,
        start_time: datetime
    ) -> ProbeResult:
        """Return empty probe result."""
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        return ProbeResult(
            underlying=underlying,
            expiry=expiry,
            strikes_probed=0,
            strikes_with_data=0,
            calls=[],
            puts=[],
            probe_duration_ms=duration_ms
        )
