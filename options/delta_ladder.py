"""
Delta Ladder for managing delta-targeted option subscriptions.

Maintains a ladder of options at target deltas (ATM, 40D, 25D, 10D) for
calls and puts across multiple products and expiries. Each slot in the
ladder maps to a specific option contract with permanent subscription.

Key principles:
- One option per slot (product, expiry, target_delta, right)
- Slots are locked with permanent subscriptions
- Track actual vs target delta for drift monitoring
- Quality metadata for institutional-grade data provenance
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

from ib_insync import Contract

from .subscription_manager import SubscriptionManager, ContractType
from .strike_prober import StrikeProber, ProbeResult, StrikeWithDelta
from .quality_enums import (
    ModelDeltaSource, ModelDeltaMethod, ModelUnderlyingRef,
    QuoteState, QualityFlag,
    DeltaSource  # Legacy, for backward compat
)

logger = logging.getLogger(__name__)


class MissingReason(str, Enum):
    """Reason why a ladder slot could not be filled."""
    NO_UNIQUE_STRIKE = "NO_UNIQUE_STRIKE"        # Collision in assignment (strike already used)
    NO_QUOTES = "NO_QUOTES"                       # No bid/ask or modelGreeks available
    OUTSIDE_TOLERANCE = "OUTSIDE_TOLERANCE"      # Best strike too far from target delta
    SURFACE_NOT_AVAILABLE = "SURFACE_NOT_AVAILABLE"  # Vol surface couldn't fit (not enough liquid strikes)


@dataclass(frozen=True)
class LadderKey:
    """Unique identifier for a delta ladder slot."""
    product: str      # CL, NG, RB, HO
    expiry: str       # YYYYMMDD
    target_delta: float  # 0.50, 0.40, 0.25, 0.10
    right: str        # C or P

    def __str__(self) -> str:
        return f"{self.product}:{self.expiry}:{self.target_delta:.2f}:{self.right}"


@dataclass
class LadderSlot:
    """
    A slot in the delta ladder holding an option contract with quality metadata.

    Two-track delta ontology:
    - model_delta: OUR Black-76 calculation (from live quotes or vol surface)
    - broker_delta: IBKR's modelGreeks (for comparison/fallback)

    Supports both filled slots (with contract) and placeholder slots (MISSING).
    """
    key: LadderKey

    # Contract info (None for MISSING slots)
    contract: Optional[Contract] = None
    strike: Optional[float] = None

    # ==========================================================================
    # Model Greeks Track (our QuantLib Black-76 calculation)
    # ==========================================================================
    model_delta: Optional[float] = None
    model_gamma: Optional[float] = None
    model_theta: Optional[float] = None
    model_vega: Optional[float] = None
    model_iv: Optional[float] = None
    model_delta_source: ModelDeltaSource = ModelDeltaSource.SURFACE
    model_delta_method: ModelDeltaMethod = ModelDeltaMethod.BLACK_76
    model_underlying_ref: ModelUnderlyingRef = ModelUnderlyingRef.FUT_MID
    model_delta_ts: datetime = field(default_factory=datetime.now)

    # ==========================================================================
    # Broker Greeks Track (IBKR's modelGreeks calculation)
    # ==========================================================================
    broker_delta: Optional[float] = None
    broker_gamma: Optional[float] = None
    broker_theta: Optional[float] = None
    broker_vega: Optional[float] = None
    broker_iv: Optional[float] = None
    broker_delta_ts: datetime = field(default_factory=datetime.now)

    # ==========================================================================
    # Reproducibility Inputs (for deterministic replay)
    # ==========================================================================
    underlying_price_used: Optional[float] = None
    option_mid_used: Optional[float] = None
    rate_used: Optional[float] = None
    dte_used: Optional[float] = None

    # ==========================================================================
    # Quote Quality Metadata
    # ==========================================================================
    quote_state: QuoteState = QuoteState.MISSING
    quality_flag: QualityFlag = QualityFlag.INVALID
    missing_reason: Optional[MissingReason] = None

    # Quote data for diagnostics
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    last_price: Optional[float] = None
    last_size: Optional[int] = None
    spread_pct: Optional[float] = None

    # Timestamps for staleness check (option and futures quotes should be synchronized)
    option_quote_ts: Optional[datetime] = None
    fut_quote_ts: Optional[datetime] = None

    # ==========================================================================
    # Lifecycle Tracking
    # ==========================================================================
    subscribed_at: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    is_dirty: bool = False  # Flagged for rebalancing

    # Legacy alias (kept for backward compat)
    iv: Optional[float] = None  # Deprecated: use model_iv

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def target_delta(self) -> float:
        """Target delta from the ladder key."""
        return self.key.target_delta

    @property
    def quotes_synchronized(self) -> bool:
        """Check if option and futures quotes are within 2 seconds."""
        if self.option_quote_ts is None or self.fut_quote_ts is None:
            return False
        return abs((self.option_quote_ts - self.fut_quote_ts).total_seconds()) <= 2.0

    @property
    def drift(self) -> float:
        """Absolute drift from target delta. Uses model_delta if available, else broker."""
        delta = self.for_rebalancing
        if delta is None:
            return float('inf')
        return abs(abs(delta) - self.key.target_delta)

    @property
    def is_filled(self) -> bool:
        """Check if this slot has a valid contract (not a MISSING placeholder)."""
        return self.contract is not None and self.strike is not None

    @property
    def execution_eligible(self) -> bool:
        """
        Check if this slot is eligible for trade execution.

        Requires:
        - Model delta from QUOTES (live bid/ask)
        - LIVE_TRADABLE quote state
        - OK quality flag
        - Quotes synchronized (option/futures within 2s)
        """
        return (
            self.is_filled and
            self.model_delta is not None and
            self.model_delta_source == ModelDeltaSource.QUOTES and
            self.quote_state == QuoteState.LIVE_TRADABLE and
            self.quality_flag == QualityFlag.OK and
            self.quotes_synchronized
        )

    @property
    def for_rebalancing(self) -> Optional[float]:
        """Delta for drift checking. Prefers model_delta, falls back to broker_delta."""
        return self.model_delta if self.model_delta is not None else self.broker_delta

    @property
    def for_analytics(self) -> Optional[float]:
        """Delta for reporting/ML. Always uses best available (model preferred)."""
        return self.model_delta if self.model_delta is not None else self.broker_delta

    # ==========================================================================
    # Backward Compatibility Aliases
    # ==========================================================================

    @property
    def observed_delta(self) -> Optional[float]:
        """DEPRECATED: Use model_delta or broker_delta. Returns model_delta or broker fallback."""
        return self.for_analytics

    @property
    def actual_delta(self) -> Optional[float]:
        """DEPRECATED: Alias for observed_delta (backward compatibility)."""
        return self.observed_delta

    @property
    def delta_source(self) -> DeltaSource:
        """DEPRECATED: Use model_delta_source. Maps to legacy DeltaSource enum."""
        if self.model_delta is not None:
            if self.model_delta_source == ModelDeltaSource.QUOTES:
                return DeltaSource.TIER_1_MARKET
            else:
                return DeltaSource.TIER_2_SURFACE
        return DeltaSource.TIER_3_IBKR

    @property
    def delta_timestamp(self) -> datetime:
        """DEPRECATED: Use model_delta_ts or broker_delta_ts."""
        return self.model_delta_ts

    # ==========================================================================
    # String Representation
    # ==========================================================================

    def __str__(self) -> str:
        if not self.is_filled:
            reason = f" ({self.missing_reason.value})" if self.missing_reason else ""
            return f"{self.key}: MISSING{reason}"

        dirty_flag = " [DIRTY]" if self.is_dirty else ""
        exec_flag = " [EXEC]" if self.execution_eligible else ""
        delta_display = self.model_delta if self.model_delta is not None else self.broker_delta
        source_display = f"model:{self.model_delta_source.value}" if self.model_delta else "broker"
        return (
            f"{self.key}: strike={self.strike:.2f}, "
            f"delta={delta_display:.4f} (target={self.key.target_delta:.2f}, "
            f"drift={self.drift:.4f}) [{source_display}]{dirty_flag}{exec_flag}"
        )


class DeltaLadder:
    """
    Manages delta-targeted option subscriptions across products.

    Usage:
        ladder = DeltaLadder(sub_manager, prober)

        # Build ladder for CL options
        await ladder.build_product_ladder(
            product='CL',
            futures=[cl_m1, cl_m2, ...],
            futures_prices={'CLH5': 75.50, 'CLJ5': 76.20, ...},
            trading_class='LO',
            expiries=['20250315', '20250415', ...],
            probe_band=12
        )

        # Get a specific slot
        slot = ladder.get_slot('CL', '20250315', 0.25, 'C')
    """

    # Standard delta targets
    TARGETS = [0.50, 0.40, 0.25, 0.10]

    def __init__(
        self,
        sub_manager: SubscriptionManager,
        prober: StrikeProber
    ):
        """
        Initialize delta ladder.

        Args:
            sub_manager: Subscription manager for permanent subscriptions
            prober: Strike prober for finding best delta matches
        """
        self.sub_manager = sub_manager
        self.prober = prober

        # Main storage: LadderKey -> LadderSlot
        self.slots: Dict[LadderKey, LadderSlot] = {}

        # Reverse lookup: conId -> LadderKey
        self._conid_to_key: Dict[int, LadderKey] = {}

    def get_slot(
        self,
        product: str,
        expiry: str,
        target_delta: float,
        right: str
    ) -> Optional[LadderSlot]:
        """Get slot for a specific delta bucket."""
        key = LadderKey(product, expiry, target_delta, right)
        return self.slots.get(key)

    def get_slot_by_conid(self, conid: int) -> Optional[LadderSlot]:
        """Get slot by option contract ID."""
        key = self._conid_to_key.get(conid)
        if key:
            return self.slots.get(key)
        return None

    def get_product_slots(self, product: str) -> List[LadderSlot]:
        """Get all slots for a product."""
        return [s for s in self.slots.values() if s.key.product == product]

    def get_expiry_slots(self, product: str, expiry: str) -> List[LadderSlot]:
        """Get all slots for a specific expiry."""
        return [
            s for s in self.slots.values()
            if s.key.product == product and s.key.expiry == expiry
        ]

    def get_dirty_slots(self) -> List[LadderSlot]:
        """Get all slots flagged for rebalancing."""
        return [s for s in self.slots.values() if s.is_dirty]

    def mark_dirty(self, key: LadderKey):
        """Mark a slot as needing rebalancing."""
        if key in self.slots:
            self.slots[key].is_dirty = True

    def clear_dirty(self, key: LadderKey):
        """Clear dirty flag after rebalancing."""
        if key in self.slots:
            self.slots[key].is_dirty = False

    async def build_product_ladder(
        self,
        product: str,
        futures: List[Contract],
        futures_prices: Dict[str, float],
        trading_class: str,
        expiries: List[str],
        probe_band: int = 10,
        targets: Optional[List[float]] = None
    ) -> Tuple[int, int]:
        """
        Build delta ladder for a product across expiries.

        Args:
            product: Product code (CL, NG, RB, HO)
            futures: List of futures contracts (for underlying reference)
            futures_prices: Map of futures symbol to current price
            trading_class: IBKR trading class (LO, LNE, OB, OH)
            expiries: List of option expiries (YYYYMMDD)
            probe_band: Strikes to probe around ATM
            targets: Delta targets (default: [0.50, 0.40, 0.25, 0.10])

        Returns:
            (slots_filled, slots_failed) count tuple
        """
        import asyncio
        from datetime import datetime

        targets = targets or self.TARGETS
        slots_filled = 0
        slots_failed = 0

        total_slots = len(expiries) * len(targets) * 2
        logger.info(
            f"Building delta ladder for {product}: "
            f"{len(expiries)} expiries × {len(targets)} deltas × 2 rights = "
            f"{total_slots} slots"
        )

        start_time = datetime.now()

        # Prepare probe tasks for all expiries
        probe_tasks = []
        expiry_data = []  # (expiry, underlying, price) tuples

        for expiry in expiries:
            underlying = self._find_underlying(futures, expiry)
            if not underlying:
                logger.warning(f"No underlying found for {product} {expiry}")
                slots_failed += len(targets) * 2
                continue

            price = futures_prices.get(underlying.localSymbol)
            if not price:
                logger.warning(f"No price for {underlying.localSymbol}")
                slots_failed += len(targets) * 2
                continue

            expiry_data.append((expiry, underlying, price))

        # Probe all expiries in parallel batches
        # Each probe uses ~50 subscriptions, batch size limited by cap headroom
        contracts_per_probe = (probe_band * 2 + 1) * 2  # strikes × 2 (C/P)
        headroom = self.sub_manager.headroom()
        batch_size = max(1, headroom // contracts_per_probe)

        logger.info(
            f"Probing {len(expiry_data)} expiries in batches of {batch_size} "
            f"(headroom={headroom}, ~{contracts_per_probe}/probe)"
        )

        # Process in batches
        probe_results = {}
        for i in range(0, len(expiry_data), batch_size):
            batch = expiry_data[i:i + batch_size]

            # Create probe tasks for this batch
            batch_tasks = [
                self.prober.probe_strikes(
                    underlying=underlying,
                    underlying_price=price,
                    expiry=expiry,
                    trading_class=trading_class,
                    band_width=probe_band
                )
                for expiry, underlying, price in batch
            ]

            # Run batch in parallel
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Collect results
            for (expiry, underlying, _), result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error(f"Probe failed for {expiry}: {result}")
                    slots_failed += len(targets) * 2
                else:
                    probe_results[expiry] = (result, underlying)

        # Fill slots from probe results with constraint satisfaction
        # Process targets in priority order: ATM (0.50) first, then 0.40, 0.25, 0.10
        # This ensures most important deltas get best strikes
        priority_targets = sorted(targets, reverse=True)  # [0.50, 0.40, 0.25, 0.10]

        for expiry, (probe_result, underlying) in probe_results.items():
            if probe_result.strikes_with_data == 0:
                logger.warning(f"No valid strikes for {product} {expiry}")
                # Create MISSING placeholders for all targets
                for target in priority_targets:
                    for right in ['C', 'P']:
                        key = LadderKey(product, expiry, target, right)
                        self.slots[key] = LadderSlot(
                            key=key,
                            quote_state=QuoteState.MISSING,
                            quality_flag=QualityFlag.INVALID,
                            missing_reason=MissingReason.NO_QUOTES,
                        )
                slots_failed += len(targets) * 2
                continue

            # Track assigned strikes per right to ensure uniqueness
            assigned_calls: set = set()
            assigned_puts: set = set()

            for target in priority_targets:
                for right in ['C', 'P']:
                    key = LadderKey(product, expiry, target, right)
                    assigned = assigned_calls if right == 'C' else assigned_puts

                    success = await self._fill_slot(
                        key, probe_result, underlying, assigned_strikes=assigned
                    )

                    # Track the assigned strike if successful
                    if success and key in self.slots and self.slots[key].strike:
                        assigned.add(self.slots[key].strike)
                        slots_filled += 1
                    else:
                        slots_failed += 1

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Delta ladder for {product}: {slots_filled} filled, {slots_failed} failed "
            f"({elapsed:.1f}s)"
        )

        return slots_filled, slots_failed

    async def _fill_slot(
        self,
        key: LadderKey,
        probe_result: ProbeResult,
        underlying: Contract,
        assigned_strikes: Optional[set] = None
    ) -> bool:
        """
        Fill a single ladder slot with best matching option.

        Always creates a slot (dense ladder):
        - Valid assignment → full slot with contract and quality metadata
        - No match → placeholder slot with MISSING state and reason

        Args:
            key: The ladder key for this slot
            probe_result: Results from strike probing
            underlying: The underlying futures contract
            assigned_strikes: Set of already-assigned strikes (for non-duplication)

        Returns:
            True if slot was filled with a valid contract, False if MISSING
        """
        assigned_strikes = assigned_strikes or set()

        # Find best strike for this delta target (excluding already assigned)
        best, missing_reason = self._find_best_unassigned(
            probe_result, key.target_delta, key.right, assigned_strikes
        )

        if not best:
            # Create MISSING placeholder with specific reason
            slot = LadderSlot(
                key=key,
                quote_state=QuoteState.MISSING,
                quality_flag=QualityFlag.INVALID,
                missing_reason=missing_reason or MissingReason.NO_QUOTES,
            )
            self.slots[key] = slot
            logger.warning(f"MISSING slot for {key}: {missing_reason.value if missing_reason else 'unknown'}")
            return False

        contract = best.option_contract

        # Subscribe permanently
        reason = f"ladder:{key}"
        if not self.sub_manager.subscribe(contract, ContractType.OPTION, reason=reason):
            # Create MISSING placeholder for subscription failure
            slot = LadderSlot(
                key=key,
                quote_state=QuoteState.MISSING,
                quality_flag=QualityFlag.INVALID,
                missing_reason=MissingReason.NO_QUOTES,
            )
            self.slots[key] = slot
            logger.warning(f"MISSING slot created for {key}: subscription failed")
            return False

        # Map legacy DeltaSource to two-track model
        now = datetime.now()
        model_delta = None
        model_delta_source = ModelDeltaSource.SURFACE
        broker_delta = best.broker_delta  # Always capture broker delta if available

        if best.delta_source == DeltaSource.TIER_1_MARKET:
            model_delta = best.delta
            model_delta_source = ModelDeltaSource.QUOTES
        elif best.delta_source == DeltaSource.TIER_2_SURFACE:
            model_delta = best.delta
            model_delta_source = ModelDeltaSource.SURFACE
        # For TIER_3_IBKR, model_delta stays None, broker_delta already set

        # Compute quality flag based on delta drift
        quality_flag = self._compute_preliminary_quality(
            model_delta_source if model_delta else None,
            key.target_delta,
            best.delta
        )

        # Create filled slot with full enterprise-grade data
        slot = LadderSlot(
            key=key,
            contract=contract,
            strike=best.strike,
            # Model Greeks track (our QuantLib calculation)
            model_delta=model_delta,
            model_gamma=best.gamma,
            model_theta=best.theta,
            model_vega=best.vega,
            model_iv=best.iv,
            model_delta_source=model_delta_source,
            model_delta_method=ModelDeltaMethod.BLACK_76,
            model_underlying_ref=ModelUnderlyingRef.FUT_MID,
            model_delta_ts=best.quote_timestamp or now,
            # Broker Greeks track (IBKR modelGreeks)
            broker_delta=broker_delta,
            broker_gamma=best.broker_gamma,
            broker_theta=best.broker_theta,
            broker_vega=best.broker_vega,
            broker_iv=best.broker_iv,
            broker_delta_ts=best.quote_timestamp or now,
            # Reproducibility inputs
            underlying_price_used=best.underlying_price_used,
            option_mid_used=best.mid,
            rate_used=best.rate_used,
            dte_used=best.dte_used,
            # Quote snapshot
            bid=best.bid,
            ask=best.ask,
            bid_size=best.bid_size,
            ask_size=best.ask_size,
            last_price=best.last_price,
            last_size=best.last_size,
            spread_pct=best.spread_pct,
            # Quality metadata
            quote_state=best.quote_state,
            quality_flag=quality_flag,
            missing_reason=None,
            option_quote_ts=best.quote_timestamp,
            fut_quote_ts=None,  # TODO: Pass futures quote timestamp from probe
            # Legacy alias
            iv=best.iv,
        )

        # Store slot
        self.slots[key] = slot
        self._conid_to_key[contract.conId] = key

        logger.debug(
            f"Filled {key}: strike={best.strike:.2f}, "
            f"delta={best.delta:.4f} (target={key.target_delta:.2f}) "
            f"[{best.delta_source.value}]"
        )

        return True

    def _find_best_unassigned(
        self,
        probe_result: ProbeResult,
        target_delta: float,
        right: str,
        assigned_strikes: set
    ) -> tuple[Optional[StrikeWithDelta], Optional[MissingReason]]:
        """
        Find best strike for target delta, excluding already-assigned strikes.

        Returns:
            (strike_data, missing_reason) - strike_data is None if no match,
            missing_reason indicates why (if applicable)
        """
        candidates = probe_result.calls if right == 'C' else probe_result.puts
        if not candidates:
            return None, MissingReason.NO_QUOTES

        # Filter out already assigned strikes
        available = [s for s in candidates if s.strike not in assigned_strikes]
        if not available:
            return None, MissingReason.NO_UNIQUE_STRIKE

        best = min(available, key=lambda s: s.distance_to_target(target_delta))

        # Check tolerance based on tier
        # TIER_1: max 0.10 drift, TIER_2/3: max 0.15 drift
        drift = best.distance_to_target(target_delta)
        if best.delta_source == DeltaSource.TIER_1_MARKET:
            max_drift = 0.10
        else:
            max_drift = 0.15

        if drift > max_drift:
            return None, MissingReason.OUTSIDE_TOLERANCE

        return best, None

    def _compute_preliminary_quality(
        self,
        model_delta_source: Optional[ModelDeltaSource],
        target_delta: float,
        observed_delta: float
    ) -> QualityFlag:
        """
        Compute preliminary quality flag based on delta drift.

        Uses tiered tolerances:
        - QUOTES (execution-grade): OK ≤0.05, DEGRADED ≤0.10, else INVALID
        - SURFACE or broker: OK ≤0.10, DEGRADED ≤0.15, else INVALID

        Args:
            model_delta_source: Source of model delta (None if broker_delta only)
            target_delta: Target delta for this slot
            observed_delta: Observed delta value
        """
        drift = abs(abs(observed_delta) - target_delta)

        # Execution-grade (QUOTES) has tighter tolerance
        if model_delta_source == ModelDeltaSource.QUOTES:
            if drift <= 0.05:
                return QualityFlag.OK
            elif drift <= 0.10:
                return QualityFlag.DEGRADED
            else:
                return QualityFlag.INVALID
        else:
            # SURFACE or broker delta (analytics track)
            if drift <= 0.10:
                return QualityFlag.OK
            elif drift <= 0.15:
                return QualityFlag.DEGRADED
            else:
                return QualityFlag.INVALID

    async def update_slot(
        self,
        key: LadderKey,
        new_strike_data: StrikeWithDelta
    ) -> bool:
        """
        Update a slot with a new option contract (after rebalancing).

        Uses atomic swap to maintain subscription count.

        Args:
            key: The ladder key for this slot
            new_strike_data: New strike data with quality metadata
        """
        if key not in self.slots:
            logger.error(f"Cannot update non-existent slot: {key}")
            return False

        old_slot = self.slots[key]
        if not old_slot.is_filled:
            logger.error(f"Cannot update MISSING slot: {key}")
            return False

        old_contract = old_slot.contract
        new_contract = new_strike_data.option_contract

        # Atomic swap
        reason = f"rebalance:{key}"
        if not self.sub_manager.swap(
            old_contract, new_contract, ContractType.OPTION, reason=reason
        ):
            logger.error(f"Swap failed for {key}")
            return False

        # Update reverse lookup
        del self._conid_to_key[old_contract.conId]
        self._conid_to_key[new_contract.conId] = key

        # Map legacy DeltaSource to new two-track model
        model_delta = None
        model_delta_source = ModelDeltaSource.SURFACE
        broker_delta = None
        now = datetime.now()

        if new_strike_data.delta_source == DeltaSource.TIER_1_MARKET:
            model_delta = new_strike_data.delta
            model_delta_source = ModelDeltaSource.QUOTES
        elif new_strike_data.delta_source == DeltaSource.TIER_2_SURFACE:
            model_delta = new_strike_data.delta
            model_delta_source = ModelDeltaSource.SURFACE
        else:
            broker_delta = new_strike_data.delta

        # Compute quality flag
        quality_flag = self._compute_preliminary_quality(
            model_delta_source if model_delta else None,
            key.target_delta,
            new_strike_data.delta
        )

        # Update slot with two-track delta model
        self.slots[key] = LadderSlot(
            key=key,
            contract=new_contract,
            strike=new_strike_data.strike,
            # Model delta track
            model_delta=model_delta,
            model_delta_source=model_delta_source,
            model_delta_method=ModelDeltaMethod.BLACK_76,
            model_underlying_ref=ModelUnderlyingRef.FUT_MID,
            model_delta_ts=new_strike_data.quote_timestamp or now,
            # Broker delta track
            broker_delta=broker_delta,
            broker_delta_ts=new_strike_data.quote_timestamp or now,
            # Greeks
            iv=new_strike_data.iv,
            # Quality metadata
            quote_state=new_strike_data.quote_state,
            quality_flag=quality_flag,
            missing_reason=None,
            bid=new_strike_data.bid,
            ask=new_strike_data.ask,
            spread_pct=new_strike_data.spread_pct,
            option_quote_ts=new_strike_data.quote_timestamp,
            fut_quote_ts=None,
            is_dirty=False  # Just rebalanced
        )

        logger.info(
            f"Updated {key}: {old_slot.strike:.2f} -> {new_strike_data.strike:.2f}, "
            f"delta {old_slot.for_analytics:.4f} -> {new_strike_data.delta:.4f} "
            f"[{new_strike_data.delta_source.value}]"
        )

        return True

    def update_delta(
        self,
        key: LadderKey,
        new_delta: float,
        new_iv: float,
        delta_source: Optional[DeltaSource] = None,
        quote_state: Optional[QuoteState] = None,
        option_quote_ts: Optional[datetime] = None,
        fut_quote_ts: Optional[datetime] = None
    ):
        """
        Update delta for a slot (from drift monitoring).

        Maps legacy DeltaSource to the two-track model:
        - TIER_1_MARKET → model_delta with QUOTES source
        - TIER_2_SURFACE → model_delta with SURFACE source
        - TIER_3_IBKR → broker_delta

        Args:
            key: The ladder key
            new_delta: New delta value
            new_iv: New IV value
            delta_source: Legacy delta source (determines which track to update)
            quote_state: Optional new quote state
            option_quote_ts: Timestamp of option quote
            fut_quote_ts: Timestamp of futures quote (for staleness check)
        """
        if key not in self.slots:
            return

        slot = self.slots[key]
        if not slot.is_filled:
            return  # Can't update MISSING slot

        now = datetime.now()

        # Map legacy source to new two-track model and update appropriate field
        if delta_source == DeltaSource.TIER_1_MARKET:
            slot.model_delta = new_delta
            slot.model_delta_source = ModelDeltaSource.QUOTES
            slot.model_delta_ts = now
        elif delta_source == DeltaSource.TIER_2_SURFACE:
            slot.model_delta = new_delta
            slot.model_delta_source = ModelDeltaSource.SURFACE
            slot.model_delta_ts = now
        elif delta_source == DeltaSource.TIER_3_IBKR:
            slot.broker_delta = new_delta
            slot.broker_delta_ts = now
        else:
            # No source specified - update model delta with current source
            slot.model_delta = new_delta
            slot.model_delta_ts = now

        slot.iv = new_iv
        slot.last_update = now

        if quote_state is not None:
            slot.quote_state = quote_state
        if option_quote_ts is not None:
            slot.option_quote_ts = option_quote_ts
        if fut_quote_ts is not None:
            slot.fut_quote_ts = fut_quote_ts

        # Recompute quality flag
        slot.quality_flag = self._compute_preliminary_quality(
            slot.model_delta_source if slot.model_delta else None,
            slot.target_delta,
            new_delta
        )

    def remove_slot(self, key: LadderKey) -> bool:
        """Remove a slot and unsubscribe its contract."""
        if key not in self.slots:
            return False

        slot = self.slots[key]

        # Only unsubscribe if slot has a contract (not MISSING)
        if slot.is_filled and slot.contract:
            self.sub_manager.unsubscribe(slot.contract)
            if slot.contract.conId in self._conid_to_key:
                del self._conid_to_key[slot.contract.conId]

        # Remove from storage
        del self.slots[key]

        logger.info(f"Removed slot {key}")
        return True

    def remove_product(self, product: str) -> int:
        """Remove all slots for a product."""
        keys_to_remove = [k for k in self.slots.keys() if k.product == product]

        for key in keys_to_remove:
            self.remove_slot(key)

        return len(keys_to_remove)

    def _find_underlying(
        self,
        futures: List[Contract],
        option_expiry: str
    ) -> Optional[Contract]:
        """
        Find the futures contract that underlies options with given expiry.

        For most energy options, the underlying is the futures with same
        or nearest expiry month.
        """
        # Parse option expiry month
        opt_month = option_expiry[:6]  # YYYYMM

        # Find futures with matching or closest expiry
        best_match = None
        best_diff = float('inf')

        for fut in futures:
            if not fut.lastTradeDateOrContractMonth:
                continue

            fut_month = fut.lastTradeDateOrContractMonth[:6]

            # Prefer exact month match
            if fut_month == opt_month:
                return fut

            # Otherwise find closest
            diff = abs(int(fut_month) - int(opt_month))
            if diff < best_diff:
                best_diff = diff
                best_match = fut

        return best_match

    def status(self) -> dict:
        """Get ladder status summary with quality metrics (two-track model)."""
        by_product = {}
        quality_counts = {
            'filled': 0,
            'missing': 0,
            'execution_eligible': 0,
            'quotes_synchronized': 0,
            # New two-track breakdown
            'by_model_source': {s.value: 0 for s in ModelDeltaSource},
            'broker_only': 0,  # Slots with only broker_delta
            # Legacy breakdown (for backward compat)
            'by_tier': {t.value: 0 for t in DeltaSource},
            'by_quality': {q.value: 0 for q in QualityFlag},
            'by_quote_state': {s.value: 0 for s in QuoteState},
        }

        for slot in self.slots.values():
            if slot.key.product not in by_product:
                by_product[slot.key.product] = {
                    'total': 0, 'filled': 0, 'missing': 0,
                    'dirty': 0, 'exec_eligible': 0, 'expiries': set()
                }

            prod_stats = by_product[slot.key.product]
            prod_stats['total'] += 1
            prod_stats['expiries'].add(slot.key.expiry)

            if slot.is_filled:
                prod_stats['filled'] += 1
                quality_counts['filled'] += 1

                # Two-track breakdown
                if slot.model_delta is not None:
                    quality_counts['by_model_source'][slot.model_delta_source.value] += 1
                else:
                    quality_counts['broker_only'] += 1

                # Staleness check
                if slot.quotes_synchronized:
                    quality_counts['quotes_synchronized'] += 1

                # Legacy breakdown (via backward-compat property)
                quality_counts['by_tier'][slot.delta_source.value] += 1
                quality_counts['by_quality'][slot.quality_flag.value] += 1
                quality_counts['by_quote_state'][slot.quote_state.value] += 1

                if slot.execution_eligible:
                    prod_stats['exec_eligible'] += 1
                    quality_counts['execution_eligible'] += 1
            else:
                prod_stats['missing'] += 1
                quality_counts['missing'] += 1
                quality_counts['by_quote_state'][QuoteState.MISSING.value] += 1

            if slot.is_dirty:
                prod_stats['dirty'] += 1

        # Convert sets to counts
        for product in by_product:
            by_product[product]['expiries'] = len(by_product[product]['expiries'])

        return {
            'total_slots': len(self.slots),
            'filled_slots': quality_counts['filled'],
            'missing_slots': quality_counts['missing'],
            'execution_eligible': quality_counts['execution_eligible'],
            'quotes_synchronized': quality_counts['quotes_synchronized'],
            'dirty_slots': len(self.get_dirty_slots()),
            'fill_rate': quality_counts['filled'] / len(self.slots) if self.slots else 0,
            'exec_rate': quality_counts['execution_eligible'] / len(self.slots) if self.slots else 0,
            # Two-track breakdown
            'by_model_source': quality_counts['by_model_source'],
            'broker_only': quality_counts['broker_only'],
            # Legacy breakdown
            'by_tier': quality_counts['by_tier'],
            'by_quality': quality_counts['by_quality'],
            'by_quote_state': quality_counts['by_quote_state'],
            'by_product': by_product
        }

    def __len__(self) -> int:
        return len(self.slots)

    def __iter__(self):
        return iter(self.slots.values())
