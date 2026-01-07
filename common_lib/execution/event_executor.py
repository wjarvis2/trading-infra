"""
Event-Driven Executor.

Bridges the new type system (Signal, ExecutionEvent) to existing cost models.

Design principles:
- execute(signal, bars) → List[ExecutionEvent]
- Returns FillEvent or RejectEvent, never raises
- Uses existing cost models (SlippageModel, CommissionModel)
- mark_to_market() for position pricing

This is the adapter between generic signals and execution simulation.

"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Union
from datetime import datetime
import pandas as pd

from common_lib.interfaces.instrument import Tradable, ExecutionMode
from common_lib.interfaces.signal import Signal, SignalIntent
from common_lib.interfaces.execution_events import (
    ExecutionEvent,
    FillEvent,
    RejectEvent,
    RollFillEvent,
    LegFill,
    FillType,
    RejectReason,
    create_fill_event,
    create_reject_event,
    create_single_leg_fill,
    create_spread_fill,
    create_roll_fill_event,
)
from common_lib.execution.product_spec import ProductSpec, ProductCatalog, create_default_catalog
from common_lib.execution.executor_config import ExecutorConfig, DEFAULT_CONFIG
from common_lib.execution.cost_models import (
    SlippageModel,
    CommissionModel,
    BarRangeSlippageModel,
    PerContractCommissionModel,
    create_slippage_model,
)


class EventExecutorProtocol(Protocol):
    """Protocol for event-driven executors."""

    def execute(
        self,
        signal: Signal,
        bars: Dict[str, pd.Series],
    ) -> List[ExecutionEvent]:
        """
        Execute a signal and return execution events.

        Parameters
        ----------
        signal : Signal
            The signal to execute
        bars : dict[str, pd.Series]
            Current bars by instrument_id

        Returns
        -------
        list[ExecutionEvent]
            FillEvent(s) or RejectEvent
        """
        ...

    def mark_to_market(
        self,
        instrument_ids: List[str],
        bars: Dict[str, pd.Series],
    ) -> Dict[str, float]:
        """
        Get current mark prices for instruments.

        Parameters
        ----------
        instrument_ids : list[str]
            Instruments to price
        bars : dict[str, pd.Series]
            Current bars by instrument_id

        Returns
        -------
        dict[str, float]
            Current prices by instrument_id
        """
        ...


@dataclass
class EventExecutor:
    """
    Event-driven execution simulator.

    Takes generic Signals and returns ExecutionEvents (fills or rejects).
    Uses pluggable cost models for slippage and commission calculation.

    Parameters
    ----------
    product : ProductSpec
        Product specification (multiplier, tick size, etc.)
    config : ExecutorConfig
        Behavioral configuration
    slippage_model : SlippageModel
        Pluggable slippage calculator
    commission_model : CommissionModel
        Pluggable commission calculator

    Examples
    --------
    >>> executor = create_event_executor("CL")
    >>> events = executor.execute(signal, {"CL1_CL2": bar})
    >>> for event in events:
    ...     position = ledger.apply(event)
    """

    product: ProductSpec
    config: ExecutorConfig
    slippage_model: SlippageModel
    commission_model: CommissionModel

    def execute(
        self,
        signal: Signal,
        bars: Dict[str, pd.Series],
    ) -> List[ExecutionEvent]:
        """
        Execute a signal and return execution events.

        Dispatches based on signal intent:
        - OPEN: Single fill event (entry)
        - CLOSE: Single fill event (exit)
        - ROLL: Two fill events (exit current + enter new)

        Returns RejectEvent if execution cannot proceed (no market data, etc.).
        """
        # Validate we have market data
        instrument_id = signal.tradable.instrument_id
        if instrument_id not in bars:
            return [
                create_reject_event(
                    signal=signal,
                    reason=RejectReason.NO_MARKET_DATA,
                    message=f"No bar data for {instrument_id}",
                )
            ]

        bar = bars[instrument_id]

        # Dispatch based on intent
        if signal.intent == SignalIntent.OPEN:
            return self._execute_open(signal, bar)
        elif signal.intent == SignalIntent.CLOSE:
            return self._execute_close(signal, bar)
        elif signal.intent == SignalIntent.ROLL:
            return self._execute_roll(signal, bars)
        else:
            return [
                create_reject_event(
                    signal=signal,
                    reason=RejectReason.INVALID_ORDER,
                    message=f"Unknown signal intent: {signal.intent}",
                )
            ]

    def _execute_open(
        self,
        signal: Signal,
        bar: pd.Series,
    ) -> List[ExecutionEvent]:
        """Execute an OPEN (entry) signal."""
        tradable = signal.tradable
        direction = signal.direction
        quantity = abs(int(signal.target_position))

        if quantity == 0:
            return [
                create_reject_event(
                    signal=signal,
                    reason=RejectReason.INVALID_ORDER,
                    message="Cannot open position with quantity 0",
                )
            ]

        # Determine if aggressive (for slippage calculation)
        # Use explicit execution_style if provided, otherwise fall back to strategy inference
        if signal.execution_style is not None:
            is_aggressive = signal.execution_style == "aggressive"
        else:
            # Legacy fallback: infer from strategy name (will be removed after migration)
            is_aggressive = "momentum" in signal.strategy.lower() or "mom" in signal.strategy.lower()

        # Calculate slippage
        slippage = self.slippage_model(
            direction=direction,
            bar=bar,
            product=self.product,
            is_aggressive=is_aggressive,
        )

        # Calculate commission
        is_spread = tradable.execution_mode in (ExecutionMode.SYNTHETIC, ExecutionMode.NATIVE)
        commission = self.commission_model(
            quantity=quantity,
            product=self.product,
            is_spread=is_spread,
        )

        # Fill price
        spread_close = bar["close"]
        fill_price = spread_close + slippage
        slippage_dollars = abs(slippage) * quantity * self.product.multiplier

        # Create fill event based on tradable type
        # Use signal.timestamp for backtest correctness (not datetime.now())
        if tradable.is_spread:
            return [
                self._create_spread_fill_event(
                    signal=signal,
                    bar=bar,
                    fill_price=fill_price,
                    quantity=quantity,
                    commission=commission,
                    slippage_dollars=slippage_dollars,
                    timestamp=signal.timestamp,
                )
            ]
        else:
            return [
                self._create_single_fill_event(
                    signal=signal,
                    fill_price=fill_price,
                    quantity=quantity,
                    commission=commission,
                    slippage_dollars=slippage_dollars,
                    timestamp=signal.timestamp,
                )
            ]

    def _execute_close(
        self,
        signal: Signal,
        bar: pd.Series,
    ) -> List[ExecutionEvent]:
        """Execute a CLOSE (exit) signal."""
        tradable = signal.tradable
        # For exits, direction is opposite of what we're closing
        # But slippage model needs the exit direction
        # We don't know the position direction from signal alone
        # Convention: CLOSE signals have target_position=0, need to infer from context
        # For now, assume we're selling (direction=-1 for slippage)
        exit_direction = -1  # Pessimistic: assume adverse slippage

        # Note: In real usage, the runner would pass position direction
        # For now, we use a reasonable default

        # Calculate slippage (exits are typically less aggressive)
        slippage = self.slippage_model(
            direction=exit_direction,
            bar=bar,
            product=self.product,
            is_aggressive=False,
        )

        # For exits, we need quantity from the position (not signal)
        # Convention: signal.metadata may contain 'quantity' or we default to 1
        quantity = signal.metadata.get("quantity", 1)

        # Calculate commission
        is_spread = tradable.execution_mode in (ExecutionMode.SYNTHETIC, ExecutionMode.NATIVE)
        commission = self.commission_model(
            quantity=quantity,
            product=self.product,
            is_spread=is_spread,
        )

        # Fill price
        spread_close = bar["close"]
        fill_price = spread_close + slippage
        slippage_dollars = abs(slippage) * quantity * self.product.multiplier

        # Create fill event
        # Use signal.timestamp for backtest correctness (not datetime.now())
        if tradable.is_spread:
            return [
                self._create_spread_fill_event(
                    signal=signal,
                    bar=bar,
                    fill_price=fill_price,
                    quantity=quantity,
                    commission=commission,
                    slippage_dollars=slippage_dollars,
                    timestamp=signal.timestamp,
                )
            ]
        else:
            return [
                self._create_single_fill_event(
                    signal=signal,
                    fill_price=fill_price,
                    quantity=quantity,
                    commission=commission,
                    slippage_dollars=slippage_dollars,
                    timestamp=signal.timestamp,
                )
            ]

    def _execute_roll(
        self,
        signal: Signal,
        bars: Dict[str, pd.Series],
    ) -> List[ExecutionEvent]:
        """
        Execute a ROLL signal (atomic exit + entry).

        Returns two FillEvents: one for closing current, one for opening new.
        """
        # Validate roll_to exists
        if signal.roll_to is None:
            return [
                create_reject_event(
                    signal=signal,
                    reason=RejectReason.INVALID_ORDER,
                    message="ROLL signal missing roll_to tradable",
                )
            ]

        # Validate we have bars for both instruments
        current_id = signal.tradable.instrument_id
        new_id = signal.roll_to.instrument_id

        if current_id not in bars:
            return [
                create_reject_event(
                    signal=signal,
                    reason=RejectReason.NO_MARKET_DATA,
                    message=f"No bar data for current position {current_id}",
                )
            ]

        if new_id not in bars:
            return [
                create_reject_event(
                    signal=signal,
                    reason=RejectReason.NO_MARKET_DATA,
                    message=f"No bar data for roll target {new_id}",
                )
            ]

        # Get quantity from metadata (position size being rolled)
        quantity = signal.metadata.get("quantity", 1)
        direction = signal.metadata.get("direction", 1)

        # === EXIT LEG (closing current position) ===
        current_bar = bars[current_id]
        exit_slippage = self.slippage_model(
            direction=-direction,  # Closing is opposite direction
            bar=current_bar,
            product=self.product,
            is_aggressive=False,
        )

        is_exit_spread = signal.tradable.is_spread
        exit_commission = self.commission_model(
            quantity=quantity,
            product=self.product,
            is_spread=is_exit_spread,
        )

        exit_fill_price = current_bar["close"] + exit_slippage
        exit_slippage_dollars = abs(exit_slippage) * quantity * self.product.multiplier

        # Build exit leg fills
        exit_leg_fills = self._build_leg_fills(
            tradable=signal.tradable,
            bar=current_bar,
            fill_price=exit_fill_price,
            quantity=quantity,
            commission=exit_commission,
            direction=-direction,  # Closing flips direction
        )

        # === ENTRY LEG (opening new position) ===
        new_bar = bars[new_id]
        entry_slippage = self.slippage_model(
            direction=direction,
            bar=new_bar,
            product=self.product,
            is_aggressive=False,
        )

        is_entry_spread = signal.roll_to.is_spread
        entry_commission = self.commission_model(
            quantity=quantity,
            product=self.product,
            is_spread=is_entry_spread,
        )

        entry_fill_price = new_bar["close"] + entry_slippage
        entry_slippage_dollars = abs(entry_slippage) * quantity * self.product.multiplier

        # Build entry leg fills
        entry_leg_fills = self._build_leg_fills(
            tradable=signal.roll_to,
            bar=new_bar,
            fill_price=entry_fill_price,
            quantity=quantity,
            commission=entry_commission,
            direction=direction,
        )

        # Return single atomic RollFillEvent
        roll_event = create_roll_fill_event(
            signal=signal,
            exit_tradable=signal.tradable,
            exit_fill_price=exit_fill_price,
            exit_quantity=quantity,
            exit_leg_fills=exit_leg_fills,
            exit_commission=exit_commission,
            exit_slippage=exit_slippage_dollars,
            entry_tradable=signal.roll_to,
            entry_fill_price=entry_fill_price,
            entry_quantity=quantity,
            entry_leg_fills=entry_leg_fills,
            entry_commission=entry_commission,
            entry_slippage=entry_slippage_dollars,
            timestamp=signal.timestamp,
        )

        return [roll_event]

    def _create_spread_fill_event(
        self,
        signal: Signal,
        bar: pd.Series,
        fill_price: float,
        quantity: int,
        commission: float,
        slippage_dollars: float,
        timestamp: Optional[datetime] = None,
    ) -> FillEvent:
        """Create a FillEvent for a spread tradable."""
        tradable = signal.tradable

        # Extract or estimate leg prices
        front_close = bar.get("front_close", fill_price + 0.5)
        back_close = bar.get("back_close", front_close - fill_price)

        # Create leg fills - commission is split across legs
        leg_fills = []
        if tradable.legs:
            num_legs = len(tradable.legs)
            for i, leg_def in enumerate(tradable.legs):
                leg_price = front_close if i == 0 else back_close
                side = "BUY" if leg_def.ratio > 0 else "SELL"
                leg_fill = LegFill(
                    instrument_id=leg_def.instrument.instrument_id,
                    side=side,
                    quantity=abs(leg_def.ratio) * quantity,
                    price=float(leg_price),
                    commission=commission / num_legs,
                )
                leg_fills.append(leg_fill)
        else:
            # Single instrument treated as one leg
            side = "BUY" if signal.direction > 0 else "SELL"
            leg_fill = LegFill(
                instrument_id=tradable.instrument_id,
                side=side,
                quantity=quantity,
                price=fill_price,
                commission=commission,
            )
            leg_fills.append(leg_fill)

        return create_fill_event(
            signal=signal,
            quantity_filled=quantity,
            fill_price=fill_price,
            leg_fills=leg_fills,
            timestamp=timestamp,
            slippage=slippage_dollars,
        )

    def _create_single_fill_event(
        self,
        signal: Signal,
        fill_price: float,
        quantity: int,
        commission: float,
        slippage_dollars: float,
        timestamp: Optional[datetime] = None,
    ) -> FillEvent:
        """Create a FillEvent for a single instrument."""
        side = "BUY" if signal.direction > 0 else "SELL"
        leg_fill = LegFill(
            instrument_id=signal.tradable.instrument_id,
            side=side,
            quantity=quantity,
            price=fill_price,
            commission=commission,
        )
        return create_fill_event(
            signal=signal,
            quantity_filled=quantity,
            fill_price=fill_price,
            leg_fills=[leg_fill],
            timestamp=timestamp,
            slippage=slippage_dollars,
        )

    def _build_leg_fills(
        self,
        tradable: Tradable,
        bar: pd.Series,
        fill_price: float,
        quantity: int,
        commission: float,
        direction: int,
    ) -> List[LegFill]:
        """
        Build leg fills for a tradable (spread or single).

        Parameters
        ----------
        tradable : Tradable
            The instrument being filled
        bar : pd.Series
            Bar data with optional front_close/back_close
        fill_price : float
            Composite fill price
        quantity : int
            Quantity filled
        commission : float
            Total commission for all legs
        direction : int
            1 for long, -1 for short/closing

        Returns
        -------
        list[LegFill]
            Leg-level fill details
        """
        leg_fills = []

        if tradable.legs:
            # Spread: extract or estimate leg prices
            front_close = bar.get("front_close", fill_price + 0.5)
            back_close = bar.get("back_close", front_close - fill_price)
            num_legs = len(tradable.legs)

            for i, leg_def in enumerate(tradable.legs):
                leg_price = front_close if i == 0 else back_close
                # Leg ratio sign * direction determines side
                # Long spread (dir=1) with ratio=1 (front) -> BUY
                # Long spread (dir=1) with ratio=-1 (back) -> SELL
                # Exit (dir=-1) with ratio=1 (front) -> SELL
                effective_direction = leg_def.ratio * direction
                side = "BUY" if effective_direction > 0 else "SELL"
                leg_fill = LegFill(
                    instrument_id=leg_def.instrument.instrument_id,
                    side=side,
                    quantity=abs(leg_def.ratio) * quantity,
                    price=float(leg_price),
                    commission=commission / num_legs,
                )
                leg_fills.append(leg_fill)
        else:
            # Single instrument
            side = "BUY" if direction > 0 else "SELL"
            leg_fill = LegFill(
                instrument_id=tradable.instrument_id,
                side=side,
                quantity=quantity,
                price=fill_price,
                commission=commission,
            )
            leg_fills.append(leg_fill)

        return leg_fills

    def mark_to_market(
        self,
        instrument_ids: List[str],
        bars: Dict[str, pd.Series],
    ) -> Dict[str, float]:
        """
        Get current mark prices for instruments.

        For spreads, uses the 'close' price from the bar.
        For single instruments, uses 'close' price.

        Parameters
        ----------
        instrument_ids : list[str]
            Instruments to price
        bars : dict[str, pd.Series]
            Current bars by instrument_id

        Returns
        -------
        dict[str, float]
            Current prices by instrument_id
        """
        prices = {}
        for instrument_id in instrument_ids:
            if instrument_id in bars:
                bar = bars[instrument_id]
                prices[instrument_id] = float(bar["close"])
        return prices


# =============================================================================
# Factory Functions
# =============================================================================


def create_event_executor(
    product_root: str = "CL",
    config: Optional[ExecutorConfig] = None,
    catalog: Optional[ProductCatalog] = None,
) -> EventExecutor:
    """
    Create an EventExecutor with product-specific defaults.

    Parameters
    ----------
    product_root : str
        Product root (CL, RB, HO, etc.)
    config : ExecutorConfig, optional
        Behavioral configuration (defaults to DEFAULT_CONFIG)
    catalog : ProductCatalog, optional
        Product catalog (defaults to global catalog)

    Returns
    -------
    EventExecutor
        Configured executor
    """
    if config is None:
        config = DEFAULT_CONFIG

    if catalog is None:
        catalog = create_default_catalog()

    product = catalog.get(product_root)
    slippage_model = create_slippage_model(config)
    commission_model = PerContractCommissionModel()

    return EventExecutor(
        product=product,
        config=config,
        slippage_model=slippage_model,
        commission_model=commission_model,
    )
