"""
Execution Event Types.

Design principles:
- Events stream from executor to ledger: executor.execute(signal) -> [events]
- Each event is immutable and carries full context for ledger.apply(event)
- Multi-leg fills are atomic - one FillEvent with all LegFill details
- Rejects and partials are explicit event types (not null or residual signals)
- Events reference Tradable (not symbol strings) for type safety

Event hierarchy:
- ExecutionEvent (base)
  - FillEvent (complete or partial fill)
  - RejectEvent (execution rejected)

"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

from .instrument import Tradable
from .signal import Signal


class RejectReason(Enum):
    """Why an execution was rejected."""
    NO_MARKET_DATA = "no_market_data"
    INSUFFICIENT_LIQUIDITY = "insufficient_liquidity"
    RISK_LIMIT = "risk_limit"
    TIMEOUT = "timeout"
    BROKER_REJECT = "broker_reject"
    INVALID_ORDER = "invalid_order"
    PRICE_LIMIT = "price_limit"


class FillType(Enum):
    """Type of fill."""
    FULL = "full"        # Complete fill
    PARTIAL = "partial"  # Partial fill (more to come)
    FINAL = "final"      # Final fill of a partial sequence


@dataclass(frozen=True)
class LegFill:
    """
    Fill details for one leg of a tradable.

    For single-leg tradables, there's one LegFill.
    For synthetic spreads, there's one per leg.

    Attributes
    ----------
    instrument_id : str
        The leg's instrument identifier (e.g., "CLX24")
    side : str
        "BUY" or "SELL"
    quantity : int
        Number of contracts filled
    price : float
        Fill price per contract
    commission : float
        Commission for this leg
    """
    instrument_id: str
    side: Literal["BUY", "SELL"]
    quantity: int
    price: float
    commission: float = 0.0


@dataclass(frozen=True)
class FillEvent:
    """
    Execution fill event.

    Represents a successful execution - complete or partial.
    For multi-leg tradables, all leg fills are atomic in one event.

    Attributes
    ----------
    event_id : str
        Unique event identifier
    timestamp : datetime
        When the fill occurred
    signal : Signal
        Original signal that triggered this fill
    tradable : Tradable
        What was traded
    fill_type : FillType
        FULL, PARTIAL, or FINAL
    quantity_filled : int
        Total quantity filled (in tradable units, e.g., spread contracts)
    quantity_remaining : int
        Remaining unfilled quantity (0 for FULL fills)
    fill_price : float
        Composite fill price (for spreads: front - back)
    leg_fills : tuple[LegFill, ...]
        Individual leg fill details
    total_commission : float
        Total commission across all legs
    slippage : float
        Estimated slippage from theoretical price
    metadata : dict
        Additional execution details

    Examples
    --------
    Single-leg fill:
    >>> fill = FillEvent(
    ...     event_id="fill_001",
    ...     timestamp=now,
    ...     signal=signal,
    ...     tradable=cl_tradable,
    ...     fill_type=FillType.FULL,
    ...     quantity_filled=1,
    ...     quantity_remaining=0,
    ...     fill_price=72.50,
    ...     leg_fills=(LegFill("CLX24", "BUY", 1, 72.50, 2.00),),
    ...     total_commission=2.00,
    ...     slippage=0.02,
    ... )

    Multi-leg spread fill:
    >>> fill = FillEvent(
    ...     event_id="fill_002",
    ...     timestamp=now,
    ...     signal=signal,
    ...     tradable=spread_tradable,
    ...     fill_type=FillType.FULL,
    ...     quantity_filled=1,
    ...     quantity_remaining=0,
    ...     fill_price=-0.50,  # Front - Back
    ...     leg_fills=(
    ...         LegFill("CLX24", "BUY", 1, 72.50, 2.00),
    ...         LegFill("CLZ24", "SELL", 1, 73.00, 2.00),
    ...     ),
    ...     total_commission=4.00,
    ...     slippage=0.01,
    ... )
    """
    event_id: str
    timestamp: datetime
    signal: Signal
    tradable: Tradable
    fill_type: FillType
    quantity_filled: int
    quantity_remaining: int
    fill_price: float
    leg_fills: Tuple[LegFill, ...]
    total_commission: float
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """True if this is a complete fill (no remaining quantity)."""
        return self.fill_type == FillType.FULL or self.quantity_remaining == 0

    @property
    def is_partial(self) -> bool:
        """True if this is a partial fill with more expected."""
        return self.fill_type == FillType.PARTIAL and self.quantity_remaining > 0

    @property
    def instrument_id(self) -> str:
        """Convenience: get the tradable's instrument_id."""
        return self.tradable.instrument_id

    @property
    def total_notional(self) -> float:
        """Total notional value (price * quantity)."""
        return abs(self.fill_price * self.quantity_filled)


@dataclass(frozen=True)
class RejectEvent:
    """
    Execution rejection event.

    Represents a failed execution attempt.

    Attributes
    ----------
    event_id : str
        Unique event identifier
    timestamp : datetime
        When the rejection occurred
    signal : Signal
        Original signal that was rejected
    tradable : Tradable
        What we tried to trade
    reason : RejectReason
        Why the execution was rejected
    message : str
        Human-readable rejection message
    metadata : dict
        Additional context (e.g., available liquidity, risk limits)

    Examples
    --------
    >>> reject = RejectEvent(
    ...     event_id="rej_001",
    ...     timestamp=now,
    ...     signal=signal,
    ...     tradable=tradable,
    ...     reason=RejectReason.NO_MARKET_DATA,
    ...     message="No market data for CLX24",
    ... )
    """
    event_id: str
    timestamp: datetime
    signal: Signal
    tradable: Tradable
    reason: RejectReason
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def instrument_id(self) -> str:
        """Convenience: get the tradable's instrument_id."""
        return self.tradable.instrument_id


@dataclass(frozen=True)
class RollFillEvent:
    """
    Roll fill event - atomic position roll from one tradable to another.

    A roll is conceptually one operation but physically two fills:
    - Exit the old position (at exit_fill_price)
    - Enter the new position (at entry_fill_price)

    This event captures both with proper pricing for P&L calculation.
    Unlike emitting two separate FillEvents, this prevents the ledger from
    processing them as independent operations.

    Attributes
    ----------
    event_id : str
        Unique event identifier
    timestamp : datetime
        When the roll occurred
    signal : Signal
        Original ROLL signal that triggered this

    Exit leg (closing old position):
    exit_tradable : Tradable
        The position being closed
    exit_fill_price : float
        Price at which old position was exited (for P&L)
    exit_quantity : int
        Quantity closed
    exit_leg_fills : tuple[LegFill, ...]
        Leg-level fill details for exit
    exit_commission : float
        Commission for exit
    exit_slippage : float
        Slippage on exit

    Entry leg (opening new position):
    entry_tradable : Tradable
        The new position being opened (same as signal.roll_to)
    entry_fill_price : float
        Price at which new position was entered
    entry_quantity : int
        Quantity opened (typically same as exit_quantity)
    entry_leg_fills : tuple[LegFill, ...]
        Leg-level fill details for entry
    entry_commission : float
        Commission for entry
    entry_slippage : float
        Slippage on entry

    Examples
    --------
    >>> roll = RollFillEvent(
    ...     event_id="roll_001",
    ...     timestamp=now,
    ...     signal=roll_signal,
    ...     exit_tradable=old_spread,
    ...     exit_fill_price=-0.45,
    ...     exit_quantity=1,
    ...     exit_leg_fills=(LegFill("CLX24", "SELL", 1, 72.50, 2.0), ...),
    ...     exit_commission=4.0,
    ...     exit_slippage=0.01,
    ...     entry_tradable=new_spread,
    ...     entry_fill_price=-0.52,
    ...     entry_quantity=1,
    ...     entry_leg_fills=(LegFill("CLZ24", "BUY", 1, 73.00, 2.0), ...),
    ...     entry_commission=4.0,
    ...     entry_slippage=0.01,
    ... )
    """
    event_id: str
    timestamp: datetime
    signal: Signal

    # Exit leg (closing old position)
    exit_tradable: Tradable
    exit_fill_price: float
    exit_quantity: int
    exit_leg_fills: Tuple[LegFill, ...]
    exit_commission: float
    exit_slippage: float

    # Entry leg (opening new position)
    entry_tradable: Tradable
    entry_fill_price: float
    entry_quantity: int
    entry_leg_fills: Tuple[LegFill, ...]
    entry_commission: float
    entry_slippage: float

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def exit_instrument_id(self) -> str:
        """Instrument ID of position being closed."""
        return self.exit_tradable.instrument_id

    @property
    def entry_instrument_id(self) -> str:
        """Instrument ID of position being opened."""
        return self.entry_tradable.instrument_id

    @property
    def total_commission(self) -> float:
        """Total commission for both legs of roll."""
        return self.exit_commission + self.entry_commission

    @property
    def total_slippage(self) -> float:
        """Total slippage for both legs of roll."""
        return self.exit_slippage + self.entry_slippage


# =============================================================================
# Type Aliases
# =============================================================================

# Union type for all execution events
ExecutionEvent = FillEvent | RejectEvent | RollFillEvent


# =============================================================================
# Factory Functions
# =============================================================================

def create_fill_event(
    signal: Signal,
    quantity_filled: int,
    fill_price: float,
    leg_fills: List[LegFill],
    timestamp: Optional[datetime] = None,
    quantity_remaining: int = 0,
    slippage: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> FillEvent:
    """
    Create a fill event.

    Parameters
    ----------
    signal : Signal
        Original signal
    quantity_filled : int
        Quantity filled
    fill_price : float
        Composite fill price
    leg_fills : list[LegFill]
        Individual leg fills
    timestamp : datetime, optional
        Fill time (defaults to now)
    quantity_remaining : int
        Remaining quantity (default 0 = full fill)
    slippage : float
        Estimated slippage
    metadata : dict, optional
        Additional context

    Returns
    -------
    FillEvent
        The fill event
    """
    ts = timestamp or datetime.now()
    event_id = f"fill_{signal.tradable.instrument_id}_{ts.strftime('%Y%m%d_%H%M%S_%f')}"

    # Determine fill type
    if quantity_remaining == 0:
        fill_type = FillType.FULL
    else:
        fill_type = FillType.PARTIAL

    return FillEvent(
        event_id=event_id,
        timestamp=ts,
        signal=signal,
        tradable=signal.tradable,
        fill_type=fill_type,
        quantity_filled=quantity_filled,
        quantity_remaining=quantity_remaining,
        fill_price=fill_price,
        leg_fills=tuple(leg_fills),
        total_commission=sum(lf.commission for lf in leg_fills),
        slippage=slippage,
        metadata=metadata or {},
    )


def create_reject_event(
    signal: Signal,
    reason: RejectReason,
    message: str,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> RejectEvent:
    """
    Create a rejection event.

    Parameters
    ----------
    signal : Signal
        Original signal
    reason : RejectReason
        Rejection reason
    message : str
        Human-readable message
    timestamp : datetime, optional
        Rejection time (defaults to now)
    metadata : dict, optional
        Additional context

    Returns
    -------
    RejectEvent
        The rejection event
    """
    ts = timestamp or datetime.now()
    event_id = f"rej_{signal.tradable.instrument_id}_{ts.strftime('%Y%m%d_%H%M%S_%f')}"

    return RejectEvent(
        event_id=event_id,
        timestamp=ts,
        signal=signal,
        tradable=signal.tradable,
        reason=reason,
        message=message,
        metadata=metadata or {},
    )


def create_single_leg_fill(
    signal: Signal,
    side: Literal["BUY", "SELL"],
    quantity: int,
    price: float,
    commission: float = 0.0,
    timestamp: Optional[datetime] = None,
    slippage: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> FillEvent:
    """
    Convenience: create a fill event for single-leg tradable.

    Parameters
    ----------
    signal : Signal
        Original signal
    side : str
        "BUY" or "SELL"
    quantity : int
        Contracts filled
    price : float
        Fill price
    commission : float
        Commission (default 0)
    timestamp : datetime, optional
        Fill time
    slippage : float
        Estimated slippage
    metadata : dict, optional
        Additional context

    Returns
    -------
    FillEvent
        Single-leg fill event
    """
    leg = LegFill(
        instrument_id=signal.tradable.instrument_id,
        side=side,
        quantity=quantity,
        price=price,
        commission=commission,
    )

    return create_fill_event(
        signal=signal,
        quantity_filled=quantity,
        fill_price=price,
        leg_fills=[leg],
        timestamp=timestamp,
        quantity_remaining=0,
        slippage=slippage,
        metadata=metadata,
    )


def create_spread_fill(
    signal: Signal,
    front_price: float,
    back_price: float,
    quantity: int = 1,
    commission_per_leg: float = 2.0,
    timestamp: Optional[datetime] = None,
    slippage: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> FillEvent:
    """
    Convenience: create a fill event for synthetic spread.

    Assumes standard calendar spread: long front, short back.
    Direction is determined by signal.target_position.

    Parameters
    ----------
    signal : Signal
        Original signal (must have Tradable with legs)
    front_price : float
        Front leg fill price
    back_price : float
        Back leg fill price
    quantity : int
        Spread contracts filled (default 1)
    commission_per_leg : float
        Commission per leg (default 2.0)
    timestamp : datetime, optional
        Fill time
    slippage : float
        Estimated slippage
    metadata : dict, optional
        Additional context

    Returns
    -------
    FillEvent
        Spread fill event

    Raises
    ------
    ValueError
        If tradable doesn't have legs
    """
    tradable = signal.tradable
    if not tradable.legs or len(tradable.legs) < 2:
        raise ValueError(f"Tradable {tradable.instrument_id} doesn't have spread legs")

    # Determine direction from signal
    is_long = signal.target_position > 0

    front_leg = tradable.front_leg
    back_leg = tradable.back_leg

    if front_leg is None or back_leg is None:
        raise ValueError(f"Could not identify front/back legs for {tradable.instrument_id}")

    # Long spread = buy front, sell back
    # Short spread = sell front, buy back
    leg_fills = [
        LegFill(
            instrument_id=front_leg.instrument.instrument_id,
            side="BUY" if is_long else "SELL",
            quantity=quantity,
            price=front_price,
            commission=commission_per_leg,
        ),
        LegFill(
            instrument_id=back_leg.instrument.instrument_id,
            side="SELL" if is_long else "BUY",
            quantity=quantity,
            price=back_price,
            commission=commission_per_leg,
        ),
    ]

    # Spread price = front - back
    spread_price = front_price - back_price

    return create_fill_event(
        signal=signal,
        quantity_filled=quantity,
        fill_price=spread_price,
        leg_fills=leg_fills,
        timestamp=timestamp,
        quantity_remaining=0,
        slippage=slippage,
        metadata=metadata,
    )


def create_roll_fill_event(
    signal: Signal,
    exit_tradable: Tradable,
    exit_fill_price: float,
    exit_quantity: int,
    exit_leg_fills: List[LegFill],
    exit_commission: float,
    exit_slippage: float,
    entry_tradable: Tradable,
    entry_fill_price: float,
    entry_quantity: int,
    entry_leg_fills: List[LegFill],
    entry_commission: float,
    entry_slippage: float,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> RollFillEvent:
    """
    Create a roll fill event.

    Parameters
    ----------
    signal : Signal
        Original ROLL signal
    exit_tradable : Tradable
        Position being closed
    exit_fill_price : float
        Exit fill price
    exit_quantity : int
        Quantity exited
    exit_leg_fills : list[LegFill]
        Exit leg fills
    exit_commission : float
        Exit commission
    exit_slippage : float
        Exit slippage in dollars
    entry_tradable : Tradable
        Position being opened (same as signal.roll_to)
    entry_fill_price : float
        Entry fill price
    entry_quantity : int
        Quantity entered
    entry_leg_fills : list[LegFill]
        Entry leg fills
    entry_commission : float
        Entry commission
    entry_slippage : float
        Entry slippage in dollars
    timestamp : datetime, optional
        Roll time (defaults to signal.timestamp)
    metadata : dict, optional
        Additional context

    Returns
    -------
    RollFillEvent
        The roll fill event
    """
    ts = timestamp or signal.timestamp
    if ts is None:
        raise ValueError("RollFillEvent requires timestamp (signal.timestamp is None)")

    event_id = f"roll_{exit_tradable.instrument_id}_to_{entry_tradable.instrument_id}_{ts.strftime('%Y%m%d_%H%M%S_%f')}"

    return RollFillEvent(
        event_id=event_id,
        timestamp=ts,
        signal=signal,
        exit_tradable=exit_tradable,
        exit_fill_price=exit_fill_price,
        exit_quantity=exit_quantity,
        exit_leg_fills=tuple(exit_leg_fills),
        exit_commission=exit_commission,
        exit_slippage=exit_slippage,
        entry_tradable=entry_tradable,
        entry_fill_price=entry_fill_price,
        entry_quantity=entry_quantity,
        entry_leg_fills=tuple(entry_leg_fills),
        entry_commission=entry_commission,
        entry_slippage=entry_slippage,
        metadata=metadata or {},
    )
