"""
Event-Driven Position Ledger.

Design principles:
- Positions tracked by Tradable, not symbol strings
- apply(event) pattern for all state changes
- Core position fields only - strategy-specific data in metadata
- Separate mark_to_market() for MTM updates
- Roll linkage via logical_position_id

This is a fresh implementation designed around the new type system,
referencing strategies/spread_regime/backtest/position_ledger.py for behavior.

"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import uuid

from common_lib.interfaces.instrument import Tradable
from common_lib.interfaces.signal import Signal, SignalIntent
from common_lib.interfaces.execution_events import (
    ExecutionEvent,
    FillEvent,
    RejectEvent,
    RollFillEvent,
    FillType,
)


class PositionStatus(Enum):
    """Position lifecycle status."""
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class LedgerPosition:
    """
    Core position with essential fields only.

    Strategy-specific data (entry_z, mm_net_pct, etc.) lives in metadata,
    not as dedicated fields. This keeps the core position generic.

    Attributes
    ----------
    position_id : str
        Unique position identifier
    logical_id : str
        Groups positions across rolls (same conceptual trade)
    tradable : Tradable
        What was traded
    strategy : str
        Strategy that generated this position
    direction : int
        +1 for long, -1 for short
    quantity : int
        Number of contracts/spreads
    entry_time : datetime
        When position was opened
    entry_price : float
        Entry fill price
    status : PositionStatus
        OPEN or CLOSED
    current_price : float
        Current mark price
    unrealized_pnl : float
        Current unrealized P&L in dollars
    last_update : datetime
        Last MTM update time
    total_commission : float
        Accumulated commission
    total_slippage : float
        Accumulated slippage
    exit_time : datetime, optional
        When position was closed
    exit_price : float, optional
        Exit fill price
    exit_reason : str, optional
        Why position was closed
    realized_pnl : float, optional
        Final realized P&L (before costs)
    rolled_from : str, optional
        Position ID this was rolled from
    rolled_to : str, optional
        Position ID this was rolled to
    mae : float
        Maximum Adverse Excursion (worst unrealized loss)
    mfe : float
        Maximum Favorable Excursion (best unrealized gain)
    mae_time : datetime, optional
        When MAE occurred
    mfe_time : datetime, optional
        When MFE occurred
    metadata : dict
        Strategy-specific data from signal
    """
    position_id: str
    logical_id: str
    tradable: Tradable
    strategy: str
    direction: int
    quantity: int
    entry_time: datetime
    entry_price: float
    status: PositionStatus = PositionStatus.OPEN

    # Tracking (mutable during position lifetime)
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    last_update: Optional[datetime] = None

    # Costs
    total_commission: float = 0.0
    total_slippage: float = 0.0

    # Exit details (filled on close)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    realized_pnl: Optional[float] = None

    # Roll linkage
    rolled_from: Optional[str] = None
    rolled_to: Optional[str] = None

    # MAE/MFE tracking
    mae: float = 0.0
    mfe: float = 0.0
    mae_time: Optional[datetime] = None
    mfe_time: Optional[datetime] = None

    # Strategy-specific data
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def instrument_id(self) -> str:
        """Convenience: get tradable's instrument_id."""
        return self.tradable.instrument_id

    @property
    def is_open(self) -> bool:
        """True if position is still open."""
        return self.status == PositionStatus.OPEN

    @property
    def is_closed(self) -> bool:
        """True if position is closed."""
        return self.status == PositionStatus.CLOSED

    @property
    def net_pnl(self) -> float:
        """Net P&L after costs."""
        base_pnl = self.realized_pnl if self.realized_pnl is not None else self.unrealized_pnl
        return base_pnl - self.total_commission - self.total_slippage

    @property
    def hold_duration(self) -> Optional[float]:
        """Hold duration in days."""
        end_time = self.exit_time or self.last_update
        if end_time is None:
            return None
        delta = end_time - self.entry_time
        return delta.total_seconds() / 86400  # Convert to days


class EventLedger:
    """
    Position ledger driven by execution events.

    All state changes happen via apply(event). This ensures:
    - Consistent handling of fills, rejects, partials
    - Clear audit trail (event -> state change)
    - Easy testing (inject events, check state)

    Parameters
    ----------
    multiplier : float
        Contract multiplier for P&L calculation (default 1000 for CL)
    commission_per_side : float
        Commission per contract per side (default 2.50)

    Examples
    --------
    >>> ledger = EventLedger(multiplier=1000)
    >>>
    >>> # Process execution events
    >>> events = executor.execute(signal, bars)
    >>> for event in events:
    ...     position = ledger.apply(event)
    >>>
    >>> # Update mark-to-market
    >>> ledger.mark_to_market({"CL1_CL2": 0.75}, timestamp)
    >>>
    >>> # Query positions
    >>> open_positions = ledger.get_open_positions()
    >>> total_pnl = ledger.get_total_unrealized_pnl()
    """

    def __init__(
        self,
        multiplier: float = 1000.0,
        commission_per_side: float = 2.50,
    ):
        self.multiplier = multiplier
        self.commission_per_side = commission_per_side

        self._positions: Dict[str, LedgerPosition] = {}
        self._logical_id_counter = 0
        self._events: List[ExecutionEvent] = []  # Audit trail

    def apply(self, event: ExecutionEvent) -> Optional[LedgerPosition]:
        """
        Apply an execution event to the ledger.

        Dispatches to appropriate handler based on event type and signal intent.

        Parameters
        ----------
        event : ExecutionEvent
            FillEvent, RollFillEvent, or RejectEvent to process

        Returns
        -------
        LedgerPosition or None
            Affected position (None for rejects)
        """
        self._events.append(event)

        if isinstance(event, RollFillEvent):
            return self._apply_roll_fill(event)
        elif isinstance(event, FillEvent):
            return self._apply_fill(event)
        elif isinstance(event, RejectEvent):
            return self._apply_reject(event)
        else:
            raise ValueError(f"Unknown event type: {type(event)}")

    def _apply_fill(self, event: FillEvent) -> LedgerPosition:
        """Process a fill event based on signal intent."""
        signal = event.signal
        intent = signal.intent

        if intent == SignalIntent.OPEN:
            return self._open_position(event)
        elif intent == SignalIntent.CLOSE:
            return self._close_position(event)
        elif intent == SignalIntent.ROLL:
            # ROLL signals should come as RollFillEvent, not FillEvent
            raise ValueError(
                f"ROLL intent received as FillEvent. "
                f"Executor should emit RollFillEvent for rolls. "
                f"Signal: {signal.tradable.instrument_id}"
            )
        else:
            raise ValueError(f"Unknown signal intent: {intent}")

    def _apply_reject(self, event: RejectEvent) -> None:
        """Process a rejection event (no state change, just logged)."""
        # Rejects don't change position state, but are recorded in audit trail
        return None

    def _open_position(self, event: FillEvent) -> LedgerPosition:
        """Open a new position from a fill event."""
        signal = event.signal

        # Generate IDs
        position_id = str(uuid.uuid4())[:8]
        self._logical_id_counter += 1
        logical_id = f"L{self._logical_id_counter:06d}"

        # Calculate commission for entry
        entry_commission = self.commission_per_side * event.quantity_filled

        position = LedgerPosition(
            position_id=position_id,
            logical_id=logical_id,
            tradable=event.tradable,
            strategy=signal.strategy,
            direction=1 if signal.target_position > 0 else -1,
            quantity=event.quantity_filled,
            entry_time=event.timestamp,
            entry_price=event.fill_price,
            current_price=event.fill_price,
            last_update=event.timestamp,
            total_commission=entry_commission,
            total_slippage=event.slippage,
            metadata=dict(signal.metadata),  # Copy metadata from signal
        )

        self._positions[position_id] = position
        return position

    def _close_position(self, event: FillEvent) -> LedgerPosition:
        """Close an existing position from a fill event."""
        signal = event.signal

        # Find the open position for this tradable
        position = self._find_open_position(event.tradable.instrument_id)
        if position is None:
            raise ValueError(f"No open position found for {event.tradable.instrument_id}")

        # Calculate realized P&L
        price_change = event.fill_price - position.entry_price
        gross_pnl = position.direction * price_change * position.quantity * self.multiplier

        # Add exit commission
        exit_commission = self.commission_per_side * position.quantity
        position.total_commission += exit_commission
        position.total_slippage += event.slippage

        # Update position
        position.status = PositionStatus.CLOSED
        position.exit_time = event.timestamp
        position.exit_price = event.fill_price
        position.exit_reason = signal.reason
        position.realized_pnl = gross_pnl
        position.current_price = event.fill_price
        position.last_update = event.timestamp

        # Merge exit signal metadata into position (preserves entry metadata)
        if signal.metadata:
            position.metadata.update(signal.metadata)

        return position

    def _apply_roll_fill(self, event: RollFillEvent) -> LedgerPosition:
        """
        Apply a roll fill event: close current position, open new with linkage.

        Unlike processing two separate FillEvents, this uses the proper
        exit_fill_price for P&L calculation and entry_fill_price for the new position.
        """
        signal = event.signal

        # Find and close the old position
        old_position = self._find_open_position(event.exit_tradable.instrument_id)
        if old_position is None:
            raise ValueError(f"No open position to roll for {event.exit_tradable.instrument_id}")

        # Close old position with EXIT price (not entry price)
        price_change = event.exit_fill_price - old_position.entry_price
        gross_pnl = old_position.direction * price_change * old_position.quantity * self.multiplier

        old_position.total_commission += event.exit_commission
        old_position.total_slippage += event.exit_slippage

        old_position.status = PositionStatus.CLOSED
        old_position.exit_time = event.timestamp
        old_position.exit_price = event.exit_fill_price
        old_position.exit_reason = "roll"
        old_position.realized_pnl = gross_pnl
        old_position.current_price = event.exit_fill_price
        old_position.last_update = event.timestamp

        # Open new position with ENTRY price (potentially different from exit)
        new_position_id = str(uuid.uuid4())[:8]

        # Build new metadata: inherit from old, but MUST update front_expiry to prevent infinite rolls
        new_metadata = dict(old_position.metadata)
        new_front_expiry = signal.metadata.get("new_front_expiry")
        if new_front_expiry is None:
            raise ValueError(
                f"ROLL signal for {signal.tradable.instrument_id} missing new_front_expiry in metadata. "
                f"This would cause infinite roll loops. Signal metadata: {signal.metadata}"
            )
        new_metadata["front_expiry"] = new_front_expiry

        new_position = LedgerPosition(
            position_id=new_position_id,
            logical_id=old_position.logical_id,  # Same logical trade
            tradable=event.entry_tradable,
            strategy=old_position.strategy,
            direction=old_position.direction,  # Same direction
            quantity=event.entry_quantity,
            entry_time=event.timestamp,
            entry_price=event.entry_fill_price,  # Use actual entry fill price
            current_price=event.entry_fill_price,
            last_update=event.timestamp,
            total_commission=event.entry_commission,
            total_slippage=event.entry_slippage,
            rolled_from=old_position.position_id,
            metadata=new_metadata,  # Inherit metadata with updated front_expiry
        )

        # Link positions
        old_position.rolled_to = new_position_id
        self._positions[new_position_id] = new_position

        return new_position

    def _find_open_position(self, instrument_id: str) -> Optional[LedgerPosition]:
        """Find an open position by instrument ID."""
        for position in self._positions.values():
            if position.is_open and position.instrument_id == instrument_id:
                return position
        return None

    def mark_to_market(
        self,
        prices: Dict[str, float],
        timestamp: datetime,
    ) -> None:
        """
        Update mark-to-market for all open positions.

        Parameters
        ----------
        prices : dict[str, float]
            Current prices by instrument_id
        timestamp : datetime
            Current timestamp
        """
        for position in self._positions.values():
            if not position.is_open:
                continue

            if position.instrument_id not in prices:
                continue

            current_price = prices[position.instrument_id]

            # Calculate unrealized P&L
            price_change = current_price - position.entry_price
            position.unrealized_pnl = (
                position.direction * price_change * position.quantity * self.multiplier
            )

            # Update MAE/MFE
            if position.unrealized_pnl < position.mae:
                position.mae = position.unrealized_pnl
                position.mae_time = timestamp
            if position.unrealized_pnl > position.mfe:
                position.mfe = position.unrealized_pnl
                position.mfe_time = timestamp

            position.current_price = current_price
            position.last_update = timestamp

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_position(self, position_id: str) -> Optional[LedgerPosition]:
        """Get position by ID."""
        return self._positions.get(position_id)

    def get_open_positions(
        self,
        instrument_id: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> List[LedgerPosition]:
        """
        Get open positions with optional filters.

        Parameters
        ----------
        instrument_id : str, optional
            Filter by instrument ID
        strategy : str, optional
            Filter by strategy

        Returns
        -------
        list[LedgerPosition]
            Matching open positions
        """
        positions = [p for p in self._positions.values() if p.is_open]

        if instrument_id:
            positions = [p for p in positions if p.instrument_id == instrument_id]
        if strategy:
            # Normalize both sides for alias-drift protection
            sid = (strategy or "").strip().upper()
            positions = [p for p in positions if (p.strategy or "").strip().upper() == sid]

        return positions

    def get_closed_positions(self) -> List[LedgerPosition]:
        """Get all closed positions."""
        return [p for p in self._positions.values() if p.is_closed]

    def get_all_positions(self) -> List[LedgerPosition]:
        """Get all positions."""
        return list(self._positions.values())

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across open positions."""
        return sum(p.unrealized_pnl for p in self.get_open_positions())

    def get_total_realized_pnl(self) -> float:
        """Get total realized P&L (net of costs) across closed positions."""
        return sum(p.net_pnl for p in self.get_closed_positions())

    def get_events(self) -> List[ExecutionEvent]:
        """Get audit trail of all events."""
        return list(self._events)

    def has_open_position(self, instrument_id: str) -> bool:
        """Check if there's an open position for an instrument."""
        return self._find_open_position(instrument_id) is not None

    # =========================================================================
    # Roll Helpers
    # =========================================================================

    def get_roll_chain(self, position_id: str) -> List[LedgerPosition]:
        """
        Get all positions in a roll chain.

        Traces back through rolled_from and forward through rolled_to
        to get complete chain of positions for a logical trade.
        """
        chain = []

        # Find the start of the chain (no rolled_from)
        current = self._positions.get(position_id)
        visited_backward = set()
        while current and current.rolled_from and current.position_id not in visited_backward:
            visited_backward.add(current.position_id)
            current = self._positions.get(current.rolled_from)

        # Traverse forward from start (fresh seen set)
        visited_forward = set()
        while current and current.position_id not in visited_forward:
            visited_forward.add(current.position_id)
            chain.append(current)
            if current.rolled_to:
                current = self._positions.get(current.rolled_to)
            else:
                break

        return chain

    def get_positions_by_logical_id(self, logical_id: str) -> List[LedgerPosition]:
        """Get all positions sharing a logical ID."""
        return [p for p in self._positions.values() if p.logical_id == logical_id]


# =============================================================================
# Factory Function
# =============================================================================

def create_ledger(
    product: str = "CL",
    multiplier: Optional[float] = None,
    commission_per_side: float = 2.50,
) -> EventLedger:
    """
    Create a ledger with product-specific defaults.

    Parameters
    ----------
    product : str
        Product code (CL, RB, HO, etc.)
    multiplier : float, optional
        Contract multiplier (auto-detected from product if not provided)
    commission_per_side : float
        Commission per contract per side

    Returns
    -------
    EventLedger
        Configured ledger
    """
    # Default multipliers by product
    MULTIPLIERS = {
        "CL": 1000.0,
        "RB": 42000.0,
        "HO": 42000.0,
        "NG": 10000.0,
        "LC": 1.0,  # Lithium carbonate (GFEX)
    }

    if multiplier is None:
        multiplier = MULTIPLIERS.get(product, 1000.0)

    return EventLedger(
        multiplier=multiplier,
        commission_per_side=commission_per_side,
    )
