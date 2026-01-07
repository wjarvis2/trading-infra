"""
Signal Types for Position Intent.

Design principles:
- Signal = position intent (what you want to be), not trade direction
- target_position expresses desired state (+1, -1, 0, or contract count)
- SignalIntent distinguishes OPEN/CLOSE/ROLL without enum explosion
- Metadata carries strategy-specific context without polluting core types

This replaces the alpha-style Signal (strength/confidence) with position-intent.

"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional
from datetime import datetime
from enum import Enum

from .instrument import Tradable


# Explicit execution style to avoid stringly-typed strategy inference
ExecutionStyle = Literal["aggressive", "conservative"]


class SignalIntent(Enum):
    """
    What kind of position change is intended.

    OPEN: Establish or increase position (target_position != 0)
    CLOSE: Reduce or flatten position (typically target_position = 0)
    ROLL: Close current tradable, open new one (atomic replacement)
    """
    OPEN = "open"
    CLOSE = "close"
    ROLL = "roll"


@dataclass(frozen=True)
class Signal:
    """
    Position intent signal.

    Expresses desired position state, not trade direction.
    Runner computes delta from current position and executes.

    Attributes
    ----------
    timestamp : datetime
        When the signal was generated
    tradable : Tradable
        What to trade (includes execution semantics)
    intent : SignalIntent
        Type of position change (OPEN, CLOSE, ROLL)
    target_position : float
        Desired position after execution:
        - +1.0 = long 1 contract
        - -1.0 = short 1 contract
        - 0.0 = flat
        - +2.0 = long 2 contracts (scale in)
    strategy : str
        Strategy identifier (e.g., "MR", "MOM")
    reason : str
        Human-readable reason (e.g., "z_score_entry", "max_hold_exit")
    roll_to : Tradable, optional
        For ROLL intent: the new tradable to enter
    metadata : dict
        Strategy-specific context (entry z-score, regime, etc.)
    confidence : float
        Conviction level 0.0-1.0 for position sizing (default 1.0 = full)
    magnitude : float, optional
        Expected move size in strategy-specific units
    expires_at : datetime, optional
        Signal validity window (None = no expiry)

    Examples
    --------
    Entry signal (go long 1):
    >>> Signal(
    ...     timestamp=now,
    ...     tradable=cl1_cl2_spread,
    ...     intent=SignalIntent.OPEN,
    ...     target_position=+1.0,
    ...     strategy="MR",
    ...     reason="z_score_below_minus_2",
    ... )

    Exit signal (flatten):
    >>> Signal(
    ...     timestamp=now,
    ...     tradable=cl1_cl2_spread,
    ...     intent=SignalIntent.CLOSE,
    ...     target_position=0.0,
    ...     strategy="MR",
    ...     reason="z_score_crossed_zero",
    ... )

    Roll signal (close old spread, open new):
    >>> Signal(
    ...     timestamp=now,
    ...     tradable=cl1_cl2_spread,
    ...     intent=SignalIntent.ROLL,
    ...     target_position=0.0,  # exit current
    ...     strategy="MR",
    ...     reason="front_month_expiry",
    ...     roll_to=cl2_cl3_spread,
    ... )

    Scale-in signal (add to existing long):
    >>> Signal(
    ...     timestamp=now,
    ...     tradable=cl1_cl2_spread,
    ...     intent=SignalIntent.OPEN,
    ...     target_position=+2.0,  # was +1, now want +2
    ...     strategy="MR",
    ...     reason="confirmation_signal",
    ... )
    """
    timestamp: datetime
    tradable: Tradable
    intent: SignalIntent
    target_position: float
    strategy: str
    reason: str
    roll_to: Optional[Tradable] = None
    execution_style: Optional[ExecutionStyle] = None  # "aggressive" or "conservative"
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Conviction/sizing hints (LEAN-inspired)
    confidence: float = 1.0  # 0.0-1.0, for position sizing (1.0 = full conviction)
    magnitude: Optional[float] = None  # Expected move size (strategy-specific units)
    expires_at: Optional[datetime] = None  # Signal validity window

    def __post_init__(self):
        # Validate ROLL has roll_to
        if self.intent == SignalIntent.ROLL and self.roll_to is None:
            raise ValueError("ROLL intent requires roll_to tradable")

    @property
    def instrument_id(self) -> str:
        """Convenience: get the tradable's instrument_id."""
        return self.tradable.instrument_id

    @property
    def is_entry(self) -> bool:
        """True if this opens or adds to a position."""
        return self.intent == SignalIntent.OPEN and self.target_position != 0

    @property
    def is_exit(self) -> bool:
        """True if this closes a position."""
        return self.intent == SignalIntent.CLOSE and self.target_position == 0

    @property
    def is_roll(self) -> bool:
        """True if this is a roll (atomic exit + entry)."""
        return self.intent == SignalIntent.ROLL

    @property
    def direction(self) -> int:
        """
        Direction implied by target position.

        Returns +1 for long, -1 for short, 0 for flat.
        """
        if self.target_position > 0:
            return 1
        elif self.target_position < 0:
            return -1
        return 0


# =============================================================================
# Factory Functions
# =============================================================================

def create_entry_signal(
    tradable: Tradable,
    direction: int,
    strategy: str,
    reason: str,
    timestamp: Optional[datetime] = None,
    quantity: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
    execution_style: Optional[ExecutionStyle] = None,
    confidence: float = 1.0,
    magnitude: Optional[float] = None,
    expires_at: Optional[datetime] = None,
) -> Signal:
    """
    Create an entry signal.

    Parameters
    ----------
    tradable : Tradable
        What to trade
    direction : int
        +1 for long, -1 for short
    strategy : str
        Strategy identifier (e.g., "MR", "MOM")
    reason : str
        Human-readable reason
    timestamp : datetime, optional
        Signal time (defaults to now)
    quantity : float
        Number of contracts (default 1)
    metadata : dict, optional
        Additional context
    execution_style : "aggressive" | "conservative", optional
        Explicit execution style. If None, executor may infer from strategy.
    confidence : float
        Conviction level 0.0-1.0 for position sizing (default 1.0)
    magnitude : float, optional
        Expected move size in strategy-specific units
    expires_at : datetime, optional
        Signal validity window

    Returns
    -------
    Signal
        Entry signal
    """
    return Signal(
        timestamp=timestamp or datetime.now(),
        tradable=tradable,
        intent=SignalIntent.OPEN,
        target_position=direction * quantity,
        strategy=strategy,
        reason=reason,
        execution_style=execution_style,
        metadata=metadata or {},
        confidence=confidence,
        magnitude=magnitude,
        expires_at=expires_at,
    )


def create_exit_signal(
    tradable: Tradable,
    strategy: str,
    reason: str,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None,
    execution_style: Optional[ExecutionStyle] = None,
    confidence: float = 1.0,
    magnitude: Optional[float] = None,
    expires_at: Optional[datetime] = None,
) -> Signal:
    """
    Create an exit signal (flatten position).

    Parameters
    ----------
    tradable : Tradable
        What to exit
    strategy : str
        Strategy identifier (e.g., "MR", "MOM")
    reason : str
        Human-readable reason
    timestamp : datetime, optional
        Signal time (defaults to now)
    metadata : dict, optional
        Additional context
    execution_style : "aggressive" | "conservative", optional
        Explicit execution style. If None, executor may infer from strategy.
    confidence : float
        Conviction level 0.0-1.0 for position sizing (default 1.0)
    magnitude : float, optional
        Expected move size in strategy-specific units
    expires_at : datetime, optional
        Signal validity window

    Returns
    -------
    Signal
        Exit signal (target_position=0)
    """
    return Signal(
        timestamp=timestamp or datetime.now(),
        tradable=tradable,
        intent=SignalIntent.CLOSE,
        target_position=0.0,
        strategy=strategy,
        reason=reason,
        execution_style=execution_style,
        metadata=metadata or {},
        confidence=confidence,
        magnitude=magnitude,
        expires_at=expires_at,
    )


def create_roll_signal(
    tradable: Tradable,
    roll_to: Tradable,
    strategy: str,
    reason: str,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None,
    execution_style: Optional[ExecutionStyle] = None,
    confidence: float = 1.0,
    magnitude: Optional[float] = None,
    expires_at: Optional[datetime] = None,
) -> Signal:
    """
    Create a roll signal (atomic exit + entry).

    Parameters
    ----------
    tradable : Tradable
        Current position to exit
    roll_to : Tradable
        New position to enter
    strategy : str
        Strategy identifier (e.g., "MR", "MOM")
    reason : str
        Human-readable reason (e.g., "front_month_expiry")
    timestamp : datetime, optional
        Signal time (defaults to now)
    metadata : dict, optional
        Additional context
    execution_style : "aggressive" | "conservative", optional
        Explicit execution style. If None, executor may infer from strategy.
    confidence : float
        Conviction level 0.0-1.0 for position sizing (default 1.0)
    magnitude : float, optional
        Expected move size in strategy-specific units
    expires_at : datetime, optional
        Signal validity window

    Returns
    -------
    Signal
        Roll signal
    """
    return Signal(
        timestamp=timestamp or datetime.now(),
        tradable=tradable,
        intent=SignalIntent.ROLL,
        target_position=0.0,  # Exit current
        strategy=strategy,
        reason=reason,
        roll_to=roll_to,
        execution_style=execution_style,
        metadata=metadata or {},
        confidence=confidence,
        magnitude=magnitude,
        expires_at=expires_at,
    )
