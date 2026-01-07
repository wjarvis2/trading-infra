"""
Core Instrument and Tradable Types.

Design principles:
- Instrument = pure identity (WHAT it is)
- Tradable = execution wrapper (HOW you trade it)
- LegDefinition = one leg in a multi-leg tradable
- Clean separation allows same instrument to be traded different ways

"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple
from datetime import date
from enum import Enum


class ExecutionMode(Enum):
    """How the tradable should be executed."""
    SINGLE = "single"        # Single-leg outright (future, option)
    NATIVE = "native"        # Exchange-listed spread/combo
    SYNTHETIC = "synthetic"  # Multi-leg coordinated execution


@dataclass(frozen=True)
class Instrument:
    """
    Pure identity - what IS this thing.

    Immutable. Does not know HOW it's traded.

    Attributes
    ----------
    instrument_id : str
        Unique identifier. Examples:
        - "CLX24" (specific contract)
        - "CL1" (generic front month)
        - "CL1_CL2" (spread identity)
    root : str
        Product root symbol (e.g., "CL", "RB", "LC", "ALB")
    expiry : date, optional
        Contract expiration date. None for:
        - Generic positions (CL1, CL2)
        - Spreads (expiry lives on legs)
        - Equities

    Examples
    --------
    >>> cl_front = Instrument("CLX24", "CL", date(2024, 11, 20))
    >>> cl_generic = Instrument("CL1", "CL")
    >>> spread = Instrument("CL1_CL2", "CL")
    >>> equity = Instrument("ALB", "ALB")
    """
    instrument_id: str
    root: str
    expiry: Optional[date] = None

    def __str__(self) -> str:
        return self.instrument_id


@dataclass(frozen=True)
class LegDefinition:
    """
    One leg in a multi-leg tradable.

    Attributes
    ----------
    instrument : Instrument
        The instrument for this leg
    ratio : int
        Position ratio: +1 (long), -1 (short), +2 (double long), etc.
        For standard calendar spread: front=+1, back=-1

    Examples
    --------
    >>> front = Instrument("CLX24", "CL", date(2024, 11, 20))
    >>> back = Instrument("CLZ24", "CL", date(2024, 12, 18))
    >>> leg1 = LegDefinition(front, ratio=+1)
    >>> leg2 = LegDefinition(back, ratio=-1)
    """
    instrument: Instrument
    ratio: int

    def __post_init__(self):
        if self.ratio == 0:
            raise ValueError("Leg ratio cannot be 0")


@dataclass(frozen=True)
class Tradable:
    """
    How you trade something - wraps Instrument with execution semantics.

    Separates identity (what) from execution (how). The same instrument
    could be traded as NATIVE (exchange spread) or SYNTHETIC (leg-by-leg).

    Attributes
    ----------
    instrument : Instrument
        The identity of what you're trading as one unit
    execution_mode : ExecutionMode
        How to execute: SINGLE, NATIVE, or SYNTHETIC
    legs : tuple[LegDefinition, ...]
        Leg definitions for SYNTHETIC mode. Empty for SINGLE/NATIVE.

    Examples
    --------
    Single future:
    >>> cl = Instrument("CLX24", "CL", date(2024, 11, 20))
    >>> tradable = Tradable(cl, ExecutionMode.SINGLE)

    Synthetic calendar spread:
    >>> spread_id = Instrument("CL1_CL2", "CL")
    >>> front = Instrument("CLX24", "CL", date(2024, 11, 20))
    >>> back = Instrument("CLZ24", "CL", date(2024, 12, 18))
    >>> tradable = Tradable(
    ...     instrument=spread_id,
    ...     execution_mode=ExecutionMode.SYNTHETIC,
    ...     legs=(LegDefinition(front, +1), LegDefinition(back, -1)),
    ... )

    Native exchange spread (same identity, different execution):
    >>> native_tradable = Tradable(
    ...     instrument=spread_id,
    ...     execution_mode=ExecutionMode.NATIVE,
    ... )
    """
    instrument: Instrument
    execution_mode: ExecutionMode
    legs: Tuple[LegDefinition, ...] = ()

    def __post_init__(self):
        if self.execution_mode == ExecutionMode.SYNTHETIC and len(self.legs) < 2:
            raise ValueError("SYNTHETIC execution requires at least 2 legs")

    @property
    def instrument_id(self) -> str:
        """Convenience: get the instrument_id directly."""
        return self.instrument.instrument_id

    @property
    def root(self) -> str:
        """Convenience: get the root directly."""
        return self.instrument.root

    @property
    def is_spread(self) -> bool:
        """Check if this is a multi-leg tradable."""
        return self.execution_mode in (ExecutionMode.NATIVE, ExecutionMode.SYNTHETIC)

    @property
    def front_leg(self) -> Optional[LegDefinition]:
        """Get front leg (first leg with positive ratio)."""
        for leg in self.legs:
            if leg.ratio > 0:
                return leg
        return self.legs[0] if self.legs else None

    @property
    def back_leg(self) -> Optional[LegDefinition]:
        """Get back leg (first leg with negative ratio)."""
        for leg in self.legs:
            if leg.ratio < 0:
                return leg
        return self.legs[1] if len(self.legs) >= 2 else None


# =============================================================================
# Factory Functions
# =============================================================================

def create_single(
    instrument_id: str,
    root: str,
    expiry: Optional[date] = None,
) -> Tradable:
    """
    Create a single-leg tradable (outright future/option).

    Parameters
    ----------
    instrument_id : str
        Contract identifier (e.g., "CLX24")
    root : str
        Product root (e.g., "CL")
    expiry : date, optional
        Expiration date

    Returns
    -------
    Tradable
        Single-leg tradable
    """
    return Tradable(
        instrument=Instrument(instrument_id, root, expiry),
        execution_mode=ExecutionMode.SINGLE,
    )


def create_synthetic_spread(
    spread_id: str,
    root: str,
    front_id: str,
    back_id: str,
    front_expiry: Optional[date] = None,
    back_expiry: Optional[date] = None,
    front_ratio: int = +1,
    back_ratio: int = -1,
) -> Tradable:
    """
    Create a synthetic (leg-by-leg) calendar spread.

    Parameters
    ----------
    spread_id : str
        Spread identifier (e.g., "CL1_CL2")
    root : str
        Product root (e.g., "CL")
    front_id : str
        Front leg identifier
    back_id : str
        Back leg identifier
    front_expiry, back_expiry : date, optional
        Leg expiration dates
    front_ratio, back_ratio : int
        Leg ratios (default +1/-1 for standard calendar)

    Returns
    -------
    Tradable
        Synthetic spread tradable

    Examples
    --------
    >>> spread = create_synthetic_spread(
    ...     spread_id="CL1_CL2",
    ...     root="CL",
    ...     front_id="CLX24",
    ...     back_id="CLZ24",
    ... )
    """
    return Tradable(
        instrument=Instrument(spread_id, root),
        execution_mode=ExecutionMode.SYNTHETIC,
        legs=(
            LegDefinition(Instrument(front_id, root, front_expiry), front_ratio),
            LegDefinition(Instrument(back_id, root, back_expiry), back_ratio),
        ),
    )


def create_native_spread(
    spread_id: str,
    root: str,
) -> Tradable:
    """
    Create a native (exchange-listed) spread.

    For spreads that trade as a single instrument on the exchange
    (e.g., CME calendar spreads with their own quotes).

    Parameters
    ----------
    spread_id : str
        Spread identifier (e.g., "CLX24-CLZ24")
    root : str
        Product root (e.g., "CL")

    Returns
    -------
    Tradable
        Native spread tradable
    """
    return Tradable(
        instrument=Instrument(spread_id, root),
        execution_mode=ExecutionMode.NATIVE,
    )


# =============================================================================
# DEPRECATED - Keep for backward compatibility during migration
# =============================================================================

# These aliases allow existing code to continue working
# TODO: Remove after migration complete

class NativeKind(Enum):
    """DEPRECATED: Use ExecutionMode instead."""
    SINGLE_LEG = "single_leg"
    EXCHANGE_SPREAD = "exchange_spread"
    BAG_COMBO = "bag_combo"


@dataclass(frozen=True)
class InstrumentDefinition:
    """
    DEPRECATED: Use Instrument + Tradable instead.

    This class combines identity and execution semantics.
    Kept for backward compatibility during migration.
    """
    instrument_id: str
    execution_mode: ExecutionMode
    native_kind: Optional[NativeKind] = None
    conid: Optional[int] = None
    exchange: Optional[str] = None
    sec_type: Optional[str] = None
    legs: Tuple['LegDefinitionOld', ...] = ()
    multiplier: float = 1000.0

    def __post_init__(self):
        if self.execution_mode == ExecutionMode.SYNTHETIC:
            if not self.legs or len(self.legs) < 2:
                raise ValueError("Synthetic spreads require at least 2 legs")

    @property
    def is_spread(self) -> bool:
        if self.execution_mode == ExecutionMode.SYNTHETIC:
            return True
        if self.native_kind in (NativeKind.EXCHANGE_SPREAD, NativeKind.BAG_COMBO):
            return True
        return False

    def to_tradable(self) -> Tradable:
        """Convert to new Tradable type."""
        root = self.instrument_id.split("_")[0]
        instrument = Instrument(self.instrument_id, root)

        if self.execution_mode == ExecutionMode.SYNTHETIC and self.legs:
            new_legs = tuple(
                LegDefinition(
                    Instrument(leg.instrument_id, root),
                    leg.ratio
                )
                for leg in self.legs
            )
            return Tradable(instrument, self.execution_mode, new_legs)
        else:
            mode = ExecutionMode.NATIVE if self.execution_mode == ExecutionMode.NATIVE else ExecutionMode.SINGLE
            return Tradable(instrument, mode)


@dataclass(frozen=True)
class LegDefinitionOld:
    """DEPRECATED: Use LegDefinition instead."""
    instrument_id: str
    ratio: int
    multiplier: float = 1000.0
    conid: Optional[int] = None


def create_calendar_spread(
    front_id: str,
    back_id: str,
    execution_mode: ExecutionMode = ExecutionMode.SYNTHETIC,
    front_conid: Optional[int] = None,
    back_conid: Optional[int] = None,
    spread_conid: Optional[int] = None,
    exchange: str = "NYMEX",
    multiplier: float = 1000.0,
) -> InstrumentDefinition:
    """
    DEPRECATED: Use create_synthetic_spread or create_native_spread instead.
    """
    spread_id = f"{front_id}_{back_id}"

    if execution_mode == ExecutionMode.NATIVE:
        return InstrumentDefinition(
            instrument_id=spread_id,
            execution_mode=ExecutionMode.NATIVE,
            native_kind=NativeKind.EXCHANGE_SPREAD,
            conid=spread_conid,
            exchange=exchange,
            multiplier=multiplier,
        )
    else:
        return InstrumentDefinition(
            instrument_id=spread_id,
            execution_mode=ExecutionMode.SYNTHETIC,
            legs=(
                LegDefinitionOld(front_id, ratio=+1, multiplier=multiplier, conid=front_conid),
                LegDefinitionOld(back_id, ratio=-1, multiplier=multiplier, conid=back_conid),
            ),
            multiplier=multiplier,
        )
