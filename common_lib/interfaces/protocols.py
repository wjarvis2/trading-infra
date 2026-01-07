"""
Position and P&L Protocols.

Minimal interfaces that downstream components depend on.
Strategies implement these with their own concrete types.

Design principles:
- Protocols contain raw facts only, no derived fields
- Composable: consumers depend only on what they need
- Helper functions compute derived values (net_pnl, etc.)

"""

from typing import Protocol, Optional, Union, runtime_checkable
from datetime import datetime
from enum import Enum


class PositionStatus(Enum):
    """Position lifecycle status."""
    OPEN = "open"
    CLOSED = "closed"
    ROLLED = "rolled"


# =============================================================================
# P&L Protocols
# =============================================================================

@runtime_checkable
class HasUnrealizedPnl(Protocol):
    """
    For components that only need current unrealized P&L.

    Used by: EquityCurve, monitoring/prometheus
    """
    unrealized_pnl: float


@runtime_checkable
class HasPnlSummary(Protocol):
    """
    For components that need realized + unrealized P&L.

    Used by: position_tracker, portfolio reports
    """
    realized_pnl: Optional[float]
    unrealized_pnl: float


@runtime_checkable
class HasCosts(Protocol):
    """
    For components that need cost breakdown.

    Optional - not all engines track commission/slippage separately.
    """
    commission: float
    slippage: float


# =============================================================================
# Position Lifecycle Protocols
# =============================================================================

@runtime_checkable
class HasPositionIdentity(Protocol):
    """
    Minimal position identity.

    position_id can be int, str, or UUID depending on engine.
    """
    position_id: Union[int, str]


@runtime_checkable
class HasPositionLifecycle(Protocol):
    """
    For components that need position state/timing.

    Used by: trade logs, position queries
    """
    position_id: Union[int, str]
    status: PositionStatus
    entry_time: datetime
    exit_time: Optional[datetime]
    exit_reason: Optional[str]


# =============================================================================
# Trade Context Protocols
# =============================================================================

@runtime_checkable
class HasTradeContext(Protocol):
    """
    For components that need trade direction/size.

    Used by: P&L calculations, position sizing
    """
    direction: int  # 1=LONG, -1=SHORT
    quantity: int


@runtime_checkable
class HasEntryExit(Protocol):
    """
    For components that need price info.

    Used by: return calculations, trade records
    """
    entry_price: float
    exit_price: Optional[float]


# =============================================================================
# Risk Metrics Protocols (deferred - only spread_regime uses currently)
# =============================================================================

# NOTE: HasRiskMetrics (mae, mfe) is intentionally NOT defined here.
# From the audit, MAE/MFE are only consumed within spread_regime.
# When a cross-strategy consumer emerges, add:
#
# @runtime_checkable
# class HasRiskMetrics(Protocol):
#     """For MAE/MFE analysis. Values in P&L space (dollars)."""
#     mae_pnl: float  # Maximum Adverse Excursion (most negative P&L)
#     mfe_pnl: float  # Maximum Favorable Excursion (most positive P&L)
