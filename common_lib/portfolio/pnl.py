"""
P&L calculation helpers.

Pure functions that compute derived P&L values from Protocol-compliant objects.
Keeps Protocols as "raw facts" while providing consistent calculations.

"""

from typing import Optional, Union
from common_lib.interfaces.protocols import HasPnlSummary, HasCosts


def calc_net_pnl(
    position: HasPnlSummary,
    costs: Optional[HasCosts] = None,
) -> float:
    """
    Calculate net P&L from a position.

    If position has realized_pnl, uses that. Otherwise uses unrealized_pnl.
    Subtracts costs if provided.

    Parameters
    ----------
    position : HasPnlSummary
        Object with realized_pnl and unrealized_pnl
    costs : HasCosts, optional
        Object with commission and slippage (can be same object as position)

    Returns
    -------
    float
        Net P&L after costs

    Examples
    --------
    >>> calc_net_pnl(position)  # No costs
    350.0
    >>> calc_net_pnl(position, position)  # Position implements both protocols
    335.0
    """
    # Use realized if available (closed position), else unrealized (open)
    gross_pnl = (
        position.realized_pnl
        if position.realized_pnl is not None
        else position.unrealized_pnl
    )

    if costs is None:
        return gross_pnl

    return gross_pnl - costs.commission - costs.slippage


def calc_total_pnl(position: HasPnlSummary) -> float:
    """
    Calculate total P&L (realized + unrealized).

    Parameters
    ----------
    position : HasPnlSummary
        Object with realized_pnl and unrealized_pnl

    Returns
    -------
    float
        Total P&L
    """
    realized = position.realized_pnl if position.realized_pnl is not None else 0.0
    return realized + position.unrealized_pnl


def calc_return_pct(
    pnl: float,
    entry_price: float,
    quantity: int,
    multiplier: float = 1.0,
) -> float:
    """
    Calculate return percentage from P&L.

    Parameters
    ----------
    pnl : float
        Net or gross P&L in dollars
    entry_price : float
        Entry price
    quantity : int
        Position quantity (absolute value used)
    multiplier : float
        Contract multiplier (default 1.0)

    Returns
    -------
    float
        Return as decimal (0.05 = 5%)
    """
    notional = abs(quantity) * entry_price * multiplier
    if notional == 0:
        return 0.0
    return pnl / notional
