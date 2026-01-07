"""Portfolio management components."""

from .rebalancer import (
    RebalanceConfig,
    RebalanceMethod,
    RebalanceSignal,
    PortfolioRebalancer,
    calculate_pca_based_weights,
    calculate_target_weights
)

from .seasonal_weights import (
    ContractWeight,
    SeasonalWeightCalculator
)

from .equity import (
    EquityPoint,
    EquityCurve,
)

from .pnl import (
    calc_net_pnl,
    calc_total_pnl,
    calc_return_pct,
)

from .event_ledger import (
    EventLedger,
    LedgerPosition,
    PositionStatus,
    create_ledger,
)

# Re-export Protocol from canonical location for backwards compatibility
from common_lib.interfaces.protocols import HasUnrealizedPnl

__all__ = [
    # Rebalancer
    'RebalanceConfig',
    'RebalanceMethod',
    'RebalanceSignal',
    'PortfolioRebalancer',
    'calculate_pca_based_weights',
    'calculate_target_weights',
    # Seasonal weights
    'ContractWeight',
    'SeasonalWeightCalculator',
    # Equity tracking
    'EquityPoint',
    'EquityCurve',
    # P&L helpers
    'calc_net_pnl',
    'calc_total_pnl',
    'calc_return_pct',
    # Event-driven position ledger
    'EventLedger',
    'LedgerPosition',
    'PositionStatus',
    'create_ledger',
    # Protocol (re-exported for convenience)
    'HasUnrealizedPnl',
]