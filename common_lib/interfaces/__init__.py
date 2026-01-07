"""
Core interfaces for the trading system.

These abstract base classes define the contracts that all implementations
must follow, ensuring compatibility between backtesting and live trading.
"""

from .base_strategy import IStrategy
from .data_feed import IDataFeed
from .executor import IExecutor
from .risk_model import IRiskModel
from .types import Bar, Signal, Fill, Position, Order

# Position Protocols - minimal interfaces for downstream components
from .protocols import (
    PositionStatus,
    HasUnrealizedPnl,
    HasPnlSummary,
    HasCosts,
    HasPositionIdentity,
    HasPositionLifecycle,
    HasTradeContext,
    HasEntryExit,
)

# Instrument types - core trading primitives
from .instrument import (
    Instrument,
    Tradable,
    LegDefinition,
    ExecutionMode,
    create_single,
    create_synthetic_spread,
    create_native_spread,
)

# Signal types - position intent
from .signal import (
    Signal as PositionSignal,  # Alias to avoid conflict with types.Signal
    SignalIntent,
    ExecutionStyle,
    create_entry_signal,
    create_exit_signal,
    create_roll_signal,
)

# Execution events - event stream from executor
from .execution_events import (
    ExecutionEvent,
    FillEvent,
    RejectEvent,
    LegFill,
    FillType,
    RejectReason,
    create_fill_event,
    create_reject_event,
    create_single_leg_fill,
    create_spread_fill,
)

# Risk gate - portfolio-level risk constraints
from .risk_gate import (
    RiskGate,
    RiskGateResult,
    RiskGateEvent,
    RiskGateAction,
    approve,
    reject,
    modify,
)

# Versioned contracts
from .contracts import (
    CONTRACT_VERSION,
    DataFeedContract,
    ExecutorContract,
    DATAFEED_REQUIRED_FIELDS,
    RUNRESULT_REQUIRED_FIELDS,
    validate_datafeed_output,
    validate_run_result,
    ContractViolation,
    ContractInfo,
    CONTRACTS,
    get_contract_version,
)

__all__ = [
    # Abstract interfaces
    'IStrategy',
    'IDataFeed',
    'IExecutor',
    'IRiskModel',
    # Data types
    'Bar',
    'Signal',
    'Fill',
    'Position',
    'Order',
    # Position Protocols
    'PositionStatus',
    'HasUnrealizedPnl',
    'HasPnlSummary',
    'HasCosts',
    'HasPositionIdentity',
    'HasPositionLifecycle',
    'HasTradeContext',
    'HasEntryExit',
    # Instrument types
    'Instrument',
    'Tradable',
    'LegDefinition',
    'ExecutionMode',
    'create_single',
    'create_synthetic_spread',
    'create_native_spread',
    # Signal types
    'PositionSignal',
    'SignalIntent',
    'ExecutionStyle',
    'create_entry_signal',
    'create_exit_signal',
    'create_roll_signal',
    # Execution events
    'ExecutionEvent',
    'FillEvent',
    'RejectEvent',
    'LegFill',
    'FillType',
    'RejectReason',
    'create_fill_event',
    'create_reject_event',
    'create_single_leg_fill',
    'create_spread_fill',
    # Risk gate
    'RiskGate',
    'RiskGateResult',
    'RiskGateEvent',
    'RiskGateAction',
    'approve',
    'reject',
    'modify',
    # Versioned contracts
    'CONTRACT_VERSION',
    'DataFeedContract',
    'ExecutorContract',
    'DATAFEED_REQUIRED_FIELDS',
    'RUNRESULT_REQUIRED_FIELDS',
    'validate_datafeed_output',
    'validate_run_result',
    'ContractViolation',
    'ContractInfo',
    'CONTRACTS',
    'get_contract_version',
]

