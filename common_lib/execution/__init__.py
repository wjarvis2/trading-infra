"""
Execution Layer - Spread Order Execution Simulation.

This module provides:
- ProductSpec: Per-instrument properties (multiplier, tick_size, commission)
- ExecutorConfig: Behavioral settings (slippage model, fill model)
- Cost Models: Pluggable slippage and commission calculators
- Executors: SyntheticSpreadExecutor, NativeSpreadExecutor
- Factory: resolve_executor for InstrumentDefinition routing

Architecture follows research/decisions/003_execution_architecture.md:
- ExecutorConfig = behavioral only (HOW you simulate)
- ProductSpec = per-instrument properties (WHAT you trade)
- Cost models = pluggable, testable in isolation
- No hardcoded product assumptions

Usage
-----
>>> from common_lib.execution import (
...     create_spread_executor,
...     CL_SPEC,
...     DEFAULT_CONFIG,
... )
>>> executor = create_spread_executor("CL")
>>> fill = executor.execute_entry(direction=-1, bar=bar, strategy="MR")
"""

# Product specifications
from common_lib.execution.product_spec import (
    ProductSpec,
    ProductCatalog,
    CL_SPEC,
    RB_SPEC,
    HO_SPEC,
    LC_SPEC,
    create_default_catalog,
)

# Executor configuration
from common_lib.execution.executor_config import (
    ExecutorConfig,
    DEFAULT_CONFIG,
    CONSERVATIVE_CONFIG,
    OPTIMISTIC_CONFIG,
    FIXED_SLIPPAGE_CONFIG,
)

# Cost models
from common_lib.execution.cost_models import (
    SlippageModel,
    CommissionModel,
    BarRangeSlippageModel,
    FixedSlippageModel,
    ZeroSlippageModel,
    PerContractCommissionModel,
    FixedCommissionModel,
    ZeroCommissionModel,
    create_slippage_model,
    create_commission_model,
)

# Executors
from common_lib.execution.spread_executor import (
    FillResult,
    SpreadExecutorProtocol,
    SyntheticSpreadExecutor,
    NativeSpreadExecutor,
    MissingLegDataError,
)

# Factory
from common_lib.execution.executor_factory import (
    resolve_executor,
    create_spread_executor,
)

# Event-driven executor (new type system)
from common_lib.execution.event_executor import (
    EventExecutor,
    EventExecutorProtocol,
    create_event_executor,
)

__all__ = [
    # Product specs
    "ProductSpec",
    "ProductCatalog",
    "CL_SPEC",
    "RB_SPEC",
    "HO_SPEC",
    "LC_SPEC",
    "create_default_catalog",
    # Config
    "ExecutorConfig",
    "DEFAULT_CONFIG",
    "CONSERVATIVE_CONFIG",
    "OPTIMISTIC_CONFIG",
    "FIXED_SLIPPAGE_CONFIG",
    # Cost models
    "SlippageModel",
    "CommissionModel",
    "BarRangeSlippageModel",
    "FixedSlippageModel",
    "ZeroSlippageModel",
    "PerContractCommissionModel",
    "FixedCommissionModel",
    "ZeroCommissionModel",
    "create_slippage_model",
    "create_commission_model",
    # Executors
    "FillResult",
    "SpreadExecutorProtocol",
    "SyntheticSpreadExecutor",
    "NativeSpreadExecutor",
    "MissingLegDataError",
    # Factory
    "resolve_executor",
    "create_spread_executor",
    # Event-driven executor (new type system)
    "EventExecutor",
    "EventExecutorProtocol",
    "create_event_executor",
]
