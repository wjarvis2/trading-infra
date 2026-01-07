"""
Generic backtest/simulation runner framework.

Provides strategy-agnostic orchestration:
- DataFeed → Strategy → Executor → Ledger

"""

from .generic_runner import (
    GenericRunner,
    RunResult,
    StrategyProtocol,
    DataFeedProtocol,
)
from .ray_optimizer import RayOptimizer, OptimizationResult

__all__ = [
    "GenericRunner",
    "RunResult",
    "StrategyProtocol",
    "DataFeedProtocol",
    "RayOptimizer",
    "OptimizationResult",
]
