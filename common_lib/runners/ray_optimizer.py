"""
Ray-based Distributed Parameter Optimizer

Enables massively parallel backtesting across a Ray cluster using GenericRunner.

Example:
--------
>>> import ray
>>> ray.init(address="auto")
>>>
>>> optimizer = RayOptimizer(
...     strategy_factory=lambda cfg: MyStrategy(cfg),
...     data_feed_factory=lambda: MyDataFeed(),
...     base_config=config
... )
>>>
>>> results = optimizer.optimize(
...     param_grid={'threshold': [1.0, 1.5, 2.0]},
... )

Date: 2026-01-04
"""

import ray
import numpy as np
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, asdict
import logging
import time
import copy

from common_lib.runners.generic_runner import (
    GenericRunner,
    RunResult,
    DataFeedProtocol,
    StrategyProtocol,
)
from common_lib.execution.event_executor import create_event_executor
from common_lib.portfolio.event_ledger import create_ledger

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result from a single parameter combination."""
    parameters: Dict[str, Any]
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    total_trades: int
    runtime_seconds: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def expand_param_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Expand parameter grid into list of combinations."""
    import itertools

    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    values = [v if isinstance(v, list) else [v] for v in param_grid.values()]

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def merge_config(base: Dict, updates: Dict) -> Dict:
    """Deep merge updates into base config."""
    result = copy.deepcopy(base)

    for key, value in updates.items():
        if '.' in key:
            # Handle nested keys like 'parameters.threshold'
            keys = key.split('.')
            current = result
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        elif isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value

    return result


@ray.remote
def run_backtest_task(
    strategy_module: str,
    strategy_class_name: str,
    data_feed_module: str,
    data_feed_class_name: str,
    data_feed_kwargs: Dict[str, Any],
    config: Dict[str, Any],
    params: Dict[str, Any],
    product: str = "CL",
    initial_capital: float = 100_000.0,
) -> OptimizationResult:
    """
    Ray remote task to run a single backtest.

    Strategy and data feed are passed by module/class name for serialization.
    """
    import importlib
    import time

    start = time.time()

    try:
        # Import strategy class
        strat_mod = importlib.import_module(strategy_module)
        strategy_class = getattr(strat_mod, strategy_class_name)

        # Import data feed class
        feed_mod = importlib.import_module(data_feed_module)
        feed_class = getattr(feed_mod, data_feed_class_name)

        # Merge params into config
        merged_config = merge_config(config, params)

        # Create components
        strategy = strategy_class(merged_config)
        data_feed = feed_class(**data_feed_kwargs)
        executor = create_event_executor(product)
        ledger = create_ledger(product)

        # Run backtest
        runner = GenericRunner(
            strategy=strategy,
            data_feed=data_feed,
            executor=executor,
            ledger=ledger,
            initial_capital=initial_capital,
        )

        result = runner.run()

        return OptimizationResult(
            parameters=params,
            sharpe_ratio=result.sharpe_ratio,
            total_return=result.total_return_pct,
            max_drawdown=result.max_drawdown_pct,
            total_trades=result.total_trades,
            runtime_seconds=time.time() - start,
        )

    except Exception as e:
        return OptimizationResult(
            parameters=params,
            sharpe_ratio=-999.0,
            total_return=-1.0,
            max_drawdown=1.0,
            total_trades=0,
            runtime_seconds=time.time() - start,
            error=str(e),
        )


class RayOptimizer:
    """
    Distributed parameter optimizer using Ray.

    Uses GenericRunner internally - the new event-driven architecture.
    """

    def __init__(
        self,
        strategy_module: str,
        strategy_class_name: str,
        data_feed_module: str,
        data_feed_class_name: str,
        data_feed_kwargs: Dict[str, Any],
        base_config: Dict[str, Any],
        product: str = "CL",
        initial_capital: float = 100_000.0,
    ):
        """
        Initialize optimizer.

        Args:
            strategy_module: Module path for strategy (e.g., "strategies.spread_regime.strategy")
            strategy_class_name: Class name (e.g., "SpreadRegimeStrategy")
            data_feed_module: Module path for data feed
            data_feed_class_name: Data feed class name
            data_feed_kwargs: Kwargs to pass to data feed constructor
            base_config: Base configuration dict
            product: Product for executor/ledger (e.g., "CL")
            initial_capital: Starting capital
        """
        self.strategy_module = strategy_module
        self.strategy_class_name = strategy_class_name
        self.data_feed_module = data_feed_module
        self.data_feed_class_name = data_feed_class_name
        self.data_feed_kwargs = data_feed_kwargs
        self.base_config = base_config
        self.product = product
        self.initial_capital = initial_capital

    def optimize(
        self,
        param_grid: Dict[str, List[Any]],
        max_combinations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run distributed parameter optimization.

        Args:
            param_grid: Parameter grid to search
            max_combinations: Maximum combinations to test

        Returns:
            Dictionary with optimization results
        """
        if not ray.is_initialized():
            logger.warning("Ray not initialized. Starting local cluster...")
            ray.init()

        # Expand parameter grid
        combinations = expand_param_grid(param_grid)

        if max_combinations and len(combinations) > max_combinations:
            indices = np.random.choice(len(combinations), max_combinations, replace=False)
            combinations = [combinations[i] for i in indices]

        total = len(combinations)
        logger.info(f"Starting Ray optimization: {total} combinations")
        logger.info(f"Cluster resources: {ray.cluster_resources()}")

        start_time = time.time()

        # Submit all tasks
        futures = [
            run_backtest_task.remote(
                strategy_module=self.strategy_module,
                strategy_class_name=self.strategy_class_name,
                data_feed_module=self.data_feed_module,
                data_feed_class_name=self.data_feed_class_name,
                data_feed_kwargs=self.data_feed_kwargs,
                config=self.base_config,
                params=params,
                product=self.product,
                initial_capital=self.initial_capital,
            )
            for params in combinations
        ]

        # Collect results with progress
        results = []
        completed = 0

        while futures:
            done, futures = ray.wait(futures, num_returns=1, timeout=60.0)

            for future in done:
                try:
                    result = ray.get(future)
                    results.append(result)
                    completed += 1

                    if completed % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        logger.info(
                            f"Progress: {completed}/{total} ({100*completed/total:.1f}%) "
                            f"- {rate:.1f}/sec"
                        )
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    completed += 1

        total_time = time.time() - start_time

        # Process results
        valid = [r for r in results if r.error is None]
        errors = len(results) - len(valid)

        if not valid:
            return {"best_parameters": None, "error": "All backtests failed"}

        best = max(valid, key=lambda r: r.sharpe_ratio)
        sharpes = [r.sharpe_ratio for r in valid]

        return {
            "best_parameters": best.parameters,
            "best_sharpe": best.sharpe_ratio,
            "best_return": best.total_return,
            "best_drawdown": best.max_drawdown,
            "all_results": [r.to_dict() for r in results],
            "summary": {
                "total": total,
                "valid": len(valid),
                "errors": errors,
                "time_seconds": total_time,
                "rate_per_second": total / total_time,
                "sharpe_mean": np.mean(sharpes),
                "sharpe_max": np.max(sharpes),
            },
        }
