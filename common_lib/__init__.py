"""
Common Library for Trading System

A unified package containing all shared components used by both
backtesting and live trading engines.

Package Structure (New Architecture):
- interfaces/: Protocols (ExecutionEvents, Signal, Instrument, Bar)
- execution/: Event-driven execution (EventExecutor, cost_models)
- runners/: Strategy runners (GenericRunner)
- data_feeds/: Composable data feeds
- data_fetchers/: Unified data fetching
- portfolio/: Equity tracking and PnL calculation
- strategy/: Strategy identity and metadata

Legacy (in archive/):
- risk/: Old position sizing (replaced by execution cost models)
- slippage/: Old slippage (replaced by cost_models.py)
- factories/: Old wiring (replaced by executor_factory)
- events/: Old event bus (replaced by ExecutionEvents)
- logging/: Old logging (replaced by logging_config)
"""

__version__ = '0.2.0'  # Bumped for new architecture

