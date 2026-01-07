# Common Library (common_lib)

The foundational library for the energy trading system, providing production-ready components for strategy development, backtesting, and live trading.

## Overview

`common_lib` implements a clean architecture with:
- **Interface-driven design** ensuring compatibility between backtesting and live trading
- **Separation of concerns** between data I/O and business logic
- **Single source of truth** for all calculations
- **Comprehensive testing** and validation frameworks

## Core Modules

### interfaces/
Core contracts that all implementations must follow:
- `IStrategy` - Base class for all trading strategies
- `IDataProvider` - Minimal interface for strategy data access
- `IExecutor` - Order execution interface
- `IRiskModel` - Risk management interface
- `types.py` - Common data structures (Bar, Signal, Fill, Position, Order)

### data_providers/
Concrete implementations of IDataProvider:
- `TimescaleProvider` - Live database queries for production trading
- `InMemoryProvider` - Pre-loaded data for backtesting
- `SpreadProvider` - Adds spread calculation capability to any provider

### data_fetchers/
Centralized database I/O operations:
- `BarFetcher` - OHLCV bar data with connection pooling
- `PCAFetcher` - PCA factor data retrieval
- `InventoryFetcher` - Fundamental inventory data
- `CurveFetcher` - Forward curve snapshots
- `RegimeFetcher` - Volatility regime indicators

### research/
Production-grade backtesting infrastructure:
- `backtesting/` - NotebookBacktester for Jupyter-based strategy testing
- `data/` - BacktestDataLoader for efficient data loading
- `utils/` - Performance metrics, trade matching, parameter optimization
- `signals/` - Signal generation utilities

### factors/
Stateless calculators for financial factors:
- `PCACalculator` - Term structure PCA (level, slope, curvature)
- `CarryPCACalculator` - PCA with carry adjustments
- `VolatilityRegime` - Market regime classification

### indicators/
Technical analysis and market indicators:
- `technical.py` - ATR, Bollinger Bands, RSI, VWAP, correlations
- `momentum.py` - Multiple momentum calculations including PCA momentum
- `seasonality.py` - Seasonal pattern analysis
- `volatility.py` - EWMA and other volatility measures
- `microstructure.py` - Market microstructure analysis
- `session.py` - Trading session metrics

### execution/
Order execution logic:
- `SimpleExecutor` - Fixed slippage model for backtesting
- `SpreadDecomposer` - Breaks spread orders into individual legs

### risk/
Risk management components:
- `PositionSizer` - Multiple sizing methods (fixed risk, Kelly, volatility target)
- `CorrelationMonitor` - Cross-asset correlation tracking
- `HolidayAdjuster` - Holiday-aware position adjustments

### slippage/
Execution cost models:
- `DynamicSlippage` - Calibrated slippage models with market impact

### pricing/
Canonical pricing calculations:
- `SpotPrice` - Theoretically correct spot price using carry theory

### curve_builder/
SQL/Python parity for curve calculations:
- SHA-256 validation of SQL files
- Alembic migrations for schema changes
- Property-based testing with Hypothesis
- CLI tools for validation

### portfolio/
Portfolio construction and management:
- `PortfolioRebalancer` - Multiple rebalancing strategies
- `SeasonalWeightCalculator` - Seasonal weight adjustments

### factories/
Component creation with environment-aware configuration:
- Creates configured instances based on environment (dev/paper/prod)
- Handles dependency injection and configuration management

### utils/
Common utilities:
- `contract_parser.py` - Contract nomenclature parsing

## Usage Examples

### Implementing a Strategy
```python
from common_lib.interfaces import IStrategy
from common_lib.interfaces.types import Bar, Signal, Side

class MyStrategy(IStrategy):
    def __init__(self, config, data_provider):
        self.config = config
        self.data = data_provider
    
    def on_bar(self, bar: Bar) -> List[Signal]:
        # Get additional data
        inventory = self.data.get_latest_inventory()
        pca_factors = self.data.get_latest_pca_factors()
        
        # Generate signals
        if self.should_buy(bar, inventory, pca_factors):
            return [Signal(
                timestamp=bar.timestamp,
                strategy_id=self.name,
                symbol=bar.symbol,
                side=Side.BUY,
                strength=0.05,
                confidence=0.8
            )]
        return []
```

### Running a Backtest
```python
from common_lib.research.backtesting import NotebookBacktester

# Configure and run backtest
backtester = NotebookBacktester(
    strategy_class=MyStrategy,
    base_config={'param1': 10, 'param2': 0.5},
    bar_type='15s',
    warmup_bars=240  # 1 hour warmup
)

result = backtester.run_backtest(
    start_date='2025-07-01',
    end_date='2025-08-01',
    latency_buffer=1  # Trade on 1-bar stale data
)

# Analyze results
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Total PnL: ${result.metrics['total_pnl']:,.2f}")
```

### Using Indicators
```python
from common_lib.indicators.technical import calculate_atr, calculate_bollinger_bands
from common_lib.indicators.momentum import calculate_pca_momentum

# Calculate technical indicators
atr = calculate_atr(bars_df, period=14)
bb_data = calculate_bollinger_bands(bars_df, period=20, num_std=2)

# Calculate PCA momentum
momentum = calculate_pca_momentum(pca_factors_df, lookback=120)
```

### Risk Management
```python
from common_lib.risk import PositionSizer

# Create position sizer
sizer = PositionSizer(
    method='volatility_target',
    base_capital=1000000,
    target_volatility=0.15
)

# Calculate position size
size = sizer.calculate_position_size(
    signal_strength=0.05,
    current_volatility=0.20,
    execution_costs=50.0
)
```

## Design Principles

1. **No Synthetic Data**: All calculations use real data or raise exceptions
2. **Interface Compliance**: All strategies must implement IStrategy
3. **Centralized Data Access**: All database operations through data_fetchers
4. **Stateless Calculations**: Business logic separated from I/O
5. **Environment Awareness**: Components adapt to dev/paper/prod environments

## Testing

Run tests with:
```bash
# Run all common_lib tests
pytest common_lib/

# Run with coverage
pytest --cov=common_lib common_lib/

# Run specific module tests
pytest common_lib/indicators/tests/
```

## Configuration

All configuration is centralized in the `/config/` directory:
- Environment configs: `dev.yaml`, `paper.yaml`, `prod.yaml`
- Strategy configs: `/config/strategies/{strategy_name}.yaml`
- Calibration data: `/config/calibration/`

## Contributing

When adding new components:
1. Implement appropriate interface from `interfaces/`
2. Add to relevant module (don't create ad-hoc implementations)
3. Include comprehensive tests
4. Update this README if adding new modules
5. Ensure no synthetic data or hardcoded values

## License

Proprietary - All rights reserved