# Trading System Interfaces

This module defines the core interfaces that ensure compatibility between backtesting and live trading engines.

## Core Interfaces

### IStrategy
The main interface that all trading strategies must implement:
- `on_bar()`: Process new market data and generate signals
- `on_fill()`: Handle execution fills
- `get_state()`/`set_state()`: State management for checkpointing

### IDataFeed
Interface for market data providers:
- `get_historical_bars()`: Historical data for backtesting
- `subscribe()`: Real-time data for live trading
- Supports multiple bar sizes (5s, 1m, 1h, 1d)

### IExecutor
Interface for order execution:
- `submit_order()`: Submit orders to market
- `get_positions()`: Track current positions
- Works identically in backtest (simulated) and live (real broker)

### IRiskModel
Interface for risk management:
- `calculate_position_size()`: Size positions based on risk
- `check_risk_limits()`: Enforce portfolio constraints
- `calculate_stop_loss()`: Determine stop levels

## Usage Example

```python
from common_lib.interfaces import IStrategy, Bar, Signal, Side

class MyStrategy(IStrategy):
    def on_bar(self, bar: Bar) -> List[Signal]:
        # Strategy logic here
        if bar.close > self.sma:
            return [Signal(
                timestamp=bar.timestamp,
                strategy_id=self.name,
                symbol=bar.symbol,
                side=Side.BUY,
                strength=0.05,  # 5% expected return
                confidence=0.8
            )]
        return []
```

## Key Benefits

1. **Consistency**: Same code runs in backtest and live
2. **Testability**: Easy to mock interfaces for unit tests
3. **Modularity**: Swap implementations without changing strategies
4. **Type Safety**: Full type hints for better IDE support