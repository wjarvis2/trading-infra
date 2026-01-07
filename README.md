# Trading Infrastructure

Core infrastructure for systematic commodity futures trading. Event-driven architecture designed for backtesting and live execution.

## Architecture Overview

```
common_lib/
├── interfaces/          # Core type system and protocols
├── execution/           # Event-driven executor with cost models
├── portfolio/           # Position ledger and P&L tracking
├── risk/                # Risk gates and position sizing
├── runners/             # Backtest runner + Ray distributed optimizer
├── factors/             # PCA and carry factor calculators
├── signal_mining/       # IC scoring and hypothesis validation
└── data_fetchers/       # Unified data access layer

options/
├── delta_ladder.py      # Delta-targeted option subscription management
├── strike_prober.py     # Strike selection with Greeks calculation
├── quality_enums.py     # Two-track delta model (market vs model)
├── vol_surface.py       # Volatility surface fitting
└── quote_validator.py   # Quote quality gates
```

## Key Design Decisions

### Event-Driven Execution
- All state changes flow through `apply(event)` pattern
- Atomic roll handling via `RollFillEvent` (single event for exit + entry)
- Clear separation: executor reports fills, ledger interprets them

### Risk Management
- `RiskGate` protocol for portfolio-level constraints
- Can approve, modify, or reject signals
- Decisions tracked for post-hoc analysis

### Two-Track Delta Model (Options)
- `model_delta`: Our Black-76 calculation from live quotes (execution-grade)
- `broker_delta`: Vendor Greeks for comparison (never execution-eligible)
- Explicit quality gates: `LIVE_TRADABLE`, `INDICATIVE`, `MODEL_ONLY`

### Distributed Backtesting
- `RayOptimizer` for parallel parameter sweeps across cluster
- Stateless task execution - no shared state between runs
- Early stopping support for compute efficiency

## Modules

### `common_lib.interfaces`
Core types: `Signal`, `Tradable`, `RiskGate`, `FillEvent`, `RollFillEvent`

### `common_lib.execution`
Event executor with pluggable cost models. Handles entry, exit, and roll execution.

### `common_lib.portfolio`
`EventLedger` tracks positions with roll linkage via `logical_position_id`. Supports mark-to-market updates.

### `common_lib.risk`
`BasicRiskGate` implementation with:
- Position count limits
- Drawdown kill switch
- Max position size (with auto-scaling)
- Strategy concentration limits

### `common_lib.runners`
- `GenericRunner`: Core backtest loop with optional risk gate hook
- `RayOptimizer`: Distributed parameter optimization

### `common_lib.signal_mining`
- `HypothesisSpec`: Structured hypothesis definition
- `ICScorer`: Information coefficient with bootstrap standard errors
- `ForwardReturns`: Target variable calculation

### `options/`
Enterprise-grade options data infrastructure:
- Quality-gated delta ladder with explicit data provenance
- Execution eligibility requires: market-derived delta + live quotes + synchronized timestamps
- Vol surface fitting with liquidity gates

## Requirements

- Python 3.10+
- Core: `numpy`, `pandas`, `pydantic`
- Execution: `ib_insync` (Interactive Brokers)
- Options: `QuantLib` (Greeks calculation)
- Distributed: `ray` (optional, for parallel optimization)

## Usage

This is infrastructure code extracted from a larger trading system. It demonstrates architectural patterns for systematic trading - not a turnkey solution.

## License

MIT
