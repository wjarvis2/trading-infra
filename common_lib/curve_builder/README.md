# Curve Builder

A versioned package ensuring SQL/Python parity for curve calculations in the trading system.

## Overview

The curve_builder package maintains consistency between SQL queries and Python implementations for:
- 5-second mid-price curves across front 8 contracts
- Spot price adjustments using canonical calculations
- PCA factor calculations
- Historical data management with tiered storage

## Key Features

1. **SQL/Python Parity**: Ensures SQL queries and Python code produce identical results
2. **SHA-256 Validation**: Detects when SQL files change, prompting Python updates
3. **Property-Based Testing**: Uses Hypothesis to test edge cases
4. **Alembic Migrations**: Version-controlled database schema changes
5. **CLI Tools**: Validate parity, check hashes, run tests

## Installation

```bash
cd /home/will/trading_system
pip install -e common_lib
```

## Usage

### CLI Commands

```bash
# Initialize hash tracking for SQL files
curve-builder init

# Validate SQL/Python parity
curve-builder validate

# Check SQL file hashes
curve-builder check-hashes

# Update registry with current hashes
curve-builder check-hashes --update

# Show SQL for a registered query
curve-builder show-sql curve_mid_5s

# Test Python implementation
curve-builder test-python --start-time "2025-08-01T00:00:00" --end-time "2025-08-01T01:00:00"
```

### Python API

```python
from common_lib.curve_builder import CurveCalculator, validate_sql_python_parity

# Create calculator
calculator = CurveCalculator(lookback_days=2)

# Calculate curve from bar data
curve_df = calculator.calculate_curve_mid_5s(
    bar_5s_df,
    continuous_contracts_df,
    reference_time
)

# Apply spot adjustment
adjusted_df = calculator.calculate_spot_adjustment(
    curve_df,
    expiry_map
)

# Validate SQL/Python parity
is_valid = validate_sql_python_parity(db_conn_str)
```

## Architecture

### Directory Structure

```
curve_builder/
├── sql/
│   ├── registry.yaml      # SQL file registry with hashes
│   └── hash_validator.py  # SHA-256 validation
├── python/
│   ├── curve_calculator.py  # Python implementations
│   └── validator.py         # Parity validation
├── migrations/
│   ├── alembic.ini         # Alembic config
│   ├── env.py              # Migration environment
│   └── versions/           # Migration scripts
├── tests/
│   └── test_curve_calculator.py  # Property-based tests
└── cli.py                  # Command-line interface
```

### SQL Registry

The `sql/registry.yaml` file tracks:
- SQL file locations
- SHA-256 hashes
- Last validation timestamps
- Descriptions

### Validation Process

1. **Hash Check**: Verify SQL files haven't changed
2. **Data Fetch**: Get results from both SQL and Python
3. **Comparison**: Check numerical tolerance (default 1e-6)
4. **Reporting**: Log any discrepancies

## Development

### Adding New Queries

1. Add entry to `sql/registry.yaml`:
```yaml
queries:
  new_query:
    path: "path/to/query.sql"
    description: "Description of query"
    sha256: null
    last_validated: null
```

2. Implement Python equivalent in `curve_calculator.py`

3. Add validation test in `validator.py`

4. Run `curve-builder init` to calculate initial hash

### Running Tests

```bash
# Run all tests
pytest common_lib/curve_builder/tests/

# Run with coverage
pytest --cov=common_lib.curve_builder common_lib/curve_builder/tests/

# Run property-based tests with more examples
pytest common_lib/curve_builder/tests/ --hypothesis-show-statistics
```

### Database Migrations

```bash
# Create new migration
cd common_lib/curve_builder/migrations
alembic revision -m "Description of changes"

# Run migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Integration Points

- **PCA Factors**: Uses curve_mid_5s as input
- **Fair Value Curves**: Shares spot calculation logic
- **Continuous Contracts**: Depends on v_continuous_contracts_5s
- **Historical Infrastructure**: Manages tiered storage for backtesting

## Performance Considerations

- Materialized view refreshes every 5 minutes
- 2-day lookback window for live trading
- Historical data uses partitioned hypertables
- LZ4 compression for archived data

## Monitoring

The package tracks:
- SQL file changes
- Validation failures
- Performance metrics
- Data quality scores

Check logs for validation results and use Prometheus metrics for monitoring.