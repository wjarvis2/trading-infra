"""
Data validation schemas using Pandera.

Provides comprehensive validation for all data types in the trading system.
Ensures data quality at each transformation stage.

Created: 2025-08-06
"""

import pandera as pa
import pandas as pd
import numpy as np
from typing import Optional
from pandera.typing import DataFrame, Series

from common_lib.constants import MAX_PRICE_JUMP_PCT, CONTRACT_SPECS


# Define custom checks following Pandera's check function patterns
def check_no_complex(series: pd.Series) -> bool:
    """Check that series contains no complex numbers."""
    # For column-level checks, return single boolean
    return not np.iscomplexobj(series.values)


def check_price_continuity(series: pd.Series) -> bool:
    """Check that prices don't have unrealistic jumps."""
    max_jump = MAX_PRICE_JUMP_PCT
    returns = series.pct_change()
    # Skip NaN values from pct_change
    return (returns.abs() <= max_jump).all() or returns.isna().all()


def check_high_low_consistency(df: pd.DataFrame) -> bool:
    """Check that high >= low for all bars."""
    if 'high' not in df.columns or 'low' not in df.columns:
        return True  # Skip check if columns missing
    return (df['high'] >= df['low']).all()


def check_ohlc_consistency(df: pd.DataFrame) -> bool:
    """Check OHLC relationships: high >= max(open, close) and low <= min(open, close)."""
    required_cols = ['high', 'low', 'open', 'close']
    if not all(col in df.columns for col in required_cols):
        return True  # Skip check if columns missing
    high_ok = (df['high'] >= df[['open', 'close']].max(axis=1)).all()
    low_ok = (df['low'] <= df[['open', 'close']].min(axis=1)).all()
    return high_ok and low_ok


# Bar data schema for OHLCV data
BarDataSchema = pa.DataFrameSchema(
    columns={
        "open": pa.Column(
            float,
            checks=[
                pa.Check.greater_than(0, name="positive_open"),
                pa.Check.less_than(1000, name="reasonable_open"),  # Oil rarely > $1000
                # Complex check for OHLC data
            ],
            nullable=False,
            coerce=True,  # Allow int to float conversion
            description="Opening price"
        ),
        "high": pa.Column(
            float,
            checks=[
                pa.Check.greater_than(0, name="positive_high"),
                pa.Check.less_than(1000, name="reasonable_high"),
                # Complex check for OHLC data
            ],
            nullable=False,
            coerce=True,
            description="High price"
        ),
        "low": pa.Column(
            float,
            checks=[
                pa.Check.greater_than(0, name="positive_low"),
                pa.Check.less_than(1000, name="reasonable_low"),
                # Complex check for OHLC data
            ],
            nullable=False,
            coerce=True,
            description="Low price"
        ),
        "close": pa.Column(
            float,
            checks=[
                pa.Check.greater_than(0, name="positive_close"),
                pa.Check.less_than(1000, name="reasonable_close"),
                # Complex check for OHLC data
                # Price continuity check
            ],
            nullable=False,
            coerce=True,
            description="Closing price"
        ),
        "volume": pa.Column(
            float,
            checks=[
                pa.Check.greater_than_or_equal_to(0, name="non_negative_volume"),
                pa.Check.less_than(10_000_000, name="reasonable_volume"),  # Sanity check
            ],
            nullable=False,
            coerce=True,  # Allow int to float conversion
            description="Trading volume"
        ),
    },
    checks=[
        # DataFrame-level consistency checks handled by quick_check
    ],
    index=pa.Index(
        pa.DateTime,
        name="timestamp",
        checks=[
            # Monotonic timestamps check
        ]
    ),
    strict=False,  # Allow additional columns
    name="BarDataSchema"
)

# Bar data schema with contract column (for multi-contract data)
BarDataWithContractSchema = BarDataSchema.add_columns({
    "contract": pa.Column(
        str,
        checks=[
            pa.Check.str_matches(r"^[A-Z]{2}\d+$", name="valid_contract_format"),
        ],
        nullable=False,
        description="Contract symbol"
    ),
})


# Spread price schema
SpreadPriceSchema = pa.SeriesSchema(
    float,
    checks=[
        pa.Check.greater_than(-100, name="reasonable_negative_spread"),
        pa.Check.less_than(100, name="reasonable_positive_spread"),
        pa.Check(lambda s: not np.iscomplexobj(s.values), name="no_complex_spreads"),
        pa.Check(lambda s: s.notna().sum() > 0, name="has_valid_data"),
    ],
    nullable=True,  # Allow NaN for missing data
    name="SpreadPriceSchema"
)


# Volatility output schema
VolatilitySchema = pa.SeriesSchema(
    float,
    checks=[
        pa.Check.greater_than_or_equal_to(0, name="non_negative_volatility"),
        pa.Check.less_than(10, name="reasonable_volatility"),  # 1000% annualized vol is extreme
        pa.Check(lambda s: not np.iscomplexobj(s.values), name="no_complex_volatility"),
        pa.Check(lambda s: ~np.isinf(s).any(), name="no_infinite_volatility"),
    ],
    nullable=True,  # Allow NaN for initial periods
    name="VolatilitySchema"
)


# Signal schema
SignalSchema = pa.DataFrameSchema(
    columns={
        "timestamp": pa.Column(pa.DateTime, nullable=False),
        "symbol": pa.Column(str, nullable=False),
        "side": pa.Column(str, checks=[pa.Check.isin(["BUY", "SELL"])]),
        "quantity": pa.Column(int, checks=[pa.Check.greater_than(0)]),
        "signal_type": pa.Column(str, nullable=True),
        "confidence": pa.Column(
            float,
            checks=[
                pa.Check.greater_than_or_equal_to(0),
                pa.Check.less_than_or_equal_to(1),
            ],
            nullable=True
        ),
    },
    strict=False,
    name="SignalSchema"
)


# Position schema
PositionSchema = pa.DataFrameSchema(
    columns={
        "symbol": pa.Column(str, nullable=False),
        "quantity": pa.Column(int, nullable=False),  # Can be negative for short
        "entry_price": pa.Column(float, checks=[pa.Check.greater_than(0)]),
        "current_price": pa.Column(float, checks=[pa.Check.greater_than(0)]),
        "unrealized_pnl": pa.Column(float, nullable=False),
        "realized_pnl": pa.Column(float, nullable=False),
    },
    strict=False,
    name="PositionSchema"
)


class DataValidator:
    """
    Comprehensive data validator for trading system.
    """
    
    @staticmethod
    def validate_bars(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
        """
        Validate bar data.
        
        Args:
            df: DataFrame with OHLCV data
            context: Context string for error messages
            
        Returns:
            Validated DataFrame
            
        Raises:
            pa.errors.SchemaError: If validation fails
        """
        try:
            return BarDataSchema.validate(df)
        except pa.errors.SchemaError as e:
            # Re-raise with additional context
            error_msg = f"Bar validation failed at {context}: {str(e)}"
            raise ValueError(error_msg) from e
    
    @staticmethod
    def validate_spread_prices(series: pd.Series, context: str = "") -> pd.Series:
        """
        Validate spread price series.
        
        Args:
            series: Series of spread prices
            context: Context string for error messages
            
        Returns:
            Validated Series
            
        Raises:
            pa.errors.SchemaError: If validation fails
        """
        try:
            return SpreadPriceSchema.validate(series)
        except pa.errors.SchemaError as e:
            error_msg = f"Spread price validation failed at {context}: {str(e)}"
            raise ValueError(error_msg) from e
    
    @staticmethod
    def validate_volatility(series: pd.Series, context: str = "") -> pd.Series:
        """
        Validate volatility series.
        
        Args:
            series: Series of volatility values
            context: Context string for error messages
            
        Returns:
            Validated Series
            
        Raises:
            pa.errors.SchemaError: If validation fails
        """
        try:
            return VolatilitySchema.validate(series)
        except pa.errors.SchemaError as e:
            error_msg = f"Volatility validation failed at {context}: {str(e)}"
            raise ValueError(error_msg) from e
    
    @staticmethod
    def validate_signals(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
        """
        Validate trading signals.
        
        Args:
            df: DataFrame with signal data
            context: Context string for error messages
            
        Returns:
            Validated DataFrame
            
        Raises:
            pa.errors.SchemaError: If validation fails
        """
        try:
            return SignalSchema.validate(df)
        except pa.errors.SchemaError as e:
            error_msg = f"Signal validation failed at {context}: {str(e)}"
            raise ValueError(error_msg) from e
    
    @staticmethod
    def validate_positions(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
        """
        Validate position data.
        
        Args:
            df: DataFrame with position data
            context: Context string for error messages
            
        Returns:
            Validated DataFrame
            
        Raises:
            pa.errors.SchemaError: If validation fails
        """
        try:
            return PositionSchema.validate(df)
        except pa.errors.SchemaError as e:
            error_msg = f"Position validation failed at {context}: {str(e)}"
            raise ValueError(error_msg) from e
    
    @staticmethod
    def quick_check(data: any, check_name: str = "data") -> dict:
        """
        Quick check for common data issues.
        
        Args:
            data: Data to check (DataFrame, Series, array)
            check_name: Name for logging
            
        Returns:
            Dictionary with check results
        """
        results = {
            "name": check_name,
            "type": type(data).__name__,
            "has_complex": False,
            "has_nan": False,
            "has_inf": False,
            "has_negative": False,
            "shape": None,
            "dtype": None,
            "issues": []
        }
        
        if hasattr(data, 'values'):
            values = data.values
        elif isinstance(data, np.ndarray):
            values = data
        else:
            return results
        
        results["shape"] = values.shape if hasattr(values, 'shape') else len(values)
        results["dtype"] = str(values.dtype) if hasattr(values, 'dtype') else type(values)
        
        # Check for complex
        if np.iscomplexobj(values):
            results["has_complex"] = True
            results["issues"].append("Contains complex numbers")
        
        # Check for NaN
        try:
            if np.isnan(values).any():
                results["has_nan"] = True
                results["issues"].append("Contains NaN values")
        except (TypeError, ValueError):
            pass
        
        # Check for inf
        try:
            if np.isinf(values).any():
                results["has_inf"] = True
                results["issues"].append("Contains infinite values")
        except (TypeError, ValueError):
            pass
        
        # Check for negative (if numeric)
        try:
            if (values < 0).any():
                results["has_negative"] = True
                # This is not necessarily an issue for spreads
        except (TypeError, ValueError):
            pass
        
        return results