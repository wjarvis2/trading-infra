"""
Schema Enforcer for Data Loading Architecture
Validates and enforces canonical schemas for each data type
"""

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime, timezone


class SchemaValidationError(Exception):
    """Raised when data doesn't conform to schema"""
    pass


class SchemaEnforcer:
    """Enforces canonical schemas for trading data types"""
    
    # Canonical schema for bar data
    BARS_SCHEMA = {
        'required_columns': ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'],
        'optional_columns': ['vwap', 'trades', 'bid', 'ask'],
        'index_names': ['timestamp', 'symbol'],
        'dtypes': {
            'open': 'float64',
            'high': 'float64', 
            'low': 'float64',
            'close': 'float64',
            'volume': 'float64',
            'vwap': 'float64',
            'trades': 'int64',
            'bid': 'float64',
            'ask': 'float64'
        }
    }
    
    # Schema for inventory/fundamental data
    INVENTORY_SCHEMA = {
        'required_columns': ['date', 'series_code', 'value'],
        'optional_columns': ['units', 'source'],
        'index_names': ['date', 'series_code'],
        'dtypes': {
            'value': 'float64',
            'units': 'object',
            'source': 'object'
        }
    }
    
    # Schema for curve data
    CURVE_SCHEMA = {
        'required_columns': ['date', 'contract', 'price', 'days_to_expiry'],
        'optional_columns': ['volume', 'open_interest'],
        'index_names': ['date', 'contract'],
        'dtypes': {
            'price': 'float64',
            'days_to_expiry': 'int64',
            'volume': 'float64',
            'open_interest': 'float64'
        }
    }
    
    # Schema for spread data
    SPREAD_SCHEMA = {
        'required_columns': ['timestamp', 'spread_type', 'value'],
        'optional_columns': ['leg1', 'leg2', 'leg3', 'ratio'],
        'index_names': ['timestamp', 'spread_type'],
        'dtypes': {
            'value': 'float64',
            'ratio': 'object'
        }
    }
    
    def enforce_bars(self, df: pd.DataFrame, 
                    check_duplicates: bool = True,
                    check_monotonic: bool = True,
                    check_constraints: bool = True) -> pd.DataFrame:
        """
        Enforce bar data schema with validation
        
        Args:
            df: Input dataframe to validate
            check_duplicates: Check for duplicate (timestamp, symbol) pairs
            check_monotonic: Check timestamp monotonicity per symbol
            check_constraints: Check data constraints (OHLC relationships, positive volume)
            
        Returns:
            Validated dataframe with MultiIndex
            
        Raises:
            SchemaValidationError: If validation fails
        """
        if df is None or df.empty:
            raise SchemaValidationError("Cannot enforce schema on empty dataframe")
            
        df = df.copy()
        
        # Check required columns
        missing_cols = set(self.BARS_SCHEMA['required_columns']) - set(df.columns)
        if missing_cols:
            raise SchemaValidationError(f"Missing required columns: {missing_cols}")
            
        # Ensure UTC timezone on timestamp
        if 'timestamp' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                if df['timestamp'].dt.tz is None:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                elif str(df['timestamp'].dt.tz) != 'UTC':
                    df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                
        # Type coercion for numeric columns
        for col, dtype in self.BARS_SCHEMA['dtypes'].items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError) as e:
                    raise SchemaValidationError(f"Cannot cast {col} to {dtype}: {e}")
                    
        # Set MultiIndex
        df = df.set_index(self.BARS_SCHEMA['index_names'])
        
        # Check for duplicates
        if check_duplicates:
            duplicates = df.index.duplicated()
            if duplicates.any():
                dup_count = duplicates.sum()
                first_dup = df[duplicates].index[0]
                raise SchemaValidationError(
                    f"Found {dup_count} duplicate (timestamp, symbol) pairs. "
                    f"First duplicate: {first_dup}"
                )
                
        # Check monotonic timestamps per symbol
        if check_monotonic:
            for symbol in df.index.get_level_values('symbol').unique():
                symbol_data = df.xs(symbol, level='symbol')
                if not symbol_data.index.is_monotonic_increasing:
                    raise SchemaValidationError(
                        f"Timestamps not monotonic for symbol {symbol}"
                    )
                    
        # Check OHLC constraints
        if check_constraints:
            # High >= Low
            invalid_hl = df['high'] < df['low']
            if invalid_hl.any():
                raise SchemaValidationError(
                    f"Found {invalid_hl.sum()} bars where high < low"
                )
                
            # High >= Open, Close
            invalid_high = (df['high'] < df['open']) | (df['high'] < df['close'])
            if invalid_high.any():
                raise SchemaValidationError(
                    f"Found {invalid_high.sum()} bars where high < open or close"
                )
                
            # Low <= Open, Close  
            invalid_low = (df['low'] > df['open']) | (df['low'] > df['close'])
            if invalid_low.any():
                raise SchemaValidationError(
                    f"Found {invalid_low.sum()} bars where low > open or close"
                )
                
            # Volume >= 0
            if (df['volume'] < 0).any():
                raise SchemaValidationError("Found negative volume values")
                
        return df
    
    def enforce_inventory(self, df: pd.DataFrame,
                         check_duplicates: bool = True) -> pd.DataFrame:
        """Enforce inventory/fundamental data schema"""
        if df is None or df.empty:
            raise SchemaValidationError("Cannot enforce schema on empty dataframe")
            
        df = df.copy()
        
        # Check required columns
        missing_cols = set(self.INVENTORY_SCHEMA['required_columns']) - set(df.columns)
        if missing_cols:
            raise SchemaValidationError(f"Missing required columns: {missing_cols}")
            
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Type coercion
        for col, dtype in self.INVENTORY_SCHEMA['dtypes'].items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
                
        # Set MultiIndex
        df = df.set_index(self.INVENTORY_SCHEMA['index_names'])
        
        # Check duplicates
        if check_duplicates and df.index.duplicated().any():
            raise SchemaValidationError("Found duplicate (date, series_code) pairs")
            
        return df
    
    def enforce_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Enforce schemas on all data types in dictionary
        
        Args:
            data: Dictionary of dataframes by type
            
        Returns:
            Dictionary with all dataframes validated
        """
        result = {}
        
        if 'bars' in data and data['bars'] is not None:
            result['bars'] = self.enforce_bars(data['bars'])
            
        if 'inventory' in data and data['inventory'] is not None:
            result['inventory'] = self.enforce_inventory(data['inventory'])
            
        # Pass through any other data unchanged for now
        for key, value in data.items():
            if key not in result:
                result[key] = value
                
        return result