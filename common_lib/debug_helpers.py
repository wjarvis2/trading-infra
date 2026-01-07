"""
Debug helpers for faster development iteration.
Provides utilities for state inspection, data validation, and performance profiling.
"""
import functools
import time
import logging
import traceback
from typing import Any, Callable, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime
import json
import sys
from pathlib import Path


logger = logging.getLogger(__name__)


def debug_logger(name: str = None):
    """
    Create a debug logger with detailed formatting.
    
    Usage:
        log = debug_logger(__name__)
        log.debug("Processing data", extra={'shape': df.shape})
    """
    log_name = name or 'debug'
    log = logging.getLogger(log_name)
    log.setLevel(logging.DEBUG)
    
    if not log.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
    
    return log


def trace_calls(func: Callable) -> Callable:
    """
    Decorator to trace function calls with arguments and return values.
    
    Usage:
        @trace_calls
        def calculate_spread(front, back):
            return front - back
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log = debug_logger(func.__module__)
        
        # Log function entry
        arg_str = ', '.join([repr(a)[:50] for a in args[:3]])  # First 3 args
        kwarg_str = ', '.join([f"{k}={repr(v)[:30]}" for k, v in list(kwargs.items())[:3]])
        log.debug(f"→ {func.__name__}({arg_str}, {kwarg_str})")
        
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            
            # Log function exit
            result_str = repr(result)[:100] if result is not None else 'None'
            log.debug(f"← {func.__name__} returned {result_str} [{elapsed:.3f}s]")
            
            return result
            
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            log.error(f"✗ {func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    
    return wrapper


def validate_data(func: Callable) -> Callable:
    """
    Decorator to validate input/output data for common issues.
    Checks for NaN, inf, data types, and shape consistency.
    
    Usage:
        @validate_data
        def process_bars(df: pd.DataFrame) -> pd.DataFrame:
            return df.resample('1h').agg({'close': 'last'})
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log = debug_logger(func.__module__)
        
        # Validate inputs
        for i, arg in enumerate(args):
            _validate_argument(arg, f"arg_{i}", log)
        
        for key, val in kwargs.items():
            _validate_argument(val, key, log)
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Validate output
        if result is not None:
            _validate_argument(result, "return_value", log)
        
        return result
    
    return wrapper


def _validate_argument(arg: Any, name: str, log: logging.Logger):
    """Validate a single argument for common data issues."""
    if isinstance(arg, pd.DataFrame):
        if arg.empty:
            log.warning(f"{name} is empty DataFrame")
        if arg.isnull().any().any():
            null_cols = arg.columns[arg.isnull().any()].tolist()
            log.warning(f"{name} has NaN values in columns: {null_cols}")
        if np.isinf(arg.select_dtypes(include=[np.number])).any().any():
            log.warning(f"{name} has inf values")
            
    elif isinstance(arg, pd.Series):
        if arg.empty:
            log.warning(f"{name} is empty Series")
        if arg.isnull().any():
            log.warning(f"{name} has {arg.isnull().sum()} NaN values")
        if pd.api.types.is_numeric_dtype(arg) and np.isinf(arg).any():
            log.warning(f"{name} has inf values")
            
    elif isinstance(arg, np.ndarray):
        if arg.size == 0:
            log.warning(f"{name} is empty array")
        if np.isnan(arg).any():
            log.warning(f"{name} has NaN values")
        if np.isinf(arg).any():
            log.warning(f"{name} has inf values")


def profile_performance(func: Callable) -> Callable:
    """
    Decorator to profile function performance and memory usage.
    
    Usage:
        @profile_performance
        def expensive_calculation(data):
            return data.rolling(100).mean()
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        log = debug_logger(func.__module__)
        
        # Memory before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time execution
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before
        
        log.info(
            f"⚡ {func.__name__} - Time: {elapsed:.3f}s, "
            f"Memory: {mem_used:+.1f}MB (total: {mem_after:.1f}MB)"
        )
        
        # Warn if slow or memory intensive
        if elapsed > 1.0:
            log.warning(f"⚠ {func.__name__} is slow: {elapsed:.3f}s")
        if mem_used > 100:
            log.warning(f"⚠ {func.__name__} uses lots of memory: {mem_used:.1f}MB")
        
        return result
    
    return wrapper


class DataInspector:
    """
    Interactive data inspector for debugging.
    
    Usage:
        inspector = DataInspector()
        inspector.inspect(df, "spread_data")
        inspector.compare(df1, df2, "before", "after")
    """
    
    def __init__(self):
        self.log = debug_logger('DataInspector')
        self.snapshots = {}
    
    def inspect(self, data: Any, name: str = "data"):
        """Inspect data object with detailed statistics."""
        self.log.info(f"\n{'='*60}")
        self.log.info(f"Inspecting: {name}")
        self.log.info(f"{'='*60}")
        
        if isinstance(data, pd.DataFrame):
            self._inspect_dataframe(data, name)
        elif isinstance(data, pd.Series):
            self._inspect_series(data, name)
        elif isinstance(data, np.ndarray):
            self._inspect_array(data, name)
        elif isinstance(data, dict):
            self._inspect_dict(data, name)
        else:
            self.log.info(f"Type: {type(data)}")
            self.log.info(f"Value: {repr(data)[:200]}")
        
        # Save snapshot for comparison
        self.snapshots[name] = data
    
    def _inspect_dataframe(self, df: pd.DataFrame, name: str):
        """Inspect DataFrame with statistics."""
        self.log.info(f"Type: DataFrame")
        self.log.info(f"Shape: {df.shape}")
        self.log.info(f"Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        self.log.info(f"Columns: {df.columns.tolist()}")
        self.log.info(f"Index: {df.index.dtype} [{df.index[0]} ... {df.index[-1]}]" if not df.empty else "empty")
        
        # Data types
        self.log.info("\nData Types:")
        for col, dtype in df.dtypes.items():
            null_count = df[col].isnull().sum()
            null_pct = null_count / len(df) * 100 if len(df) > 0 else 0
            self.log.info(f"  {col}: {dtype} ({null_pct:.1f}% null)")
        
        # Statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.log.info("\nNumeric Statistics:")
            stats = df[numeric_cols].describe()
            self.log.info(f"\n{stats}")
        
        # Sample data
        self.log.info("\nFirst 3 rows:")
        self.log.info(f"\n{df.head(3)}")
    
    def _inspect_series(self, series: pd.Series, name: str):
        """Inspect Series with statistics."""
        self.log.info(f"Type: Series")
        self.log.info(f"Length: {len(series)}")
        self.log.info(f"Dtype: {series.dtype}")
        self.log.info(f"Memory: {series.memory_usage(deep=True) / 1024:.2f} KB")
        
        if pd.api.types.is_numeric_dtype(series):
            self.log.info(f"Min: {series.min():.4f}")
            self.log.info(f"Max: {series.max():.4f}")
            self.log.info(f"Mean: {series.mean():.4f}")
            self.log.info(f"Std: {series.std():.4f}")
            self.log.info(f"NaN count: {series.isnull().sum()}")
        
        self.log.info(f"\nFirst 5 values: {series.head().tolist()}")
    
    def _inspect_array(self, arr: np.ndarray, name: str):
        """Inspect numpy array."""
        self.log.info(f"Type: ndarray")
        self.log.info(f"Shape: {arr.shape}")
        self.log.info(f"Dtype: {arr.dtype}")
        self.log.info(f"Memory: {arr.nbytes / 1024:.2f} KB")
        
        if np.issubdtype(arr.dtype, np.number):
            self.log.info(f"Min: {arr.min():.4f}")
            self.log.info(f"Max: {arr.max():.4f}")
            self.log.info(f"Mean: {arr.mean():.4f}")
            self.log.info(f"Std: {arr.std():.4f}")
            self.log.info(f"NaN count: {np.isnan(arr).sum()}")
    
    def _inspect_dict(self, d: dict, name: str):
        """Inspect dictionary."""
        self.log.info(f"Type: dict")
        self.log.info(f"Keys: {list(d.keys())}")
        for key, val in list(d.items())[:5]:
            self.log.info(f"  {key}: {type(val).__name__} = {repr(val)[:50]}")
    
    def compare(self, data1: Any, data2: Any, name1: str = "before", name2: str = "after"):
        """Compare two data objects."""
        self.log.info(f"\n{'='*60}")
        self.log.info(f"Comparing: {name1} vs {name2}")
        self.log.info(f"{'='*60}")
        
        if type(data1) != type(data2):
            self.log.warning(f"Type mismatch: {type(data1)} vs {type(data2)}")
            return
        
        if isinstance(data1, pd.DataFrame):
            self._compare_dataframes(data1, data2, name1, name2)
        elif isinstance(data1, pd.Series):
            self._compare_series(data1, data2, name1, name2)
        elif isinstance(data1, np.ndarray):
            self._compare_arrays(data1, data2, name1, name2)
    
    def _compare_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str):
        """Compare two DataFrames."""
        self.log.info(f"Shape: {df1.shape} → {df2.shape}")
        
        # Column differences
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        if cols1 != cols2:
            added = cols2 - cols1
            removed = cols1 - cols2
            if added:
                self.log.info(f"Added columns: {added}")
            if removed:
                self.log.info(f"Removed columns: {removed}")
        
        # Compare common columns
        common_cols = cols1 & cols2
        for col in common_cols:
            if not df1[col].equals(df2[col]):
                self.log.info(f"Column '{col}' changed:")
                if pd.api.types.is_numeric_dtype(df1[col]):
                    diff = df2[col] - df1[col]
                    self.log.info(f"  Mean diff: {diff.mean():.4f}")
                    self.log.info(f"  Max diff: {diff.abs().max():.4f}")


def breakpoint_on_error(func: Callable) -> Callable:
    """
    Decorator to drop into debugger on error.
    
    Usage:
        @breakpoint_on_error
        def risky_function(data):
            return data / 0  # Will drop into pdb
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            import pdb
            import sys
            
            # Print exception info
            print(f"\n{'='*60}")
            print(f"Exception in {func.__name__}: {e}")
            print(f"{'='*60}")
            traceback.print_exc()
            print(f"{'='*60}\n")
            
            # Drop into debugger
            print("Dropping into debugger. Type 'h' for help.")
            pdb.post_mortem(sys.exc_info()[2])
            raise
    
    return wrapper


# Quick debugging function for notebooks
def debug_here():
    """
    Drop into debugger at current location.
    Useful for notebooks and scripts.
    
    Usage:
        from common_lib.debug_helpers import debug_here
        debug_here()  # Drops into pdb
    """
    import pdb
    pdb.set_trace()


# Export commonly used helpers
__all__ = [
    'debug_logger',
    'trace_calls',
    'validate_data',
    'profile_performance',
    'DataInspector',
    'breakpoint_on_error',
    'debug_here'
]