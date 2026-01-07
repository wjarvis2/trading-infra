"""
Bar data fetching logic - centralized access to OHLCV bar data.

This module provides unified interfaces for fetching bar data from:
- TimescaleDB continuous contracts views
- Raw bar tables (bar_15s, bar_30s, etc.)
- Aggregated bar data

IMPORTANT: Ensure database has indices on:
- (bucket, root, position) for continuous contract views
- (ts, contract_code) for individual bar tables

Author: Trading System Team
Date: 2025-08-02
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union, Literal, Tuple, NamedTuple
from datetime import datetime, timedelta
import logging
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
import time
from dataclasses import dataclass
import hashlib
import weakref
from collections import OrderedDict

# Import NoDataError from calendar module for consistency
from common_lib.utils.market_calendar import NoDataError

logger = logging.getLogger(__name__)

# Type definitions
BarFreq = Literal['5s', '15s', '30s', '1m', '5m', '1h', '1d']

# Global engine cache with weak references for cleanup
_engine_cache: Dict[str, weakref.ref] = {}


class BarFetchError(Exception):
    """Custom exception for bar fetching errors."""
    pass


@dataclass
class DataQualityResult:
    """Results from data quality validation."""
    valid: bool
    issues: List[str]
    rows: int
    time_range: Optional[Tuple[datetime, datetime]]
    
    def __str__(self):
        status = "VALID" if self.valid else "INVALID"
        return f"DataQuality[{status}]: {self.rows} rows, {len(self.issues)} issues"


def _get_engine(dsn: str) -> Engine:
    """Get or create a shared engine for the given DSN."""
    # Check if we have a live engine
    if dsn in _engine_cache:
        engine_ref = _engine_cache[dsn]
        engine = engine_ref()
        if engine is not None:
            return engine
    
    # Create new engine with connection pooling
    engine = create_engine(
        dsn,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,      # Prevent indefinite hangs
        pool_pre_ping=True,   # Verify connections before use
        pool_recycle=3600     # Recycle connections after 1 hour
    )
    
    # Store weak reference
    _engine_cache[dsn] = weakref.ref(engine)
    logger.info(f"Created new engine pool for DSN: {dsn.split('@')[1]}")
    
    return engine


def dispose_engines():
    """Dispose all cached engines. Call this in cleanup scenarios."""
    for dsn, engine_ref in list(_engine_cache.items()):
        engine = engine_ref()
        if engine is not None:
            engine.dispose()
            logger.info(f"Disposed engine for DSN: {dsn.split('@')[1]}")
    _engine_cache.clear()


class BarFetcher:
    """
    Unified interface for fetching OHLCV bar data.
    
    Provides consistent access to bar data across different:
    - Time frequencies (15s, 30s, 1m, 5m, etc.)
    - Contract types (continuous, individual)
    - Data sources (views, raw tables)
    
    Examples
    --------
    >>> fetcher = BarFetcher(db_conn_str)
    >>> bars = fetcher.fetch_continuous_bars('CL', [1, 2], '2024-01-01', '2024-01-31')
    >>> 
    >>> # Use as context manager for automatic cleanup
    >>> with BarFetcher(db_conn_str) as fetcher:
    ...     bars = fetcher.fetch_latest_bars('CL', [1, 2, 3])
    ...     print(fetcher.cache_info())
    """
    
    def __init__(self, db_conn_str: str, use_memory_cache: bool = False, max_cache_mb: int = 500):
        """
        Initialize bar fetcher.
        
        Parameters
        ----------
        db_conn_str : str
            Database connection string
        use_memory_cache : bool
            Enable in-memory caching for repeated queries
        max_cache_mb : int
            Maximum cache size in MB
        """
        self.db_conn_str = db_conn_str
        self.use_memory_cache = use_memory_cache
        self.max_cache_mb = max_cache_mb
        # Use OrderedDict for LRU cache behavior
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._cache_size_bytes = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
    @property
    def engine(self) -> Engine:
        """Get shared engine from global pool."""
        return _get_engine(self.db_conn_str)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clear cache."""
        self._cache.clear()
        self._cache_size_bytes = 0
    
    def cache_info(self) -> Dict[str, any]:
        """Get cache statistics."""
        return {
            'enabled': self.use_memory_cache,
            'size_mb': self._cache_size_bytes / (1024 * 1024),
            'max_size_mb': self.max_cache_mb,
            'entries': len(self._cache),
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) 
                       if (self._cache_hits + self._cache_misses) > 0 else 0
        }
    
    def _get_cache_key(self, **kwargs) -> str:
        """Generate cache key from query parameters."""
        # Create stable hash from parameters
        key_str = str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Check if query result is in cache."""
        if not self.use_memory_cache:
            return None
            
        if cache_key in self._cache:
            # Move to end for LRU behavior
            self._cache.move_to_end(cache_key)
            self._cache_hits += 1
            return self._cache[cache_key].copy()
        
        self._cache_misses += 1
        return None
    
    def _add_to_cache(self, cache_key: str, df: pd.DataFrame):
        """Add dataframe to cache with LRU eviction."""
        if not self.use_memory_cache:
            return
        
        # Estimate dataframe size
        df_size = df.memory_usage(deep=True).sum()
        
        # Check if we need to evict entries
        max_size_bytes = self.max_cache_mb * 1024 * 1024
        while self._cache_size_bytes + df_size > max_size_bytes and self._cache:
            # Evict least recently used (first item)
            oldest_key, evicted_df = self._cache.popitem(last=False)
            self._cache_size_bytes -= evicted_df.memory_usage(deep=True).sum()
        
        # Add to cache
        self._cache[cache_key] = df.copy()
        self._cache_size_bytes += df_size
    
    def fetch_continuous_bars(
        self,
        root: str,
        positions: List[int],
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        bar_freq: BarFreq = '15s',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch continuous contract bar data.
        
        Parameters
        ----------
        root : str
            Contract root (e.g., 'CL')
        positions : list[int]
            Contract positions (1=front, 2=second, etc.)
        start_time : str or datetime
            Start timestamp
        end_time : str or datetime
            End timestamp
        bar_freq : BarFreq
            Bar frequency ('5s', '15s', '30s', '1m', '5m')
        columns : list[str], optional
            Specific columns to fetch. Default: all OHLCV columns
        
        Returns
        -------
        pd.DataFrame
            Bar data with MultiIndex (timestamp, position)
            
        Raises
        ------
        BarFetchError
            If query fails or invalid parameters
        """
        # Check cache
        cache_key = self._get_cache_key(
            method='continuous',
            root=root,
            positions=positions,
            start_time=str(start_time),
            end_time=str(end_time),
            bar_freq=bar_freq,
            columns=columns
        )
        
        cached_result = self._check_cache(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for continuous bars query")
            return cached_result
        
        # Map bar frequencies to view names
        # Note: Only 5s view exists currently. Other frequencies need to be created.
        view_map = {
            '5s': 'v_continuous_contracts_5s',
            '15s': 'v_continuous_contracts_15s',
            '30s': 'v_continuous_contracts_30s',
            '1m': 'v_continuous_contracts_1m',
            '5m': 'v_continuous_contracts_5m',
            '10m': 'v_continuous_contracts_10m'
        }
        
        if bar_freq not in view_map:
            raise BarFetchError(f"Unsupported bar frequency: {bar_freq}")
        
        view_name = view_map[bar_freq]
        
        # Default columns
        if columns is None:
            columns = ['bucket', 'continuous_position', 'open', 'high', 'low', 'close', 'volume']
        
        # Validate positions
        if not positions or any(p < 1 or p > 20 for p in positions):
            raise BarFetchError(f"Invalid positions: {positions}. Must be between 1-20")
        
        # Build query with expanding parameter for IN clause
        col_str = ', '.join(columns)
        
        # Note: No root filter needed since continuous contract views are already CL-specific
        # The views are pre-filtered during creation (WHERE c.root = 'CL')
        query = text(f"""
        SELECT {col_str}
        FROM market_data.{view_name}
        WHERE continuous_position IN :positions
          AND bucket >= :start_time
          AND bucket <= :end_time
        ORDER BY bucket, continuous_position
        """).bindparams(bindparam("positions", expanding=True))
        
        # Execute with timing
        start = time.perf_counter()
        try:
            df = pd.read_sql(
                query,
                self.engine,
                params={
                    'positions': positions,
                    'start_time': start_time,
                    'end_time': end_time
                },
                parse_dates=['bucket']
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(f"Fetched {len(df)} continuous bars in {elapsed_ms:.1f}ms")
            
            # Log slow queries
            if elapsed_ms > 500:
                logger.info(f"Slow query detected: {elapsed_ms:.1f}ms for {len(df)} continuous bars")
            
        except Exception as e:
            raise BarFetchError(f"Failed to fetch continuous bars: {e}")
        
        # CRITICAL: Enforce "real data only" principle
        # If no data found, this is an error - caller should only request valid trading days
        if df.empty:
            raise NoDataError(
                f"No data found for {root} positions {positions} between {start_time} and {end_time}. "
                f"This should not happen if the optimizer is using valid trading days. "
                f"Check that the date range includes actual market sessions."
            )
        
        # Set MultiIndex
        if not df.empty:
            # Rename continuous_position back to position for consistent API
            if 'continuous_position' in df.columns:
                df.rename(columns={'continuous_position': 'position'}, inplace=True)
            
            if 'position' in df.columns:
                df.set_index(['bucket', 'position'], inplace=True)
        
        # Add to cache
        self._add_to_cache(cache_key, df)
        
        return df
    
    def fetch_individual_bars(
        self,
        contracts: List[str],
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        bar_freq: BarFreq = '15s',
        columns: Optional[List[str]] = None,
        chunksize: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch individual contract bar data.
        
        Parameters
        ----------
        contracts : list[str]
            Contract codes (e.g., ['CLF25', 'CLG25'])
        start_time : str or datetime
            Start timestamp
        end_time : str or datetime
            End timestamp
        bar_freq : BarFreq
            Bar frequency
        columns : list[str], optional
            Columns to fetch
        chunksize : int, optional
            Read in chunks for large queries
        
        Returns
        -------
        pd.DataFrame
            Bar data with MultiIndex (timestamp, contract_code)
        """
        # Map frequencies to table names
        table_map = {
            '15s': 'bar_15s',
            '30s': 'bar_30s',
            '1m': 'bar_1m',
            '5m': 'bar_5m',
            '1h': 'bar_1h',
            '1d': 'bar_daily'
        }
        
        if bar_freq not in table_map:
            raise BarFetchError(f"Unsupported bar frequency: {bar_freq}")
        
        table_name = table_map[bar_freq]
        
        # Default columns
        if columns is None:
            columns = ['ts', 'contract_code', 'open', 'high', 'low', 'close', 'volume']
        
        # Build query with expanding parameter
        col_str = ', '.join(columns)
        
        query = text(f"""
        SELECT {col_str}
        FROM market_data.{table_name}
        WHERE contract_code IN :contracts
          AND ts >= :start_time
          AND ts <= :end_time
        ORDER BY ts, contract_code
        """).bindparams(bindparam("contracts", expanding=True))
        
        # Execute query
        start = time.perf_counter()
        try:
            if chunksize:
                # Read in chunks for large queries
                chunks = []
                for chunk_df in pd.read_sql(
                    query,
                    self.engine,
                    params={
                        'contracts': contracts,
                        'start_time': start_time,
                        'end_time': end_time
                    },
                    parse_dates=['ts'],
                    chunksize=chunksize
                ):
                    chunks.append(chunk_df)
                df = pd.concat(chunks, ignore_index=True)
                # Restore sort order
                df = df.sort_values(['ts', 'contract_code'])
            else:
                df = pd.read_sql(
                    query,
                    self.engine,
                    params={
                        'contracts': contracts,
                        'start_time': start_time,
                        'end_time': end_time
                    },
                    parse_dates=['ts']
                )
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(f"Fetched {len(df)} individual bars in {elapsed_ms:.1f}ms")
            
            if elapsed_ms > 500:
                logger.info(f"Slow query detected: {elapsed_ms:.1f}ms for {len(df)} individual bars")
                
        except Exception as e:
            raise BarFetchError(f"Failed to fetch individual bars: {e}")
        
        # Set MultiIndex
        if not df.empty and 'contract_code' in df.columns:
            df.set_index(['ts', 'contract_code'], inplace=True)
        
        return df
    
    def fetch_curve_bars(
        self,
        root: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        bar_freq: BarFreq = '15s',
        max_position: int = 8
    ) -> pd.DataFrame:
        """
        Fetch full curve bar data (all positions).
        
        Parameters
        ----------
        root : str
            Contract root
        start_time : str or datetime
            Start timestamp
        end_time : str or datetime
            End timestamp
        bar_freq : BarFreq
            Bar frequency
        max_position : int
            Maximum position to fetch (default: 8)
        
        Returns
        -------
        pd.DataFrame
            Wide format with columns: position_1_close, position_2_close, etc.
        """
        # Fetch all positions
        positions = list(range(1, max_position + 1))
        
        # Get continuous bars
        df = self.fetch_continuous_bars(
            root=root,
            positions=positions,
            start_time=start_time,
            end_time=end_time,
            bar_freq=bar_freq,
            columns=['bucket', 'position', 'close']
        )
        
        if df.empty:
            return pd.DataFrame()
        
        # Pivot to wide format
        df_wide = df['close'].unstack(level='position')
        df_wide.columns = [f'position_{pos}_close' for pos in df_wide.columns]
        
        # Forward fill then backward fill (limited) to handle gaps
        df_wide = df_wide.ffill().bfill(limit=1)
        
        # Optional: Convert to float32 to save memory
        # df_wide = df_wide.astype('float32')
        
        return df_wide
    
    def fetch_latest_bars(
        self,
        root: str,
        positions: List[int],
        bar_freq: BarFreq = '15s',
        lookback_bars: int = 1,
        extra_buffer_multiplier: float = 2.0,
        timezone: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch most recent bars for live trading.
        
        Parameters
        ----------
        root : str
            Contract root
        positions : list[int]
            Contract positions
        bar_freq : BarFreq
            Bar frequency
        lookback_bars : int
            Number of recent bars to fetch
        extra_buffer_multiplier : float
            Extra buffer for ensuring we get enough data
        timezone : str, optional
            Timezone for 'now' calculation (e.g., 'America/New_York')
        
        Returns
        -------
        pd.DataFrame
            Recent bar data
        """
        # Map frequency to interval
        interval_map = {
            '5s': timedelta(seconds=5),
            '15s': timedelta(seconds=15),
            '30s': timedelta(seconds=30),
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '1h': timedelta(hours=1),
            '1d': timedelta(days=1)
        }
        
        if bar_freq not in interval_map:
            raise BarFetchError(f"Unsupported bar frequency: {bar_freq}")
        
        interval = interval_map[bar_freq]
        
        # Calculate time range with buffer
        if timezone:
            import pytz
            tz = pytz.timezone(timezone)
            end_time = datetime.now(tz)
        else:
            end_time = datetime.now()
            
        start_time = end_time - (interval * lookback_bars * extra_buffer_multiplier)
        
        # Fetch data
        df = self.fetch_continuous_bars(
            root=root,
            positions=positions,
            start_time=start_time,
            end_time=end_time,
            bar_freq=bar_freq
        )
        
        # Get last N bars per position
        if not df.empty:
            df = df.groupby(level='position').tail(lookback_bars)
        
        return df
    
    def validate_data_quality(
        self,
        df: pd.DataFrame,
        check_gaps: bool = True,
        check_outliers: bool = True,
        max_gap_bars: int = 10,
        max_return_pct: float = 0.10
    ) -> DataQualityResult:
        """
        Validate bar data quality.
        
        Parameters
        ----------
        df : pd.DataFrame
            Bar data to validate
        check_gaps : bool
            Check for time gaps
        check_outliers : bool
            Check for price outliers
        max_gap_bars : int
            Maximum allowed gap in bars
        max_return_pct : float
            Maximum allowed single-bar return
        
        Returns
        -------
        DataQualityResult
            Validation results
        """
        issues = []
        
        if df.empty:
            return DataQualityResult(
                valid=False,
                issues=['Empty dataframe'],
                rows=0,
                time_range=None
            )
        
        # Check for gaps
        if check_gaps and 'bucket' in df.index.names:
            time_idx = df.index.get_level_values('bucket')
            time_diff = time_idx.to_series().diff()
            median_diff = time_diff.median()
            
            if pd.notna(median_diff):
                gaps = time_diff[time_diff > median_diff * max_gap_bars]
                if len(gaps) > 0:
                    issues.append(f"Found {len(gaps)} time gaps larger than {max_gap_bars} bars")
        
        # Check for outliers
        if check_outliers and 'close' in df.columns:
            close_prices = df['close']
            
            # Check for zero/negative prices
            if (close_prices <= 0).any():
                issues.append("Found zero or negative prices")
            
            # Check for extreme moves
            returns = close_prices.pct_change()
            extreme_moves = returns[returns.abs() > max_return_pct]
            if len(extreme_moves) > 0:
                issues.append(f"Found {len(extreme_moves)} extreme price moves (>{max_return_pct:.0%})")
        
        # Check for missing OHLC consistency
        ohlc_cols = ['open', 'high', 'low', 'close']
        if all(col in df.columns for col in ohlc_cols):
            # High should be >= max(open, close)
            invalid_high = df['high'] < df[['open', 'close']].max(axis=1)
            if invalid_high.any():
                issues.append(f"Found {invalid_high.sum()} bars with invalid high prices")
            
            # Low should be <= min(open, close)
            invalid_low = df['low'] > df[['open', 'close']].min(axis=1)
            if invalid_low.any():
                issues.append(f"Found {invalid_low.sum()} bars with invalid low prices")
        
        # Get time range
        time_range = None
        if not df.empty:
            if 'bucket' in df.index.names:
                time_idx = df.index.get_level_values('bucket')
            elif 'ts' in df.index.names:
                time_idx = df.index.get_level_values('ts')
            else:
                time_idx = df.index
            
            time_range = (time_idx.min(), time_idx.max())
        
        return DataQualityResult(
            valid=len(issues) == 0,
            issues=issues,
            rows=len(df),
            time_range=time_range
        )

    def fetch_front8_bars(
        self,
        root: str,
        positions: List[int],
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        columns: Optional[List[str]] = None,
        trading_hours_only: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch bar data from bars_5s_front8 table.

        This table contains 5-second bars for front 8 contracts with
        both Databento historical data and IBKR live data.

        Parameters
        ----------
        root : str
            Contract root (e.g., 'CL', 'RB', 'HO', 'NG')
        positions : list[int]
            Contract positions (1=front month, 2=second month, etc.)
        start_time : str or datetime
            Start timestamp
        end_time : str or datetime
            End timestamp
        columns : list[str], optional
            Columns to fetch. Default: ts, root, position, open, high, low, close, volume
        trading_hours_only : bool
            If True, filter to CME trading hours (18:00-17:00 ET)

        Returns
        -------
        pd.DataFrame
            Bar data with MultiIndex (ts, position)

        Raises
        ------
        BarFetchError
            If query fails or no data found
        """
        # Default columns
        if columns is None:
            columns = ['ts', 'root', 'position', 'symbol', 'open', 'high', 'low', 'close', 'volume']

        col_str = ', '.join(columns)

        # Trading hours filter (CME: 18:00-17:00 ET = 23 hours)
        hours_filter = ""
        if trading_hours_only:
            hours_filter = """
                AND (
                    EXTRACT(HOUR FROM ts AT TIME ZONE 'America/New_York') >= 18
                    OR EXTRACT(HOUR FROM ts AT TIME ZONE 'America/New_York') < 17
                )
            """

        query = text(f"""
            SELECT {col_str}
            FROM market_data.v_continuous_5s
            WHERE root = :root
              AND position IN :positions
              AND ts >= :start_time
              AND ts < :end_time
              {hours_filter}
            ORDER BY ts, position
        """).bindparams(bindparam("positions", expanding=True))

        start = time.perf_counter()
        try:
            df = pd.read_sql(
                query,
                self.engine,
                params={
                    'root': root,
                    'positions': positions,
                    'start_time': start_time,
                    'end_time': end_time,
                },
                parse_dates=['ts']
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(f"Fetched {len(df)} front8 bars in {elapsed_ms:.1f}ms")

            if elapsed_ms > 500:
                logger.info(f"Slow query: {elapsed_ms:.1f}ms for {len(df)} front8 bars")

        except Exception as e:
            raise BarFetchError(f"Failed to fetch front8 bars: {e}")

        if df.empty:
            raise NoDataError(
                f"No data found for {root} positions {positions} "
                f"between {start_time} and {end_time} in v_continuous_5s"
            )

        # Set MultiIndex
        if 'position' in df.columns:
            df.set_index(['ts', 'position'], inplace=True)

        return df