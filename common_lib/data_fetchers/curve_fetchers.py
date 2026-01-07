"""
Curve data fetching logic - single source of truth for all curve data access.

This module centralizes all curve data fetching from various sources:
- TimescaleDB (live and historical)
- Parquet files (historical exports)
- Materialized views (performance optimization)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import psycopg2
from sqlalchemy import create_engine
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CurveFetcher:
    """
    Unified interface for fetching curve data from all sources.
    
    This class provides a single interface for accessing curve data
    regardless of the underlying storage mechanism.
    """
    
    def __init__(self, db_conn_str: Optional[str] = None):
        """
        Initialize curve fetcher.
        
        Args:
            db_conn_str: Database connection string (optional)
        """
        self.db_conn_str = db_conn_str
        self._engine = None
    
    @property
    def engine(self):
        """Lazy-load SQLAlchemy engine."""
        if self._engine is None and self.db_conn_str:
            self._engine = create_engine(self.db_conn_str)
        return self._engine
    
    def fetch_from_db(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        contracts: List[str] = None,
        source_table: str = "market_data.curve_mid_5s"
    ) -> pd.DataFrame:
        """
        Fetch curve data from database.
        
        Args:
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)
            contracts: List of contracts to fetch (default: all CL1-CL8)
            source_table: Table to query from
            
        Returns:
            DataFrame with curve data (index: ts, columns: cl1-cl8)
            
        Raises:
            ValueError: If no database connection configured
            RuntimeError: If query fails
        """
        if not self.db_conn_str:
            raise ValueError("No database connection configured")
            
        if contracts is None:
            contracts = [f"cl{i}" for i in range(1, 9)]
            
        # Build column list
        columns = ", ".join(contracts)
        
        query = f"""
        SELECT ts, {columns}
        FROM {source_table}
        WHERE ts >= %s AND ts <= %s
        ORDER BY ts
        """
        
        try:
            df = pd.read_sql(
                query,
                self.engine,
                params=(start_time, end_time),
                parse_dates=['ts'],
                index_col='ts'
            )
            
            # Ensure timezone aware
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch curve data: {e}")
            raise RuntimeError(f"Database query failed: {str(e)}")
    
    def fetch_from_parquet(
        self,
        file_path: Path,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        contracts: List[str] = None
    ) -> pd.DataFrame:
        """
        Fetch curve data from parquet file.
        
        Args:
            file_path: Path to parquet file
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)
            contracts: List of contracts to fetch
            
        Returns:
            DataFrame with curve data
            
        Raises:
            FileNotFoundError: If parquet file doesn't exist
            RuntimeError: If read fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")
            
        try:
            # Read parquet file
            df = pd.read_parquet(file_path)
            
            # Ensure timestamp index
            if 'ts' in df.columns and df.index.name != 'ts':
                df = df.set_index('ts')
            
            # Filter time range
            mask = (df.index >= start_time) & (df.index <= end_time)
            df = df[mask]
            
            # Filter contracts if specified
            if contracts:
                available_cols = [col for col in contracts if col in df.columns]
                df = df[available_cols]
            
            # Ensure timezone aware
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to read parquet file: {e}")
            raise RuntimeError(f"Parquet read failed: {str(e)}")
    
    def fetch_from_materialized_view(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        view_name: str = "market_data.mv_curve_mid_5s_1min"
    ) -> pd.DataFrame:
        """
        Fetch from materialized view for better performance.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            view_name: Materialized view name
            
        Returns:
            DataFrame with curve data
        """
        # Materialized views are just special tables
        return self.fetch_from_db(start_time, end_time, source_table=view_name)
    
    def fetch_expiry_data(
        self,
        reference_date: pd.Timestamp,
        contracts: List[str] = None
    ) -> Dict[str, float]:
        """
        Fetch contract expiry data for a given date.
        
        Args:
            reference_date: Date to get expiry mapping for
            contracts: List of contracts (default: cl1-cl8)
            
        Returns:
            Dict mapping contract names to days to expiry
        """
        if not self.db_conn_str:
            raise ValueError("No database connection configured")
            
        if contracts is None:
            contracts = [f"cl{i}" for i in range(1, 9)]
            
        # Extract position numbers
        positions = [int(c[2:]) for c in contracts if c.startswith('cl')]
        
        query = """
        WITH latest_mapping AS (
            SELECT DISTINCT ON (v.continuous_position)
                v.continuous_position,
                v.instrument_id,
                c.expiry,
                v.bucket
            FROM market_data.v_continuous_contracts v
            JOIN market_data.futures_contracts c ON v.instrument_id = c.instrument_id
            WHERE v.bucket::date = %s::date
              AND v.continuous_position = ANY(%s)
            ORDER BY v.continuous_position, v.bucket DESC
        )
        SELECT 
            lm.continuous_position,
            (lm.expiry - %s::date)::int as days_to_expiry
        FROM latest_mapping lm
        ORDER BY continuous_position
        """
        
        try:
            with psycopg2.connect(self.db_conn_str) as conn:
                df = pd.read_sql(
                    query,
                    conn,
                    params=(reference_date, positions, reference_date)
                )
                
            # Build expiry map
            expiry_map = {}
            for _, row in df.iterrows():
                pos = int(row['continuous_position'])
                expiry_map[f'cl{pos}'] = float(row['days_to_expiry'])
                
            # Fill any missing with defaults
            defaults = {
                'cl1': 21, 'cl2': 52, 'cl3': 82, 'cl4': 113,
                'cl5': 143, 'cl6': 174, 'cl7': 204, 'cl8': 235
            }
            for contract in contracts:
                if contract not in expiry_map and contract in defaults:
                    expiry_map[contract] = defaults[contract]
                    logger.warning(f"Using default expiry for {contract}: {defaults[contract]} days")
                    
            return expiry_map
            
        except Exception as e:
            logger.error(f"Failed to fetch expiry data: {e}")
            # Return defaults on error
            return {
                'cl1': 21, 'cl2': 52, 'cl3': 82, 'cl4': 113,
                'cl5': 143, 'cl6': 174, 'cl7': 204, 'cl8': 235
            }
    
    def fetch_with_quality_check(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        min_completeness: float = 0.8,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Fetch curve data with quality metrics.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            min_completeness: Minimum data completeness required
            **kwargs: Additional arguments for fetch_from_db
            
        Returns:
            Tuple of (curve_data, quality_metrics)
            
        Raises:
            ValueError: If data quality is below threshold
        """
        # Fetch data
        df = self.fetch_from_db(start_time, end_time, **kwargs)
        
        if df.empty:
            raise ValueError("No data found in specified time range")
        
        # Calculate quality metrics
        metrics = {}
        
        # Data completeness
        for col in df.columns:
            completeness = df[col].notna().sum() / len(df)
            metrics[f'{col}_completeness'] = completeness
            
        overall_completeness = df.notna().sum().sum() / (len(df) * len(df.columns))
        metrics['overall_completeness'] = overall_completeness
        
        # Check threshold
        if overall_completeness < min_completeness:
            raise ValueError(
                f"Data completeness {overall_completeness:.2%} below "
                f"required threshold {min_completeness:.2%}"
            )
        
        # Time gaps
        if len(df) > 1:
            time_diffs = df.index.to_series().diff()
            metrics['max_time_gap'] = time_diffs.max().total_seconds()
            metrics['mean_time_gap'] = time_diffs.mean().total_seconds()
        
        return df, metrics
    
    def fetch_for_backtest(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        lookback_days: int = 30,
        contracts: List[str] = None
    ) -> pd.DataFrame:
        """
        Fetch curve data for backtesting with proper lookback.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            lookback_days: Days of history needed for indicators
            contracts: Contracts to fetch
            
        Returns:
            DataFrame with curve data including lookback period
        """
        # Adjust start date for lookback
        adjusted_start = start_date - pd.Timedelta(days=lookback_days)
        
        # Fetch data
        df = self.fetch_from_db(adjusted_start, end_date, contracts)
        
        # Mark the actual backtest period
        df['is_backtest_period'] = df.index >= start_date
        
        return df