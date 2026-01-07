"""
PCA factor data fetcher.

Fetches pre-calculated PCA factors from market_data.curve_f8_pca table
and related metadata. Provides both real-time and historical factor access.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from .bar_fetchers import BarFetchError
import sqlalchemy as sa
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


@dataclass
class PCAFactorData:
    """Container for PCA factor data."""
    timestamp: pd.Timestamp
    factors: pd.Series  # PC1, PC2, PC3, etc.
    loadings: pd.DataFrame  # Contract loadings for each factor
    variance_explained: pd.Series  # Variance explained by each factor
    bar_freq: str
    window_bars: int


class PCAFetcher:
    """
    Fetches PCA factors from the database.
    
    This fetcher retrieves pre-calculated PCA factors from market_data.curve_f8_pca
    which are computed every 5 minutes on 15-second bars.
    """
    
    def __init__(self, engine: Engine, source_table: str = 'curve_f8_pca'):
        """
        Initialize PCA fetcher.
        
        Parameters
        ----------
        engine : sa.Engine
            SQLAlchemy engine for database access
        source_table : str
            Table to fetch PCA factors from (default: curve_f8_pca)
        """
        self.engine = engine
        self.source_table = source_table
        
    def get_latest_factors(
        self,
        bar_freq: str = '15 seconds',
        window_bars: int = 240
    ) -> Optional[PCAFactorData]:
        """
        Get the most recent PCA factors.
        
        Parameters
        ----------
        bar_freq : str
            Bar frequency to filter by
        window_bars : int
            Window size to filter by
            
        Returns
        -------
        Optional[PCAFactorData]
            Latest PCA factor data or None if not available
        """
        query = f"""
        WITH latest_ts AS (
            SELECT MAX(ts) as max_ts
            FROM market_data.{self.source_table}
            WHERE bar_freq = %s::interval
            AND window_bars = %s
        ),
        factors AS (
            SELECT 
                ts,
                bar_freq,
                window_bars,
                factor_idx,
                score,
                exp_var,
                loadings
            FROM market_data.{self.source_table}
            WHERE ts = (SELECT max_ts FROM latest_ts)
            AND bar_freq = %s::interval
            AND window_bars = %s
            ORDER BY factor_idx
        )
        SELECT * FROM factors;
        """
        
        try:
            engine = self.engine
            df = pd.read_sql_query(
                query,
                engine,
                params=(bar_freq, window_bars, bar_freq, window_bars)
            )
            
            if df.empty:
                logger.warning(f"No PCA factors found for {bar_freq}/{window_bars}")
                return None
                
            return self._process_factor_data(df)
            
        except Exception as e:
            logger.error(f"Error fetching latest PCA factors: {e}")
            raise BarFetchError(f"Failed to fetch PCA factors: {e}")
    
    def get_historical_factors(
        self,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        bar_freq: str = '15 seconds',
        window_bars: int = 240,
        factor_idx: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get historical PCA factors.
        
        Parameters
        ----------
        start_date : Union[str, datetime, pd.Timestamp]
            Start date for historical data
        end_date : Union[str, datetime, pd.Timestamp]
            End date for historical data
        bar_freq : str
            Bar frequency to filter by
        window_bars : int
            Window size to filter by
        factor_idx : Optional[int]
            Specific factor index to retrieve (None for all)
            
        Returns
        -------
        pd.DataFrame
            Historical PCA factors with columns: ts, PC1, PC2, PC3, exp_var_1, etc.
        """
        # Convert dates
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        
        # Build query
        factor_filter = "" if factor_idx is None else f"AND factor_idx = {factor_idx}"
        
        query = f"""
        SELECT 
            ts,
            factor_idx,
            score,
            exp_var,
            loadings
        FROM market_data.{self.source_table}
        WHERE ts >= %s::timestamp
        AND ts <= %s::timestamp
        AND bar_freq = %s::interval
        AND window_bars = %s
        {factor_filter}
        ORDER BY ts, factor_idx;
        """
        
        try:
            engine = self.engine
            df = pd.read_sql_query(
                query,
                engine,
                params=(start_date, end_date, bar_freq, window_bars)
            )
            
            if df.empty:
                logger.warning(f"No PCA factors found for period {start_date} to {end_date}")
                return pd.DataFrame()
                
            # Pivot to wide format
            return self._pivot_to_wide_format(df)
            
        except Exception as e:
            logger.error(f"Error fetching historical PCA factors: {e}")
            raise BarFetchError(f"Failed to fetch historical PCA factors: {e}")
    
    def get_factor_loadings(
        self,
        timestamp: Union[str, datetime, pd.Timestamp],
        bar_freq: str = '15 seconds',
        window_bars: int = 240
    ) -> pd.DataFrame:
        """
        Get PCA loadings for a specific timestamp.
        
        Parameters
        ----------
        timestamp : Union[str, datetime, pd.Timestamp]
            Timestamp to get loadings for
        bar_freq : str
            Bar frequency
        window_bars : int
            Window size
            
        Returns
        -------
        pd.DataFrame
            Loadings matrix with contracts as rows and factors as columns
        """
        timestamp = pd.Timestamp(timestamp)
        
        query = f"""
        SELECT 
            factor_idx,
            loadings
        FROM market_data.{self.source_table}
        WHERE ts = %s::timestamp
        AND bar_freq = %s::interval
        AND window_bars = %s
        ORDER BY factor_idx;
        """
        
        try:
            engine = self.engine
            df = pd.read_sql_query(
                query,
                engine,
                params=(timestamp, bar_freq, window_bars)
            )
            
            if df.empty:
                logger.warning(f"No loadings found for {timestamp}")
                return pd.DataFrame()
                
            # Convert to loadings matrix
            return self._process_loadings(df)
            
        except Exception as e:
            logger.error(f"Error fetching PCA loadings: {e}")
            raise BarFetchError(f"Failed to fetch PCA loadings: {e}")
    
    def get_variance_explained_history(
        self,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        bar_freq: str = '15 seconds',
        window_bars: int = 240
    ) -> pd.DataFrame:
        """
        Get historical variance explained by each factor.
        
        Parameters
        ----------
        start_date : Union[str, datetime, pd.Timestamp]
            Start date
        end_date : Union[str, datetime, pd.Timestamp]
            End date
        bar_freq : str
            Bar frequency
        window_bars : int
            Window size
            
        Returns
        -------
        pd.DataFrame
            Time series of variance explained by each factor
        """
        df = self.get_historical_factors(
            start_date, end_date, bar_freq, window_bars
        )
        
        if df.empty:
            return pd.DataFrame()
            
        # Extract variance explained columns
        var_cols = [col for col in df.columns if col.startswith('exp_var_')]
        return df[['ts'] + var_cols].set_index('ts')
    
    def _process_factor_data(self, df: pd.DataFrame) -> PCAFactorData:
        """Process raw factor data into structured format."""
        # Extract unique timestamp
        ts = df['ts'].iloc[0]
        bar_freq = df['bar_freq'].iloc[0]
        window_bars = df['window_bars'].iloc[0]
        
        # Create factors series
        factors = pd.Series(
            df['score'].values,
            index=[f'PC{i+1}' for i in range(len(df))]
        )
        
        # Create variance explained series
        var_explained = pd.Series(
            df['exp_var'].values,
            index=[f'PC{i+1}' for i in range(len(df))]
        )
        
        # Process loadings
        loadings_df = self._process_loadings(df)
        
        return PCAFactorData(
            timestamp=ts,
            factors=factors,
            loadings=loadings_df,
            variance_explained=var_explained,
            bar_freq=str(bar_freq),
            window_bars=window_bars
        )
    
    def _process_loadings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process loadings arrays into DataFrame."""
        # Assuming loadings are stored as arrays
        contracts = [f'CL{i}' for i in range(1, 9)]  # CL1-CL8
        
        loadings_data = {}
        for _, row in df.iterrows():
            factor_name = f'PC{row["factor_idx"] + 1}'
            loadings_data[factor_name] = row['loadings']
            
        # Create DataFrame
        loadings_df = pd.DataFrame(loadings_data, index=contracts)
        return loadings_df
    
    def _pivot_to_wide_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot factor data to wide format."""
        # Pivot scores
        scores_pivot = df.pivot(
            index='ts',
            columns='factor_idx',
            values='score'
        )
        scores_pivot.columns = [f'PC{i+1}' for i in scores_pivot.columns]
        
        # Pivot variance explained
        var_pivot = df.pivot(
            index='ts',
            columns='factor_idx',
            values='exp_var'
        )
        var_pivot.columns = [f'exp_var_{i+1}' for i in var_pivot.columns]
        
        # Combine
        result = pd.concat([scores_pivot, var_pivot], axis=1)
        result = result.reset_index()
        
        return result


class PCAMetadataFetcher:
    """
    Fetches PCA calculation metadata for monitoring and analysis.
    """
    
    def __init__(self, engine: Engine):
        """Initialize metadata fetcher."""
        self.engine = engine
    
    def get_latest_metadata(self) -> Dict[str, any]:
        """
        Get latest PCA calculation metadata.
        
        Returns
        -------
        Dict[str, any]
            Metadata including calculation time, data quality, etc.
        """
        query = """
        SELECT 
            ts,
            bar_freq,
            window_bars,
            n_contracts,
            total_variance,
            condition_number,
            calc_duration_ms,
            data_quality_score,
            created_at
        FROM market_data.pca_metadata
        ORDER BY created_at DESC
        LIMIT 1;
        """
        
        try:
            engine = self.engine
            df = pd.read_sql_query(query, engine)
            
            if df.empty:
                return {}
                
            return df.iloc[0].to_dict()
            
        except Exception as e:
            logger.error(f"Error fetching PCA metadata: {e}")
            return {}
    
    def get_calculation_health(
        self,
        lookback_hours: int = 24
    ) -> pd.DataFrame:
        """
        Get PCA calculation health metrics over recent period.
        
        Parameters
        ----------
        lookback_hours : int
            Hours to look back
            
        Returns
        -------
        pd.DataFrame
            Health metrics time series
        """
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        
        query = """
        SELECT 
            ts,
            total_variance,
            condition_number,
            calc_duration_ms,
            data_quality_score
        FROM market_data.pca_metadata
        WHERE created_at >= %s
        ORDER BY ts;
        """
        
        try:
            engine = self.engine
            df = pd.read_sql_query(query, engine, params=(cutoff,))
            
            if not df.empty:
                df['ts'] = pd.to_datetime(df['ts'])
                df = df.set_index('ts')
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching calculation health: {e}")
            return pd.DataFrame()