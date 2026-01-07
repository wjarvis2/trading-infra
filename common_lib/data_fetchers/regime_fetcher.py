"""
Data fetcher for precomputed volatility regimes.

Fetches volatility regime classifications and transitions from the database,
supporting both historical analysis and real-time trading operations.
"""

import pandas as pd
import psycopg2
from psycopg2 import pool
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class RegimeFetcher:
    """
    Fetches precomputed volatility regime data from the database.
    
    This fetcher retrieves regime classifications that have been precomputed
    by an ETL process, avoiding the need to calculate regimes on-the-fly
    during trading operations.
    """
    
    def __init__(self, connection_pool: psycopg2.pool.ThreadedConnectionPool):
        """
        Initialize the regime fetcher.
        
        Parameters
        ----------
        connection_pool : psycopg2.pool.ThreadedConnectionPool
            Database connection pool
        """
        self.pool = connection_pool
        
        # Cache for recent regime data
        self.regime_cache: Dict[str, pd.DataFrame] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_ttl = timedelta(minutes=5)  # 5-minute cache
    
    @contextmanager
    def get_connection(self):
        """Get a database connection from the pool."""
        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.pool.putconn(conn)
    
    def fetch_current_regime(
        self,
        symbol: str,
        as_of: Optional[datetime] = None
    ) -> Dict[str, any]:
        """
        Fetch the current volatility regime for a symbol.
        
        Parameters
        ----------
        symbol : str
            Symbol/spread identifier (e.g., 'CL1-CL2')
        as_of : datetime, optional
            Timestamp to fetch regime for. If None, uses latest.
            
        Returns
        -------
        dict
            Dictionary with regime info:
            - regime: str (LOW, NORMAL, HIGH, EXTREME)
            - volatility: float
            - percentile: float
            - duration: int (bars in current regime)
            - timestamp: datetime
            - pca_speed: float
        """
        query = """
        SELECT 
            regime,
            volatility,
            percentile,
            duration,
            pca_speed,
            timestamp,
            confidence
        FROM analytics.volatility_regimes
        WHERE symbol = %s
        """
        
        params = [symbol]
        
        if as_of:
            query += " AND timestamp <= %s"
            params.append(as_of)
        
        query += " ORDER BY timestamp DESC LIMIT 1"
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                row = cursor.fetchone()
                
                if not row:
                    logger.warning(f"No regime data found for {symbol}")
                    return None
                
                return {
                    'regime': row[0],
                    'volatility': row[1],
                    'percentile': row[2],
                    'duration': row[3],
                    'pca_speed': row[4],
                    'timestamp': row[5],
                    'confidence': row[6]
                }
    
    def fetch_regime_history(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        include_transitions: bool = False
    ) -> pd.DataFrame:
        """
        Fetch historical regime data for analysis.
        
        Parameters
        ----------
        symbol : str
            Symbol/spread identifier
        start_date : datetime
            Start of history period
        end_date : datetime
            End of history period
        include_transitions : bool
            Whether to include regime transition markers
            
        Returns
        -------
        pd.DataFrame
            DataFrame with regime history
        """
        # Check cache first
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if cache_key in self.regime_cache:
            if datetime.now() < self.cache_expiry[cache_key]:
                return self.regime_cache[cache_key].copy()
        
        query = """
        SELECT 
            timestamp,
            regime,
            volatility,
            percentile,
            duration,
            pca_speed,
            confidence,
            band_upper,
            band_lower,
            band_multiplier
        FROM analytics.volatility_regimes
        WHERE symbol = %s
            AND timestamp >= %s
            AND timestamp <= %s
        ORDER BY timestamp
        """
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=[symbol, start_date, end_date],
                index_col='timestamp'
            )
        
        if include_transitions and not df.empty:
            # Mark regime transitions
            df['regime_changed'] = df['regime'] != df['regime'].shift(1)
            df['prev_regime'] = df['regime'].shift(1)
        
        # Update cache
        self.regime_cache[cache_key] = df.copy()
        self.cache_expiry[cache_key] = datetime.now() + self.cache_ttl
        
        return df
    
    def fetch_regime_transitions(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        from_regime: Optional[str] = None,
        to_regime: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch regime transition events for analysis.
        
        Parameters
        ----------
        symbol : str
            Symbol/spread identifier
        start_date : datetime
            Start of period
        end_date : datetime
            End of period
        from_regime : str, optional
            Filter by source regime
        to_regime : str, optional
            Filter by destination regime
            
        Returns
        -------
        pd.DataFrame
            DataFrame with transition events
        """
        query = """
        SELECT 
            transition_time,
            from_regime,
            to_regime,
            from_duration,
            volatility_before,
            volatility_after,
            pca_speed_at_transition,
            price_move_pct,
            transition_type
        FROM analytics.regime_transitions
        WHERE symbol = %s
            AND transition_time >= %s
            AND transition_time <= %s
        """
        
        params = [symbol, start_date, end_date]
        
        if from_regime:
            query += " AND from_regime = %s"
            params.append(from_regime)
        
        if to_regime:
            query += " AND to_regime = %s"
            params.append(to_regime)
        
        query += " ORDER BY transition_time"
        
        with self.get_connection() as conn:
            return pd.read_sql_query(
                query,
                conn,
                params=params,
                index_col='transition_time'
            )
    
    def fetch_regime_statistics(
        self,
        symbols: List[str],
        lookback_days: int = 30
    ) -> pd.DataFrame:
        """
        Fetch regime distribution statistics for multiple symbols.
        
        Parameters
        ----------
        symbols : List[str]
            List of symbols to analyze
        lookback_days : int
            Number of days to look back
            
        Returns
        -------
        pd.DataFrame
            DataFrame with regime statistics by symbol
        """
        start_date = datetime.now() - timedelta(days=lookback_days)
        
        query = """
        SELECT 
            symbol,
            regime,
            COUNT(*) as count,
            AVG(volatility) as avg_volatility,
            AVG(duration) as avg_duration,
            MAX(duration) as max_duration,
            AVG(pca_speed) as avg_pca_speed,
            COUNT(DISTINCT DATE(timestamp)) as days_in_regime
        FROM analytics.volatility_regimes
        WHERE symbol = ANY(%s)
            AND timestamp >= %s
        GROUP BY symbol, regime
        ORDER BY symbol, regime
        """
        
        with self.get_connection() as conn:
            return pd.read_sql_query(
                query,
                conn,
                params=[symbols, start_date]
            )
    
    def fetch_latest_regimes(
        self,
        symbols: List[str]
    ) -> Dict[str, Dict]:
        """
        Fetch latest regime for multiple symbols efficiently.
        
        Parameters
        ----------
        symbols : List[str]
            List of symbols
            
        Returns
        -------
        dict
            Dictionary mapping symbol to regime info
        """
        query = """
        WITH latest_regimes AS (
            SELECT DISTINCT ON (symbol)
                symbol,
                regime,
                volatility,
                percentile,
                duration,
                pca_speed,
                timestamp,
                confidence
            FROM analytics.volatility_regimes
            WHERE symbol = ANY(%s)
            ORDER BY symbol, timestamp DESC
        )
        SELECT * FROM latest_regimes
        """
        
        results = {}
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, [symbols])
                
                for row in cursor.fetchall():
                    results[row[0]] = {
                        'regime': row[1],
                        'volatility': row[2],
                        'percentile': row[3],
                        'duration': row[4],
                        'pca_speed': row[5],
                        'timestamp': row[6],
                        'confidence': row[7]
                    }
        
        return results
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cached regime data.
        
        Parameters
        ----------
        symbol : str, optional
            Symbol to clear cache for. If None, clears all.
        """
        if symbol:
            # Clear specific symbol
            keys_to_remove = [k for k in self.regime_cache.keys() if k.startswith(symbol)]
            for key in keys_to_remove:
                del self.regime_cache[key]
                del self.cache_expiry[key]
        else:
            # Clear all
            self.regime_cache.clear()
            self.cache_expiry.clear()
    
    def check_table_exists(self) -> bool:
        """
        Check if required database tables exist.
        
        Returns
        -------
        bool
            True if tables exist
        """
        query = """
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.tables 
            WHERE table_schema = 'analytics' 
                AND table_name IN ('volatility_regimes', 'regime_transitions')
        )
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchone()[0]