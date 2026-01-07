"""
Pure I/O fetcher for inventory/fundamental data
NO business logic calculations - use common_lib.factors for analysis
"""

import pandas as pd
from datetime import datetime
from typing import Optional, List
from sqlalchemy import text, create_engine
import logging
import os

from ..data_api.fetcher_protocols import (
    IInventoryFetcher, 
    DataNotAvailableError, 
    FetcherConfigError
)

logger = logging.getLogger(__name__)


class InventoryFetcher:
    """
    Pure I/O fetcher for fundamental inventory data.
    Implements IInventoryFetcher protocol.
    NO business logic - only database queries.
    """
    
    def __init__(self, connection=None, db_conn_str=None):
        """
        Initialize with database connection.
        
        Args:
            connection: SQLAlchemy connection/engine
            db_conn_str: Database connection string
                        If both None, uses PG_DSN environment variable
        """
        self._owns_connection = False
        
        if connection:
            self._connection = connection
        elif db_conn_str:
            self.engine = create_engine(db_conn_str)
            self._owns_connection = True
        else:
            dsn = os.getenv('PG_DSN')
            if not dsn:
                raise FetcherConfigError(
                    "No database connection provided and PG_DSN not set"
                )
            self.engine = create_engine(dsn)
            self._owns_connection = True
    
    @property
    def conn(self):
        """Get connection (create if using engine)"""
        if hasattr(self, '_connection'):
            return self._connection
        return self.engine
    
    def fetch(self, 
             start_date: datetime, 
             end_date: datetime, 
             series_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch inventory data - PURE I/O ONLY.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            series_codes: Optional list of series to fetch
        
        Returns:
            DataFrame with columns: [date, series_code, value]
            Exactly matches IInventoryFetcher protocol
        
        Raises:
            DataNotAvailableError: If no data found or table missing
        """
        try:
            if series_codes:
                query = text("""
                    SELECT 
                        obs_date as date, 
                        series_code, 
                        value
                    FROM model.crude_balance_driver
                    WHERE obs_date >= :start_date 
                      AND obs_date <= :end_date
                      AND series_code = ANY(:series_codes)
                    ORDER BY obs_date, series_code
                """)
                params = {
                    'start_date': start_date.date() if hasattr(start_date, 'date') else start_date,
                    'end_date': end_date.date() if hasattr(end_date, 'date') else end_date,
                    'series_codes': series_codes
                }
            else:
                query = text("""
                    SELECT 
                        obs_date as date, 
                        series_code, 
                        value
                    FROM model.crude_balance_driver
                    WHERE obs_date >= :start_date 
                      AND obs_date <= :end_date
                    ORDER BY obs_date, series_code
                """)
                params = {
                    'start_date': start_date.date() if hasattr(start_date, 'date') else start_date,
                    'end_date': end_date.date() if hasattr(end_date, 'date') else end_date
                }
            
            df = pd.read_sql(query, self.conn, params=params)
            
            if df.empty:
                series_msg = f" for series {series_codes}" if series_codes else ""
                raise DataNotAvailableError(
                    f"No inventory data found between {start_date} and {end_date}{series_msg}"
                )
            
            return self._validate_result(df)
            
        except DataNotAvailableError:
            raise
        except Exception as e:
            if "does not exist" in str(e) or "no such table" in str(e).lower():
                raise DataNotAvailableError(f"Inventory table not available: {e}")
            logger.error(f"Error fetching inventory data: {e}")
            raise DataNotAvailableError(f"Failed to fetch inventory data: {e}")
    
    def fetch_wide(self, 
                  start_date: datetime, 
                  end_date: datetime,
                  series_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch inventory data in wide format (series as columns).
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            series_codes: Optional list of series to fetch
        
        Returns:
            DataFrame with date index and series_codes as columns
        """
        df = self.fetch(start_date, end_date, series_codes)
        return df.pivot(index='date', columns='series_code', values='value')
    
    def fetch_from_view(self,
                       start_date: datetime,
                       end_date: datetime) -> pd.DataFrame:
        """
        Fetch from core_energy.v_model_wide curated view.
        Returns all fundamental columns already joined.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
        
        Returns:
            DataFrame with all v_model_wide columns, date as index
        """
        query = text("""
            SELECT *
            FROM core_energy.v_model_wide
            WHERE date >= :start_date 
              AND date <= :end_date
            ORDER BY date
        """)
        
        params = {
            'start_date': start_date.date() if hasattr(start_date, 'date') else start_date,
            'end_date': end_date.date() if hasattr(end_date, 'date') else end_date
        }
        
        try:
            df = pd.read_sql(query, self.conn, params=params)
            
            if df.empty:
                raise DataNotAvailableError(
                    f"No data in v_model_wide between {start_date} and {end_date}"
                )
            
            # Ensure date column is datetime and set as index
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            return df
            
        except DataNotAvailableError:
            raise
        except Exception as e:
            if "does not exist" in str(e) or "no such table" in str(e).lower():
                raise DataNotAvailableError(f"View v_model_wide not available: {e}")
            logger.error(f"Error fetching from v_model_wide: {e}")
            raise DataNotAvailableError(f"Failed to fetch from v_model_wide: {e}")
    
    def get_available_series(self) -> pd.DataFrame:
        """
        Get metadata about available series - PURE I/O.
        
        Returns:
            DataFrame with columns: [series_code, record_count, first_date, last_date]
        """
        query = text("""
            SELECT 
                series_code,
                COUNT(*) as record_count,
                MIN(obs_date) as first_date,
                MAX(obs_date) as last_date
            FROM model.crude_balance_driver
            GROUP BY series_code
            ORDER BY series_code
        """)
        
        try:
            df = pd.read_sql(query, self.conn)
            
            if not df.empty:
                df['first_date'] = pd.to_datetime(df['first_date'])
                df['last_date'] = pd.to_datetime(df['last_date'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting available series: {e}")
            raise DataNotAvailableError(f"Failed to get available series: {e}")
    
    def get_latest_values(self, 
                         series_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get most recent value per series - PURE I/O.
        
        Args:
            series_codes: Optional list of series to filter
        
        Returns:
            DataFrame with columns: [series_code, date, value]
        """
        if series_codes:
            query = text("""
                SELECT DISTINCT ON (series_code) 
                    series_code,
                    obs_date as date,
                    value
                FROM model.crude_balance_driver
                WHERE series_code = ANY(:series_codes)
                ORDER BY series_code, obs_date DESC
            """)
            params = {'series_codes': series_codes}
        else:
            query = text("""
                SELECT DISTINCT ON (series_code) 
                    series_code,
                    obs_date as date,
                    value
                FROM model.crude_balance_driver
                ORDER BY series_code, obs_date DESC
            """)
            params = {}
        
        try:
            df = pd.read_sql(query, self.conn, params=params)
            
            if df.empty:
                series_msg = f" for series {series_codes}" if series_codes else ""
                raise DataNotAvailableError(f"No data available{series_msg}")
            
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'])
            
            return df
            
        except DataNotAvailableError:
            raise
        except Exception as e:
            logger.error(f"Error getting latest values: {e}")
            raise DataNotAvailableError(f"Failed to get latest values: {e}")
    
    def _validate_result(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate fetched data matches expected schema.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Validated DataFrame with proper types
        
        Raises:
            DataNotAvailableError: If schema validation fails
        """
        required_columns = {'date', 'series_code', 'value'}
        missing = required_columns - set(df.columns)
        
        if missing:
            raise DataNotAvailableError(
                f"Missing required columns: {missing}"
            )
        
        # Ensure proper data types
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Check for invalid values after conversion
        if df['value'].isna().any():
            bad_rows = df[df['value'].isna()]['series_code'].unique()
            logger.warning(f"Invalid numeric values for series: {bad_rows}")
        
        return df