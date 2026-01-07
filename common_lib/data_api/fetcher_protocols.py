"""
Fetcher Protocols for Type Contracts
Defines interfaces that all data fetchers must implement
"""

from typing import Protocol, List, Optional, Dict, Any
from datetime import datetime
import pandas as pd


class IBarFetcher(Protocol):
    """Protocol for fetching OHLCV bar data"""
    
    def fetch(self,
             start_date: datetime,
             end_date: datetime,
             contracts: List[str],
             bar_type: str = '15s') -> pd.DataFrame:
        """
        Fetch bar data for specified contracts
        
        Returns:
            DataFrame with columns: [timestamp, symbol, open, high, low, close, volume]
            Should NOT have MultiIndex - that's enforced by SchemaEnforcer
        """
        ...


class IInventoryFetcher(Protocol):
    """Protocol for fetching fundamental inventory data"""
    
    def fetch(self,
             start_date: datetime,
             end_date: datetime,
             series_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch inventory/fundamental data
        
        Returns:
            DataFrame with columns: [date, series_code, value]
            Optional columns: [units, source]
        """
        ...


class ICurveFetcher(Protocol):
    """Protocol for fetching forward curve data"""
    
    def fetch(self,
             start_date: datetime,
             end_date: datetime,
             roots: List[str]) -> pd.DataFrame:
        """
        Fetch forward curve data for commodity roots
        
        Returns:
            DataFrame with columns: [date, contract, price, days_to_expiry]
            Optional columns: [volume, open_interest]
        """
        ...


class ISpreadFetcher(Protocol):
    """Protocol for fetching spread data"""
    
    def fetch(self,
             start_date: datetime,
             end_date: datetime,
             spread_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Fetch or calculate spread values
        
        Args:
            spread_configs: List of spread configurations, each with:
                - type: 'calendar', 'crack', 'butterfly', etc.
                - leg1, leg2, leg3: Contract symbols
                - ratio: Weight ratios for each leg
                
        Returns:
            DataFrame with columns: [timestamp, spread_type, value]
            Optional columns: [leg1, leg2, leg3, ratio]
        """
        ...


class IFactorFetcher(Protocol):
    """Protocol for fetching factor/PCA data"""
    
    def fetch(self,
             start_date: datetime,
             end_date: datetime,
             factor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch factor or PCA data
        
        Returns:
            DataFrame with columns: [timestamp, factor_name, value]
            Or wide format with factor names as columns
        """
        ...


class FetcherError(Exception):
    """Base exception for fetcher errors"""
    pass


class DataNotAvailableError(FetcherError):
    """Raised when requested data is not available"""
    pass


class FetcherConfigError(FetcherError):
    """Raised when fetcher configuration is invalid"""
    pass