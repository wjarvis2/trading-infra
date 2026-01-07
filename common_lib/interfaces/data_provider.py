"""
Data provider interface for strategy data access.

Minimal interface exposing only core data primitives needed by strategies.
Keeps provider lean and enables seamless swapping between backtest and live.
"""

from typing import Protocol, Optional, List, Dict, Any
import pandas as pd
from dataclasses import dataclass


class IDataProvider(Protocol):
    """
    Minimal data access interface for trading strategies.
    
    Implementations:
    - InMemoryProvider: Pre-loaded data for backtesting
    - TimescaleProvider: Live database queries
    - MockProvider: Unit testing
    """
    
    def get_bars(
        self,
        contracts: List[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        bar_size: str = '15s'
    ) -> pd.DataFrame:
        """
        Get OHLCV bars for specified contracts.
        
        Args:
            contracts: List of contracts (e.g., ['CL1', 'CL2'])
            start: Start timestamp (inclusive, UTC)
            end: End timestamp (inclusive, UTC)
            bar_size: Bar frequency ('15s', '1m', '5m', '1h', '1d')
            
        Returns:
            DataFrame with columns: [timestamp, contract, open, high, low, close, volume]
            Indexed by timestamp, sorted ascending
        """
        ...
    
    def get_inventory(
        self,
        date: pd.Timestamp,
        series: str = 'crude_ending_commercial_stocks'
    ) -> Optional[float]:
        """
        Get inventory level for a specific date.
        
        Args:
            date: Date to get inventory for (UTC)
            series: Inventory series name
            
        Returns:
            Inventory level in appropriate units, or None if not available
        """
        ...
    
    def get_curve_data(
        self,
        date: pd.Timestamp,
        contracts: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get forward curve snapshot for a specific date.
        
        Args:
            date: Date to get curve for (UTC)
            contracts: Optional list of specific contracts (default: all CL contracts)
            
        Returns:
            DataFrame with columns: [contract, expiry, price, volume, open_interest]
            Sorted by expiry
        """
        ...
    
    def get_pca_factors(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        bar_freq: str = '15s',
        window_bars: int = 240
    ) -> pd.DataFrame:
        """
        Get PCA factors for the specified time range.
        
        Args:
            start: Start timestamp (inclusive, UTC)
            end: End timestamp (inclusive, UTC)
            bar_freq: Bar frequency used for PCA calculation
            window_bars: Rolling window size used for PCA
            
        Returns:
            DataFrame with columns:
            - timestamp: UTC datetime64[ns, UTC]
            - factor_idx: int (1, 2, 3... for PC1, PC2, PC3...)
            - score: float32 (factor score/value)
            - exp_var: float32 (variance explained by this factor, 0-1)
            - loadings: dict (JSON dict of contract->loading mappings)
            
            Sorted by timestamp, factor_idx ascending
        """
        ...