"""
Data feed interface for providing market data to strategies.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime, timedelta

from .types import Bar


class IDataFeed(ABC):
    """
    Abstract interface for data feeds.
    
    Can be implemented by:
    - Historical data feed (for backtesting)
    - Live data feed (for production trading)
    - Simulated data feed (for testing)
    """
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to data source."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to data source."""
        pass
    
    @abstractmethod
    def subscribe(self, symbols: List[str], 
                  callback: Callable[[Bar], None]) -> None:
        """
        Subscribe to real-time data for given symbols.
        
        Args:
            symbols: List of symbols to subscribe to
            callback: Function to call with new bars
        """
        pass
    
    @abstractmethod
    def unsubscribe(self, symbols: List[str]) -> None:
        """
        Unsubscribe from symbols.
        
        Args:
            symbols: List of symbols to unsubscribe from
        """
        pass
    
    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        bar_size: str = "5s"
    ) -> List[Bar]:
        """
        Get historical bars for a symbol.
        
        Args:
            symbol: Symbol to get data for
            start: Start datetime
            end: End datetime
            bar_size: Bar size (e.g., "5s", "1m", "1h", "1d")
            
        Returns:
            List of bars ordered by timestamp
        """
        pass
    
    @abstractmethod
    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        """
        Get most recent bar for a symbol.
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            Latest bar or None if no data
        """
        pass
    
    @abstractmethod
    def get_snapshot(self, symbols: List[str]) -> Dict[str, Bar]:
        """
        Get current market snapshot for multiple symbols.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dictionary mapping symbol to latest bar
        """
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if data feed is connected."""
        pass
    
    @property
    @abstractmethod
    def supported_bar_sizes(self) -> List[str]:
        """List of supported bar sizes."""
        pass
    
    # Optional methods with default implementations
    
    def get_contract_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get contract details for a symbol.
        
        Default implementation returns None.
        
        Args:
            symbol: Symbol to get details for
            
        Returns:
            Contract details or None
        """
        return None
    
    def get_trading_calendar(
        self,
        exchange: str,
        start: datetime,
        end: datetime
    ) -> List[datetime]:
        """
        Get trading calendar for an exchange.
        
        Default implementation returns empty list.
        
        Args:
            exchange: Exchange name
            start: Start date
            end: End date
            
        Returns:
            List of trading days
        """
        return []

