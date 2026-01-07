"""
Base strategy interface that all strategies must implement.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Mapping
from datetime import datetime

from .types import Bar, Signal, Fill, Position, PortfolioState
from .data_provider import IDataProvider


class IStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Both backtest and live trading engines will call these methods
    in the same sequence, ensuring identical behavior.
    
    Constructor Requirement:
        All strategies MUST accept exactly two arguments:
        - config: Mapping[str, Any] - Strategy configuration
        - data_provider: IDataProvider - Data access interface
        
    Example:
        def __init__(self, config: Mapping[str, Any], data_provider: IDataProvider):
            self.config = config
            self.data = data_provider
    """
    
    @abstractmethod
    def __init__(self, config: Mapping[str, Any], data_provider: IDataProvider) -> None:
        """
        Initialize strategy with configuration and data provider.
        
        Args:
            config: Strategy configuration dictionary
            data_provider: Data access interface (same for backtest/live)
        """
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize strategy state after construction.
        
        Called after __init__ to set up any additional state,
        indicators, or pre-calculations needed by the strategy.
        """
        pass
    
    @abstractmethod
    def on_start(self, portfolio: PortfolioState) -> None:
        """
        Called when the strategy starts trading.
        
        Args:
            portfolio: Initial portfolio state
        """
        pass
    
    @abstractmethod
    def on_bar(self, bar: Bar) -> List[Signal]:
        """
        Process new bar data and generate signals.
        
        This is the main entry point for strategy logic.
        
        Args:
            bar: New OHLCV bar
            
        Returns:
            List of trading signals (can be empty)
        """
        pass
    
    @abstractmethod
    def on_fill(self, fill: Fill) -> None:
        """
        Handle execution fill.
        
        Update internal state based on execution.
        
        Args:
            fill: Execution fill details
        """
        pass
    
    @abstractmethod
    def on_position_update(self, position: Position) -> None:
        """
        Handle position update.
        
        Called when position changes due to fills or market moves.
        
        Args:
            position: Updated position
        """
        pass
    
    @abstractmethod
    def on_day_end(self, portfolio: PortfolioState) -> None:
        """
        End of day processing.
        
        Opportunity to log metrics, save state, etc.
        
        Args:
            portfolio: End of day portfolio state
        """
        pass
    
    @abstractmethod
    def on_stop(self) -> None:
        """
        Called when strategy stops.
        
        Clean up resources, save final state, etc.
        """
        pass
    
    # State management for disaster recovery and backtest checkpointing
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get current strategy state.
        
        Returns:
            Serializable dictionary of strategy state
        """
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore strategy state.
        
        Args:
            state: Previously saved state from get_state()
        """
        pass
    
    # Optional methods with default implementations
    
    def on_timer(self, timestamp: datetime) -> List[Signal]:
        """
        Handle timer events (for time-based actions).
        
        Default implementation does nothing.
        
        Args:
            timestamp: Current time
            
        Returns:
            List of signals (if any)
        """
        return []
    
    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate signal before submission.
        
        Default implementation accepts all signals.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid
        """
        return True
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for identification."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Strategy version."""
        pass
    
    @property
    def required_symbols(self) -> List[str]:
        """
        List of symbols this strategy needs data for.
        
        Default returns empty list (subscribe to all).
        """
        return []
    
    @property
    def required_lookback(self) -> int:
        """
        Number of historical bars needed before strategy can trade.
        
        Default is 0 (no lookback required).
        """
        return 0

