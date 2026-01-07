"""
Risk model interface for position sizing and risk management.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .types import Signal, Position, PortfolioState, Bar


class IRiskModel(ABC):
    """
    Abstract interface for risk models.
    
    Responsible for:
    - Position sizing
    - Risk limit checking
    - Stop loss calculation
    - Portfolio heat management
    """
    
    @abstractmethod
    def initialize(self, config: Dict) -> None:
        """
        Initialize risk model with configuration.
        
        Args:
            config: Risk model configuration including:
                - risk_per_trade: Max risk per trade (e.g., 0.02 for 2%)
                - max_positions: Maximum number of positions
                - max_correlation: Maximum portfolio correlation
                - stop_type: Type of stop loss (fixed, atr, volatility)
        """
        pass
    
    @abstractmethod
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        portfolio: PortfolioState,
        volatility: float,
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Tuple[int, float]:
        """
        Calculate position size for a signal.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            portfolio: Current portfolio state
            volatility: Asset volatility (annualized)
            correlation_matrix: Correlations with existing positions
            
        Returns:
            Tuple of (position_size, stop_price)
        """
        pass
    
    @abstractmethod
    def check_risk_limits(
        self,
        signal: Signal,
        proposed_size: int,
        portfolio: PortfolioState
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if trade passes risk limits.
        
        Args:
            signal: Trading signal
            proposed_size: Proposed position size
            portfolio: Current portfolio state
            
        Returns:
            Tuple of (is_allowed, rejection_reason)
        """
        pass
    
    @abstractmethod
    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        bars: List[Bar],
        atr_multiplier: float = 2.0
    ) -> float:
        """
        Calculate stop loss price.
        
        Args:
            symbol: Symbol to trade
            entry_price: Entry price
            side: "BUY" or "SELL"
            bars: Recent price bars for ATR calculation
            atr_multiplier: ATR multiplier for stop distance
            
        Returns:
            Stop loss price
        """
        pass
    
    @abstractmethod
    def update_trailing_stop(
        self,
        position: Position,
        current_price: float,
        highest_price: float,
        lowest_price: float
    ) -> Optional[float]:
        """
        Update trailing stop for a position.
        
        Args:
            position: Current position
            current_price: Current market price
            highest_price: Highest price since entry
            lowest_price: Lowest price since entry
            
        Returns:
            New stop price or None if no update needed
        """
        pass
    
    @abstractmethod
    def get_portfolio_risk_metrics(
        self,
        portfolio: PortfolioState,
        market_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate portfolio-level risk metrics.
        
        Args:
            portfolio: Current portfolio state
            market_prices: Current market prices
            
        Returns:
            Dictionary with metrics like:
            - total_risk: Total portfolio risk
            - var_95: 95% Value at Risk
            - max_drawdown: Maximum drawdown
            - sharpe_ratio: Sharpe ratio
            - correlation_risk: Portfolio correlation
        """
        pass
    
    @abstractmethod
    def should_reduce_risk(
        self,
        portfolio: PortfolioState,
        recent_pnl: List[float]
    ) -> bool:
        """
        Determine if risk should be reduced.
        
        Args:
            portfolio: Current portfolio state
            recent_pnl: Recent P&L history
            
        Returns:
            True if risk should be reduced
        """
        pass
    
    # Optional methods with default implementations
    
    def adjust_size_for_regime(
        self,
        base_size: int,
        volatility_regime: str,
        trend_strength: float
    ) -> int:
        """
        Adjust position size based on market regime.
        
        Default implementation returns base size.
        
        Args:
            base_size: Base position size
            volatility_regime: Current vol regime (low/normal/high)
            trend_strength: Trend strength (0-1)
            
        Returns:
            Adjusted position size
        """
        return base_size
    
    def get_correlation_penalty(
        self,
        symbol: str,
        existing_positions: Dict[str, Position],
        correlation_matrix: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Calculate correlation penalty for a new position.
        
        Default implementation returns 0 (no penalty).
        
        Args:
            symbol: Symbol to trade
            existing_positions: Current positions
            correlation_matrix: Correlation matrix
            
        Returns:
            Penalty factor (0-1, where 1 = no penalty)
        """
        return 1.0
    
    @property
    def risk_parameters(self) -> Dict[str, float]:
        """
        Get current risk parameters.
        
        Returns dictionary of risk parameters.
        """
        return {}

