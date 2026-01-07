"""
Execution interface for submitting and managing orders.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Callable
from datetime import datetime

from .types import Order, Fill, Position, Side, OrderType


class IExecutor(ABC):
    """
    Abstract interface for order execution.
    
    Can be implemented by:
    - Live broker connection (e.g., IBKR)
    - Simulated executor (for backtesting)
    - Paper trading executor
    """
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to execution venue."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to execution venue."""
        pass
    
    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """
        Submit an order for execution.
        
        Args:
            order: Order to submit
            
        Returns:
            Order ID assigned by the executor
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancellation was successful
        """
        pass
    
    @abstractmethod
    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> bool:
        """
        Modify a pending order.
        
        Args:
            order_id: ID of order to modify
            quantity: New quantity (optional)
            limit_price: New limit price (optional)
            stop_price: New stop price (optional)
            
        Returns:
            True if modification was successful
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get current status of an order.
        
        Args:
            order_id: ID of order to check
            
        Returns:
            Order object with current status or None
        """
        pass
    
    @abstractmethod
    def get_open_orders(self) -> List[Order]:
        """
        Get all open orders.
        
        Returns:
            List of open orders
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """
        Get all current positions.
        
        Returns:
            Dictionary mapping symbol to position
        """
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            Position or None if no position
        """
        pass
    
    @abstractmethod
    def set_fill_callback(self, callback: Callable[[Fill], None]) -> None:
        """
        Set callback for execution fills.
        
        Args:
            callback: Function to call when fills occur
        """
        pass
    
    @abstractmethod
    def set_order_callback(self, callback: Callable[[Order], None]) -> None:
        """
        Set callback for order status updates.
        
        Args:
            callback: Function to call when order status changes
        """
        pass
    
    @abstractmethod
    def tick_pending_orders(self, market_data: Dict[str, 'Bar']) -> Optional[List[Fill]]:
        """
        Process pending orders on each market tick.
        Should be called every bar to check if pending orders can now be executed.
        
        Args:
            market_data: Current market bars by symbol
            
        Returns:
            List of fills from executed orders, or None if no fills
        """
        pass
    
    @abstractmethod
    def get_pending_orders(self) -> List[Order]:
        """
        Get all orders waiting for execution.
        
        Returns:
            List of pending orders (empty list if none)
        """
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if executor is connected."""
        pass
    
    # Convenience methods with default implementations
    
    def submit_market_order(
        self,
        symbol: str,
        side: Side,
        quantity: int,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Submit a market order.
        
        Args:
            symbol: Symbol to trade
            side: BUY or SELL
            quantity: Number of contracts
            metadata: Optional metadata
            
        Returns:
            Order ID
        """
        order = Order(
            order_id="",  # Will be assigned by executor
            timestamp=datetime.utcnow(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            metadata=metadata
        )
        return self.submit_order(order)
    
    def submit_limit_order(
        self,
        symbol: str,
        side: Side,
        quantity: int,
        limit_price: float,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Submit a limit order.
        
        Args:
            symbol: Symbol to trade
            side: BUY or SELL
            quantity: Number of contracts
            limit_price: Limit price
            metadata: Optional metadata
            
        Returns:
            Order ID
        """
        order = Order(
            order_id="",  # Will be assigned by executor
            timestamp=datetime.utcnow(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            metadata=metadata
        )
        return self.submit_order(order)
    
    def flatten_position(self, symbol: str) -> Optional[str]:
        """
        Close position for a symbol.
        
        Args:
            symbol: Symbol to flatten
            
        Returns:
            Order ID or None if no position
        """
        position = self.get_position(symbol)
        if position is None or position.quantity == 0:
            return None
        
        # Submit opposite side market order
        side = Side.SELL if position.quantity > 0 else Side.BUY
        quantity = abs(position.quantity)
        
        return self.submit_market_order(symbol, side, quantity)
    
    def flatten_all_positions(self) -> List[str]:
        """
        Close all open positions.
        
        Returns:
            List of order IDs
        """
        order_ids = []
        positions = self.get_positions()
        
        for symbol, position in positions.items():
            if position.quantity != 0:
                order_id = self.flatten_position(symbol)
                if order_id:
                    order_ids.append(order_id)
        
        return order_ids

