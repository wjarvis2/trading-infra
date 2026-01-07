#!/usr/bin/env python3
"""
Market hours utilities for NYMEX energy futures trading.
Trading hours are Sunday 5:00 PM ET through Friday 4:00 PM ET with daily breaks 4:00-5:00 PM ET.
"""

from datetime import datetime, time, timedelta
import pytz

class MarketHours:
    """Check if NYMEX energy futures market is open"""
    
    # Market hours in ET timezone
    ET = pytz.timezone('US/Eastern')
    
    # Daily trading sessions (in ET)
    MARKET_OPEN = time(17, 0)   # 5:00 PM
    MARKET_CLOSE = time(16, 0)  # 4:00 PM
    
    @classmethod
    def is_market_open(cls, dt=None):
        """
        Check if market is currently open for trading.
        
        Returns:
            bool: True if market is open, False otherwise
        """
        if dt is None:
            dt = datetime.now(cls.ET)
        elif dt.tzinfo is None:
            dt = cls.ET.localize(dt)
        else:
            dt = dt.astimezone(cls.ET)
            
        weekday = dt.weekday()  # 0=Monday, 6=Sunday
        current_time = dt.time()
        
        # Market closed all day Saturday
        if weekday == 5:  # Saturday
            return False
            
        # Sunday: opens at 5:00 PM ET
        if weekday == 6:  # Sunday
            return current_time >= cls.MARKET_OPEN
            
        # Monday-Thursday: Two sessions with break
        if weekday in [0, 1, 2, 3]:  # Mon-Thu
            # Morning session: midnight to 4:00 PM
            if current_time < cls.MARKET_CLOSE:
                return True
            # Evening session: 5:00 PM to midnight
            elif current_time >= cls.MARKET_OPEN:
                return True
            # Daily break: 4:00 PM to 5:00 PM
            else:
                return False
                
        # Friday: closes at 4:00 PM ET for weekend
        if weekday == 4:  # Friday
            # Only morning session, no evening session
            return current_time < cls.MARKET_CLOSE
            
        return False
    
    @classmethod
    def next_market_open(cls, dt=None):
        """
        Get the next market open time.
        
        Returns:
            datetime: Next market open time in ET
        """
        if dt is None:
            dt = datetime.now(cls.ET)
        elif dt.tzinfo is None:
            dt = cls.ET.localize(dt)
        else:
            dt = dt.astimezone(cls.ET)
            
        # If market is currently open, return None
        if cls.is_market_open(dt):
            return None
            
        weekday = dt.weekday()
        current_time = dt.time()
        
        # During daily break (4-5 PM Mon-Thu)
        if weekday in [0, 1, 2, 3] and cls.MARKET_CLOSE <= current_time < cls.MARKET_OPEN:
            return dt.replace(hour=17, minute=0, second=0, microsecond=0)
            
        # Friday after 4:00 PM -> Sunday 5:00 PM
        if weekday == 4 and current_time >= cls.MARKET_CLOSE:
            days_until_sunday = 2
            next_open = dt + timedelta(days=days_until_sunday)
            return next_open.replace(hour=17, minute=0, second=0, microsecond=0)
            
        # Saturday -> Sunday 5:00 PM
        if weekday == 5:
            next_open = dt + timedelta(days=1)
            return next_open.replace(hour=17, minute=0, second=0, microsecond=0)
            
        # Sunday before 5:00 PM -> Sunday 5:00 PM
        if weekday == 6 and current_time < cls.MARKET_OPEN:
            return dt.replace(hour=17, minute=0, second=0, microsecond=0)
            
        # Should not reach here
        return None
    
    @classmethod
    def seconds_until_market_open(cls, dt=None):
        """
        Get seconds until next market open.
        
        Returns:
            int: Seconds until market opens, or 0 if market is open
        """
        if cls.is_market_open(dt):
            return 0
            
        next_open = cls.next_market_open(dt)
        if next_open:
            if dt is None:
                dt = datetime.now(cls.ET)
            elif dt.tzinfo is None:
                dt = cls.ET.localize(dt)
            else:
                dt = dt.astimezone(cls.ET)
                
            delta = next_open - dt
            return int(delta.total_seconds())
        
        return 0
    
    @classmethod
    def market_status_string(cls, dt=None):
        """Get human-readable market status"""
        if cls.is_market_open(dt):
            return "Market is OPEN"
        else:
            next_open = cls.next_market_open(dt)
            if next_open:
                return f"Market is CLOSED. Opens at {next_open.strftime('%Y-%m-%d %I:%M %p %Z')}"
            return "Market is CLOSED"