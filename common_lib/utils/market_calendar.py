"""
Market Calendar Utility - Single Source of Truth for Trading Days

This module provides deterministic, calendar-aware date handling for all
backtesting and optimization components. It ensures we only test on valid
trading days, preventing empty data errors while maintaining our principle
of using only real market data.

Design Principles:
- NO synthetic data: Never create fake trading days
- Deterministic: Same input always produces same output
- Single source of truth: All date validation happens here
- Fail fast: Raise exceptions for invalid date ranges

Date: 2025-08-06
"""

import pandas as pd
from typing import Tuple, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Try to use pandas_market_calendars if available, otherwise use simple logic
try:
    import pandas_market_calendars as mcal
    CALENDAR_AVAILABLE = True
    # Use NYSE calendar as it covers most US market holidays
    # Energy futures generally follow similar holiday schedule
    NYMEX_CALENDAR = mcal.get_calendar('NYSE')
    logger.info("Using NYSE calendar for accurate holiday detection")
except ImportError:
    CALENDAR_AVAILABLE = False
    NYMEX_CALENDAR = None
    logger.warning("pandas_market_calendars not available, using simple weekend detection")


class NoDataError(Exception):
    """Raised when no market data is available for requested dates."""
    pass


def is_weekend(date: pd.Timestamp) -> bool:
    """Check if date is a weekend (Saturday=5, Sunday=6)."""
    return date.weekday() >= 5


def is_us_holiday(date: pd.Timestamp) -> bool:
    """
    Check if date is a US market holiday.
    
    Uses pandas_market_calendars if available, otherwise checks major holidays.
    """
    if CALENDAR_AVAILABLE and NYMEX_CALENDAR:
        # Check if date is in valid trading days
        schedule = NYMEX_CALENDAR.schedule(start_date=date, end_date=date)
        return len(schedule) == 0
    else:
        # Simple holiday check for major US holidays
        # This is not comprehensive but covers main market closures
        year = date.year
        holidays = [
            pd.Timestamp(year, 1, 1),   # New Year's Day
            pd.Timestamp(year, 7, 4),   # Independence Day
            pd.Timestamp(year, 12, 25), # Christmas
        ]
        
        # Add observed holidays (if holiday falls on weekend)
        observed_holidays = []
        for holiday in holidays:
            if holiday.weekday() == 5:  # Saturday
                observed_holidays.append(holiday - timedelta(days=1))
            elif holiday.weekday() == 6:  # Sunday
                observed_holidays.append(holiday + timedelta(days=1))
        
        holidays.extend(observed_holidays)
        return date.normalize() in holidays


def next_session_day(date: pd.Timestamp) -> pd.Timestamp:
    """
    Return the date if it's a valid session day, else the next valid session day.
    
    Args:
        date: Date to check
        
    Returns:
        Same date if valid trading day, otherwise next valid trading day
        
    Raises:
        ValueError: If no valid trading day found within 10 days
    """
    date = pd.Timestamp(date).normalize()
    
    # Try up to 10 days to find next trading day
    for i in range(10):
        check_date = date + timedelta(days=i)
        
        if CALENDAR_AVAILABLE and NYMEX_CALENDAR:
            # Use market calendar for accurate check
            schedule = NYMEX_CALENDAR.schedule(
                start_date=check_date, 
                end_date=check_date
            )
            if len(schedule) > 0:
                return check_date
        else:
            # Simple check: not weekend and not major holiday
            if not is_weekend(check_date) and not is_us_holiday(check_date):
                return check_date
    
    raise ValueError(f"No valid trading day found within 10 days of {date}")


def previous_session_day(date: pd.Timestamp) -> pd.Timestamp:
    """
    Return the date if it's a valid session day, else the previous valid session day.
    
    Args:
        date: Date to check
        
    Returns:
        Same date if valid trading day, otherwise previous valid trading day
        
    Raises:
        ValueError: If no valid trading day found within 10 days
    """
    date = pd.Timestamp(date).normalize()
    
    # Try up to 10 days to find previous trading day
    for i in range(10):
        check_date = date - timedelta(days=i)
        
        if CALENDAR_AVAILABLE and NYMEX_CALENDAR:
            # Use market calendar for accurate check
            schedule = NYMEX_CALENDAR.schedule(
                start_date=check_date, 
                end_date=check_date
            )
            if len(schedule) > 0:
                return check_date
        else:
            # Simple check: not weekend and not major holiday
            if not is_weekend(check_date) and not is_us_holiday(check_date):
                return check_date
    
    raise ValueError(f"No valid trading day found within 10 days before {date}")


def adjust_date_range(
    start_date: pd.Timestamp, 
    end_date: pd.Timestamp
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Adjust date range to valid trading days.
    
    Rules:
    1. Start date shifted forward to next session day
    2. End date shifted backward to previous session day
    3. If adjusted range is invalid (start >= end), raise error
    
    Args:
        start_date: Requested start date
        end_date: Requested end date
        
    Returns:
        Tuple of (adjusted_start, adjusted_end)
        
    Raises:
        ValueError: If no valid trading days in range
    """
    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()
    
    # Adjust to valid trading days
    adjusted_start = next_session_day(start_date)
    adjusted_end = previous_session_day(end_date)
    
    # Validate range - allow same day ranges (single day backtests)
    if adjusted_start > adjusted_end:
        raise ValueError(
            f"No valid trading days between {start_date} and {end_date}. "
            f"Adjusted range {adjusted_start} to {adjusted_end} is invalid."
        )
    
    # Log if dates were adjusted
    if adjusted_start != start_date or adjusted_end != end_date:
        logger.info(
            f"Adjusted date range from {start_date:%Y-%m-%d} - {end_date:%Y-%m-%d} "
            f"to {adjusted_start:%Y-%m-%d} - {adjusted_end:%Y-%m-%d} (trading days only)"
        )
    
    return adjusted_start, adjusted_end


def get_trading_days(
    start_date: pd.Timestamp, 
    end_date: pd.Timestamp
) -> List[pd.Timestamp]:
    """
    Get list of all trading days in date range.
    
    Args:
        start_date: Start of range (inclusive)
        end_date: End of range (inclusive)
        
    Returns:
        List of trading days as Timestamps
    """
    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()
    
    if CALENDAR_AVAILABLE and NYMEX_CALENDAR:
        # Use market calendar for accurate list
        schedule = NYMEX_CALENDAR.schedule(start_date=start_date, end_date=end_date)
        return list(schedule.index)
    else:
        # Generate list manually
        trading_days = []
        current = start_date
        while current <= end_date:
            if not is_weekend(current) and not is_us_holiday(current):
                trading_days.append(current)
            current += timedelta(days=1)
        return trading_days


def validate_trading_day(date: pd.Timestamp) -> bool:
    """
    Check if a date is a valid trading day.
    
    Args:
        date: Date to validate
        
    Returns:
        True if valid trading day, False otherwise
    """
    date = pd.Timestamp(date).normalize()
    
    if CALENDAR_AVAILABLE and NYMEX_CALENDAR:
        schedule = NYMEX_CALENDAR.schedule(start_date=date, end_date=date)
        return len(schedule) > 0
    else:
        return not is_weekend(date) and not is_us_holiday(date)