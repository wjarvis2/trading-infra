"""
Central constants for the trading system.

This module provides single source of truth for all system-wide constants,
particularly time-related constants used in annualization and volatility calculations.

Created: 2025-08-06
"""

# Trading hours and days
TRADING_HOURS_PER_DAY = 6.5  # CME energy markets: 17:00-23:30 CT (6.5 hours)
TRADING_DAYS_PER_YEAR = 252  # Standard trading days per year

# Bars per trading day for each frequency
# Based on 6.5 trading hours per day
BARS_PER_DAY = {
    "5s": 4680,   # 6.5 hours * 3600 seconds / 5 seconds
    "15s": 1560,  # 6.5 hours * 3600 seconds / 15 seconds  
    "30s": 780,   # 6.5 hours * 3600 seconds / 30 seconds
    "1m": 390,    # 6.5 hours * 60 minutes
    "1T": 390,    # Pandas format for 1 minute
    "5m": 78,     # 6.5 hours * 60 minutes / 5
    "5T": 78,     # Pandas format for 5 minutes
    "10m": 39,    # 6.5 hours * 60 minutes / 10
    "10T": 39,    # Pandas format for 10 minutes
    "15m": 26,    # 6.5 hours * 60 minutes / 15
    "15T": 26,    # Pandas format for 15 minutes
    "30m": 13,    # 6.5 hours * 60 minutes / 30
    "30T": 13,    # Pandas format for 30 minutes
    "1h": 6.5,    # 6.5 hours (partial bar for last hour)
    "1H": 6.5,    # Pandas format for 1 hour
}

# Bars per year for each frequency (for annualization)
BARS_PER_YEAR = {
    freq: bars_per_day * TRADING_DAYS_PER_YEAR 
    for freq, bars_per_day in BARS_PER_DAY.items()
}

# Seconds per bar for each frequency (for time-based calculations)
SECONDS_PER_BAR = {
    "5s": 5,
    "15s": 15,
    "30s": 30,
    "1m": 60,
    "1T": 60,
    "5m": 300,
    "5T": 300,
    "10m": 600,
    "10T": 600,
    "15m": 900,
    "15T": 900,
    "30m": 1800,
    "30T": 1800,
    "1h": 3600,
    "1H": 3600,
}

# Contract specifications
CONTRACT_SPECS = {
    "CL": {
        "multiplier": 1000,  # 1000 barrels per contract
        "tick_size": 0.01,   # $0.01 per barrel
        "currency": "USD",
        "exchange": "NYMEX",
        "description": "Light Sweet Crude Oil",
        "expiry_day": 20,    # Approximate - expires ~3 business days before 25th
    },
    "RB": {
        "multiplier": 42000,  # 42,000 gallons per contract  
        "tick_size": 0.0001,  # $0.0001 per gallon
        "currency": "USD", 
        "exchange": "NYMEX",
        "description": "RBOB Gasoline",
        "expiry_day": 20,    # Last business day of month preceding delivery
    },
    "HO": {
        "multiplier": 42000,  # 42,000 gallons per contract
        "tick_size": 0.0001,  # $0.0001 per gallon
        "currency": "USD",
        "exchange": "NYMEX", 
        "description": "NY Harbor ULSD Heating Oil",
        "expiry_day": 20,    # Last business day of month preceding delivery
    },
}

# Slippage and execution constants
DEFAULT_SLIPPAGE_TICKS = {
    "CL": 1,     # 1 tick = $0.01 per barrel
    "RB": 10,    # 10 ticks = $0.001 per gallon
    "HO": 10,    # 10 ticks = $0.001 per gallon
}

# Risk management constants
MAX_POSITION_SIZE = {
    "CL": 100,   # Maximum 100 contracts per position
    "RB": 50,    # Maximum 50 contracts per position
    "HO": 50,    # Maximum 50 contracts per position
}

# Data quality thresholds
MIN_BARS_FOR_VOLATILITY = 20  # Minimum bars required for volatility calculation
MIN_BARS_FOR_ATR = 14         # Minimum bars required for ATR calculation
MAX_FORWARD_FILL_BARS = 2     # Maximum bars to forward fill in resampling
MAX_PRICE_JUMP_PCT = 0.5      # Maximum allowed price jump (50%) before flagging as suspicious

# Memory and performance constants
MAX_BAR_HISTORY = 1000        # Maximum bars to keep in memory per contract
DEFAULT_BATCH_SIZE = 10000    # Default batch size for data operations
GPU_MIN_DATA_SIZE = 100000    # Minimum data points before using GPU acceleration

def get_bars_per_year(frequency: str) -> int:
    """
    Get the number of bars per year for a given frequency.
    
    Args:
        frequency: Bar frequency (e.g., '5s', '1m', '5m')
        
    Returns:
        Number of bars per year
        
    Raises:
        ValueError: If frequency is not recognized
    """
    if frequency not in BARS_PER_YEAR:
        raise ValueError(f"Unknown frequency: {frequency}. Valid frequencies: {list(BARS_PER_YEAR.keys())}")
    return BARS_PER_YEAR[frequency]

def get_annualization_factor(frequency: str) -> float:
    """
    Get the annualization factor (sqrt of bars per year) for volatility calculations.
    
    Args:
        frequency: Bar frequency (e.g., '5s', '1m', '5m')
        
    Returns:
        Square root of bars per year for annualization
        
    Raises:
        ValueError: If frequency is not recognized
    """
    import numpy as np
    bars_per_year = get_bars_per_year(frequency)
    return np.sqrt(bars_per_year)

def normalize_frequency(frequency: str) -> str:
    """
    Normalize frequency string to pandas-compatible format.
    
    Args:
        frequency: Bar frequency (e.g., '1m', '5m')
        
    Returns:
        Pandas-compatible frequency string (e.g., '1T', '5T')
    """
    # Convert minute notation to pandas format
    if frequency.endswith('m') and frequency[:-1].isdigit():
        return frequency[:-1] + 'T'
    return frequency