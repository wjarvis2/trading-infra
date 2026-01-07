"""
Session-based analysis and metrics.
"""

import pandas as pd
from .technical import vwap_session


def calculate_session_metrics(
    df: pd.DataFrame,
    session_start_hour: int = 18,  # 6 PM ET (futures session start)
    timezone: str = 'America/New_York'
) -> pd.DataFrame:
    """
    Calculate session-based metrics (VWAP, range, etc).
    
    Parameters
    ----------
    df : pd.DataFrame
        Intraday data with OHLCV columns
    session_start_hour : int
        Hour when new session starts (in specified timezone)
    timezone : str
        Timezone for session calculations
        
    Returns
    -------
    pd.DataFrame
        Original data with added session metrics
    """
    # Convert to session timezone for marking
    df_tz = df.copy()
    df_tz.index = df_tz.index.tz_convert(timezone)
    
    # Mark session starts
    session_starts = (df_tz.index.hour == session_start_hour) & (
        df_tz.index.hour != df_tz.index.shift(1).hour
    )
    
    # Calculate mid price if not present
    if 'mid' not in df.columns:
        df['mid'] = (df['high'] + df['low']) / 2
    
    # Session VWAP
    df['session_vwap'] = vwap_session(
        df['mid'].values,
        df['volume'].values,
        session_starts.values
    )
    
    # Session high/low (expanding window within session)
    df['session_high'] = df.groupby(session_starts.cumsum())['high'].expanding().max().values
    df['session_low'] = df.groupby(session_starts.cumsum())['low'].expanding().min().values
    
    # Session range
    df['session_range'] = df['session_high'] - df['session_low']
    
    # Position within session range
    df['session_position'] = (df['mid'] - df['session_low']) / df['session_range']
    df['session_position'] = df['session_position'].clip(0, 1)
    
    return df


def identify_trading_sessions(
    timestamps: pd.DatetimeIndex,
    timezone: str = 'America/New_York'
) -> pd.Series:
    """
    Identify trading sessions (Asian, European, US).
    
    Parameters
    ----------
    timestamps : pd.DatetimeIndex
        UTC timestamps
    timezone : str
        Reference timezone for session definitions
        
    Returns
    -------
    pd.Series
        Session labels ('asian', 'european', 'us', 'off_hours')
    """
    # Convert to reference timezone
    local_times = timestamps.tz_convert(timezone)
    hours = local_times.hour
    
    # Define sessions (in ET)
    # Asian: 7 PM - 3 AM
    # European: 3 AM - 9 AM  
    # US: 9 AM - 5 PM
    # Off hours: 5 PM - 7 PM
    
    sessions = pd.Series('off_hours', index=timestamps)
    
    # Asian session (crosses midnight)
    sessions[(hours >= 19) | (hours < 3)] = 'asian'
    
    # European session
    sessions[(hours >= 3) & (hours < 9)] = 'european'
    
    # US session
    sessions[(hours >= 9) & (hours < 17)] = 'us'
    
    return sessions