"""
Core technical indicators optimized with Numba.
"""

import numpy as np
from numba import njit
from typing import Tuple


@njit
def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Numba-accelerated Average True Range calculation.
    
    Parameters
    ----------
    highs : np.ndarray
        High prices
    lows : np.ndarray
        Low prices
    closes : np.ndarray
        Close prices
    window : int
        ATR lookback window (default: 20)
        
    Returns
    -------
    np.ndarray
        ATR values (first window-1 values are NaN)
    """
    n = len(highs)
    if n < window:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr[i] = max(hl, hc, lc)
    
    # Calculate ATR using exponential moving average
    atr_values = np.full(n, np.nan)
    
    # Initial ATR is simple average
    atr_values[window-1] = np.mean(tr[:window])
    
    # Subsequent values use EMA formula
    alpha = 1.0 / window
    for i in range(window, n):
        atr_values[i] = alpha * tr[i] + (1 - alpha) * atr_values[i-1]
    
    return atr_values


@njit
def bollinger_bands(prices: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.
    
    Parameters
    ----------
    prices : np.ndarray
        Price series (typically close prices)
    window : int
        Moving average window
    num_std : float
        Number of standard deviations for bands
        
    Returns
    -------
    tuple
        (middle_band, upper_band, lower_band)
    """
    n = len(prices)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(window-1, n):
        window_data = prices[i-window+1:i+1]
        mean = np.mean(window_data)
        std = np.std(window_data)
        
        middle[i] = mean
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std
    
    return middle, upper, lower


@njit
def rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index.
    
    Parameters
    ----------
    prices : np.ndarray
        Price series
    window : int
        RSI period
        
    Returns
    -------
    np.ndarray
        RSI values (0-100)
    """
    n = len(prices)
    if n < window + 1:
        return np.full(n, np.nan)
    
    rsi_values = np.full(n, np.nan)
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Initial averages
    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])
    
    # Calculate RSI
    if avg_loss == 0:
        rsi_values[window] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_values[window] = 100 - (100 / (1 + rs))
    
    # Calculate subsequent values using EMA
    alpha = 1.0 / window
    
    for i in range(window, len(deltas)):
        gain = gains[i]
        loss = losses[i]
        
        avg_gain = alpha * gain + (1 - alpha) * avg_gain
        avg_loss = alpha * loss + (1 - alpha) * avg_loss
        
        if avg_loss == 0:
            rsi_values[i+1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i+1] = 100 - (100 / (1 + rs))
    
    return rsi_values


@njit
def vwap_session(prices: np.ndarray, volumes: np.ndarray, session_starts: np.ndarray) -> np.ndarray:
    """
    Calculate Volume Weighted Average Price with session resets.
    
    Parameters
    ----------
    prices : np.ndarray
        Price series (typically close or mid)
    volumes : np.ndarray
        Volume series
    session_starts : np.ndarray
        Boolean array indicating session starts
        
    Returns
    -------
    np.ndarray
        VWAP values
    """
    n = len(prices)
    vwap = np.empty(n)
    
    cum_pv = 0.0
    cum_vol = 0.0
    
    for i in range(n):
        if session_starts[i] or i == 0:
            # Reset at session start
            cum_pv = prices[i] * volumes[i]
            cum_vol = volumes[i]
        else:
            # Accumulate
            cum_pv += prices[i] * volumes[i]
            cum_vol += volumes[i]
        
        if cum_vol > 0:
            vwap[i] = cum_pv / cum_vol
        else:
            vwap[i] = prices[i]
    
    return vwap


@njit
def rolling_correlation(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """
    Fast rolling correlation calculation.
    
    Parameters
    ----------
    x, y : np.ndarray
        Input arrays
    window : int
        Rolling window size
        
    Returns
    -------
    np.ndarray
        Rolling correlations
    """
    n = len(x)
    if n != len(y) or n < window:
        return np.full(n, np.nan)
    
    corr = np.full(n, np.nan)
    
    for i in range(window-1, n):
        x_window = x[i-window+1:i+1]
        y_window = y[i-window+1:i+1]
        
        # Calculate correlation coefficient
        x_mean = np.mean(x_window)
        y_mean = np.mean(y_window)
        
        numerator = np.sum((x_window - x_mean) * (y_window - y_mean))
        x_std = np.sqrt(np.sum((x_window - x_mean) ** 2))
        y_std = np.sqrt(np.sum((y_window - y_mean) ** 2))
        
        if x_std > 0 and y_std > 0:
            corr[i] = numerator / (x_std * y_std)
        else:
            corr[i] = 0.0
    
    return corr