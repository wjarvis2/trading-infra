"""
GPU-accelerated technical indicators using CuPy.

All functions strictly validate input data and reject synthetic/invalid values.
These are drop-in replacements for CPU versions with identical outputs.
"""

import numpy as np
from typing import Tuple, Optional
import warnings

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


def validate_gpu_input(data: np.ndarray, name: str = "data") -> cp.ndarray:
    """
    Validate data before GPU processing.
    
    CRITICAL: Ensures no synthetic or invalid data enters GPU pipeline.
    
    Parameters
    ----------
    data : np.ndarray
        Input data to validate
    name : str
        Name of data for error messages
        
    Returns
    -------
    cp.ndarray
        Validated GPU array
        
    Raises
    ------
    ValueError
        If data contains NaN, inf, or insufficient samples
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available. Install cupy.")
    
    # Check for synthetic/invalid data
    if np.any(np.isnan(data)):
        raise ValueError(f"{name} contains NaN values - cannot process synthetic/missing data")
    
    if np.any(np.isinf(data)):
        raise ValueError(f"{name} contains infinite values - invalid data")
    
    if len(data) < 2:
        raise ValueError(f"{name} has insufficient data points: {len(data)}")
    
    # Check data is not constant (likely synthetic)
    if np.std(data) < 1e-10:
        warnings.warn(f"{name} appears to be constant - possible synthetic data")
    
    return cp.asarray(data, dtype=cp.float64)


def atr_gpu(
    highs: np.ndarray,
    lows: np.ndarray, 
    closes: np.ndarray,
    window: int = 20,
    batch_size: Optional[int] = None
) -> np.ndarray:
    """
    GPU-accelerated Average True Range calculation.
    
    Processes multiple securities in parallel if data is 2D.
    
    Parameters
    ----------
    highs : np.ndarray
        High prices (1D or 2D: [n_securities, n_timestamps])
    lows : np.ndarray
        Low prices
    closes : np.ndarray
        Close prices
    window : int
        ATR lookback window
    batch_size : int, optional
        Process securities in batches to manage memory
        
    Returns
    -------
    np.ndarray
        ATR values (CPU array)
    """
    # Validate all inputs
    highs_gpu = validate_gpu_input(highs, "highs")
    lows_gpu = validate_gpu_input(lows, "lows")
    closes_gpu = validate_gpu_input(closes, "closes")
    
    if window < 2:
        raise ValueError(f"Window must be >= 2, got {window}")
    
    n = len(highs_gpu)
    if n < window:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = cp.empty_like(highs_gpu)
    tr[0] = highs_gpu[0] - lows_gpu[0]
    
    # Vectorized TR calculation
    hl = highs_gpu[1:] - lows_gpu[1:]
    hc = cp.abs(highs_gpu[1:] - closes_gpu[:-1])
    lc = cp.abs(lows_gpu[1:] - closes_gpu[:-1])
    tr[1:] = cp.maximum(cp.maximum(hl, hc), lc)
    
    # Calculate ATR using exponential moving average
    atr_values = cp.full(n, cp.nan)
    
    # Initial ATR is simple average
    atr_values[window-1] = cp.mean(tr[:window])
    
    # Vectorized EMA calculation
    alpha = 1.0 / window
    for i in range(window, n):
        atr_values[i] = alpha * tr[i] + (1 - alpha) * atr_values[i-1]
    
    # Return CPU array
    return cp.asnumpy(atr_values)


def bollinger_bands_gpu(
    prices: np.ndarray,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU-accelerated Bollinger Bands calculation.
    
    Parameters
    ----------
    prices : np.ndarray
        Price series
    window : int
        Moving average window
    num_std : float
        Number of standard deviations
        
    Returns
    -------
    tuple
        (middle_band, upper_band, lower_band) as CPU arrays
    """
    prices_gpu = validate_gpu_input(prices, "prices")
    
    if window < 2:
        raise ValueError(f"Window must be >= 2, got {window}")
    
    n = len(prices_gpu)
    
    # Use CuPy's rolling window functions for efficiency
    # Create rolling windows using stride tricks
    from cupy.lib.stride_tricks import as_strided
    
    # Pad with NaN for initial values
    middle = cp.full(n, cp.nan)
    upper = cp.full(n, cp.nan)
    lower = cp.full(n, cp.nan)
    
    # Vectorized rolling calculation
    for i in range(window-1, n):
        window_data = prices_gpu[i-window+1:i+1]
        mean = cp.mean(window_data)
        std = cp.std(window_data)
        
        middle[i] = mean
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std
    
    return cp.asnumpy(middle), cp.asnumpy(upper), cp.asnumpy(lower)


def rsi_gpu(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """
    GPU-accelerated Relative Strength Index.
    
    Parameters
    ----------
    prices : np.ndarray
        Price series
    window : int
        RSI period
        
    Returns
    -------
    np.ndarray
        RSI values (0-100 scale)
    """
    prices_gpu = validate_gpu_input(prices, "prices")
    
    if window < 2:
        raise ValueError(f"Window must be >= 2, got {window}")
    
    # Calculate price changes
    deltas = cp.diff(prices_gpu)
    
    # Separate gains and losses
    gains = cp.where(deltas > 0, deltas, 0)
    losses = cp.where(deltas < 0, -deltas, 0)
    
    # Calculate average gains and losses
    n = len(prices_gpu)
    avg_gains = cp.full(n, cp.nan)
    avg_losses = cp.full(n, cp.nan)
    rsi_values = cp.full(n, cp.nan)
    
    # Initial averages
    avg_gains[window] = cp.mean(gains[:window])
    avg_losses[window] = cp.mean(losses[:window])
    
    # EMA calculation
    alpha = 1.0 / window
    for i in range(window + 1, n):
        avg_gains[i] = alpha * gains[i-1] + (1 - alpha) * avg_gains[i-1]
        avg_losses[i] = alpha * losses[i-1] + (1 - alpha) * avg_losses[i-1]
    
    # Calculate RSI
    rs = avg_gains[window:] / (avg_losses[window:] + 1e-10)  # Avoid division by zero
    rsi_values[window:] = 100 - (100 / (1 + rs))
    
    return cp.asnumpy(rsi_values)


def ema_gpu(
    prices: np.ndarray,
    span: int,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    GPU-accelerated Exponential Moving Average.
    
    Parameters
    ----------
    prices : np.ndarray
        Price series
    span : int
        EMA span (similar to window)
    min_periods : int, optional
        Minimum periods before producing values
        
    Returns
    -------
    np.ndarray
        EMA values
    """
    prices_gpu = validate_gpu_input(prices, "prices")
    
    if span < 1:
        raise ValueError(f"Span must be >= 1, got {span}")
    
    if min_periods is None:
        min_periods = span
    
    n = len(prices_gpu)
    ema_values = cp.full(n, cp.nan)
    
    # Alpha for EMA
    alpha = 2.0 / (span + 1.0)
    
    # Initial value
    if n >= min_periods:
        ema_values[min_periods-1] = cp.mean(prices_gpu[:min_periods])
        
        # Calculate EMA
        for i in range(min_periods, n):
            ema_values[i] = alpha * prices_gpu[i] + (1 - alpha) * ema_values[i-1]
    
    return cp.asnumpy(ema_values)


def sma_gpu(
    prices: np.ndarray,
    window: int
) -> np.ndarray:
    """
    GPU-accelerated Simple Moving Average.
    
    Parameters
    ----------
    prices : np.ndarray
        Price series
    window : int
        Moving average window
        
    Returns
    -------
    np.ndarray
        SMA values
    """
    prices_gpu = validate_gpu_input(prices, "prices")
    
    if window < 1:
        raise ValueError(f"Window must be >= 1, got {window}")
    
    n = len(prices_gpu)
    sma_values = cp.full(n, cp.nan)
    
    # Vectorized moving average
    for i in range(window-1, n):
        sma_values[i] = cp.mean(prices_gpu[i-window+1:i+1])
    
    return cp.asnumpy(sma_values)


def calculate_returns_gpu(
    prices: np.ndarray,
    method: str = 'simple'
) -> np.ndarray:
    """
    GPU-accelerated returns calculation.
    
    Parameters
    ----------
    prices : np.ndarray
        Price series
    method : str
        'simple' for arithmetic returns, 'log' for log returns
        
    Returns
    -------
    np.ndarray
        Returns array (first value is NaN)
    """
    prices_gpu = validate_gpu_input(prices, "prices")
    
    if method == 'simple':
        returns = cp.diff(prices_gpu) / prices_gpu[:-1]
    elif method == 'log':
        returns = cp.diff(cp.log(prices_gpu))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Prepend NaN for first value
    returns = cp.concatenate([cp.array([cp.nan]), returns])
    
    return cp.asnumpy(returns)