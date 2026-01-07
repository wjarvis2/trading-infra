"""
GPU-accelerated volatility indicators using CuPy.

Provides GPU versions of EWMA volatility, rolling std, and other volatility measures.
All functions validate input data to ensure no synthetic values.
"""

import numpy as np
from typing import Optional
import warnings

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

from .technical_gpu import validate_gpu_input


def ewma_volatility_gpu(
    returns: np.ndarray,
    span: int = 20,
    min_periods: int = 15,
    annualize: bool = True,
    periods_per_year: int = 252,
    eps: float = 1e-12
) -> np.ndarray:
    """
    GPU-accelerated EWMA volatility calculation.
    
    Critical for volatility regime detection. Ensures no synthetic data
    by validating all inputs and using only real market returns.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns (not prices) - MUST be from real market data
    span : int
        Span for EWMA calculation
    min_periods : int
        Minimum observations required
    annualize : bool
        Whether to annualize volatility
    periods_per_year : int
        Periods per year (252 for daily, 8760 for hourly)
    eps : float
        Small value to prevent negative variances
        
    Returns
    -------
    np.ndarray
        EWMA volatility values (CPU array)
    """
    # Critical validation - no synthetic returns allowed
    returns_gpu = validate_gpu_input(returns, "returns")
    
    # Check returns are reasonable (not synthetic)
    max_abs_return = cp.max(cp.abs(returns_gpu[~cp.isnan(returns_gpu)]))
    if max_abs_return > 0.5:  # 50% return in single period is suspicious
        warnings.warn(f"Extreme returns detected (max: {float(max_abs_return):.2%}). Verify data integrity.")
    
    if span < 2:
        raise ValueError(f"Span must be >= 2, got {span}")
    
    if min_periods is None:
        min_periods = span
    
    n = len(returns_gpu)
    
    # Calculate squared returns for variance
    returns_squared = returns_gpu * returns_gpu
    
    # EWMA calculation
    alpha = 2.0 / (span + 1.0)
    ewma_var = cp.full(n, cp.nan)
    
    # Need min_periods for first calculation
    if n >= min_periods:
        # Initial variance
        ewma_var[min_periods-1] = cp.mean(returns_squared[:min_periods])
        
        # EWMA update
        for i in range(min_periods, n):
            ewma_var[i] = alpha * returns_squared[i] + (1 - alpha) * ewma_var[i-1]
    
    # Ensure non-negative variance
    ewma_var = cp.maximum(ewma_var, eps)
    
    # Convert variance to volatility
    ewma_vol = cp.sqrt(ewma_var)
    
    # Annualize if requested
    if annualize:
        ewma_vol = ewma_vol * cp.sqrt(periods_per_year)
    
    return cp.asnumpy(ewma_vol)


def rolling_std_gpu(
    returns: np.ndarray,
    window: int = 20,
    min_periods: Optional[int] = None,
    annualize: bool = True,
    periods_per_year: int = 252
) -> np.ndarray:
    """
    GPU-accelerated rolling standard deviation.
    
    Parameters
    ----------
    returns : np.ndarray
        Returns array (MUST be real market data)
    window : int
        Rolling window size
    min_periods : int, optional
        Minimum periods for calculation
    annualize : bool
        Whether to annualize
    periods_per_year : int
        Periods per year
        
    Returns
    -------
    np.ndarray
        Rolling volatility values
    """
    returns_gpu = validate_gpu_input(returns, "returns")
    
    if window < 2:
        raise ValueError(f"Window must be >= 2, got {window}")
    
    if min_periods is None:
        min_periods = window
    
    n = len(returns_gpu)
    rolling_vol = cp.full(n, cp.nan)
    
    # Calculate rolling std
    for i in range(min_periods-1, n):
        start_idx = max(0, i - window + 1)
        window_returns = returns_gpu[start_idx:i+1]
        
        if len(window_returns) >= min_periods:
            rolling_vol[i] = cp.std(window_returns)
    
    # Annualize if requested
    if annualize:
        rolling_vol = rolling_vol * cp.sqrt(periods_per_year)
    
    return cp.asnumpy(rolling_vol)


def parkinson_volatility_gpu(
    highs: np.ndarray,
    lows: np.ndarray,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252
) -> np.ndarray:
    """
    GPU-accelerated Parkinson volatility estimator.
    
    Uses high-low range for volatility estimation.
    More efficient than close-to-close volatility.
    
    Parameters
    ----------
    highs : np.ndarray
        High prices (MUST be real market data)
    lows : np.ndarray
        Low prices (MUST be real market data)
    window : int
        Rolling window
    annualize : bool
        Whether to annualize
    periods_per_year : int
        Periods per year
        
    Returns
    -------
    np.ndarray
        Parkinson volatility values
    """
    highs_gpu = validate_gpu_input(highs, "highs")
    lows_gpu = validate_gpu_input(lows, "lows")
    
    # Validate high >= low (data integrity check)
    if cp.any(highs_gpu < lows_gpu):
        raise ValueError("High prices less than low prices - data integrity issue")
    
    if window < 2:
        raise ValueError(f"Window must be >= 2, got {window}")
    
    n = len(highs_gpu)
    parkinson_vol = cp.full(n, cp.nan)
    
    # Parkinson's constant
    const = 1.0 / (4.0 * cp.log(2.0))
    
    # Calculate log(high/low)^2
    log_hl_squared = cp.log(highs_gpu / lows_gpu) ** 2
    
    # Rolling calculation
    for i in range(window-1, n):
        window_data = log_hl_squared[i-window+1:i+1]
        parkinson_vol[i] = cp.sqrt(const * cp.mean(window_data))
    
    # Annualize if requested
    if annualize:
        parkinson_vol = parkinson_vol * cp.sqrt(periods_per_year)
    
    return cp.asnumpy(parkinson_vol)


def garman_klass_volatility_gpu(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252
) -> np.ndarray:
    """
    GPU-accelerated Garman-Klass volatility estimator.
    
    Uses OHLC data for more efficient volatility estimation.
    
    Parameters
    ----------
    opens : np.ndarray
        Open prices (MUST be real market data)
    highs : np.ndarray
        High prices
    lows : np.ndarray
        Low prices
    closes : np.ndarray
        Close prices
    window : int
        Rolling window
    annualize : bool
        Whether to annualize
    periods_per_year : int
        Periods per year
        
    Returns
    -------
    np.ndarray
        Garman-Klass volatility values
    """
    # Validate all OHLC data
    opens_gpu = validate_gpu_input(opens, "opens")
    highs_gpu = validate_gpu_input(highs, "highs")
    lows_gpu = validate_gpu_input(lows, "lows")
    closes_gpu = validate_gpu_input(closes, "closes")
    
    # Data integrity checks
    if cp.any(highs_gpu < lows_gpu):
        raise ValueError("High prices less than low prices - data integrity issue")
    if cp.any(highs_gpu < opens_gpu) or cp.any(highs_gpu < closes_gpu):
        warnings.warn("High prices less than open/close - possible data issue")
    if cp.any(lows_gpu > opens_gpu) or cp.any(lows_gpu > closes_gpu):
        warnings.warn("Low prices greater than open/close - possible data issue")
    
    if window < 2:
        raise ValueError(f"Window must be >= 2, got {window}")
    
    n = len(opens_gpu)
    gk_vol = cp.full(n, cp.nan)
    
    # Garman-Klass formula components
    # Term 1: 0.5 * log(H/L)^2
    term1 = 0.5 * cp.log(highs_gpu / lows_gpu) ** 2
    
    # Term 2: -(2*log(2)-1) * log(C/O)^2
    term2 = -(2 * cp.log(2) - 1) * cp.log(closes_gpu / opens_gpu) ** 2
    
    # Combined GK statistic
    gk_stat = term1 + term2
    
    # Rolling calculation
    for i in range(window-1, n):
        window_data = gk_stat[i-window+1:i+1]
        gk_vol[i] = cp.sqrt(cp.mean(window_data))
    
    # Annualize if requested
    if annualize:
        gk_vol = gk_vol * cp.sqrt(periods_per_year)
    
    return cp.asnumpy(gk_vol)


def realized_volatility_gpu(
    returns: np.ndarray,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252
) -> np.ndarray:
    """
    GPU-accelerated realized volatility calculation.
    
    Sum of squared returns over rolling window.
    
    Parameters
    ----------
    returns : np.ndarray
        High-frequency returns (MUST be real market data)
    window : int
        Rolling window
    annualize : bool
        Whether to annualize
    periods_per_year : int
        Periods per year
        
    Returns
    -------
    np.ndarray
        Realized volatility values
    """
    returns_gpu = validate_gpu_input(returns, "returns")
    
    if window < 2:
        raise ValueError(f"Window must be >= 2, got {window}")
    
    n = len(returns_gpu)
    realized_vol = cp.full(n, cp.nan)
    
    # Square returns
    returns_squared = returns_gpu * returns_gpu
    
    # Rolling sum of squared returns
    for i in range(window-1, n):
        window_data = returns_squared[i-window+1:i+1]
        realized_vol[i] = cp.sqrt(cp.sum(window_data))
    
    # Annualize if requested
    if annualize:
        # Realized vol uses different annualization
        realized_vol = realized_vol * cp.sqrt(periods_per_year / window)
    
    return cp.asnumpy(realized_vol)