"""
Volatility indicators and calculations.

Provides various volatility measures including EWMA volatility,
rolling standard deviation, Parkinson volatility, and related calculations.
All implementations are optimized with Numba where possible.
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple, Optional
from common_lib.logging_config import InstrumentedLogger

instrumented_logger = InstrumentedLogger(__name__)


@njit
def ewma_volatility(
    returns: np.ndarray,
    span: int = 20,
    min_periods: int = 15,
    annualize: bool = True,
    periods_per_year: int = 252,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Calculate Exponentially Weighted Moving Average (EWMA) volatility.
    
    This is the primary volatility measure used in the Volatility Regime strategy.
    Uses an exponential weighting scheme that gives more weight to recent observations.
    
    Guarantees:
    - Non-negative variance (no complex numbers from sqrt)
    - Returns NaN until min_periods observations reached
    - Handles constant-price windows gracefully
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns or price changes (for spreads use price changes)
    span : int
        Span for EWMA calculation (similar to window size)
        Default of 20 is commonly used for daily data
    min_periods : int
        Minimum number of observations required to produce a value
    annualize : bool
        Whether to annualize the volatility
    periods_per_year : int
        Number of periods per year for annualization (252 for daily, 8760 for hourly)
    eps : float
        Small positive value to clip negative variances (default 1e-12)
        
    Returns
    -------
    np.ndarray
        EWMA volatility values (annualized if requested)
    """
    n = len(returns)
    
    # Sensible default for min_periods if not specified
    if min_periods is None:
        min_periods = span
    
    if n < min_periods:
        return np.full(n, np.nan)
    
    # Calculate alpha (smoothing factor)
    alpha = 2.0 / (span + 1)
    
    # Initialize output array
    vol = np.full(n, np.nan)
    
    # Calculate initial variance as simple variance of first span observations
    if n >= span:
        initial_var = np.var(returns[:span])
        # Clip negative variance (shouldn't happen but ensures safety)
        if initial_var < eps:
            initial_var = 0.0
        vol[span-1] = np.sqrt(initial_var)
        
        # Calculate EWMA variance
        ewma_var = initial_var
        for i in range(span, n):
            # Update EWMA variance
            ewma_var = alpha * (returns[i-1] ** 2) + (1 - alpha) * ewma_var
            
            # Clip tiny negatives caused by FP error
            if ewma_var < eps:
                ewma_var = 0.0
            
            vol[i] = np.sqrt(ewma_var)
    
    # Ensure NaN for first min_periods-1 values
    vol[:min_periods-1] = np.nan
    
    # Annualize if requested
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    
    return vol


@njit
def rolling_volatility(
    returns: np.ndarray,
    window: int = 20,
    min_periods: Optional[int] = None,
    annualize: bool = True,
    periods_per_year: int = 252
) -> np.ndarray:
    """
    Calculate rolling window standard deviation volatility.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns
    window : int
        Rolling window size
    min_periods : int, optional
        Minimum periods required. If None, uses window size
    annualize : bool
        Whether to annualize the volatility
    periods_per_year : int
        Number of periods per year for annualization
        
    Returns
    -------
    np.ndarray
        Rolling volatility values
    """
    n = len(returns)
    if min_periods is None:
        min_periods = window
    
    vol = np.full(n, np.nan)
    
    for i in range(window-1, n):
        # Check if we have enough non-NaN values
        window_data = returns[max(0, i-window+1):i+1]
        non_nan_count = 0
        for val in window_data:
            if not np.isnan(val):
                non_nan_count += 1
        
        if non_nan_count >= min_periods:
            # Calculate standard deviation
            vol[i] = np.std(window_data)
    
    # Annualize if requested
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    
    return vol


@njit
def parkinson_volatility(
    highs: np.ndarray,
    lows: np.ndarray,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252
) -> np.ndarray:
    """
    Calculate Parkinson volatility estimator using high-low prices.
    
    More efficient than close-to-close volatility as it uses intraday information.
    Formula: sqrt(1/(4*n*ln(2)) * sum((ln(H/L))^2))
    
    Parameters
    ----------
    highs : np.ndarray
        High prices
    lows : np.ndarray
        Low prices
    window : int
        Rolling window size
    annualize : bool
        Whether to annualize the volatility
    periods_per_year : int
        Number of periods per year
        
    Returns
    -------
    np.ndarray
        Parkinson volatility estimates
    """
    n = len(highs)
    if n != len(lows) or n < window:
        return np.full(n, np.nan)
    
    vol = np.full(n, np.nan)
    factor = 1.0 / (4.0 * np.log(2))
    
    for i in range(window-1, n):
        sum_sq = 0.0
        valid_count = 0
        
        for j in range(i-window+1, i+1):
            if highs[j] > 0 and lows[j] > 0 and highs[j] >= lows[j]:
                log_hl = np.log(highs[j] / lows[j])
                sum_sq += log_hl * log_hl
                valid_count += 1
        
        if valid_count > 0:
            vol[i] = np.sqrt(factor * sum_sq / valid_count)
    
    # Annualize if requested
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    
    return vol


@njit
def volatility_percentile(
    volatility: np.ndarray,
    lookback_window: int = 504
) -> np.ndarray:
    """
    Calculate rolling percentile rank of volatility.
    
    Used in the Volatility Regime strategy to classify regimes based on
    where current volatility stands relative to historical distribution.
    
    Parameters
    ----------
    volatility : np.ndarray
        Volatility time series
    lookback_window : int
        Window for percentile calculation (default 504 = 2 years of trading days)
        
    Returns
    -------
    np.ndarray
        Percentile values between 0 and 1
    """
    n = len(volatility)
    percentiles = np.full(n, np.nan)
    
    for i in range(lookback_window-1, n):
        # Get window of data
        window_data = volatility[i-lookback_window+1:i+1]
        
        # Count non-NaN values
        valid_values = []
        for val in window_data:
            if not np.isnan(val):
                valid_values.append(val)
        
        if len(valid_values) > 0:
            current_val = volatility[i]
            if not np.isnan(current_val):
                # Calculate percentile rank
                count_below = 0
                for val in valid_values:
                    if val < current_val:
                        count_below += 1
                
                percentiles[i] = count_below / len(valid_values)
    
    return percentiles


def calculate_spread_volatility(
    spread_prices: pd.Series,
    method: str = 'ewma',
    span: int = 20,
    min_periods: int = 15,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate volatility for a spread time series using absolute price changes.
    
    Uses linear (additive) differences instead of multiplicative returns
    to handle spreads that can cross zero or go negative (e.g., calendar spreads
    in backwardation).
    
    Parameters
    ----------
    spread_prices : pd.Series
        Spread price series (can be negative)
    method : str
        Volatility calculation method ('ewma' or 'rolling')
    span : int
        Span/window for volatility calculation
    min_periods : int
        Minimum periods required
    annualize : bool
        Whether to annualize volatility
        
    Returns
    -------
    pd.Series
        Volatility series in $/bbl (same units as input prices)
    """
    # Log input characteristics
    instrumented_logger.log_data_transform(
        stage="spread_volatility_start",
        input_data=spread_prices,
        metadata={
            "method": method,
            "span": span,
            "min_periods": min_periods,
            "annualize": annualize,
            "has_negative": bool((spread_prices < 0).any()) if len(spread_prices) > 0 else False,
            "crosses_zero": bool((spread_prices.shift() * spread_prices < 0).any()) if len(spread_prices) > 1 else False
        }
    )
    
    # Calculate absolute price changes (not percentage returns!)
    # This handles negative spreads and zero-crossings correctly
    price_changes = spread_prices.diff().fillna(0).values
    
    # Calculate volatility based on method
    if method == 'ewma':
        vol_values = ewma_volatility(
            returns=price_changes,  # Using price changes, not returns
            span=span,
            min_periods=min_periods,
            annualize=annualize
        )
    elif method == 'rolling':
        vol_values = rolling_volatility(
            returns=price_changes,  # Using price changes, not returns
            window=span,
            min_periods=min_periods,
            annualize=annualize
        )
    else:
        raise ValueError(f"Unknown volatility method: {method}")
    
    # Check for complex numbers before returning
    if np.iscomplexobj(vol_values):
        instrumented_logger.log_complex_number_detection(
            location="calculate_spread_volatility",
            variable="vol_values",
            value=vol_values,
            context={
                "method": method,
                "input_min": float(spread_prices.min()) if len(spread_prices) > 0 else None,
                "input_max": float(spread_prices.max()) if len(spread_prices) > 0 else None
            }
        )
    
    result = pd.Series(vol_values, index=spread_prices.index)
    
    # Log output characteristics
    instrumented_logger.log_data_transform(
        stage="spread_volatility_complete",
        output_data=result,
        metadata={
            "has_nan": bool(result.isna().any()),
            "has_complex": bool(np.iscomplexobj(result.values))
        }
    )
    
    # Return as pandas Series
    return result


def calculate_volatility_bands(
    price: float,
    volatility: float,
    multiplier: float = 1.0,
    time_horizon: int = 1
) -> Tuple[float, float]:
    """
    Calculate volatility-based trading bands around a price level.
    
    Used in the Volatility Regime strategy to set entry/exit levels.
    
    Parameters
    ----------
    price : float
        Center price (e.g., fair value)
    volatility : float
        Annualized volatility
    multiplier : float
        Number of standard deviations for bands
    time_horizon : int
        Time horizon in days for band calculation
        
    Returns
    -------
    tuple
        (lower_band, upper_band)
    """
    # Convert annualized vol to period vol
    period_vol = volatility * np.sqrt(time_horizon / 252)
    
    # Calculate bands
    band_width = price * period_vol * multiplier
    lower_band = price - band_width
    upper_band = price + band_width
    
    return lower_band, upper_band


def calculate_sparse_spread_volatility(
    spread_prices: pd.Series,
    bar_seconds: int = 5,
    span: int = 288,
    min_periods: int = 15,
    annualize: bool = True,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Calculate volatility for sparse spread data with calendar-time normalization.
    
    This function handles sparse data (e.g., back-month contracts) where bars
    may be missing. It normalizes price changes by actual time elapsed.
    
    Parameters
    ----------
    spread_prices : pd.Series
        Spread price series with datetime index (can have gaps)
    bar_seconds : int
        Expected seconds per bar (e.g., 5 for 5s bars, 15 for 15s bars)
    span : int
        Span for EWMA calculation
    min_periods : int
        Minimum periods required
    annualize : bool
        Whether to annualize volatility
    periods_per_year : int
        Number of periods per year for annualization
        
    Returns
    -------
    pd.Series
        Calendar-time normalized volatility series
        
    Notes
    -----
    This is specifically for handling sparse back-month contracts where
    5-second bars might have gaps. The volatility is adjusted for the
    actual time between observations.
    """
    # Drop NaN prices
    clean_prices = spread_prices.dropna()
    
    if len(clean_prices) < min_periods:
        return pd.Series(np.nan, index=spread_prices.index)
    
    # Calculate price changes
    price_changes = clean_prices.diff()
    
    # Calculate actual time elapsed between observations
    time_diffs = clean_prices.index.to_series().diff().dt.total_seconds()
    
    # Replace zero/NaN time diffs with expected bar size
    time_diffs = time_diffs.fillna(bar_seconds).replace(0, bar_seconds)
    
    # Normalize price changes by sqrt of time ratio
    # This scales changes to a consistent 1-bar basis
    time_adjustment = np.sqrt(time_diffs / bar_seconds)
    adjusted_changes = price_changes / time_adjustment
    
    # Calculate EWMA volatility on adjusted changes
    vol_values = ewma_volatility(
        returns=adjusted_changes.fillna(0).values,
        span=span,
        min_periods=min_periods,
        annualize=annualize,
        periods_per_year=periods_per_year
    )
    
    # Create output series with original index
    result = pd.Series(np.nan, index=spread_prices.index)
    result.loc[clean_prices.index] = vol_values
    
    # Forward fill to match original index if needed
    result = result.ffill()
    
    return result