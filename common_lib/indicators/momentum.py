"""
Enhanced momentum indicators for trading strategies.

This module provides sophisticated momentum calculations including:
- Multi-timeframe momentum with weighted aggregation
- Z-score normalized momentum
- Momentum divergence detection
- PCA factor momentum
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class MomentumConfig:
    """Configuration for momentum calculations."""
    lookback_windows: List[int] = None
    weights: List[float] = None
    zscore_window: int = 20
    min_periods: int = 10
    
    def __post_init__(self):
        if self.lookback_windows is None:
            self.lookback_windows = [5, 10, 20]
        if self.weights is None:
            self.weights = [0.5, 0.3, 0.2]
        
        # Validate weights sum to 1
        if abs(sum(self.weights) - 1.0) > 1e-6:
            self.weights = [w / sum(self.weights) for w in self.weights]


def calculate_momentum(
    series: pd.Series,
    window: int,
    method: str = 'roc'
) -> pd.Series:
    """
    Calculate basic momentum for a single series.
    
    Parameters
    ----------
    series : pd.Series
        Price or value series
    window : int
        Lookback window in periods
    method : str
        'roc' for rate of change, 'diff' for difference
        
    Returns
    -------
    pd.Series
        Momentum values
    """
    if method == 'roc':
        # Rate of change: (current - past) / past
        return (series - series.shift(window)) / series.shift(window).abs()
    elif method == 'diff':
        # Simple difference
        return series - series.shift(window)
    else:
        raise ValueError(f"Unknown momentum method: {method}")


def calculate_momentum_zscore(
    series: pd.Series,
    momentum_window: int,
    zscore_window: int = 20
) -> pd.Series:
    """
    Calculate z-score normalized momentum.
    
    Parameters
    ----------
    series : pd.Series
        Price or value series
    momentum_window : int
        Window for momentum calculation
    zscore_window : int
        Window for z-score normalization
        
    Returns
    -------
    pd.Series
        Z-score normalized momentum
    """
    # Calculate raw momentum
    momentum = calculate_momentum(series, momentum_window)
    
    # Calculate rolling mean and std
    rolling_mean = momentum.rolling(window=zscore_window).mean()
    rolling_std = momentum.rolling(window=zscore_window).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    
    # Calculate z-score
    zscore = (momentum - rolling_mean) / rolling_std
    
    return zscore


def calculate_weighted_momentum(
    series: pd.Series,
    config: MomentumConfig,
    normalize: bool = True
) -> pd.Series:
    """
    Calculate weighted momentum across multiple timeframes.
    
    Parameters
    ----------
    series : pd.Series
        Price or value series
    config : MomentumConfig
        Configuration with windows and weights
    normalize : bool
        Whether to z-score normalize each momentum
        
    Returns
    -------
    pd.Series
        Weighted momentum score
    """
    if len(config.lookback_windows) != len(config.weights):
        raise ValueError("Number of windows must match number of weights")
    
    momentum_components = []
    
    for window, weight in zip(config.lookback_windows, config.weights):
        if normalize:
            # Z-score normalized momentum
            mom = calculate_momentum_zscore(series, window, config.zscore_window)
        else:
            # Raw momentum
            mom = calculate_momentum(series, window)
        
        # Apply weight
        momentum_components.append(mom * weight)
    
    # Combine weighted components
    weighted_momentum = pd.concat(momentum_components, axis=1).sum(axis=1)
    
    return weighted_momentum


def calculate_momentum_divergence(
    price_series: pd.Series,
    indicator_series: pd.Series,
    window: int = 20,
    min_swing_pct: float = 0.02
) -> pd.DataFrame:
    """
    Detect momentum divergence between price and indicator.
    
    Parameters
    ----------
    price_series : pd.Series
        Price series
    indicator_series : pd.Series
        Indicator series (e.g., PCA factor)
    window : int
        Window for finding peaks/troughs
    min_swing_pct : float
        Minimum percentage move to qualify as swing
        
    Returns
    -------
    pd.DataFrame
        DataFrame with divergence signals
    """
    result = pd.DataFrame(index=price_series.index)
    
    # Find local peaks and troughs
    price_peaks = price_series.rolling(window=window, center=True).max() == price_series
    price_troughs = price_series.rolling(window=window, center=True).min() == price_series
    
    ind_peaks = indicator_series.rolling(window=window, center=True).max() == indicator_series
    ind_troughs = indicator_series.rolling(window=window, center=True).min() == indicator_series
    
    # Track previous peaks/troughs
    result['price_peak'] = price_peaks
    result['price_trough'] = price_troughs
    result['ind_peak'] = ind_peaks
    result['ind_trough'] = ind_troughs
    
    # Detect divergences
    result['bullish_divergence'] = False
    result['bearish_divergence'] = False
    
    # Logic for divergence detection would go here
    # This is a placeholder for the complex logic
    
    return result


def calculate_pca_momentum(
    pca_factors: pd.DataFrame,
    config: MomentumConfig,
    factor_names: List[str] = None
) -> pd.DataFrame:
    """
    Calculate momentum for PCA factors.
    
    Parameters
    ----------
    pca_factors : pd.DataFrame
        DataFrame with PCA factor values
    config : MomentumConfig
        Momentum configuration
    factor_names : List[str]
        Names of factors to calculate momentum for
        
    Returns
    -------
    pd.DataFrame
        Momentum scores for each factor
    """
    if factor_names is None:
        factor_names = pca_factors.columns.tolist()
    
    result = pd.DataFrame(index=pca_factors.index)
    
    for factor in factor_names:
        if factor in pca_factors.columns:
            # Calculate weighted momentum for this factor
            factor_momentum = calculate_weighted_momentum(
                pca_factors[factor],
                config,
                normalize=True
            )
            result[f'{factor}_momentum'] = factor_momentum
    
    return result


def calculate_cross_sectional_momentum(
    data: pd.DataFrame,
    window: int = 20,
    rank_normalize: bool = True
) -> pd.DataFrame:
    """
    Calculate cross-sectional momentum across multiple series.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with multiple series (e.g., different contracts)
    window : int
        Momentum window
    rank_normalize : bool
        Whether to rank normalize across series
        
    Returns
    -------
    pd.DataFrame
        Cross-sectional momentum scores
    """
    # Calculate momentum for each series
    momentum_df = pd.DataFrame(index=data.index)
    
    for col in data.columns:
        momentum_df[col] = calculate_momentum(data[col], window)
    
    if rank_normalize:
        # Rank normalize across columns
        momentum_df = momentum_df.rank(axis=1, pct=True) * 2 - 1  # Scale to [-1, 1]
    
    return momentum_df


def momentum_signal_strength(
    momentum: pd.Series,
    threshold: float = 0.0,
    lookback: int = 20
) -> pd.Series:
    """
    Calculate momentum signal strength with persistence.
    
    Parameters
    ----------
    momentum : pd.Series
        Raw momentum values
    threshold : float
        Threshold for signal activation
    lookback : int
        Lookback for persistence calculation
        
    Returns
    -------
    pd.Series
        Signal strength [0, 1]
    """
    # Binary signal
    signal = (momentum.abs() > threshold).astype(float)
    
    # Add persistence component
    persistence = signal.rolling(window=lookback).mean()
    
    # Combine magnitude and persistence
    magnitude = momentum.abs() / (momentum.abs().rolling(window=lookback).max() + 1e-10)
    strength = (magnitude + persistence) / 2
    
    # Apply direction
    strength = strength * np.sign(momentum)
    
    return strength


def calculate_momentum_quality(
    series: pd.Series,
    momentum: pd.Series,
    window: int = 60
) -> pd.Series:
    """
    Calculate momentum quality score based on smoothness and consistency.
    
    Parameters
    ----------
    series : pd.Series
        Original price series
    momentum : pd.Series
        Momentum values
    window : int
        Window for quality metrics
        
    Returns
    -------
    pd.Series
        Quality score [0, 1]
    """
    # Calculate noise ratio
    returns = series.pct_change()
    signal_var = momentum.rolling(window).var()
    noise_var = returns.rolling(window).var()
    signal_to_noise = signal_var / (noise_var + 1e-10)
    
    # Calculate directional consistency
    momentum_sign = np.sign(momentum)
    consistency = momentum_sign.rolling(window).mean().abs()
    
    # Combine metrics
    quality = (signal_to_noise.clip(0, 1) + consistency) / 2
    
    return quality