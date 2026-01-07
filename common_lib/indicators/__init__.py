"""Technical indicators and analysis tools.

This module provides both CPU and GPU implementations of technical indicators.
GPU versions are automatically used when available and data size warrants it.
"""

import numpy as np
import warnings
from typing import Union, Optional

# Import CPU versions
from .technical import (
    atr as atr_cpu,
    bollinger_bands as bollinger_bands_cpu,
    rsi as rsi_cpu,
    vwap_session,
    rolling_correlation
)

from .session import (
    calculate_session_metrics,
    identify_trading_sessions
)

from .microstructure import (
    detect_microstructure_breaks,
    calculate_curve_momentum
)

from .momentum import (
    MomentumConfig,
    calculate_momentum,
    calculate_momentum_zscore,
    calculate_weighted_momentum,
    calculate_momentum_divergence,
    calculate_pca_momentum,
    calculate_cross_sectional_momentum,
    momentum_signal_strength,
    calculate_momentum_quality
)

from .seasonality import (
    SeasonalPattern,
    calculate_pattern_strength,
    calculate_seasonal_adjustment,
    calculate_multi_pattern_adjustment,
    extract_seasonal_pattern,
    calculate_seasonal_decomposition,
    identify_seasonal_regimes,
    calculate_seasonal_zscore,
    CRUDE_OIL_PATTERNS,
    RBOB_PATTERNS
)

from .volatility import (
    ewma_volatility as ewma_volatility_cpu,
    rolling_volatility as rolling_volatility_cpu,
    parkinson_volatility as parkinson_volatility_cpu
)

# Try to import GPU versions
GPU_AVAILABLE = False
try:
    from .gpu import (
        GPU_AVAILABLE as _GPU_AVAILABLE,
        atr_gpu,
        bollinger_bands_gpu,
        rsi_gpu,
        ewma_volatility_gpu,
        rolling_std_gpu,
        parkinson_volatility_gpu
    )
    GPU_AVAILABLE = _GPU_AVAILABLE
except ImportError:
    # GPU not available, will use CPU versions
    pass


# Auto-routing functions that choose CPU or GPU based on availability and data size
def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
        window: int = 20, use_gpu: Optional[Union[bool, str]] = 'auto') -> np.ndarray:
    """
    Calculate Average True Range with automatic CPU/GPU selection.
    
    Parameters
    ----------
    highs, lows, closes : np.ndarray
        OHLC price data (MUST be real market data)
    window : int
        ATR period
    use_gpu : bool or 'auto'
        Force GPU (True), CPU (False), or auto-select ('auto')
    """
    if use_gpu == 'auto':
        # Use GPU if available and data is large enough
        use_gpu = GPU_AVAILABLE and len(highs) > 10000
    
    if use_gpu and GPU_AVAILABLE:
        try:
            return atr_gpu(highs, lows, closes, window)
        except Exception as e:
            warnings.warn(f"GPU ATR failed, falling back to CPU: {e}")
            return atr_cpu(highs, lows, closes, window)
    else:
        return atr_cpu(highs, lows, closes, window)


def bollinger_bands(prices: np.ndarray, window: int = 20, num_std: float = 2.0,
                   use_gpu: Optional[Union[bool, str]] = 'auto'):
    """
    Calculate Bollinger Bands with automatic CPU/GPU selection.
    """
    if use_gpu == 'auto':
        use_gpu = GPU_AVAILABLE and len(prices) > 10000
    
    if use_gpu and GPU_AVAILABLE:
        try:
            return bollinger_bands_gpu(prices, window, num_std)
        except Exception as e:
            warnings.warn(f"GPU Bollinger Bands failed, falling back to CPU: {e}")
            return bollinger_bands_cpu(prices, window, num_std)
    else:
        return bollinger_bands_cpu(prices, window, num_std)


def rsi(prices: np.ndarray, window: int = 14,
        use_gpu: Optional[Union[bool, str]] = 'auto') -> np.ndarray:
    """
    Calculate RSI with automatic CPU/GPU selection.
    """
    if use_gpu == 'auto':
        use_gpu = GPU_AVAILABLE and len(prices) > 10000
    
    if use_gpu and GPU_AVAILABLE:
        try:
            return rsi_gpu(prices, window)
        except Exception as e:
            warnings.warn(f"GPU RSI failed, falling back to CPU: {e}")
            return rsi_cpu(prices, window)
    else:
        return rsi_cpu(prices, window)


def ewma_volatility(returns: np.ndarray, span: int = 20, min_periods: int = 15,
                   annualize: bool = True, periods_per_year: int = 252,
                   use_gpu: Optional[Union[bool, str]] = 'auto') -> np.ndarray:
    """
    Calculate EWMA volatility with automatic CPU/GPU selection.
    """
    if use_gpu == 'auto':
        use_gpu = GPU_AVAILABLE and len(returns) > 10000
    
    if use_gpu and GPU_AVAILABLE:
        try:
            return ewma_volatility_gpu(returns, span, min_periods, 
                                      annualize, periods_per_year)
        except Exception as e:
            warnings.warn(f"GPU EWMA volatility failed, falling back to CPU: {e}")
            return ewma_volatility_cpu(returns, span, min_periods,
                                      annualize, periods_per_year)
    else:
        return ewma_volatility_cpu(returns, span, min_periods,
                                  annualize, periods_per_year)

__all__ = [
    # Technical indicators
    'atr',
    'bollinger_bands',
    'rsi',
    'vwap_session',
    'rolling_correlation',
    
    # Session analysis
    'calculate_session_metrics',
    'identify_trading_sessions',
    
    # Microstructure
    'detect_microstructure_breaks',
    'calculate_curve_momentum',
    
    # Momentum indicators
    'MomentumConfig',
    'calculate_momentum',
    'calculate_momentum_zscore',
    'calculate_weighted_momentum',
    'calculate_momentum_divergence',
    'calculate_pca_momentum',
    'calculate_cross_sectional_momentum',
    'momentum_signal_strength',
    'calculate_momentum_quality',
    
    # Seasonality indicators
    'SeasonalPattern',
    'calculate_pattern_strength',
    'calculate_seasonal_adjustment',
    'calculate_multi_pattern_adjustment',
    'extract_seasonal_pattern',
    'calculate_seasonal_decomposition',
    'identify_seasonal_regimes',
    'calculate_seasonal_zscore',
    'CRUDE_OIL_PATTERNS',
    'RBOB_PATTERNS'
]