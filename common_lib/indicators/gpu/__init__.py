"""
GPU-accelerated technical indicators.

This module provides GPU-accelerated versions of common technical indicators
using CuPy for massive parallel computation. All functions follow strict
data integrity principles:

1. NO synthetic data generation - all inputs must be real market data
2. Data validation before GPU processing - no NaN/infinite values
3. Memory-efficient batch processing with automatic chunking
4. Automatic fallback to CPU if GPU unavailable

All GPU functions have CPU equivalents in parent modules for compatibility.
"""

import warnings
from typing import Optional

# Check GPU availability
try:
    import cupy as cp
    # Check if GPU is actually available before setting memory limits
    try:
        cp.cuda.runtime.getDeviceCount()
        GPU_AVAILABLE = True
        # Set memory pool limits to prevent OOM
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(size=8 * 1024**3)  # 8GB limit
    except cp.cuda.runtime.CUDARuntimeError:
        # CuPy installed but no GPU available
        GPU_AVAILABLE = False
        warnings.warn(
            "CuPy installed but no CUDA device found. GPU acceleration unavailable. "
            "Using CPU fallback.", RuntimeWarning
        )
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    warnings.warn(
        "CuPy not installed. GPU acceleration unavailable. "
        "Install with: pip install cupy-cuda12x"
    )

from .technical_gpu import (
    atr_gpu,
    bollinger_bands_gpu,
    rsi_gpu,
    ema_gpu,
    sma_gpu,
    calculate_returns_gpu
)

from .volatility_gpu import (
    ewma_volatility_gpu,
    rolling_std_gpu,
    parkinson_volatility_gpu
)

from .batch_processor import (
    GPUBatchProcessor,
    validate_data_for_gpu,
    auto_chunk_data
)

__all__ = [
    'GPU_AVAILABLE',
    # Technical indicators
    'atr_gpu',
    'bollinger_bands_gpu', 
    'rsi_gpu',
    'ema_gpu',
    'sma_gpu',
    'calculate_returns_gpu',
    # Volatility indicators
    'ewma_volatility_gpu',
    'rolling_std_gpu',
    'parkinson_volatility_gpu',
    # Utilities
    'GPUBatchProcessor',
    'validate_data_for_gpu',
    'auto_chunk_data'
]