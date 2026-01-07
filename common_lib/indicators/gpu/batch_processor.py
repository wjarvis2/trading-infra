"""
GPU batch processing utilities for memory-efficient computation.

Handles automatic chunking, memory management, and data validation
to ensure efficient GPU utilization without OOM errors.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
import warnings

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


def validate_data_for_gpu(
    data: np.ndarray,
    max_nan_ratio: float = 0.1,
    check_synthetic: bool = True
) -> Tuple[bool, str]:
    """
    Comprehensive data validation before GPU processing.
    
    CRITICAL: Enforces data integrity requirements.
    
    Parameters
    ----------
    data : np.ndarray
        Data to validate
    max_nan_ratio : float
        Maximum allowed ratio of NaN values
    check_synthetic : bool
        Whether to check for synthetic data patterns
        
    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    # Check for NaN values
    nan_ratio = np.isnan(data).sum() / data.size
    if nan_ratio > max_nan_ratio:
        return False, f"Too many NaN values: {nan_ratio:.1%} > {max_nan_ratio:.1%}"
    
    # Check for infinite values
    if np.any(np.isinf(data)):
        return False, "Data contains infinite values"
    
    # Check minimum size
    if data.size < 100:
        return False, f"Insufficient data: {data.size} points"
    
    if check_synthetic:
        # Check for synthetic patterns
        flat_data = data.flatten()
        valid_data = flat_data[~np.isnan(flat_data)]
        
        if len(valid_data) > 0:
            # Check if data is constant (synthetic)
            if np.std(valid_data) < 1e-10:
                return False, "Data appears constant - likely synthetic"
            
            # Check for perfectly linear patterns (synthetic)
            if len(valid_data) > 10:
                diffs = np.diff(valid_data[:100])  # Check first 100 points
                if np.std(diffs) < 1e-10:
                    return False, "Data shows perfectly linear pattern - likely synthetic"
            
            # Check for repeating patterns (synthetic)
            if len(valid_data) > 20:
                pattern_len = 10
                pattern = valid_data[:pattern_len]
                next_pattern = valid_data[pattern_len:2*pattern_len]
                if np.allclose(pattern, next_pattern, rtol=1e-10):
                    return False, "Data shows repeating pattern - likely synthetic"
    
    return True, ""


def auto_chunk_data(
    n_params: int,
    n_timestamps: int,
    n_indicators: int,
    available_memory_gb: Optional[float] = None,
    dtype_bytes: int = 8
) -> Dict[str, Any]:
    """
    Automatically determine optimal chunking strategy based on memory constraints.
    
    Parameters
    ----------
    n_params : int
        Number of parameter combinations
    n_timestamps : int
        Number of time points
    n_indicators : int
        Number of indicators to calculate
    available_memory_gb : float, optional
        Available GPU memory in GB (auto-detected if None)
    dtype_bytes : int
        Bytes per element (8 for float64, 4 for float32)
        
    Returns
    -------
    dict
        Chunking strategy with keys:
        - strategy: 'no_chunking', 'time_chunking', 'param_chunking', 'hybrid'
        - param_chunks: Number of parameter chunks
        - time_chunks: Number of time chunks
        - chunk_size: Size of each chunk
    """
    if not GPU_AVAILABLE:
        return {
            'strategy': 'cpu_fallback',
            'param_chunks': 1,
            'time_chunks': 1,
            'chunk_size': n_params
        }
    
    # Auto-detect GPU memory if not provided
    if available_memory_gb is None:
        meminfo = cp.cuda.Device().mem_info
        available_memory_gb = meminfo[0] / 1024**3  # Free memory
        available_memory_gb *= 0.8  # Use only 80% to be safe
    
    available_bytes = available_memory_gb * 1024**3
    
    # Calculate memory needed for full computation
    bytes_per_indicator = n_params * n_timestamps * dtype_bytes
    total_bytes_needed = bytes_per_indicator * n_indicators
    
    # Add overhead for intermediate calculations (2x safety factor)
    total_bytes_needed *= 2
    
    if total_bytes_needed <= available_bytes:
        # Everything fits!
        return {
            'strategy': 'no_chunking',
            'param_chunks': 1,
            'time_chunks': 1,
            'chunk_size': n_params
        }
    
    # Prefer time chunking first (maintains parameter relationships)
    time_chunks_needed = int(np.ceil(total_bytes_needed / available_bytes))
    
    if time_chunks_needed <= 12:  # Monthly chunks are reasonable
        time_chunk_size = n_timestamps // time_chunks_needed
        return {
            'strategy': 'time_chunking',
            'param_chunks': 1,
            'time_chunks': time_chunks_needed,
            'chunk_size': time_chunk_size
        }
    
    # Need parameter chunking
    param_chunks_needed = int(np.ceil(np.sqrt(total_bytes_needed / available_bytes)))
    
    if param_chunks_needed * n_timestamps * n_indicators * dtype_bytes <= available_bytes:
        param_chunk_size = n_params // param_chunks_needed
        return {
            'strategy': 'param_chunking',
            'param_chunks': param_chunks_needed,
            'time_chunks': 1,
            'chunk_size': param_chunk_size
        }
    
    # Need hybrid chunking
    param_chunks = int(np.ceil(np.sqrt(total_bytes_needed / available_bytes)))
    time_chunks = int(np.ceil(total_bytes_needed / (available_bytes * param_chunks)))
    
    return {
        'strategy': 'hybrid_chunking',
        'param_chunks': param_chunks,
        'time_chunks': time_chunks,
        'chunk_size': n_params // param_chunks
    }


class GPUBatchProcessor:
    """
    Manages batch processing of indicators on GPU with automatic memory management.
    
    Ensures data integrity and prevents OOM errors through intelligent chunking.
    """
    
    def __init__(
        self,
        max_memory_gb: Optional[float] = None,
        validate_data: bool = True,
        dtype: type = np.float64
    ):
        """
        Initialize GPU batch processor.
        
        Parameters
        ----------
        max_memory_gb : float, optional
            Maximum GPU memory to use (auto-detected if None)
        validate_data : bool
            Whether to validate input data for integrity
        dtype : type
            Data type for calculations (float64 or float32)
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU not available. Install cupy.")
        
        self.validate_data = validate_data
        self.dtype = dtype
        self.dtype_bytes = 8 if dtype == np.float64 else 4
        
        # Set memory limits
        if max_memory_gb is None:
            meminfo = cp.cuda.Device().mem_info
            max_memory_gb = meminfo[1] / 1024**3  # Total memory
            max_memory_gb *= 0.8  # Use only 80%
        
        self.max_memory_gb = max_memory_gb
        
        # Configure memory pool
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(size=int(max_memory_gb * 1024**3))
    
    def process_indicators_batch(
        self,
        data: Dict[str, np.ndarray],
        param_grid: List[Dict[str, Any]],
        indicator_funcs: Dict[str, Callable],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Process multiple indicators for multiple parameter sets.
        
        Parameters
        ----------
        data : dict
            Input data arrays (e.g., {'high': ..., 'low': ..., 'close': ...})
        param_grid : list
            List of parameter dictionaries
        indicator_funcs : dict
            Mapping of indicator names to GPU functions
        **kwargs
            Additional arguments for indicators
            
        Returns
        -------
        dict
            Results with shape {indicator_name: array[n_params, n_timestamps]}
        """
        # Validate all input data
        if self.validate_data:
            for name, array in data.items():
                is_valid, error = validate_data_for_gpu(array)
                if not is_valid:
                    raise ValueError(f"Data validation failed for {name}: {error}")
        
        # Determine chunking strategy
        n_params = len(param_grid)
        n_timestamps = len(next(iter(data.values())))
        n_indicators = len(indicator_funcs)
        
        chunk_strategy = auto_chunk_data(
            n_params, n_timestamps, n_indicators,
            self.max_memory_gb, self.dtype_bytes
        )
        
        print(f"GPU Batch Processing Strategy: {chunk_strategy['strategy']}")
        print(f"  Parameter chunks: {chunk_strategy['param_chunks']}")
        print(f"  Time chunks: {chunk_strategy['time_chunks']}")
        
        # Process based on strategy
        if chunk_strategy['strategy'] == 'no_chunking':
            return self._process_all_gpu(data, param_grid, indicator_funcs, **kwargs)
        elif chunk_strategy['strategy'] == 'time_chunking':
            return self._process_time_chunks(
                data, param_grid, indicator_funcs,
                chunk_strategy['time_chunks'], **kwargs
            )
        elif chunk_strategy['strategy'] == 'param_chunking':
            return self._process_param_chunks(
                data, param_grid, indicator_funcs,
                chunk_strategy['param_chunks'], **kwargs
            )
        else:  # hybrid_chunking
            return self._process_hybrid_chunks(
                data, param_grid, indicator_funcs,
                chunk_strategy['param_chunks'],
                chunk_strategy['time_chunks'], **kwargs
            )
    
    def _process_all_gpu(
        self,
        data: Dict[str, np.ndarray],
        param_grid: List[Dict[str, Any]],
        indicator_funcs: Dict[str, Callable],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Process all data at once on GPU."""
        results = {}
        
        # Transfer data to GPU once
        gpu_data = {k: cp.asarray(v, dtype=self.dtype) for k, v in data.items()}
        
        # Calculate each indicator
        for ind_name, ind_func in indicator_funcs.items():
            n_params = len(param_grid)
            n_timestamps = len(next(iter(data.values())))
            ind_results = np.zeros((n_params, n_timestamps))
            
            for i, params in enumerate(param_grid):
                # Call indicator function with specific parameters
                result = ind_func(gpu_data, **params, **kwargs)
                ind_results[i] = cp.asnumpy(result)
            
            results[ind_name] = ind_results
        
        # Clear GPU memory
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        
        return results
    
    def _process_time_chunks(
        self,
        data: Dict[str, np.ndarray],
        param_grid: List[Dict[str, Any]],
        indicator_funcs: Dict[str, Callable],
        n_chunks: int,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Process data in time chunks to manage memory."""
        n_timestamps = len(next(iter(data.values())))
        chunk_size = n_timestamps // n_chunks
        
        # Initialize results
        n_params = len(param_grid)
        results = {
            ind_name: np.zeros((n_params, n_timestamps))
            for ind_name in indicator_funcs
        }
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = n_timestamps if chunk_idx == n_chunks - 1 else (chunk_idx + 1) * chunk_size
            
            # Slice data for this time chunk
            chunk_data = {
                k: v[..., start_idx:end_idx] for k, v in data.items()
            }
            
            # Process chunk
            chunk_results = self._process_all_gpu(
                chunk_data, param_grid, indicator_funcs, **kwargs
            )
            
            # Store results
            for ind_name, ind_result in chunk_results.items():
                results[ind_name][:, start_idx:end_idx] = ind_result
        
        return results
    
    def _process_param_chunks(
        self,
        data: Dict[str, np.ndarray],
        param_grid: List[Dict[str, Any]],
        indicator_funcs: Dict[str, Callable],
        n_chunks: int,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Process parameters in chunks to manage memory."""
        n_params = len(param_grid)
        chunk_size = n_params // n_chunks
        n_timestamps = len(next(iter(data.values())))
        
        # Initialize results
        results = {
            ind_name: np.zeros((n_params, n_timestamps))
            for ind_name in indicator_funcs
        }
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = n_params if chunk_idx == n_chunks - 1 else (chunk_idx + 1) * chunk_size
            
            # Get parameter chunk
            param_chunk = param_grid[start_idx:end_idx]
            
            # Process chunk
            chunk_results = self._process_all_gpu(
                data, param_chunk, indicator_funcs, **kwargs
            )
            
            # Store results
            for ind_name, ind_result in chunk_results.items():
                results[ind_name][start_idx:end_idx] = ind_result
        
        return results
    
    def _process_hybrid_chunks(
        self,
        data: Dict[str, np.ndarray],
        param_grid: List[Dict[str, Any]],
        indicator_funcs: Dict[str, Callable],
        n_param_chunks: int,
        n_time_chunks: int,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Process with both parameter and time chunking."""
        # Combine strategies
        n_params = len(param_grid)
        param_chunk_size = n_params // n_param_chunks
        n_timestamps = len(next(iter(data.values())))
        
        # Initialize results
        results = {
            ind_name: np.zeros((n_params, n_timestamps))
            for ind_name in indicator_funcs
        }
        
        for param_chunk_idx in range(n_param_chunks):
            param_start = param_chunk_idx * param_chunk_size
            param_end = n_params if param_chunk_idx == n_param_chunks - 1 else (param_chunk_idx + 1) * param_chunk_size
            param_chunk = param_grid[param_start:param_end]
            
            # Process this parameter chunk with time chunking
            chunk_results = self._process_time_chunks(
                data, param_chunk, indicator_funcs, n_time_chunks, **kwargs
            )
            
            # Store results
            for ind_name, ind_result in chunk_results.items():
                results[ind_name][param_start:param_end] = ind_result
        
        return results