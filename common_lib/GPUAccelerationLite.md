# GPU Acceleration Lite - Implementation Guide

## Overview

This document outlines a pragmatic "GPU-Lite" approach to accelerating strategy backtesting by GPU-accelerating only the computational bottlenecks while keeping complex business logic on CPU. This approach provides 80% of the performance benefit for 20% of the implementation effort compared to full GPU vectorization.

## Executive Summary

- **Expected Speedup**: 8-16x for large parameter grids (>1000 combinations)
- **Development Effort**: 2-3 weeks
- **Break-even Point**: ~100 parameter combinations
- **Key Insight**: Accelerate math-heavy operations (indicators, signals) on GPU while keeping stateful logic (positions, risk) on CPU

## Architecture Pattern

```python
class GPULiteBacktester:
    """Hybrid CPU-GPU backtester that accelerates bottlenecks."""
    
    def run_batch(self, param_grid):
        # 1. Pre-calculate all indicators on GPU (FAST)
        all_indicators = self.gpu_calculate_indicators_batch(data, param_grid)
        
        # 2. Generate all signals on GPU (FAST)
        all_signals = self.gpu_generate_signals_batch(all_indicators, param_grid)
        
        # 3. Loop through results on CPU (slower but manageable)
        results = []
        for i, params in enumerate(param_grid):
            # Pull specific signals from GPU
            signals = all_signals[i].get()  # Transfer to CPU
            
            # Run normal CPU backtesting logic
            result = self.cpu_backtest_with_signals(signals, params)
            results.append(result)
        
        return results
```

## What Gets GPU Accelerated

### 1. Indicator Calculations (50-100x speedup)

**Current Implementation:**
```python
# Sequential calculation for each parameter set
for param_set in parameter_grid:
    volatility = calculate_ewma(prices, span=param_set['span'])
    atr = calculate_atr(high, low, close, period=param_set['atr_period'])
    rsi = calculate_rsi(prices, period=param_set['rsi_period'])
```

**GPU-Lite Implementation:**
```python
# Vectorized calculation for all parameter sets at once
import cupy as cp

def gpu_calculate_indicators_batch(prices_cpu, param_grid):
    # Transfer price data to GPU once
    prices_gpu = cp.asarray(prices_cpu)
    
    # Extract all parameter values
    spans = cp.array([p['ewma_span'] for p in param_grid])
    atr_periods = cp.array([p['atr_period'] for p in param_grid])
    
    # Calculate all indicators in parallel
    # Shape: (n_params, n_timestamps)
    all_volatilities = gpu_ewma_batch(prices_gpu, spans)
    all_atrs = gpu_atr_batch(prices_gpu, atr_periods)
    
    return {
        'volatility': all_volatilities,
        'atr': all_atrs
    }
```

**Specific Indicators to Accelerate:**
- EWMA volatility calculations
- ATR (Average True Range)
- RSI (Relative Strength Index)
- Rolling correlations
- Z-score calculations
- Moving averages (SMA, EMA)
- Bollinger Bands
- Rolling min/max

### 2. Signal Generation (20-50x speedup)

**Current Implementation:**
```python
# Complex conditional logic
if spread > seasonal_fair_value + threshold:
    if volatility < vol_threshold and regime != 'EXTREME':
        signal = -1
```

**GPU-Lite Implementation:**
```python
# Vectorized boolean operations
def gpu_generate_signals_batch(indicators, thresholds):
    # Shape: (n_params, n_timestamps)
    spread_signals = cp.where(
        (indicators['spreads'] > indicators['fair_values'] + thresholds[:, None]) &
        (indicators['volatility'] < vol_thresholds[:, None]) &
        (indicators['regimes'] != EXTREME_REGIME),
        -1, 0
    )
    return spread_signals
```

### 3. Spread/Curve Calculations (100x speedup)

**Current Implementation:**
```python
# Calculate spreads one by one
cl1_cl2 = prices['CL1'] - prices['CL2']
cl2_cl3 = prices['CL2'] - prices['CL3']
```

**GPU-Lite Implementation:**
```python
# Batch calculate all spreads
def gpu_calculate_spreads_batch(contract_prices):
    # contract_prices shape: (n_contracts, n_timestamps)
    # Output shape: (n_spreads, n_timestamps)
    
    # Pre-define spread pairs
    spread_pairs = [(0,1), (1,2), (2,3), (0,2), (0,3)]  # CL1-CL2, CL2-CL3, etc
    
    spreads = cp.empty((len(spread_pairs), contract_prices.shape[1]))
    for i, (front, back) in enumerate(spread_pairs):
        spreads[i] = contract_prices[front] - contract_prices[back]
    
    return spreads
```

### 4. Seasonal Adjustments (30x speedup)

**Current Implementation:**
```python
# Calculate seasonal patterns for each date
for date in dates:
    pattern = calculate_seasonal_pattern(date, lookback_years=5)
    adjustment = calculate_pattern_adjustment(pattern, confidence)
```

**GPU-Lite Implementation:**
```python
def gpu_seasonal_adjustments_batch(dates, all_lookback_years, all_confidences):
    # Pre-calculate all date features
    months = cp.array([d.month for d in dates])
    days_from_month_start = cp.array([d.day for d in dates])
    
    # Vectorized pattern weights
    # Shape: (n_params, n_dates, n_patterns)
    pattern_weights = gpu_calculate_pattern_weights(
        months, days_from_month_start, all_lookback_years
    )
    
    # Apply confidences
    adjustments = pattern_weights * all_confidences[:, None, None]
    
    return adjustments.sum(axis=2)  # Sum across patterns
```

## What Stays on CPU

### 1. Position Management
- Complex state tracking
- Multi-leg position handling
- Entry/exit logic with memory
- Position-specific metadata

### 2. Risk Management
- Portfolio-level constraints
- Dynamic position sizing based on account equity
- Correlation-based adjustments
- Stop loss and take profit logic

### 3. Trade Execution
- Order management with pending orders
- Fill simulation with realistic slippage
- Partial fill handling
- Commission calculations

### 4. Performance Analytics
- Trade-by-trade P&L attribution
- Drawdown calculations
- Rolling performance metrics
- Complex derived metrics

## Memory Management Strategy

### Memory Requirements Analysis

**Data Footprint for 1 Year of 15s Bars:**
- 252 trading days × 23 hours × 240 bars/hour = 1,391,040 bars/year
- 8 contracts × 1.4M bars × 8 bytes (float64) = 89.6 MB (raw data)
- Even with 10 years: ~900 MB (easily fits in GPU)

**The Real Memory Challenge - Intermediate Arrays:**
```
10,000 parameter sets × 1.4M timestamps × 8 bytes per indicator
= 112 GB per indicator (!!)
```

This is where naive implementations fail. Storing all indicators for all parameters for all timestamps would require terabytes of GPU memory.

### Smart Chunking Solutions

#### 1. Time Chunking (Most Effective)
```python
def process_with_time_chunks(data, param_grid, chunk_months=1):
    """Process data in monthly chunks to manage memory."""
    results = []
    
    # Split data by month
    for month_start, month_end in get_month_boundaries(data):
        month_data = data[month_start:month_end]  # ~116k bars
        
        # Memory usage per indicator: 
        # 10k params × 116k bars × 8 bytes = 9.3 GB (fits in 24GB GPU)
        gpu_signals = calculate_signals_gpu(month_data, param_grid)
        
        # Run backtests for this month
        month_results = cpu_backtest_month(gpu_signals, param_grid)
        results.append(month_results)
    
    # Aggregate across months
    return aggregate_monthly_results(results)
```

#### 2. Parameter Chunking
```python
def process_with_param_chunks(data, param_grid, chunk_size=1000):
    """Process parameters in chunks when dataset is large."""
    results = []
    
    for param_chunk in chunks(param_grid, chunk_size):
        # Memory usage per indicator:
        # 1k params × 1.4M bars × 8 bytes = 11.2 GB per indicator
        gpu_results = calculate_indicators_gpu(data, param_chunk)
        cpu_results = backtest_chunk(gpu_results, param_chunk)
        results.extend(cpu_results)
        
        # Free GPU memory
        cp.get_default_memory_pool().free_all_blocks()
    
    return results
```

#### 3. Indicator Staging (Sequential Processing)
```python
def staged_indicator_calculation(data, params):
    """Calculate indicators one at a time to manage memory."""
    # Process indicators sequentially, extract signals, free memory
    
    # Stage 1: Volatility
    volatility = calculate_volatility_gpu(data, params)  # 112 GB
    vol_signals = extract_vol_signals(volatility)        # 10 GB compressed
    del volatility  # Free 112 GB
    
    # Stage 2: ATR
    atr = calculate_atr_gpu(data, params)               # 112 GB
    atr_signals = extract_atr_signals(atr)              # 10 GB compressed
    del atr  # Free 112 GB
    
    # Stage 3: Combine signals (much smaller)
    final_signals = combine_signals(vol_signals, atr_signals)  # 10 GB
    
    return final_signals
```

#### 4. Mixed Precision Optimization
```python
def optimize_precision(data):
    """Use float32 where possible to halve memory usage."""
    # Most technical indicators don't need float64 precision
    prices_f32 = data['close'].astype(cp.float32)
    
    # Indicators that work well with float32:
    # - Moving averages, RSI, ATR, Bollinger Bands
    # - Spread calculations, z-scores
    
    # Keep float64 for:
    # - Cumulative returns, portfolio values
    # - Final P&L calculations
    
    return prices_f32
```

### Optimal Memory Budget

**For Different GPU Configurations:**

| GPU Model | Memory | Safe Working Set | Optimal Strategy |
|-----------|--------|------------------|------------------|
| RTX 3070 | 8 GB | 6 GB | 500 params × 1 month chunks |
| RTX 4080 | 16 GB | 13 GB | 1000 params × 2 month chunks |
| RTX 4090 | 24 GB | 20 GB | 2000 params × 3 month chunks |
| A100 | 40 GB | 35 GB | 5000 params × 6 month chunks |

### Advanced Memory Manager Implementation

```python
class GPUMemoryManager:
    def __init__(self, max_gpu_memory_gb=None):
        if max_gpu_memory_gb is None:
            # Auto-detect GPU memory
            max_gpu_memory_gb = cp.cuda.Device().mem_info[1] / 1024**3
        
        self.max_memory = max_gpu_memory_gb * 1024**3
        self.safety_factor = 0.8  # Use only 80% of available memory
        self.indicator_size_bytes = 8  # float64
        
    def calculate_optimal_chunks(self, n_params, n_timestamps, n_indicators):
        """Calculate optimal chunking strategy based on available memory."""
        # Memory per full indicator array
        bytes_per_indicator = n_params * n_timestamps * self.indicator_size_bytes
        
        # Available memory for indicators
        available_bytes = self.max_memory * self.safety_factor
        
        # Can we fit everything?
        total_bytes_needed = bytes_per_indicator * n_indicators
        
        if total_bytes_needed <= available_bytes:
            # Everything fits!
            return {
                'strategy': 'no_chunking',
                'param_chunks': 1,
                'time_chunks': 1
            }
        
        # Need chunking - prefer time chunking first
        time_chunks = int(np.ceil(total_bytes_needed / available_bytes))
        
        if time_chunks <= 12:  # Monthly chunks are reasonable
            return {
                'strategy': 'time_chunking',
                'param_chunks': 1,
                'time_chunks': time_chunks
            }
        
        # Need both time and parameter chunking
        param_chunks = int(np.ceil(np.sqrt(total_bytes_needed / available_bytes)))
        time_chunks = int(np.ceil(total_bytes_needed / (available_bytes * param_chunks)))
        
        return {
            'strategy': 'hybrid_chunking',
            'param_chunks': param_chunks,
            'time_chunks': time_chunks
        }
    
    def process_with_auto_chunking(self, data, param_grid, indicators):
        """Automatically choose best chunking strategy."""
        n_params = len(param_grid)
        n_timestamps = len(data)
        n_indicators = len(indicators)
        
        # Determine optimal strategy
        chunk_strategy = self.calculate_optimal_chunks(
            n_params, n_timestamps, n_indicators
        )
        
        print(f"Using {chunk_strategy['strategy']} strategy:")
        print(f"  Parameter chunks: {chunk_strategy['param_chunks']}")
        print(f"  Time chunks: {chunk_strategy['time_chunks']}")
        
        # Execute based on strategy
        if chunk_strategy['strategy'] == 'no_chunking':
            return self._process_all_gpu(data, param_grid, indicators)
        elif chunk_strategy['strategy'] == 'time_chunking':
            return self._process_time_chunks(
                data, param_grid, indicators, 
                chunk_strategy['time_chunks']
            )
        else:  # hybrid_chunking
            return self._process_hybrid_chunks(
                data, param_grid, indicators,
                chunk_strategy['param_chunks'],
                chunk_strategy['time_chunks']
            )
```

### Memory Usage Examples

**Example 1: Small Parameter Grid (100 params, 1 year)**
- Memory needed: 100 × 1.4M × 8 bytes = 1.12 GB per indicator
- Strategy: No chunking needed on any modern GPU
- Processing time: ~30 seconds

**Example 2: Medium Parameter Grid (1,000 params, 1 year)**
- Memory needed: 1,000 × 1.4M × 8 bytes = 11.2 GB per indicator
- Strategy: Time chunking (2 chunks) on 16GB GPU
- Processing time: ~2 minutes

**Example 3: Large Parameter Grid (10,000 params, 1 year)**
- Memory needed: 10,000 × 1.4M × 8 bytes = 112 GB per indicator
- Strategy: Hybrid chunking (5 param chunks × 3 time chunks)
- Processing time: ~15 minutes

### Key Insights

1. **Time chunking is preferred** - Maintains parameter relationships within each chunk
2. **Monthly boundaries are natural** - Aligns with reporting periods
3. **Staged indicator processing** - Calculate expensive indicators first, extract signals
4. **Mixed precision saves 50%** - Use float32 for indicators, float64 for final calculations
5. **Auto-chunking is essential** - Let the system determine optimal strategy based on available memory

With smart chunking, even 100,000 parameter combinations can be processed on consumer GPUs by breaking the problem into manageable pieces while still achieving 8-16x speedups.

## Implementation Timeline

### Week 1: Foundation (5 days)
- **Day 1-2**: Set up CuPy environment, create GPU utility functions
- **Day 3-4**: Implement batch EWMA and basic rolling statistics
- **Day 5**: Create memory management framework and chunking logic

### Week 2: Core Acceleration (5 days)
- **Day 1-2**: Vectorize signal generation logic for all strategies
- **Day 3-4**: Implement seasonal adjustment batch calculations
- **Day 5**: Add spread/curve batch calculations

### Week 3: Integration & Testing (5 days)
- **Day 1-2**: Integrate with existing NotebookBacktester
- **Day 3-4**: Performance testing and optimization
- **Day 5**: Documentation and deployment guide

## Performance Expectations

### Benchmark: 10,000 Parameter Combinations, 1 Year of 15s Data

| Component | Current Time | GPU-Lite Time | Speedup |
|-----------|-------------|---------------|----------|
| Data Loading | 30s | 30s | 1x |
| Indicator Calc | 7200s (2hr) | 120s (2min) | 60x |
| Signal Generation | 600s (10min) | 30s | 20x |
| Backtest Execution | 600s (10min) | 600s (10min) | 1x |
| **Total** | **8430s (2.3hr)** | **780s (13min)** | **10.8x** |

### Scaling Characteristics

- 100 parameters: 2x speedup (CPU-GPU transfer overhead)
- 1,000 parameters: 8x speedup
- 10,000 parameters: 11x speedup
- 100,000 parameters: 15x speedup (memory chunking overhead)

## Technical Requirements

### Hardware
- NVIDIA GPU with 8GB+ memory (RTX 3070 or better)
- CUDA 11.0 or higher
- 32GB+ system RAM for large datasets

### Software
```bash
# Install CUDA-enabled packages
pip install cupy-cuda11x  # Replace 11x with your CUDA version
pip install numpy>=1.21   # CuPy compatibility

# Optional for profiling
pip install nvtx
pip install cupy-profiler
```

### Compatibility Considerations

1. **CuPy vs NumPy**: Most NumPy operations have CuPy equivalents, but not all
2. **Pandas Operations**: No direct GPU equivalent - need to convert to arrays
3. **Date/Time Operations**: Must convert to numeric for GPU processing
4. **String Operations**: Not supported on GPU - keep on CPU

## Code Examples

### Example 1: GPU Batch EWMA

```python
def gpu_ewma_batch(prices, spans):
    """
    Calculate EWMA for multiple spans in parallel.
    
    Args:
        prices: CuPy array of shape (n_timestamps,)
        spans: CuPy array of shape (n_params,)
    
    Returns:
        CuPy array of shape (n_params, n_timestamps)
    """
    n_params = len(spans)
    n_timestamps = len(prices)
    
    # Pre-calculate alphas for all spans
    alphas = 2.0 / (spans + 1.0)
    
    # Initialize output array
    ewmas = cp.zeros((n_params, n_timestamps))
    ewmas[:, 0] = prices[0]
    
    # Vectorized EWMA calculation
    for t in range(1, n_timestamps):
        ewmas[:, t] = alphas * prices[t] + (1 - alphas) * ewmas[:, t-1]
    
    return ewmas
```

### Example 2: GPU Signal Generation

```python
def gpu_spread_signals_batch(spreads, fair_values, thresholds, volatilities, vol_limits):
    """
    Generate spread trading signals for multiple parameter sets.
    
    All inputs are CuPy arrays of shape (n_params, n_timestamps)
    Returns signals of shape (n_params, n_timestamps)
    """
    # Calculate deviations
    deviations = spreads - fair_values
    normalized_deviations = deviations / volatilities
    
    # Generate signals based on thresholds
    long_signals = cp.where(
        (normalized_deviations < -thresholds) & 
        (volatilities < vol_limits),
        1, 0
    )
    
    short_signals = cp.where(
        (normalized_deviations > thresholds) & 
        (volatilities < vol_limits),
        -1, 0
    )
    
    return long_signals + short_signals
```

## Go/No-Go Decision Criteria

### Proceed with GPU-Lite if:
- ✓ Regularly testing >1,000 parameter combinations
- ✓ Indicator calculations take >50% of total backtest time
- ✓ Have CUDA-capable GPU with 8GB+ memory
- ✓ Team has basic Python/NumPy vectorization experience
- ✓ Can dedicate 2-3 weeks for implementation

### Skip GPU-Lite if:
- ✗ Usually testing <100 parameter combinations
- ✗ Continuous aggregates already provide sufficient speed
- ✗ Complex strategy logic that's hard to vectorize
- ✗ No GPU available or limited to CPU-only deployment
- ✗ Need results within next 2 weeks

## Risk Mitigation

1. **Fallback to CPU**: Implement automatic fallback if GPU unavailable
2. **Validation**: Run subset of parameters on both CPU and GPU to verify correctness
3. **Memory Monitoring**: Add automatic chunk size adjustment based on available memory
4. **Progressive Rollout**: Start with one strategy, expand after validation

## Conclusion

The GPU-Lite approach offers significant performance improvements for parameter optimization workflows without the complexity of full GPU vectorization. By focusing on the computational bottlenecks (indicators and signals) while keeping business logic on CPU, we can achieve 8-16x speedups with manageable development effort.

This hybrid approach is particularly well-suited for systematic trading strategies where:
- Mathematical calculations dominate runtime
- Business logic is complex and stateful
- Parameter grids are large (>1000 combinations)
- Development time is limited

For most use cases, GPU-Lite provides the optimal balance between performance gain and implementation complexity.