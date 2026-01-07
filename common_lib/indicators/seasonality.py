"""
Seasonal pattern analysis and calculation functions for energy commodities.

This module provides functions to identify, quantify, and extract seasonal patterns
from time series data, particularly for oil futures calendar spreads.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class SeasonalPattern:
    """Configuration for a seasonal trading pattern."""
    name: str
    months: List[int]
    direction: str  # 'TIGHTEN', 'WIDEN', 'MIXED'
    base_adjustment: float  # $/bbl adjustment at full strength
    confidence: float  # 0-1 reliability factor
    smooth_transitions: bool = True
    utilization_dependent: bool = False
    utilization_threshold: float = 0.90


def calculate_pattern_strength(date: datetime, pattern_months: List[int], 
                             smooth_transitions: bool = True) -> float:
    """
    Calculate seasonal pattern strength with optional smooth month-boundary transitions.
    
    Args:
        date: Current date
        pattern_months: List of months in the seasonal pattern (1-12)
        smooth_transitions: Whether to smooth transitions at pattern boundaries
        
    Returns:
        Strength factor (0-1) with smooth transitions if enabled
    """
    month = date.month
    day = date.day
    
    if month not in pattern_months:
        return 0.0
    
    if not smooth_transitions:
        return 1.0
    
    # Smooth transitions at pattern boundaries
    days_in_month = pd.Timestamp(date).days_in_month
    day_fraction = day / days_in_month
    
    # Handle wrapped patterns (e.g., Nov-Feb wraps around year)
    wrapped = pattern_months[-1] > pattern_months[0]
    
    if not wrapped:
        # Normal pattern
        if month == min(pattern_months):
            return day_fraction  # Ramp up
        elif month == max(pattern_months):
            return 1.0 - day_fraction  # Ramp down
    else:
        # Wrapped pattern (e.g., [11, 12, 1, 2])
        start_months = [m for m in pattern_months if m >= pattern_months[0]]
        end_months = [m for m in pattern_months if m <= pattern_months[-1]]
        
        if month == min(start_months):
            return day_fraction  # Ramp up at start
        elif month == max(end_months):
            return 1.0 - day_fraction  # Ramp down at end
    
    # Middle months: full strength
    return 1.0


def calculate_seasonal_adjustment(date: datetime, pattern: SeasonalPattern,
                                refinery_util: Optional[float] = None) -> float:
    """
    Calculate seasonal adjustment for a given date and pattern.
    
    Args:
        date: Current date
        pattern: SeasonalPattern object
        refinery_util: Current refinery utilization rate (0-1)
        
    Returns:
        Seasonal adjustment in $/bbl
    """
    # Calculate base seasonal strength
    strength = calculate_pattern_strength(date, pattern.months, pattern.smooth_transitions)
    
    if strength == 0:
        return 0.0
    
    # Handle utilization-dependent patterns
    if pattern.utilization_dependent:
        if refinery_util is None:
            # No data available - skip pattern
            return 0.0
        elif refinery_util > pattern.utilization_threshold:
            # High utilization - pattern doesn't apply
            return 0.0
    
    # Calculate final adjustment
    adjustment = pattern.base_adjustment * strength * pattern.confidence
    
    return adjustment


def calculate_multi_pattern_adjustment(date: datetime, patterns: Dict[str, SeasonalPattern],
                                     refinery_util: Optional[float] = None) -> Dict:
    """
    Calculate total seasonal adjustment from multiple patterns with metadata.
    
    Args:
        date: Current date
        patterns: Dictionary of pattern_name -> SeasonalPattern
        refinery_util: Current refinery utilization rate
        
    Returns:
        Dictionary with total adjustment and pattern details
    """
    total_adjustment = 0.0
    total_confidence = 0.0
    active_patterns = []
    
    for name, pattern in patterns.items():
        adjustment = calculate_seasonal_adjustment(date, pattern, refinery_util)
        strength = calculate_pattern_strength(date, pattern.months, pattern.smooth_transitions)
        
        if strength > 0:
            total_adjustment += adjustment
            total_confidence += pattern.confidence * strength
            
            active_patterns.append({
                'pattern': name,
                'adjustment': adjustment,
                'strength': strength,
                'direction': pattern.direction,
                'confidence': pattern.confidence
            })
    
    # Calculate average confidence-weighted strength
    avg_confidence = total_confidence / len(active_patterns) if active_patterns else 0
    
    return {
        'total_adjustment': total_adjustment,
        'active_patterns': active_patterns,
        'pattern_count': len(active_patterns),
        'avg_confidence': avg_confidence,
        'total_confidence': total_confidence
    }


def extract_seasonal_pattern(data: pd.Series, lookback_years: int = 5,
                           recent_year_weight: float = 0.6,
                           pattern_smoothing: int = 7) -> pd.Series:
    """
    Extract seasonal pattern from historical time series data.
    
    Args:
        data: Time series data with datetime index
        lookback_years: Number of years to analyze
        recent_year_weight: Weight for recent years (0-1)
        pattern_smoothing: Days for smoothing extracted pattern
        
    Returns:
        Series with average values by day of year (1-366)
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")
    
    # Filter to lookback period
    end_date = data.index.max()
    start_date = end_date - pd.DateOffset(years=lookback_years)
    filtered_data = data[data.index >= start_date]
    
    # Extract day of year
    df = pd.DataFrame({
        'value': filtered_data.values,
        'dayofyear': filtered_data.index.dayofyear,
        'year': filtered_data.index.year
    })
    
    # Calculate year weights (exponential decay)
    years = sorted(df['year'].unique())
    if len(years) > 1:
        year_range = years[-1] - years[0]
        weights = {}
        for year in years:
            age = years[-1] - year
            weight = recent_year_weight ** (age / year_range)
            weights[year] = weight
    else:
        weights = {years[0]: 1.0}
    
    # Calculate weighted average by day of year
    seasonal_pattern = {}
    for doy in range(1, 367):
        doy_data = df[df['dayofyear'] == doy]
        if len(doy_data) > 0:
            weighted_sum = 0
            weight_sum = 0
            for year, year_data in doy_data.groupby('year'):
                if year in weights:
                    weighted_sum += year_data['value'].mean() * weights[year]
                    weight_sum += weights[year]
            seasonal_pattern[doy] = weighted_sum / weight_sum if weight_sum > 0 else np.nan
    
    # Convert to series and smooth
    pattern_series = pd.Series(seasonal_pattern)
    if pattern_smoothing > 1:
        pattern_series = pattern_series.rolling(
            window=pattern_smoothing, center=True, min_periods=1
        ).mean()
    
    return pattern_series


def calculate_seasonal_decomposition(data: pd.Series, period: int = 365) -> Dict[str, pd.Series]:
    """
    Perform seasonal decomposition of time series.
    
    Args:
        data: Time series data
        period: Seasonal period (default 365 for daily data)
        
    Returns:
        Dictionary with 'trend', 'seasonal', 'residual' components
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Ensure we have enough data
    if len(data) < 2 * period:
        raise ValueError(f"Need at least {2 * period} observations for decomposition")
    
    # Perform decomposition
    decomposition = seasonal_decompose(data, model='additive', period=period)
    
    return {
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid
    }


def identify_seasonal_regimes(data: pd.Series, patterns: Dict[str, SeasonalPattern],
                            threshold: float = 0.7) -> pd.Series:
    """
    Identify which seasonal regime is active for each date.
    
    Args:
        data: Time series data with datetime index
        patterns: Dictionary of seasonal patterns
        threshold: Minimum strength to consider pattern active
        
    Returns:
        Series with active pattern names
    """
    regimes = []
    
    for date in data.index:
        max_strength = 0
        active_pattern = 'NONE'
        
        for name, pattern in patterns.items():
            strength = calculate_pattern_strength(
                date, pattern.months, pattern.smooth_transitions
            )
            if strength > max_strength and strength >= threshold:
                max_strength = strength
                active_pattern = name
        
        regimes.append(active_pattern)
    
    return pd.Series(regimes, index=data.index)


def calculate_seasonal_zscore(current_value: float, historical_data: pd.Series,
                            current_date: datetime,
                            lookback_days: int = 365, min_samples: int = 30) -> float:
    """
    Calculate z-score of current value relative to seasonal normal.
    
    Args:
        current_value: Current observation
        historical_data: Historical time series
        current_date: Date for the current observation
        lookback_days: Days to look back for seasonal comparison
        min_samples: Minimum samples required
        
    Returns:
        Z-score relative to seasonal normal
    """
    if len(historical_data) < min_samples:
        return 0.0
    
    # Get same day of year from historical data
    current_doy = pd.Timestamp(current_date).dayofyear
    historical_values = []
    
    for date, value in historical_data.items():
        if isinstance(date, pd.Timestamp):
            if abs(date.dayofyear - current_doy) <= 7:  # +/- 1 week window
                historical_values.append(value)
    
    if len(historical_values) < min_samples // 2:
        # Fallback to broader window
        historical_values = historical_data.tail(lookback_days).values
    
    if len(historical_values) > 0:
        mean = np.mean(historical_values)
        std = np.std(historical_values)
        if std > 0:
            return (current_value - mean) / std
    
    return 0.0


# Predefined patterns for crude oil
CRUDE_OIL_PATTERNS = {
    'driving_season': SeasonalPattern(
        name='driving_season',
        months=[5, 6, 7, 8],
        direction='TIGHTEN',
        base_adjustment=-0.05,
        confidence=0.65
    ),
    'winter_heating': SeasonalPattern(
        name='winter_heating',
        months=[11, 12, 1, 2],
        direction='WIDEN',
        base_adjustment=0.05,
        confidence=0.55
    ),
    'refinery_maintenance': SeasonalPattern(
        name='refinery_maintenance',
        months=[3, 4, 9, 10],
        direction='MIXED',
        base_adjustment=-0.02,
        confidence=0.45,
        utilization_dependent=True
    )
}


# Predefined patterns for RBOB gasoline
RBOB_PATTERNS = {
    'summer_driving': SeasonalPattern(
        name='summer_driving',
        months=[5, 6, 7, 8],
        direction='TIGHTEN',
        base_adjustment=-0.08,  # Stronger than crude
        confidence=0.75
    ),
    'winter_spec_change': SeasonalPattern(
        name='winter_spec_change',
        months=[10, 11, 12, 1],
        direction='WIDEN',
        base_adjustment=0.06,
        confidence=0.60
    ),
    'spring_turnaround': SeasonalPattern(
        name='spring_turnaround',
        months=[2, 3, 4],
        direction='TIGHTEN',
        base_adjustment=-0.04,
        confidence=0.50,
        utilization_dependent=True
    )
}