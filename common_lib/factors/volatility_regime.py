"""
Volatility Regime Classification for Trading Strategies.

Classifies market conditions into volatility regimes (LOW, NORMAL, HIGH, EXTREME)
based on percentile thresholds and optional PCA Speed Index overrides.
Includes regime duration tracking and band calculation logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging

from ..indicators.volatility import volatility_percentile, calculate_volatility_bands

logger = logging.getLogger(__name__)


@dataclass
class RegimeConfig:
    """Configuration for volatility regime classification."""
    # Percentile thresholds
    low_threshold: float = 0.25      # Below this = LOW regime
    high_threshold: float = 0.75     # Above this = HIGH regime  
    extreme_threshold: float = 0.90  # Above this = EXTREME regime
    
    # PCA override settings
    pca_speed_threshold: float = 1.5           # Override to HIGH if exceeded
    pca_emergency_exit_threshold: float = 2.0  # Force exit all positions
    
    # Regime stability
    min_regime_duration: int = 24  # Minimum bars before acting on new regime
    
    # Band calculation
    band_multipliers: Dict[str, float] = None
    
    def __post_init__(self):
        if self.band_multipliers is None:
            self.band_multipliers = {
                'LOW': 1.0,
                'NORMAL': 1.0,
                'HIGH': 1.0,
                'EXTREME': 1.5
            }


class VolatilityRegimeClassifier:
    """
    Classifies market volatility into discrete regimes with stability tracking.
    
    Used by the Volatility Regime strategy to determine trading behavior
    based on current market conditions. Includes PCA Speed Index override
    capability and regime duration tracking to prevent whipsaws.
    """
    
    REGIMES = ['LOW', 'NORMAL', 'HIGH', 'EXTREME']
    
    def __init__(self, config: RegimeConfig = None):
        """
        Initialize the classifier.
        
        Parameters
        ----------
        config : RegimeConfig, optional
            Configuration object. If None, uses defaults.
        """
        self.config = config or RegimeConfig()
        
        # Track regime durations for each symbol/spread
        self.regime_duration_tracker: Dict[str, Dict] = {}
        
        # Cache recent classifications
        self.regime_cache: Dict[str, Tuple[str, datetime]] = {}
    
    def classify(
        self,
        vol_percentile: float,
        pca_speed: float = 0.0,
        symbol: Optional[str] = None
    ) -> str:
        """
        Classify current volatility regime.
        
        Parameters
        ----------
        vol_percentile : float
            Current volatility percentile (0-1)
        pca_speed : float
            PCA Speed Index value (optional override)
        symbol : str, optional
            Symbol/spread identifier for duration tracking
            
        Returns
        -------
        str
            Regime classification: 'LOW', 'NORMAL', 'HIGH', or 'EXTREME'
        """
        # Validate inputs
        if not 0 <= vol_percentile <= 1:
            logger.warning(f"Invalid volatility percentile: {vol_percentile}")
            return 'NORMAL'
        
        # Check PCA emergency exit
        if pca_speed >= self.config.pca_emergency_exit_threshold:
            logger.warning(f"PCA Speed {pca_speed:.2f} exceeds emergency threshold - EXTREME regime")
            return 'EXTREME'
        
        # Base classification
        if vol_percentile < self.config.low_threshold:
            regime = 'LOW'
        elif vol_percentile < self.config.high_threshold:
            regime = 'NORMAL'
        elif vol_percentile < self.config.extreme_threshold:
            regime = 'HIGH'
        else:
            regime = 'EXTREME'
        
        # PCA Speed override
        if pca_speed > self.config.pca_speed_threshold and regime in ['LOW', 'NORMAL']:
            logger.info(f"PCA Speed {pca_speed:.2f} overriding {regime} to HIGH")
            regime = 'HIGH'
        
        return regime
    
    def check_regime_duration(
        self,
        symbol: str,
        new_regime: str,
        current_bar: int = 1
    ) -> Tuple[bool, int]:
        """
        Check if regime has persisted long enough to trade.
        
        Prevents whipsaw trades by requiring regime persistence.
        
        Parameters
        ----------
        symbol : str
            Symbol/spread identifier
        new_regime : str
            Newly classified regime
        current_bar : int
            Current bar number (for tracking duration)
            
        Returns
        -------
        tuple
            (is_stable, duration) - whether regime is stable and current duration
        """
        if symbol not in self.regime_duration_tracker:
            self.regime_duration_tracker[symbol] = {
                'regime': new_regime,
                'duration': 1,
                'start_bar': current_bar
            }
            return False, 1
        
        tracker = self.regime_duration_tracker[symbol]
        
        if tracker['regime'] == new_regime:
            # Same regime, increment duration
            tracker['duration'] += 1
        else:
            # Regime changed, reset counter
            logger.info(f"{symbol} regime changed from {tracker['regime']} to {new_regime}")
            tracker['regime'] = new_regime
            tracker['duration'] = 1
            tracker['start_bar'] = current_bar
        
        # Check if stable
        is_stable = tracker['duration'] >= self.config.min_regime_duration
        
        return is_stable, tracker['duration']
    
    def calculate_bands(
        self,
        fair_value: float,
        volatility: float,
        regime: str,
        time_horizon: int = 1
    ) -> Dict[str, float]:
        """
        Calculate trading bands based on regime.
        
        Parameters
        ----------
        fair_value : float
            Center price (fair value)
        volatility : float
            Current annualized volatility
        regime : str
            Current regime
        time_horizon : int
            Time horizon in days
            
        Returns
        -------
        dict
            Dictionary with 'upper', 'lower', and 'width' keys
        """
        multiplier = self.config.band_multipliers.get(regime, 1.0)
        
        lower, upper = calculate_volatility_bands(
            price=fair_value,
            volatility=volatility,
            multiplier=multiplier,
            time_horizon=time_horizon
        )
        
        return {
            'upper': upper,
            'lower': lower,
            'width': upper - lower,
            'multiplier': multiplier
        }
    
    def get_regime_position_size(self, regime: str, base_size: float) -> float:
        """
        Get position size adjustment factor for regime.
        
        Parameters
        ----------
        regime : str
            Current regime
        base_size : float
            Base position size
            
        Returns
        -------
        float
            Adjusted position size
        """
        size_factors = {
            'LOW': 1.0,      # Full size in low vol
            'NORMAL': 0.5,   # Half size in normal vol
            'HIGH': 1.0,     # Full size in high vol (different signal logic)
            'EXTREME': 0.7   # Reduced size in extreme vol
        }
        
        factor = size_factors.get(regime, 0.5)
        return base_size * factor
    
    def should_exit_on_regime_change(
        self,
        entry_regime: str,
        current_regime: str,
        position_side: str
    ) -> bool:
        """
        Determine if position should be exited due to regime change.
        
        Parameters
        ----------
        entry_regime : str
            Regime when position was entered
        current_regime : str
            Current regime
        position_side : str
            'LONG' or 'SHORT'
            
        Returns
        -------
        bool
            True if position should be exited
        """
        # Always exit if entering EXTREME from non-EXTREME
        if current_regime == 'EXTREME' and entry_regime != 'EXTREME':
            return True
        
        # Exit if regime fundamentally changed
        regime_groups = {
            'low_vol': ['LOW', 'NORMAL'],
            'high_vol': ['HIGH', 'EXTREME']
        }
        
        entry_group = None
        current_group = None
        
        for group, regimes in regime_groups.items():
            if entry_regime in regimes:
                entry_group = group
            if current_regime in regimes:
                current_group = group
        
        # Exit if moved between major regime groups
        return entry_group != current_group
    
    def get_regime_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics on regime durations and transitions.
        
        Returns
        -------
        dict
            Statistics by symbol
        """
        stats = {}
        
        for symbol, tracker in self.regime_duration_tracker.items():
            stats[symbol] = {
                'current_regime': tracker['regime'],
                'duration': tracker['duration'],
                'stable': tracker['duration'] >= self.config.min_regime_duration
            }
        
        return stats
    
    def reset_tracker(self, symbol: Optional[str] = None):
        """
        Reset duration tracking for a symbol or all symbols.
        
        Parameters
        ----------
        symbol : str, optional
            Symbol to reset. If None, resets all.
        """
        if symbol:
            if symbol in self.regime_duration_tracker:
                del self.regime_duration_tracker[symbol]
        else:
            self.regime_duration_tracker.clear()


def classify_volatility_regime_series(
    volatility: pd.Series,
    lookback_window: int = 504,
    config: RegimeConfig = None,
    pca_speed: Optional[pd.Series] = None
) -> pd.Series:
    """
    Classify a full series of volatility data into regimes.
    
    Convenience function for backtesting and analysis.
    
    Parameters
    ----------
    volatility : pd.Series
        Volatility time series
    lookback_window : int
        Window for percentile calculation
    config : RegimeConfig, optional
        Configuration object
    pca_speed : pd.Series, optional
        PCA Speed Index series (must align with volatility index)
        
    Returns
    -------
    pd.Series
        Regime classifications with same index as input
    """
    # Calculate percentiles
    vol_pct = volatility_percentile(
        volatility.values,
        lookback_window=lookback_window
    )
    
    # Initialize classifier
    classifier = VolatilityRegimeClassifier(config)
    
    # Classify each point
    regimes = []
    for i in range(len(volatility)):
        if np.isnan(vol_pct[i]):
            regimes.append(np.nan)
        else:
            pca_val = 0.0
            if pca_speed is not None and i < len(pca_speed):
                pca_val = pca_speed.iloc[i] if not pd.isna(pca_speed.iloc[i]) else 0.0
            
            regime = classifier.classify(
                vol_percentile=vol_pct[i],
                pca_speed=pca_val
            )
            regimes.append(regime)
    
    return pd.Series(regimes, index=volatility.index)