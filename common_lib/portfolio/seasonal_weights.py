"""
Seasonal weight calculations for portfolio allocation.

This module provides functions to calculate contract weights based on
seasonal patterns and market conditions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class ContractWeight:
    """Weight configuration for a futures contract."""
    contract: str
    base_weight: float
    seasonal_adjustment: float
    final_weight: float
    reason: str


class SeasonalWeightCalculator:
    """Calculates optimal contract weights based on seasonal factors."""
    
    def __init__(self, lookback_years: int = 5):
        """
        Initialize seasonal weight calculator.
        
        Args:
            lookback_years: Years of history to analyze for patterns
        """
        self.lookback_years = lookback_years
        
        # Default spread weight configurations
        self.default_weights = {
            'CL1-CL2': {'base': 1.0, 'decay': 0.0},
            'CL1-CL3': {'base': 0.8, 'decay': 0.1},
            'CL2-CL4': {'base': 0.6, 'decay': 0.15},
            'CL3-CL6': {'base': 0.5, 'decay': 0.2},
            'CL1-CL4': {'base': 0.7, 'decay': 0.12},
            'CL1-CL6': {'base': 0.4, 'decay': 0.25},
            'CL2-CL3': {'base': 0.9, 'decay': 0.05},
            'CL4-CL5': {'base': 0.5, 'decay': 0.15}
        }
    
    def calculate_seasonal_weights(self, spreads: List[str], date: datetime,
                                 seasonal_patterns: Dict[str, float]) -> Dict[str, ContractWeight]:
        """
        Calculate weights for multiple spreads based on seasonal patterns.
        
        Args:
            spreads: List of spread names (e.g., ['CL1-CL2', 'CL1-CL3'])
            date: Current date for seasonal adjustment
            seasonal_patterns: Dict of pattern strengths by spread
            
        Returns:
            Dictionary mapping spread to ContractWeight
        """
        weights = {}
        
        for spread in spreads:
            base_config = self.default_weights.get(spread, {'base': 0.5, 'decay': 0.1})
            base_weight = base_config['base']
            
            # Apply seasonal adjustment
            seasonal_strength = seasonal_patterns.get(spread, 0.0)
            seasonal_adjustment = self._calculate_seasonal_adjustment(
                spread, date, seasonal_strength
            )
            
            # Calculate final weight
            final_weight = base_weight * (1 + seasonal_adjustment)
            
            # Determine reason
            if seasonal_adjustment > 0.1:
                reason = "Strong seasonal pattern"
            elif seasonal_adjustment < -0.1:
                reason = "Weak seasonal pattern"
            else:
                reason = "Normal seasonal conditions"
            
            weights[spread] = ContractWeight(
                contract=spread,
                base_weight=base_weight,
                seasonal_adjustment=seasonal_adjustment,
                final_weight=final_weight,
                reason=reason
            )
        
        # Normalize weights
        self._normalize_weights(weights)
        
        return weights
    
    def _calculate_seasonal_adjustment(self, spread: str, date: datetime,
                                     seasonal_strength: float) -> float:
        """
        Calculate seasonal adjustment factor for a spread.
        
        Args:
            spread: Spread name
            date: Current date
            seasonal_strength: Strength of seasonal pattern (-1 to 1)
            
        Returns:
            Adjustment factor (-0.5 to 0.5)
        """
        # Extract contract months from spread
        contracts = spread.split('-')
        if len(contracts) != 2:
            return 0.0
        
        # Get month distance
        front_month = int(contracts[0][2:]) if len(contracts[0]) > 2 else 1
        back_month = int(contracts[1][2:]) if len(contracts[1]) > 2 else 2
        month_spread = back_month - front_month
        
        # Seasonal adjustments based on pattern type
        month = date.month
        
        # Driving season (May-Aug): favor front spreads
        if 5 <= month <= 8:
            if month_spread <= 2:
                adjustment = 0.3 * seasonal_strength
            else:
                adjustment = -0.1 * seasonal_strength
        
        # Winter (Nov-Feb): favor back spreads
        elif month in [11, 12, 1, 2]:
            if month_spread >= 3:
                adjustment = 0.2 * seasonal_strength
            else:
                adjustment = -0.05 * seasonal_strength
        
        # Shoulder months: balanced
        else:
            adjustment = 0.0
        
        # Cap adjustment
        return np.clip(adjustment, -0.5, 0.5)
    
    def _normalize_weights(self, weights: Dict[str, ContractWeight]) -> None:
        """
        Normalize weights to sum to 1.0.
        
        Args:
            weights: Dictionary of ContractWeight objects (modified in place)
        """
        total_weight = sum(w.final_weight for w in weights.values())
        
        if total_weight > 0:
            for weight in weights.values():
                weight.final_weight /= total_weight
    
    def calculate_dynamic_weights(self, spreads: List[str], 
                                historical_performance: pd.DataFrame,
                                lookback_days: int = 60) -> Dict[str, ContractWeight]:
        """
        Calculate weights based on recent performance.
        
        Args:
            spreads: List of spread names
            historical_performance: DataFrame with spread returns
            lookback_days: Days to look back for performance
            
        Returns:
            Dictionary mapping spread to ContractWeight
        """
        weights = {}
        recent_data = historical_performance.tail(lookback_days)
        
        for spread in spreads:
            if spread not in recent_data.columns:
                continue
            
            # Calculate performance metrics
            returns = recent_data[spread].pct_change().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            win_rate = (returns > 0).mean()
            
            # Base weight on Sharpe ratio
            if sharpe > 1.5:
                base_weight = 1.2
                reason = "Excellent recent performance"
            elif sharpe > 0.8:
                base_weight = 1.0
                reason = "Good recent performance"
            elif sharpe > 0:
                base_weight = 0.8
                reason = "Positive recent performance"
            else:
                base_weight = 0.6
                reason = "Poor recent performance"
            
            # Adjust for win rate
            if win_rate > 0.6:
                base_weight *= 1.1
            elif win_rate < 0.4:
                base_weight *= 0.9
            
            weights[spread] = ContractWeight(
                contract=spread,
                base_weight=base_weight,
                seasonal_adjustment=0.0,
                final_weight=base_weight,
                reason=reason
            )
        
        # Normalize weights
        self._normalize_weights(weights)
        
        return weights
    
    def calculate_risk_parity_weights(self, spreads: List[str],
                                    volatilities: Dict[str, float],
                                    correlations: Optional[pd.DataFrame] = None) -> Dict[str, ContractWeight]:
        """
        Calculate risk parity weights for spreads.
        
        Args:
            spreads: List of spread names
            volatilities: Dictionary of spread volatilities
            correlations: Correlation matrix (optional)
            
        Returns:
            Dictionary mapping spread to ContractWeight
        """
        weights = {}
        
        # Simple inverse volatility weighting if no correlations
        if correlations is None:
            total_inv_vol = sum(1/volatilities[s] for s in spreads if s in volatilities)
            
            for spread in spreads:
                if spread in volatilities:
                    weight = (1/volatilities[spread]) / total_inv_vol
                    
                    weights[spread] = ContractWeight(
                        contract=spread,
                        base_weight=weight,
                        seasonal_adjustment=0.0,
                        final_weight=weight,
                        reason=f"Risk parity (vol={volatilities[spread]:.1%})"
                    )
        else:
            # Full risk parity with correlations
            # This is a simplified version - full implementation would use optimization
            n = len(spreads)
            vol_array = np.array([volatilities[s] for s in spreads])
            
            # Target equal risk contribution
            target_risk = 1.0 / n
            
            # Iterative solution (simplified)
            raw_weights = 1 / vol_array
            raw_weights /= raw_weights.sum()
            
            for i, spread in enumerate(spreads):
                weights[spread] = ContractWeight(
                    contract=spread,
                    base_weight=raw_weights[i],
                    seasonal_adjustment=0.0,
                    final_weight=raw_weights[i],
                    reason=f"Risk parity (equal contribution)"
                )
        
        return weights
    
    def blend_weight_schemes(self, weight_sets: List[Dict[str, ContractWeight]],
                           blend_weights: Optional[List[float]] = None) -> Dict[str, ContractWeight]:
        """
        Blend multiple weight schemes together.
        
        Args:
            weight_sets: List of weight dictionaries to blend
            blend_weights: Weights for each scheme (default: equal)
            
        Returns:
            Blended weights dictionary
        """
        if not weight_sets:
            return {}
        
        # Default to equal blending
        if blend_weights is None:
            blend_weights = [1.0 / len(weight_sets)] * len(weight_sets)
        
        # Normalize blend weights
        blend_sum = sum(blend_weights)
        blend_weights = [w / blend_sum for w in blend_weights]
        
        # Get all unique spreads
        all_spreads = set()
        for weight_set in weight_sets:
            all_spreads.update(weight_set.keys())
        
        # Blend weights
        blended = {}
        for spread in all_spreads:
            total_weight = 0.0
            total_base = 0.0
            total_seasonal = 0.0
            reasons = []
            
            for i, weight_set in enumerate(weight_sets):
                if spread in weight_set:
                    w = weight_set[spread]
                    total_weight += w.final_weight * blend_weights[i]
                    total_base += w.base_weight * blend_weights[i]
                    total_seasonal += w.seasonal_adjustment * blend_weights[i]
                    if blend_weights[i] > 0.2:  # Only include significant contributors
                        reasons.append(w.reason)
            
            blended[spread] = ContractWeight(
                contract=spread,
                base_weight=total_base,
                seasonal_adjustment=total_seasonal,
                final_weight=total_weight,
                reason=" + ".join(set(reasons)) if reasons else "Blended allocation"
            )
        
        # Normalize final weights
        self._normalize_weights(blended)
        
        return blended
    
    def get_weight_summary(self, weights: Dict[str, ContractWeight]) -> pd.DataFrame:
        """
        Create summary DataFrame of contract weights.
        
        Args:
            weights: Dictionary of ContractWeight objects
            
        Returns:
            DataFrame with weight details
        """
        data = []
        for spread, weight in weights.items():
            data.append({
                'spread': spread,
                'base_weight': weight.base_weight,
                'seasonal_adj': weight.seasonal_adjustment,
                'final_weight': weight.final_weight,
                'weight_pct': f"{weight.final_weight:.1%}",
                'reason': weight.reason
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('final_weight', ascending=False)