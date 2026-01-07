"""
Dynamic portfolio rebalancing for multi-contract strategies.

This module provides tools for:
- PCA-based dynamic weight calculation
- Periodic rebalancing with transaction cost awareness
- Risk-based position adjustment
- Correlation-aware portfolio construction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class RebalanceMethod(Enum):
    """Rebalancing methods."""
    EQUAL = "equal"
    PCA_WEIGHTED = "pca_weighted"
    VOLATILITY_PARITY = "volatility_parity"
    RISK_PARITY = "risk_parity"
    MOMENTUM_WEIGHTED = "momentum_weighted"


@dataclass
class RebalanceConfig:
    """Configuration for portfolio rebalancing."""
    method: RebalanceMethod = RebalanceMethod.PCA_WEIGHTED
    rebalance_frequency_bars: int = 480  # 2 hours for 15s bars
    min_weight: float = 0.05  # Minimum position weight
    max_weight: float = 0.40  # Maximum position weight
    transaction_cost_bps: float = 2.0  # Cost threshold to trigger rebalance
    correlation_penalty: float = 0.5  # Penalty for high correlation
    use_momentum_tilt: bool = True  # Tilt weights by momentum
    momentum_tilt_factor: float = 0.3  # How much to tilt


@dataclass
class RebalanceSignal:
    """Signal to rebalance portfolio."""
    timestamp: datetime
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    trades_required: Dict[str, float]
    estimated_cost_bps: float
    rebalance_reason: str


class PortfolioRebalancer:
    """
    Manages dynamic portfolio rebalancing based on various criteria.
    """
    
    def __init__(self, config: RebalanceConfig = None):
        """
        Initialize rebalancer.
        
        Parameters
        ----------
        config : RebalanceConfig
            Rebalancing configuration
        """
        self.config = config or RebalanceConfig()
        self.last_rebalance_time: Optional[datetime] = None
        self.rebalance_history: List[RebalanceSignal] = []
        self.bars_since_rebalance: int = 0
        
    def check_rebalance_needed(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        timestamp: datetime
    ) -> Optional[RebalanceSignal]:
        """
        Check if rebalancing is needed.
        
        Parameters
        ----------
        current_weights : Dict[str, float]
            Current portfolio weights
        target_weights : Dict[str, float]
            Target portfolio weights
        timestamp : datetime
            Current timestamp
            
        Returns
        -------
        Optional[RebalanceSignal]
            Rebalance signal if needed, None otherwise
        """
        # Check frequency constraint
        if not self._check_frequency_constraint():
            return None
            
        # Calculate required trades
        trades_required = self._calculate_trades(current_weights, target_weights)
        
        # Estimate transaction costs
        estimated_cost = self._estimate_transaction_cost(trades_required)
        
        # Check if rebalancing is worthwhile
        if not self._is_rebalance_worthwhile(trades_required, estimated_cost):
            return None
            
        # Create rebalance signal
        signal = RebalanceSignal(
            timestamp=timestamp,
            current_weights=current_weights.copy(),
            target_weights=target_weights.copy(),
            trades_required=trades_required,
            estimated_cost_bps=estimated_cost,
            rebalance_reason=self._determine_rebalance_reason(trades_required)
        )
        
        return signal
        
    def execute_rebalance(self, signal: RebalanceSignal) -> Dict[str, float]:
        """
        Execute rebalancing and update state.
        
        Parameters
        ----------
        signal : RebalanceSignal
            Rebalance signal to execute
            
        Returns
        -------
        Dict[str, float]
            New portfolio weights after rebalancing
        """
        # Record rebalance
        self.rebalance_history.append(signal)
        self.last_rebalance_time = signal.timestamp
        self.bars_since_rebalance = 0
        
        return signal.target_weights
        
    def _check_frequency_constraint(self) -> bool:
        """Check if enough time has passed since last rebalance."""
        return self.bars_since_rebalance >= self.config.rebalance_frequency_bars
        
    def _calculate_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate required trades to move from current to target weights."""
        trades = {}
        
        # Get all instruments
        all_instruments = set(current_weights.keys()) | set(target_weights.keys())
        
        for instrument in all_instruments:
            current = current_weights.get(instrument, 0.0)
            target = target_weights.get(instrument, 0.0)
            trade = target - current
            
            if abs(trade) > 0.001:  # Ignore tiny trades
                trades[instrument] = trade
                
        return trades
        
    def _estimate_transaction_cost(self, trades: Dict[str, float]) -> float:
        """Estimate transaction cost in basis points."""
        if not trades:
            return 0.0
            
        # Total turnover (sum of absolute trades)
        turnover = sum(abs(trade) for trade in trades.values())
        
        # Convert to basis points (assuming cost per unit turnover)
        cost_bps = turnover * self.config.transaction_cost_bps / 2  # Divide by 2 for one-way
        
        return cost_bps
        
    def _is_rebalance_worthwhile(
        self,
        trades: Dict[str, float],
        estimated_cost: float
    ) -> bool:
        """Determine if rebalancing is worthwhile given costs."""
        if not trades:
            return False
            
        # Check if any weight deviates significantly
        max_deviation = max(abs(trade) for trade in trades.values())
        if max_deviation > 0.10:  # 10% deviation always triggers
            return True
            
        # Check if cost is acceptable
        if estimated_cost > self.config.transaction_cost_bps * 2:
            return False
            
        return True
        
    def _determine_rebalance_reason(self, trades: Dict[str, float]) -> str:
        """Determine the primary reason for rebalancing."""
        if not trades:
            return "No trades required"
            
        max_trade = max(trades.items(), key=lambda x: abs(x[1]))
        
        if abs(max_trade[1]) > 0.10:
            return f"Large weight deviation in {max_trade[0]}"
        elif len(trades) > 4:
            return "Multiple small deviations"
        else:
            return "Regular rebalancing"
            
    def update(self, bars_elapsed: int = 1):
        """Update internal state."""
        self.bars_since_rebalance += bars_elapsed


def calculate_pca_based_weights(
    pca_loadings: pd.DataFrame,
    factor_idx: int = 1,  # PC2 for slope
    normalize: bool = True
) -> Dict[str, float]:
    """
    Calculate portfolio weights based on PCA loadings.
    
    Parameters
    ----------
    pca_loadings : pd.DataFrame
        PCA loadings matrix (contracts x factors)
    factor_idx : int
        Which principal component to use (0-indexed)
    normalize : bool
        Whether to normalize weights to sum to 1
        
    Returns
    -------
    Dict[str, float]
        Contract weights based on PCA loadings
    """
    if factor_idx >= pca_loadings.shape[1]:
        raise ValueError(f"Factor index {factor_idx} exceeds available factors")
        
    # Get loadings for selected factor
    loadings = pca_loadings.iloc[:, factor_idx].to_dict()
    
    # Convert to weights (absolute value shows importance)
    weights = {k: abs(v) for k, v in loadings.items()}
    
    if normalize and sum(weights.values()) > 0:
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
    # Apply sign from original loadings
    for k in weights:
        weights[k] *= np.sign(loadings[k])
        
    return weights


def calculate_volatility_parity_weights(
    volatilities: Dict[str, float],
    correlations: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Calculate risk parity weights based on volatility.
    
    Parameters
    ----------
    volatilities : Dict[str, float]
        Volatility for each instrument
    correlations : pd.DataFrame, optional
        Correlation matrix for more sophisticated weighting
        
    Returns
    -------
    Dict[str, float]
        Risk parity weights
    """
    # Inverse volatility weighting
    inv_vols = {k: 1.0 / v for k, v in volatilities.items() if v > 0}
    
    # Normalize
    total = sum(inv_vols.values())
    weights = {k: v / total for k, v in inv_vols.items()}
    
    # Adjust for correlations if provided
    if correlations is not None:
        # This is simplified - full risk parity requires optimization
        # Here we just penalize highly correlated assets
        avg_correlations = correlations.mean(axis=1).to_dict()
        for k in weights:
            if k in avg_correlations:
                # Reduce weight for highly correlated assets
                correlation_adj = 1.0 - (avg_correlations[k] - 1.0) / len(correlations)
                weights[k] *= correlation_adj
                
        # Re-normalize
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
    return weights


def calculate_target_weights(
    method: RebalanceMethod,
    contracts: List[str],
    pca_loadings: Optional[pd.DataFrame] = None,
    volatilities: Optional[Dict[str, float]] = None,
    momentum_scores: Optional[Dict[str, float]] = None,
    correlations: Optional[pd.DataFrame] = None,
    config: RebalanceConfig = None
) -> Dict[str, float]:
    """
    Calculate target portfolio weights based on specified method.
    
    Parameters
    ----------
    method : RebalanceMethod
        Weighting method to use
    contracts : List[str]
        List of contracts to weight
    pca_loadings : pd.DataFrame, optional
        PCA loadings for PCA-based weighting
    volatilities : Dict[str, float], optional
        Volatilities for risk-based weighting
    momentum_scores : Dict[str, float], optional
        Momentum scores for tilting
    correlations : pd.DataFrame, optional
        Correlation matrix
    config : RebalanceConfig, optional
        Configuration parameters
        
    Returns
    -------
    Dict[str, float]
        Target weights for each contract
    """
    config = config or RebalanceConfig()
    
    # Calculate base weights based on method
    if method == RebalanceMethod.EQUAL:
        weights = {c: 1.0 / len(contracts) for c in contracts}
        
    elif method == RebalanceMethod.PCA_WEIGHTED:
        if pca_loadings is None:
            raise ValueError("PCA loadings required for PCA weighting")
        weights = calculate_pca_based_weights(pca_loadings)
        
    elif method == RebalanceMethod.VOLATILITY_PARITY:
        if volatilities is None:
            raise ValueError("Volatilities required for volatility parity")
        weights = calculate_volatility_parity_weights(volatilities, correlations)
        
    else:
        # Default to equal weighting
        weights = {c: 1.0 / len(contracts) for c in contracts}
    
    # Apply momentum tilt if configured
    if config.use_momentum_tilt and momentum_scores is not None:
        weights = apply_momentum_tilt(
            weights,
            momentum_scores,
            config.momentum_tilt_factor
        )
    
    # Apply constraints
    weights = apply_weight_constraints(
        weights,
        config.min_weight,
        config.max_weight
    )
    
    return weights


def apply_momentum_tilt(
    base_weights: Dict[str, float],
    momentum_scores: Dict[str, float],
    tilt_factor: float = 0.3
) -> Dict[str, float]:
    """
    Tilt weights based on momentum scores.
    
    Parameters
    ----------
    base_weights : Dict[str, float]
        Base portfolio weights
    momentum_scores : Dict[str, float]
        Momentum scores for each instrument
    tilt_factor : float
        How much to tilt (0 = no tilt, 1 = full momentum)
        
    Returns
    -------
    Dict[str, float]
        Tilted weights
    """
    # Normalize momentum scores to [-1, 1]
    mom_values = list(momentum_scores.values())
    if mom_values:
        max_mom = max(abs(m) for m in mom_values)
        if max_mom > 0:
            norm_momentum = {k: v / max_mom for k, v in momentum_scores.items()}
        else:
            norm_momentum = momentum_scores
    else:
        return base_weights
    
    # Apply tilt
    tilted_weights = {}
    for contract, base_weight in base_weights.items():
        if contract in norm_momentum:
            # Tilt weight based on momentum
            momentum = norm_momentum[contract]
            tilt_mult = 1.0 + (momentum * tilt_factor)
            tilted_weights[contract] = base_weight * tilt_mult
        else:
            tilted_weights[contract] = base_weight
    
    # Renormalize
    total = sum(tilted_weights.values())
    if total > 0:
        tilted_weights = {k: v / total for k, v in tilted_weights.items()}
    
    return tilted_weights


def apply_weight_constraints(
    weights: Dict[str, float],
    min_weight: float = 0.0,
    max_weight: float = 1.0
) -> Dict[str, float]:
    """
    Apply min/max constraints to portfolio weights.
    
    Parameters
    ----------
    weights : Dict[str, float]
        Raw portfolio weights
    min_weight : float
        Minimum weight per position
    max_weight : float
        Maximum weight per position
        
    Returns
    -------
    Dict[str, float]
        Constrained weights
    """
    # Apply constraints
    constrained = {}
    for k, v in weights.items():
        if abs(v) < min_weight:
            constrained[k] = 0.0
        elif v > max_weight:
            constrained[k] = max_weight
        elif v < -max_weight:
            constrained[k] = -max_weight
        else:
            constrained[k] = v
    
    # Renormalize if needed
    total = sum(abs(v) for v in constrained.values())
    if total > 0 and abs(total - 1.0) > 0.001:
        constrained = {k: v / total for k, v in constrained.items()}
    
    return constrained