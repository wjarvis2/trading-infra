"""
Scoring result data models.

Contains the output of signal validation (NOT backtest results).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import date
import numpy as np


@dataclass
class MetricBundle:
    """
    Bundle of validation metrics for a signal variant.

    These are lightweight metrics computed without position simulation.
    They measure predictive content, not P&L.

    Attributes
    ----------
    ic : float
        Information coefficient (Pearson correlation)
    ic_se : float
        Bootstrap standard error for IC
    rank_ic : float
        Spearman rank correlation (more robust to outliers)
    rank_ic_se : float
        Bootstrap standard error for rank IC
    hit_rate : float
        Directional accuracy (fraction of correct predictions)
    decay_profile : dict
        IC at multiple horizons {horizon_days: ic_value}
    conditional_return_positive : float
        Mean return when signal > threshold
    conditional_return_negative : float
        Mean return when signal < -threshold
    conditional_threshold : float
        Threshold used (typically 1.0 = 1 std)
    ic_stability : float
        IC std across subperiods (lower = more stable)
    subperiod_ics : dict
        IC per subperiod {'2023-H1': ic_value, ...}
    regime_ics : dict
        IC by regime {'contango': ic_value, 'backwardation': ic_value}
    max_correlation_existing : float
        Max correlation with existing promoted signals
    most_correlated_signal : str
        Name of most correlated existing signal
    n_observations : int
        Number of signal/return pairs used
    n_effective : int
        Autocorrelation-adjusted effective sample size
    date_range_start : date
        First date in sample
    date_range_end : date
        Last date in sample
    cost_proxy_bps : float
        Average bid-ask spread for expression (tradability)
    detected_direction : str
        'procyc', 'contracyc', or 'none' based on IC sign
    """
    ic: float
    ic_se: float = 0.0
    rank_ic: float = 0.0
    rank_ic_se: float = 0.0
    hit_rate: float = 0.0

    # Decay profile (IC at each horizon)
    decay_profile: Dict[int, float] = field(default_factory=dict)

    # Conditional returns
    conditional_return_positive: float = 0.0
    conditional_return_negative: float = 0.0
    conditional_threshold: float = 1.0

    # Stability
    ic_stability: float = 0.0
    subperiod_ics: Dict[str, float] = field(default_factory=dict)

    # Regime analysis
    regime_ics: Dict[str, float] = field(default_factory=dict)

    # Redundancy
    max_correlation_existing: float = 0.0
    most_correlated_signal: Optional[str] = None

    # Sample info
    n_observations: int = 0
    n_effective: Optional[int] = None
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None

    # Cost/tradability
    cost_proxy_bps: Optional[float] = None

    # Direction detected
    detected_direction: str = 'none'

    def is_significant(
        self,
        ic_threshold: float = 0.02,
        min_observations: int = 100,
        max_stability: float = 0.5
    ) -> bool:
        """
        Check if IC is statistically meaningful.

        Args:
            ic_threshold: Minimum absolute IC required
            min_observations: Minimum sample size
            max_stability: Maximum IC std across subperiods

        Returns:
            True if signal passes significance criteria
        """
        return (
            abs(self.ic) > ic_threshold and
            self.n_observations >= min_observations and
            self.ic_stability < max_stability
        )

    def is_tradable(self, max_cost_bps: float = 50.0) -> bool:
        """
        Check if expression is tradable (low enough cost).

        Args:
            max_cost_bps: Maximum bid-ask spread in bps

        Returns:
            True if cost is below threshold
        """
        if self.cost_proxy_bps is None:
            return True  # Assume tradable if no data
        return self.cost_proxy_bps < max_cost_bps

    def best_horizon(self) -> Optional[int]:
        """Return horizon with highest absolute IC."""
        if not self.decay_profile:
            return None
        return max(self.decay_profile, key=lambda h: abs(self.decay_profile[h]))

    def ic_t_stat(self) -> float:
        """
        Compute t-statistic for IC.

        Returns IC / SE if SE > 0, else 0.
        """
        if self.ic_se > 0:
            return self.ic / self.ic_se
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for database storage."""
        return {
            'ic': self.ic,
            'ic_se': self.ic_se,
            'rank_ic': self.rank_ic,
            'rank_ic_se': self.rank_ic_se,
            'hit_rate': self.hit_rate,
            'ic_1d': self.decay_profile.get(1),
            'ic_3d': self.decay_profile.get(3),
            'ic_5d': self.decay_profile.get(5),
            'ic_10d': self.decay_profile.get(10),
            'ic_20d': self.decay_profile.get(20),
            'conditional_return_pos': self.conditional_return_positive,
            'conditional_return_neg': self.conditional_return_negative,
            'conditional_threshold': self.conditional_threshold,
            'ic_stability': self.ic_stability,
            'subperiod_ic': self.subperiod_ics,
            'regime_ic': self.regime_ics,
            'max_correlation_existing': self.max_correlation_existing,
            'most_correlated_signal': self.most_correlated_signal,
            'n_observations': self.n_observations,
            'n_effective': self.n_effective,
            'date_range_start': self.date_range_start,
            'date_range_end': self.date_range_end,
            'cost_proxy_bps': self.cost_proxy_bps,
            'detected_direction': self.detected_direction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricBundle':
        """Create from dictionary (database row)."""
        # Reconstruct decay profile from individual columns
        decay_profile = {}
        for horizon in [1, 3, 5, 10, 20]:
            key = f'ic_{horizon}d'
            if key in data and data[key] is not None:
                decay_profile[horizon] = float(data[key])

        return cls(
            ic=float(data['ic']) if data.get('ic') is not None else 0.0,
            ic_se=float(data.get('ic_se') or 0.0),
            rank_ic=float(data.get('rank_ic') or 0.0),
            rank_ic_se=float(data.get('rank_ic_se') or 0.0),
            hit_rate=float(data.get('hit_rate') or 0.0),
            decay_profile=decay_profile,
            conditional_return_positive=float(data.get('conditional_return_pos') or 0.0),
            conditional_return_negative=float(data.get('conditional_return_neg') or 0.0),
            conditional_threshold=float(data.get('conditional_threshold') or 1.0),
            ic_stability=float(data.get('ic_stability') or 0.0),
            subperiod_ics=data.get('subperiod_ic') or {},
            regime_ics=data.get('regime_ic') or {},
            max_correlation_existing=float(data.get('max_correlation_existing') or 0.0),
            most_correlated_signal=data.get('most_correlated_signal'),
            n_observations=int(data.get('n_observations') or 0),
            n_effective=int(data['n_effective']) if data.get('n_effective') else None,
            date_range_start=data.get('date_range_start'),
            date_range_end=data.get('date_range_end'),
            cost_proxy_bps=float(data['cost_proxy_bps']) if data.get('cost_proxy_bps') else None,
            detected_direction=data.get('detected_direction', 'none'),
        )


@dataclass
class ScoringResult:
    """
    Complete scoring result for a variant.

    Contains metrics bundle plus optional raw data for debugging.

    Attributes
    ----------
    variant_id : int
        Database ID of the variant
    hypothesis_id : int
        Database ID of the parent hypothesis
    metrics : MetricBundle
        All validation metrics
    raw_signal : np.ndarray, optional
        Raw signal values for debugging
    raw_returns : np.ndarray, optional
        Raw forward returns for debugging
    metadata : dict
        Additional metadata
    error : str, optional
        Error message if scoring failed
    """
    variant_id: int
    hypothesis_id: int
    metrics: MetricBundle
    raw_signal: Optional[np.ndarray] = None
    raw_returns: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if scoring completed without error."""
        return self.error is None

    def __repr__(self) -> str:
        status = "OK" if self.success else f"ERROR: {self.error}"
        return (
            f"ScoringResult(variant={self.variant_id}, "
            f"ic={self.metrics.ic:.4f}, "
            f"n={self.metrics.n_observations}, "
            f"status={status})"
        )
