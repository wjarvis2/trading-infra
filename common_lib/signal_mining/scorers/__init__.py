"""
Signal scoring module.

Pure functions for calculating signal validation metrics.
No I/O operations - all data passed as parameters.
"""

from common_lib.signal_mining.scorers.ic_scorer import (
    calculate_ic,
    calculate_rank_ic,
    calculate_hit_rate,
    calculate_decay_profile,
    calculate_decay_profile_from_df,
    calculate_conditional_returns,
    calculate_ic_stability,
    calculate_ic_stability_from_df,
    calculate_ic_with_se,
    calculate_n_effective,
    detect_direction,
    calculate_regime_ic,
)

from common_lib.signal_mining.scorers.forward_returns import (
    calculate_forward_returns,
    calculate_forward_change,
    calculate_carry_return,
    calculate_spread_forward_return,
    align_signal_returns,
    create_forward_return_df,
    ReturnType,
)

__all__ = [
    # IC scoring
    'calculate_ic',
    'calculate_rank_ic',
    'calculate_hit_rate',
    'calculate_decay_profile',
    'calculate_decay_profile_from_df',
    'calculate_conditional_returns',
    'calculate_ic_stability',
    'calculate_ic_stability_from_df',
    'calculate_ic_with_se',
    'calculate_n_effective',
    'detect_direction',
    'calculate_regime_ic',
    # Forward returns
    'calculate_forward_returns',
    'calculate_forward_change',
    'calculate_carry_return',
    'calculate_spread_forward_return',
    'align_signal_returns',
    'create_forward_return_df',
    'ReturnType',
]
