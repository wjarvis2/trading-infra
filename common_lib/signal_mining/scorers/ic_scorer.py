"""
Information Coefficient (IC) scoring for signal validation.

Pure functions - no I/O operations.
All data passed as parameters, all results returned.

Functions:
- calculate_ic: Pearson correlation between signal and returns
- calculate_rank_ic: Spearman rank correlation
- calculate_hit_rate: Directional accuracy
- calculate_decay_profile: IC at multiple horizons
- calculate_conditional_returns: Mean returns by signal strength
- calculate_ic_stability: IC variance across subperiods
- calculate_ic_with_se: IC with bootstrap standard error
- calculate_n_effective: Autocorrelation-adjusted sample size
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional, Dict, List
import warnings


def calculate_ic(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    method: str = 'pearson'
) -> Tuple[float, float]:
    """
    Calculate Information Coefficient between signal and forward returns.

    Args:
        signal: Signal values (1D array)
        forward_returns: Forward returns corresponding to signals
        method: 'pearson' or 'spearman'

    Returns:
        Tuple of (ic, p_value)

    Raises:
        ValueError: If arrays have different lengths
    """
    if len(signal) != len(forward_returns):
        raise ValueError(
            f"Signal and returns must have same length: {len(signal)} vs {len(forward_returns)}"
        )

    # Remove NaN pairs
    mask = ~(np.isnan(signal) | np.isnan(forward_returns))
    signal_clean = signal[mask]
    returns_clean = forward_returns[mask]

    if len(signal_clean) < 30:
        return np.nan, np.nan

    # Check for constant values
    if np.std(signal_clean) < 1e-10 or np.std(returns_clean) < 1e-10:
        return 0.0, 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == 'spearman':
            ic, p_value = stats.spearmanr(signal_clean, returns_clean)
        else:
            ic, p_value = stats.pearsonr(signal_clean, returns_clean)

    return float(ic), float(p_value)


def calculate_rank_ic(
    signal: np.ndarray,
    forward_returns: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate Rank IC (Spearman correlation).

    More robust to outliers than Pearson IC.

    Args:
        signal: Signal values
        forward_returns: Forward returns

    Returns:
        Tuple of (rank_ic, p_value)
    """
    return calculate_ic(signal, forward_returns, method='spearman')


def calculate_hit_rate(
    signal: np.ndarray,
    forward_returns: np.ndarray
) -> float:
    """
    Calculate directional accuracy (hit rate).

    Returns fraction of times signal direction matches return direction.
    Only considers non-zero signals.

    Args:
        signal: Signal values
        forward_returns: Forward returns

    Returns:
        Hit rate as fraction (0-1), or NaN if insufficient data
    """
    mask = ~(np.isnan(signal) | np.isnan(forward_returns))
    signal_clean = signal[mask]
    returns_clean = forward_returns[mask]

    if len(signal_clean) < 30:
        return np.nan

    # Filter to non-zero signals
    nonzero_mask = np.abs(signal_clean) > 1e-10
    if nonzero_mask.sum() < 10:
        return np.nan

    signal_dir = np.sign(signal_clean[nonzero_mask])
    return_dir = np.sign(returns_clean[nonzero_mask])

    return float((signal_dir == return_dir).mean())


def calculate_decay_profile(
    signal: np.ndarray,
    prices: np.ndarray,
    horizons: List[int] = None,
    method: str = 'pearson'
) -> Dict[int, float]:
    """
    Calculate IC at multiple forward horizons.

    Args:
        signal: Signal values
        prices: Price series (to compute returns at each horizon)
        horizons: List of forward horizons (default: [1, 3, 5, 10, 20])
        method: 'pearson' or 'spearman'

    Returns:
        Dict mapping horizon to IC value
    """
    if horizons is None:
        horizons = [1, 3, 5, 10, 20]

    decay = {}

    for horizon in horizons:
        # Calculate forward returns at this horizon
        if len(prices) <= horizon:
            decay[horizon] = np.nan
            continue

        fwd_returns = np.empty_like(prices)
        fwd_returns[:] = np.nan
        fwd_returns[:-horizon] = (prices[horizon:] - prices[:-horizon]) / prices[:-horizon]

        ic, _ = calculate_ic(signal, fwd_returns, method=method)
        decay[horizon] = ic

    return decay


def calculate_decay_profile_from_df(
    signal_df: pd.DataFrame,
    signal_col: str,
    price_col: str,
    horizons: List[int] = None,
    method: str = 'pearson'
) -> Dict[int, float]:
    """
    Calculate IC at multiple forward horizons from DataFrame.

    Args:
        signal_df: DataFrame with signal and prices
        signal_col: Column name for signal
        price_col: Column name for price
        horizons: List of forward horizons
        method: 'pearson' or 'spearman'

    Returns:
        Dict mapping horizon to IC
    """
    if horizons is None:
        horizons = [1, 3, 5, 10, 20]

    decay = {}

    for horizon in horizons:
        # Calculate forward returns
        fwd_return = signal_df[price_col].pct_change(horizon).shift(-horizon)

        ic, _ = calculate_ic(
            signal_df[signal_col].values,
            fwd_return.values,
            method=method
        )
        decay[horizon] = ic

    return decay


def calculate_conditional_returns(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    threshold_std: float = 1.0
) -> Tuple[float, float, int, int]:
    """
    Calculate mean returns conditional on signal strength.

    Args:
        signal: Signal values
        forward_returns: Forward returns
        threshold_std: Threshold in standard deviations

    Returns:
        Tuple of (mean_return_positive, mean_return_negative, n_positive, n_negative)
    """
    mask = ~(np.isnan(signal) | np.isnan(forward_returns))
    signal_clean = signal[mask]
    returns_clean = forward_returns[mask]

    if len(signal_clean) < 30:
        return np.nan, np.nan, 0, 0

    threshold = np.nanstd(signal_clean) * threshold_std

    pos_mask = signal_clean > threshold
    neg_mask = signal_clean < -threshold

    mean_pos = float(returns_clean[pos_mask].mean()) if pos_mask.sum() > 10 else np.nan
    mean_neg = float(returns_clean[neg_mask].mean()) if neg_mask.sum() > 10 else np.nan

    return mean_pos, mean_neg, int(pos_mask.sum()), int(neg_mask.sum())


def calculate_ic_stability(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    n_subperiods: int = 4,
    method: str = 'pearson'
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate IC stability across subperiods.

    Lower stability (std) is better - more consistent IC.

    Args:
        signal: Signal values
        forward_returns: Forward returns
        n_subperiods: Number of subperiods to split into

    Returns:
        Tuple of (ic_std, {subperiod_name: ic_value})
    """
    mask = ~(np.isnan(signal) | np.isnan(forward_returns))
    signal_clean = signal[mask]
    returns_clean = forward_returns[mask]

    n = len(signal_clean)
    if n < n_subperiods * 30:
        return np.nan, {}

    period_size = n // n_subperiods
    subperiod_ics = {}

    for i in range(n_subperiods):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < n_subperiods - 1 else n

        signal_sub = signal_clean[start_idx:end_idx]
        returns_sub = returns_clean[start_idx:end_idx]

        ic, _ = calculate_ic(signal_sub, returns_sub, method=method)

        # Use period number as key
        period_name = f"period_{i+1}"
        subperiod_ics[period_name] = ic

    # Calculate stability (std of ICs)
    ic_values = [v for v in subperiod_ics.values() if not np.isnan(v)]
    ic_std = float(np.std(ic_values)) if len(ic_values) > 1 else np.nan

    return ic_std, subperiod_ics


def calculate_ic_stability_from_df(
    signal_df: pd.DataFrame,
    signal_col: str,
    return_col: str,
    n_subperiods: int = 4,
    method: str = 'pearson'
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate IC stability across subperiods from DataFrame.

    Uses actual dates for subperiod labels.

    Args:
        signal_df: DataFrame with DatetimeIndex
        signal_col: Signal column name
        return_col: Return column name
        n_subperiods: Number of subperiods

    Returns:
        Tuple of (ic_std, {date_range: ic_value})
    """
    df = signal_df.dropna(subset=[signal_col, return_col])
    n = len(df)

    if n < n_subperiods * 30:
        return np.nan, {}

    period_size = n // n_subperiods
    subperiod_ics = {}

    for i in range(n_subperiods):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < n_subperiods - 1 else n

        subset = df.iloc[start_idx:end_idx]

        ic, _ = calculate_ic(
            subset[signal_col].values,
            subset[return_col].values,
            method=method
        )

        # Use date range as key
        start_date = subset.index[0].strftime('%Y-%m')
        end_date = subset.index[-1].strftime('%Y-%m')
        period_name = f"{start_date}_to_{end_date}"
        subperiod_ics[period_name] = ic

    ic_values = [v for v in subperiod_ics.values() if not np.isnan(v)]
    ic_std = float(np.std(ic_values)) if len(ic_values) > 1 else np.nan

    return ic_std, subperiod_ics


def calculate_ic_with_se(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    n_bootstrap: int = 1000,
    method: str = 'pearson',
    random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Calculate IC with bootstrap standard error.

    Args:
        signal: Signal values
        forward_returns: Forward returns
        n_bootstrap: Number of bootstrap samples
        method: 'pearson' or 'spearman'
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (ic, ic_se, p_value)
    """
    mask = ~(np.isnan(signal) | np.isnan(forward_returns))
    signal_clean = signal[mask]
    returns_clean = forward_returns[mask]

    n = len(signal_clean)
    if n < 30:
        return np.nan, np.nan, np.nan

    # Calculate point estimate
    ic, p_value = calculate_ic(signal_clean, returns_clean, method=method)

    # Bootstrap for SE
    rng = np.random.default_rng(random_state)
    bootstrap_ics = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_ic, _ = calculate_ic(
            signal_clean[idx],
            returns_clean[idx],
            method=method
        )
        if not np.isnan(boot_ic):
            bootstrap_ics.append(boot_ic)

    ic_se = float(np.std(bootstrap_ics)) if len(bootstrap_ics) > 10 else np.nan

    return ic, ic_se, p_value


def calculate_n_effective(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    max_lag: int = 20
) -> int:
    """
    Calculate autocorrelation-adjusted effective sample size.

    Uses the formula: n_eff = n / (1 + 2 * sum(autocorr_lags))

    Args:
        signal: Signal values
        forward_returns: Forward returns
        max_lag: Maximum lag for autocorrelation calculation

    Returns:
        Effective sample size (int)
    """
    mask = ~(np.isnan(signal) | np.isnan(forward_returns))
    n = mask.sum()

    if n < max_lag * 2:
        return n

    signal_clean = signal[mask]

    # Calculate autocorrelation of signal
    # Using numpy correlate for efficiency
    signal_centered = signal_clean - np.mean(signal_clean)
    autocorr = np.correlate(signal_centered, signal_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
    autocorr = autocorr / autocorr[0]  # Normalize

    # Sum of positive autocorrelations up to max_lag
    # Stop at first negative autocorrelation (Geyer's adaptive method)
    sum_autocorr = 0.0
    for lag in range(1, min(max_lag + 1, len(autocorr))):
        if autocorr[lag] < 0:
            break
        sum_autocorr += autocorr[lag]

    # Effective sample size
    n_eff = n / (1 + 2 * sum_autocorr)

    return max(1, int(n_eff))


def detect_direction(
    ic: float,
    conditional_return_pos: float,
    conditional_return_neg: float,
    ic_threshold: float = 0.015
) -> str:
    """
    Detect whether signal is procyclical or contracyclical.

    Args:
        ic: Information coefficient
        conditional_return_pos: Mean return when signal > threshold
        conditional_return_neg: Mean return when signal < -threshold
        ic_threshold: Minimum IC to declare direction

    Returns:
        'procyc', 'contracyc', or 'none'
    """
    if abs(ic) < ic_threshold:
        return 'none'

    if np.isnan(conditional_return_pos) or np.isnan(conditional_return_neg):
        # Fall back to IC sign
        return 'procyc' if ic > 0 else 'contracyc'

    # Procyclical: high signal → positive return
    # Contracyclical: high signal → negative return (mean reversion)
    if ic > 0 and conditional_return_pos > 0:
        return 'procyc'
    elif ic < 0 and conditional_return_pos < 0:
        return 'contracyc'
    elif ic > 0 and conditional_return_pos < 0:
        # Unusual: positive IC but negative conditional return
        # This suggests noisy signal, fall back to IC
        return 'procyc'
    else:
        return 'contracyc'


def calculate_regime_ic(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    regime_labels: np.ndarray,
    method: str = 'pearson'
) -> Dict[str, float]:
    """
    Calculate IC conditional on regime.

    Args:
        signal: Signal values
        forward_returns: Forward returns
        regime_labels: Regime label for each observation
        method: 'pearson' or 'spearman'

    Returns:
        Dict mapping regime name to IC
    """
    mask = ~(
        np.isnan(signal) |
        np.isnan(forward_returns) |
        pd.isna(regime_labels)
    )

    signal_clean = signal[mask]
    returns_clean = forward_returns[mask]
    regimes_clean = regime_labels[mask]

    unique_regimes = np.unique(regimes_clean)
    regime_ics = {}

    for regime in unique_regimes:
        regime_mask = regimes_clean == regime
        if regime_mask.sum() < 30:
            regime_ics[str(regime)] = np.nan
            continue

        ic, _ = calculate_ic(
            signal_clean[regime_mask],
            returns_clean[regime_mask],
            method=method
        )
        regime_ics[str(regime)] = ic

    return regime_ics
