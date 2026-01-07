"""
Forward return calculation utilities.

Pure functions for computing forward returns from price data.
Supports multiple target types: forward_return, forward_change, carry_return.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
from enum import Enum


class ReturnType(Enum):
    """Type of return to calculate."""
    SIMPLE = "simple"           # (P_t+h - P_t) / P_t
    LOG = "log"                 # ln(P_t+h / P_t)
    ARITHMETIC = "arithmetic"   # P_t+h - P_t (for spreads)


def calculate_forward_returns(
    prices: Union[np.ndarray, pd.Series],
    horizon: int,
    return_type: ReturnType = ReturnType.SIMPLE
) -> np.ndarray:
    """
    Calculate forward returns at a given horizon.

    Args:
        prices: Price series
        horizon: Forward horizon (number of periods)
        return_type: Type of return calculation

    Returns:
        Array of forward returns, aligned with input (NaN for last `horizon` values)
    """
    if isinstance(prices, pd.Series):
        prices = prices.values

    n = len(prices)
    if n <= horizon:
        return np.full(n, np.nan)

    fwd_returns = np.full(n, np.nan)

    if return_type == ReturnType.SIMPLE:
        # Simple return: (P_t+h - P_t) / P_t
        fwd_returns[:-horizon] = (prices[horizon:] - prices[:-horizon]) / prices[:-horizon]
    elif return_type == ReturnType.LOG:
        # Log return: ln(P_t+h / P_t)
        with np.errstate(divide='ignore', invalid='ignore'):
            fwd_returns[:-horizon] = np.log(prices[horizon:] / prices[:-horizon])
    elif return_type == ReturnType.ARITHMETIC:
        # Arithmetic difference: P_t+h - P_t (useful for spreads)
        fwd_returns[:-horizon] = prices[horizon:] - prices[:-horizon]

    return fwd_returns


def calculate_forward_change(
    signal: Union[np.ndarray, pd.Series],
    horizon: int
) -> np.ndarray:
    """
    Calculate forward change in signal (shock response).

    Used for testing whether current signal level predicts
    future signal changes (mean reversion in the signal itself).

    Args:
        signal: Signal series
        horizon: Forward horizon

    Returns:
        Array of forward changes: signal_t+h - signal_t
    """
    if isinstance(signal, pd.Series):
        signal = signal.values

    n = len(signal)
    if n <= horizon:
        return np.full(n, np.nan)

    fwd_change = np.full(n, np.nan)
    fwd_change[:-horizon] = signal[horizon:] - signal[:-horizon]

    return fwd_change


def calculate_carry_return(
    front_prices: Union[np.ndarray, pd.Series],
    back_prices: Union[np.ndarray, pd.Series],
    front_dte: Union[np.ndarray, pd.Series],
    back_dte: Union[np.ndarray, pd.Series],
    horizon: int
) -> np.ndarray:
    """
    Calculate realized carry return over horizon.

    Carry return = actual price appreciation vs predicted carry.
    Positive carry return means market moved more than carry implied.

    Args:
        front_prices: Front month prices
        back_prices: Back month prices
        front_dte: Days to expiry for front month
        back_dte: Days to expiry for back month
        horizon: Forward horizon

    Returns:
        Array of carry returns
    """
    if isinstance(front_prices, pd.Series):
        front_prices = front_prices.values
    if isinstance(back_prices, pd.Series):
        back_prices = back_prices.values
    if isinstance(front_dte, pd.Series):
        front_dte = front_dte.values
    if isinstance(back_dte, pd.Series):
        back_dte = back_dte.values

    n = len(front_prices)
    if n <= horizon:
        return np.full(n, np.nan)

    # Calculate annualized carry rate
    with np.errstate(divide='ignore', invalid='ignore'):
        carry_rate = np.log(back_prices / front_prices) / ((back_dte - front_dte) / 365.0)

    # Predicted price change from carry (over horizon days)
    predicted_change = carry_rate * (horizon / 365.0)

    # Actual log return
    actual_log_return = np.full(n, np.nan)
    actual_log_return[:-horizon] = np.log(front_prices[horizon:] / front_prices[:-horizon])

    # Carry return = actual - predicted
    carry_return = actual_log_return - predicted_change

    return carry_return


def calculate_spread_forward_return(
    spread_prices: Union[np.ndarray, pd.Series],
    horizon: int,
    use_arithmetic: bool = True
) -> np.ndarray:
    """
    Calculate forward returns for a spread.

    For spreads, arithmetic returns are often more appropriate
    because spread values can be negative.

    Args:
        spread_prices: Spread price series
        horizon: Forward horizon
        use_arithmetic: If True, use arithmetic difference; else use simple return

    Returns:
        Array of forward returns
    """
    return_type = ReturnType.ARITHMETIC if use_arithmetic else ReturnType.SIMPLE
    return calculate_forward_returns(spread_prices, horizon, return_type)


def align_signal_returns(
    signal: Union[np.ndarray, pd.Series],
    forward_returns: Union[np.ndarray, pd.Series],
    min_observations: int = 30
) -> tuple:
    """
    Align signal and returns, removing NaN pairs.

    Args:
        signal: Signal values
        forward_returns: Forward return values
        min_observations: Minimum required observations

    Returns:
        Tuple of (signal_clean, returns_clean, n_obs)

    Raises:
        ValueError: If insufficient observations after alignment
    """
    if isinstance(signal, pd.Series):
        signal = signal.values
    if isinstance(forward_returns, pd.Series):
        forward_returns = forward_returns.values

    mask = ~(np.isnan(signal) | np.isnan(forward_returns))
    signal_clean = signal[mask]
    returns_clean = forward_returns[mask]
    n_obs = len(signal_clean)

    if n_obs < min_observations:
        raise ValueError(
            f"Insufficient observations after alignment: {n_obs} < {min_observations}"
        )

    return signal_clean, returns_clean, n_obs


def create_forward_return_df(
    df: pd.DataFrame,
    price_col: str,
    horizons: list,
    return_type: ReturnType = ReturnType.SIMPLE,
    prefix: str = 'fwd_ret'
) -> pd.DataFrame:
    """
    Add forward return columns to DataFrame.

    Args:
        df: Input DataFrame with price column
        price_col: Column name for prices
        horizons: List of horizons to calculate
        return_type: Type of return calculation
        prefix: Prefix for new column names

    Returns:
        DataFrame with added forward return columns
    """
    result = df.copy()

    for horizon in horizons:
        col_name = f"{prefix}_{horizon}d"
        result[col_name] = calculate_forward_returns(
            df[price_col].values,
            horizon=horizon,
            return_type=return_type
        )

    return result
