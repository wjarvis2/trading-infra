"""
Rolling statistics enrichers.

Pure functions that compute rolling statistics like z-score, percentile, momentum.

"""

import pandas as pd
import numpy as np
from typing import Callable, List, Optional


def create_zscore_enricher(
    input_col: str = "close",  # Canonical: close
    output_col: str = "z_score",
    window: int = 60,
    min_periods: Optional[int] = None,
    group_by: Optional[str] = "instrument_id",  # Canonical: instrument_id
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Create a z-score enricher function.

    The enricher computes rolling z-score:
    z = (x - rolling_mean) / rolling_std

    Parameters
    ----------
    input_col : str
        Column to compute z-score for
    output_col : str
        Output column name
    window : int
        Rolling window size
    min_periods : int, optional
        Minimum observations required (default: window // 2)
    group_by : str, optional
        Column to group by before computing rolling stats

    Returns
    -------
    callable
        Enricher function: DataFrame -> DataFrame
    """
    min_periods = min_periods or window // 2

    def enricher(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if group_by and group_by in df.columns:
            # Need to sort by timestamp within each group
            df = df.sort_values([group_by, 'ts']).reset_index(drop=True)

            # Initialize output columns (canonical names)
            df[output_col] = np.nan
            df["rolling_mean"] = np.nan  # Canonical: rolling_mean
            df["rolling_std"] = np.nan   # Canonical: rolling_std

            # Compute per group using iterative approach for pandas compatibility
            for name, group in df.groupby(group_by):
                idx = group.index
                rolling = group[input_col].rolling(window=window, min_periods=min_periods)
                mean = rolling.mean()
                std = rolling.std()
                zscore = (group[input_col] - mean) / std

                df.loc[idx, output_col] = zscore.values
                df.loc[idx, "rolling_mean"] = mean.values
                df.loc[idx, "rolling_std"] = std.values

        else:
            # Global computation
            df = df.sort_values('ts').reset_index(drop=True)
            rolling = df[input_col].rolling(window=window, min_periods=min_periods)
            mean = rolling.mean()
            std = rolling.std()
            df[output_col] = (df[input_col] - mean) / std
            df["rolling_mean"] = mean  # Canonical: rolling_mean
            df["rolling_std"] = std    # Canonical: rolling_std

        return df

    return enricher


def create_percentile_enricher(
    input_col: str = "close",  # Canonical: close
    output_col: str = "pct_rank",  # Canonical: pct_rank
    window: int = 252,
    min_periods: Optional[int] = None,
    group_by: Optional[str] = "instrument_id",  # Canonical: instrument_id
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Create a percentile rank enricher function.

    Computes rolling percentile rank (0-100).

    Parameters
    ----------
    input_col : str
        Column to compute percentile for
    output_col : str
        Output column name
    window : int
        Rolling window size
    min_periods : int, optional
        Minimum observations required (default: window // 2)
    group_by : str, optional
        Column to group by before computing

    Returns
    -------
    callable
        Enricher function: DataFrame -> DataFrame
    """
    min_periods = min_periods or window // 2

    def _rolling_percentile(x):
        """Compute percentile rank of last value in window."""
        if len(x) < 2:
            return 50.0
        return (x.rank(pct=True).iloc[-1]) * 100

    def enricher(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if group_by and group_by in df.columns:
            df = df.sort_values([group_by, 'ts']).reset_index(drop=True)
            df[output_col] = np.nan

            # Compute per group using iterative approach for pandas compatibility
            for name, group in df.groupby(group_by):
                idx = group.index
                pct = group[input_col].rolling(
                    window=window, min_periods=min_periods
                ).apply(_rolling_percentile, raw=False)
                df.loc[idx, output_col] = pct.values

        else:
            df = df.sort_values('ts').reset_index(drop=True)
            df[output_col] = df[input_col].rolling(
                window=window, min_periods=min_periods
            ).apply(_rolling_percentile, raw=False)

        return df

    return enricher


def create_momentum_enricher(
    input_col: str = "close",  # Canonical: close
    output_col: str = "mom",   # Canonical: mom
    lookback: int = 10,
    group_by: Optional[str] = "instrument_id",  # Canonical: instrument_id
    also_percentile: bool = True,
    percentile_window: int = 252,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Create a momentum enricher function.

    Computes N-day change and optionally its percentile rank.

    Parameters
    ----------
    input_col : str
        Column to compute momentum for
    output_col : str
        Output column name
    lookback : int
        Lookback period for change calculation
    group_by : str, optional
        Column to group by
    also_percentile : bool
        Also compute percentile rank of momentum
    percentile_window : int
        Window for percentile calculation

    Returns
    -------
    callable
        Enricher function: DataFrame -> DataFrame
    """
    def _rolling_percentile(x):
        if len(x) < 2:
            return 50.0
        return (x.rank(pct=True).iloc[-1]) * 100

    def enricher(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if group_by and group_by in df.columns:
            df = df.sort_values([group_by, 'ts']).reset_index(drop=True)
            df[output_col] = np.nan
            if also_percentile:
                df["mom_pct"] = np.nan  # Canonical: mom_pct

            # Compute per group using iterative approach for pandas compatibility
            for name, group in df.groupby(group_by):
                idx = group.index
                mom = group[input_col].diff(lookback)
                df.loc[idx, output_col] = mom.values

                if also_percentile:
                    pct = mom.rolling(
                        window=percentile_window, min_periods=lookback
                    ).apply(_rolling_percentile, raw=False)
                    df.loc[idx, "mom_pct"] = pct.values  # Canonical: mom_pct

        else:
            df = df.sort_values('ts').reset_index(drop=True)
            df[output_col] = df[input_col].diff(lookback)

            if also_percentile:
                df["mom_pct"] = df[output_col].rolling(  # Canonical: mom_pct
                    window=percentile_window, min_periods=lookback
                ).apply(_rolling_percentile, raw=False)

        return df

    return enricher
