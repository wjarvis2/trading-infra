"""
Roll detection enricher.

Detects when positions need to roll due to approaching contract expiry.

"""

import pandas as pd
import numpy as np
from typing import Callable, Optional


def create_roll_detection_enricher(
    days_to_expiry_col: str = "days_to_expiry",
    roll_trigger_days: int = 4,
    output_prefix: str = "roll",
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Create a roll detection enricher function.

    Adds columns indicating when contracts are approaching expiry
    and rolls should be triggered.

    Parameters
    ----------
    days_to_expiry_col : str
        Column containing days to expiry
    roll_trigger_days : int
        Days before expiry to trigger roll (default 4)
    output_prefix : str
        Prefix for output columns

    Returns
    -------
    callable
        Enricher function: DataFrame -> DataFrame

    Output Columns
    --------------
    - {prefix}_window: bool, True if within roll trigger window
    - {prefix}_urgency: int, 0=no roll, 1=soon, 2=urgent, 3=critical
    """
    def enricher(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Check if expiry column exists
        if days_to_expiry_col not in df.columns:
            # No expiry data - add default columns
            df[f"{output_prefix}_window"] = False
            df[f"{output_prefix}_urgency"] = 0
            return df

        dte = df[days_to_expiry_col]

        # Roll window indicator
        df[f"{output_prefix}_window"] = dte <= roll_trigger_days

        # Roll urgency levels
        # 0 = no roll needed (dte > trigger)
        # 1 = soon (trigger < dte <= trigger * 1.5)
        # 2 = urgent (trigger * 0.5 < dte <= trigger)
        # 3 = critical (dte <= trigger * 0.5)
        conditions = [
            dte > roll_trigger_days * 1.5,
            dte > roll_trigger_days,
            dte > roll_trigger_days * 0.5,
            dte <= roll_trigger_days * 0.5,
        ]
        choices = [0, 1, 2, 3]
        df[f"{output_prefix}_urgency"] = np.select(
            conditions,
            choices,
            default=0
        )

        return df

    return enricher


def create_contract_change_detector(
    symbol_col: str = "front_symbol",
    group_by: str = "spread_id",
    output_col: str = "contract_changed",
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Create an enricher that detects when the underlying contract changes.

    This is useful for detecting term structure rolls (when CL1 changes
    from CLV24 to CLX24).

    Parameters
    ----------
    symbol_col : str
        Column containing contract symbol
    group_by : str
        Column to group by (e.g., spread_id)
    output_col : str
        Output column name

    Returns
    -------
    callable
        Enricher function: DataFrame -> DataFrame
    """
    def enricher(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if symbol_col not in df.columns:
            df[output_col] = False
            return df

        if group_by and group_by in df.columns:
            # Sort by group and timestamp
            df = df.sort_values([group_by, 'ts'])

            # Detect changes within each group
            def detect_change(group):
                return group[symbol_col] != group[symbol_col].shift(1)

            df[output_col] = df.groupby(group_by, group_keys=False).apply(detect_change)
        else:
            df = df.sort_values('ts')
            df[output_col] = df[symbol_col] != df[symbol_col].shift(1)

        # First row of each group shouldn't be marked as changed
        if group_by and group_by in df.columns:
            first_rows = df.groupby(group_by).head(1).index
            df.loc[first_rows, output_col] = False
        else:
            df.loc[df.index[0], output_col] = False

        return df

    return enricher
