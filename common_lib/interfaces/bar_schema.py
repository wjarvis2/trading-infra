"""
Canonical Bar and Feature Schema Contract.

Defines the standard field names for bars and features across the system.
All sources normalize to this schema. All strategies consume this schema.

This is the contract between:
- Data Layer (sources) → Feature Layer (enrichers)
- Feature Layer (enrichers) → Signal Layer (strategies)

"""

from dataclasses import dataclass
from typing import FrozenSet, Optional, Set
import pandas as pd


# =============================================================================
# Canonical Field Names
# =============================================================================

# Bar fields (from sources)
BAR_FIELDS = frozenset({
    "ts",              # Timestamp (datetime)
    "instrument_id",   # Instrument identifier (str) - e.g., "CL1_CL2"
    "close",           # Close price (float)
    "open",            # Open price (float) - optional
    "high",            # High price (float) - optional
    "low",             # Low price (float) - optional
    "volume",          # Volume (float)
    "front_close",     # Front leg close (float) - for spreads
    "back_close",      # Back leg close (float) - for spreads
    "front_symbol",    # Front leg contract symbol (str)
    "back_symbol",     # Back leg contract symbol (str)
})

# Feature fields (from enrichers)
FEATURE_FIELDS = frozenset({
    "z_score",         # Rolling z-score (float)
    "rolling_mean",    # Rolling mean used for z-score (float)
    "rolling_std",     # Rolling std used for z-score (float)
    "pct_rank",        # Rolling percentile rank 0-100 (float)
    "mom",             # Momentum / change over lookback (float)
    "mom_pct",         # Momentum percentile rank (float)
    "days_to_expiry",  # Days until front contract expiry (int)
    "roll_window",     # In roll trigger window (bool)
    "roll_urgency",    # Roll urgency level 0-3 (int)
    "front_expiry",    # Front contract expiry date (date)
})

# External data fields (from joined sources)
EXTERNAL_FIELDS = frozenset({
    "mm_net",          # Money manager net position % of OI (float)
    "mm_net_pct",      # Money manager net percentile (float)
    "cot_report_date", # COT report date (date)
})


# =============================================================================
# Required Fields by Layer
# =============================================================================

# Minimum required for a valid bar from sources
REQUIRED_BAR_FIELDS = frozenset({
    "ts",
    "instrument_id",
    "close",
})

# Minimum required for signal generation
REQUIRED_SIGNAL_FIELDS = frozenset({
    "ts",
    "instrument_id",
    "close",
    "z_score",
})


# =============================================================================
# Validation
# =============================================================================

def validate_bar_schema(
    df: pd.DataFrame,
    required: Optional[FrozenSet[str]] = None,
    context: str = "",
) -> None:
    """
    Validate DataFrame has required canonical fields.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required : frozenset[str], optional
        Required fields (default: REQUIRED_BAR_FIELDS)
    context : str
        Context for error messages

    Raises
    ------
    ValueError
        If required fields are missing
    """
    required = required or REQUIRED_BAR_FIELDS
    missing = required - set(df.columns)

    if missing:
        ctx = f" ({context})" if context else ""
        raise ValueError(
            f"Missing required canonical fields{ctx}: {sorted(missing)}. "
            f"Available: {sorted(df.columns)}"
        )


def validate_bar_series(
    bar: pd.Series,
    required: Optional[FrozenSet[str]] = None,
    context: str = "",
) -> None:
    """
    Validate Series has required canonical fields.

    Parameters
    ----------
    bar : pd.Series
        Bar to validate
    required : frozenset[str], optional
        Required fields (default: REQUIRED_SIGNAL_FIELDS)
    context : str
        Context for error messages

    Raises
    ------
    ValueError
        If required fields are missing
    """
    required = required or REQUIRED_SIGNAL_FIELDS
    missing = required - set(bar.index)

    if missing:
        ctx = f" ({context})" if context else ""
        raise ValueError(
            f"Missing required canonical fields{ctx}: {sorted(missing)}. "
            f"Available: {sorted(bar.index)}"
        )


# =============================================================================
# Field Mapping (for source normalization)
# =============================================================================

# Standard mappings from common source formats to canonical names
# Sources should apply these ONCE at normalization
SOURCE_TO_CANONICAL = {
    # SpreadSource mappings
    "spread_price": "close",
    "spread_id": "instrument_id",
    # Enricher mappings
    "z_score_mean": "rolling_mean",
    "z_score_std": "rolling_std",
    "spread_pct": "pct_rank",
    "spread_mom": "mom",
    "spread_mom_pct": "mom_pct",
    # COT mappings
    "report_date": "cot_report_date",
    # Contract meta mappings
    "expiry": "front_expiry",
}


def apply_canonical_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply canonical field name mappings to DataFrame.

    Use this at the source normalization boundary.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with source-specific field names

    Returns
    -------
    pd.DataFrame
        DataFrame with canonical field names
    """
    rename_map = {
        old: new
        for old, new in SOURCE_TO_CANONICAL.items()
        if old in df.columns
    }
    return df.rename(columns=rename_map)
