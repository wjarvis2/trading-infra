"""
Enrichers - pure transformation functions for feature engineering.

Each enricher:
- Takes a DataFrame
- Returns an enriched DataFrame with additional columns
- Is stateless and deterministic
- Does no I/O

"""

from .rolling_stats import (
    create_zscore_enricher,
    create_percentile_enricher,
    create_momentum_enricher,
)
from .roll_detection import create_roll_detection_enricher

__all__ = [
    "create_zscore_enricher",
    "create_percentile_enricher",
    "create_momentum_enricher",
    "create_roll_detection_enricher",
]
