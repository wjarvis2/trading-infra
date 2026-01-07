"""
Quality Enums for institutional-grade options data.

Two-track delta ontology:
- model_delta: OUR Black-76 calculation (from live quotes or vol surface)
- broker_delta: IBKR's modelGreeks (for comparison/fallback)

These enums are used across multiple modules to track data quality
and provenance. They're defined in a separate module to avoid
circular imports.
"""

from enum import Enum


# =============================================================================
# Model Delta Enums (our calculation)
# =============================================================================

class ModelDeltaSource(str, Enum):
    """Input source for our model delta calculation."""
    QUOTES = "QUOTES"      # Live bid/ask → QuantLib B76 (execution-grade)
    SURFACE = "SURFACE"    # Vol surface interpolation (analytics-only)


class ModelDeltaMethod(str, Enum):
    """Pricing model used for delta calculation."""
    BLACK_76 = "BLACK_76"     # Black-76 for futures options (current)
    SABR = "SABR"             # SABR stochastic vol (future)
    LOCAL_VOL = "LOCAL_VOL"   # Local volatility (future)


class ModelUnderlyingRef(str, Enum):
    """Reference price for underlying futures."""
    FUT_MID = "FUT_MID"       # Realtime mid price (for live greeks)
    FUT_MARK = "FUT_MARK"     # Exchange mark price
    FUT_SETTLE = "FUT_SETTLE" # EOD settlement (for reconciliation)


# =============================================================================
# Quote Quality Enums
# =============================================================================

class QuoteState(str, Enum):
    """State of the quote data."""
    LIVE_TRADABLE = "LIVE_TRADABLE"    # Valid bid/ask spread, <15s age, execution-eligible
    INDICATIVE = "INDICATIVE"          # Has quotes but may be stale (15-30s) or wide
    HALTED = "HALTED"                  # Trading halted (ticker.halted = True)
    CROSSED = "CROSSED"                # Crossed market (bid > ask)
    MODEL_ONLY = "MODEL_ONLY"          # Only IBKR modelGreeks available, no live quotes
    MISSING = "MISSING"                # No data available


class QualityFlag(str, Enum):
    """Overall quality assessment for downstream use."""
    OK = "OK"                          # Suitable for trading and ML features
    DEGRADED = "DEGRADED"              # Use with caution, analytics only
    INVALID = "INVALID"                # Do not use


# =============================================================================
# Legacy Aliases (backward compatibility)
# =============================================================================

class DeltaSource(str, Enum):
    """
    DEPRECATED: Use ModelDeltaSource + broker_delta instead.

    Legacy three-tier enum kept for backward compatibility during migration.
    Maps to new ontology:
    - TIER_1_MARKET → ModelDeltaSource.QUOTES (execution-grade)
    - TIER_2_SURFACE → ModelDeltaSource.SURFACE (analytics)
    - TIER_3_IBKR → broker_delta field (separate track)
    """
    TIER_1_MARKET = "TIER_1_MARKET"    # → ModelDeltaSource.QUOTES
    TIER_2_SURFACE = "TIER_2_SURFACE"  # → ModelDeltaSource.SURFACE
    TIER_3_IBKR = "TIER_3_IBKR"        # → broker_delta (separate field)
