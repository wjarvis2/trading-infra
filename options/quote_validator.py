"""
Quote Validator for institutional-grade options data quality.

Implements hard rules for quote state classification and quality flag
assignment. No vibes - all thresholds are explicit and configurable.

Key principles:
- Hard rules for LIVE_TRADABLE vs INDICATIVE vs MODEL_ONLY
- Tiered delta tolerance based on data source
- Sanity bounds for delta and IV values
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from ib_insync import Ticker

from .quality_enums import DeltaSource, QuoteState, QualityFlag

logger = logging.getLogger(__name__)


class QuoteValidator:
    """
    Validates quote data and assigns quality states.

    Usage:
        validator = QuoteValidator()

        # Validate a ticker
        quote_state = validator.compute_quote_state(ticker)

        # Compute quality flag for a slot
        quality_flag = validator.compute_quality_flag(
            delta_source=DeltaSource.TIER_1_MARKET,
            target_delta=0.40,
            observed_delta=0.38,
            iv=0.25
        )
    """

    # =====================================================================
    # Spread thresholds (both must pass for LIVE_TRADABLE)
    # =====================================================================
    MAX_SPREAD_PCT = 0.10           # 10% relative spread
    MAX_SPREAD_ABS = 0.50           # $0.50 absolute backstop

    # =====================================================================
    # Quote age thresholds
    # =====================================================================
    MAX_AGE_LIVE_SEC = 15           # LIVE_TRADABLE requires < 15s
    MAX_AGE_INDICATIVE_SEC = 30     # INDICATIVE allows < 30s

    # =====================================================================
    # Price validation
    # =====================================================================
    MIN_BID = 0.01                  # Minimum valid bid price
    MIN_ASK = 0.01                  # Minimum valid ask price

    # =====================================================================
    # Delta/IV sanity bounds
    # =====================================================================
    DELTA_MIN = -1.0
    DELTA_MAX = 1.0
    IV_MIN = 0.01                   # 1% annualized vol minimum
    IV_MAX = 5.0                    # 500% annualized vol maximum

    # =====================================================================
    # Delta tolerance by tier (for quality flag assignment)
    # =====================================================================
    # TIER_1 (execution): tighter tolerance
    TIER_1_OK_DRIFT = 0.05          # |Δ_obs - Δ_tgt| <= 0.05 → OK
    TIER_1_DEGRADED_DRIFT = 0.10    # |Δ_obs - Δ_tgt| <= 0.10 → DEGRADED

    # TIER_2/3 (analytics): looser tolerance
    TIER_23_OK_DRIFT = 0.10         # |Δ_obs - Δ_tgt| <= 0.10 → OK
    TIER_23_DEGRADED_DRIFT = 0.15   # |Δ_obs - Δ_tgt| <= 0.15 → DEGRADED

    def compute_quote_state(
        self,
        ticker: Ticker,
        reference_time: Optional[datetime] = None
    ) -> QuoteState:
        """
        Compute quote state from ticker data using hard rules.

        Args:
            ticker: IBKR ticker with quote data
            reference_time: Time to compute age against (default: now)

        Returns:
            QuoteState indicating data quality tier
        """
        reference_time = reference_time or datetime.now()

        # Check for halted trading first (highest priority)
        # Note: ticker.halted is nan when not halted, and bool(nan)==True, so check explicitly
        if hasattr(ticker, 'halted') and ticker.halted is True:
            return QuoteState.HALTED

        # Extract bid/ask
        bid = ticker.bid if ticker.bid and ticker.bid > 0 else None
        ask = ticker.ask if ticker.ask and ticker.ask > 0 else None

        # Check for crossed market (bid > ask) - unusual but happens
        if bid and ask and bid > ask:
            return QuoteState.CROSSED

        # No valid bid/ask → check for modelGreeks
        if not bid or not ask:
            if ticker.modelGreeks and ticker.modelGreeks.delta is not None:
                return QuoteState.MODEL_ONLY
            return QuoteState.MISSING

        # Validate minimum prices
        if bid < self.MIN_BID or ask < self.MIN_ASK:
            if ticker.modelGreeks and ticker.modelGreeks.delta is not None:
                return QuoteState.MODEL_ONLY
            return QuoteState.MISSING

        # Compute spread metrics
        mid = (bid + ask) / 2
        spread_abs = ask - bid
        spread_pct = spread_abs / mid if mid > 0 else float('inf')

        # Compute quote age
        quote_time = getattr(ticker, 'time', None)
        if quote_time:
            # Handle timezone-aware vs timezone-naive datetime comparison
            try:
                if quote_time.tzinfo is not None and reference_time.tzinfo is None:
                    # quote_time is tz-aware, reference_time is naive: convert quote_time to naive
                    quote_time_naive = quote_time.replace(tzinfo=None)
                    age_seconds = (reference_time - quote_time_naive).total_seconds()
                elif quote_time.tzinfo is None and reference_time.tzinfo is not None:
                    # quote_time is naive, reference_time is tz-aware: convert reference_time to naive
                    reference_naive = reference_time.replace(tzinfo=None)
                    age_seconds = (reference_naive - quote_time).total_seconds()
                else:
                    # Both have same tzinfo state
                    age_seconds = (reference_time - quote_time).total_seconds()
            except Exception:
                # On any datetime comparison error, treat as stale
                age_seconds = float('inf')
        else:
            # No timestamp → treat as stale
            age_seconds = float('inf')

        # LIVE_TRADABLE: tight spread AND fresh quote
        if (spread_pct <= self.MAX_SPREAD_PCT and
            spread_abs <= self.MAX_SPREAD_ABS and
            age_seconds <= self.MAX_AGE_LIVE_SEC):
            return QuoteState.LIVE_TRADABLE

        # INDICATIVE: has quotes but fails one constraint
        if age_seconds <= self.MAX_AGE_INDICATIVE_SEC:
            return QuoteState.INDICATIVE

        # Too stale → downgrade to MODEL_ONLY if available
        if ticker.modelGreeks and ticker.modelGreeks.delta is not None:
            return QuoteState.MODEL_ONLY

        return QuoteState.MISSING

    def compute_quality_flag(
        self,
        delta_source: DeltaSource,
        target_delta: float,
        observed_delta: Optional[float],
        iv: Optional[float]
    ) -> QualityFlag:
        """
        Compute quality flag based on delta drift and sanity bounds.

        Args:
            delta_source: Source tier for the delta calculation
            target_delta: Target delta we wanted
            observed_delta: Delta we actually got
            iv: Implied volatility

        Returns:
            QualityFlag indicating overall data quality
        """
        # Missing data → INVALID
        if observed_delta is None:
            return QualityFlag.INVALID

        # Sanity check delta bounds
        if not (self.DELTA_MIN <= observed_delta <= self.DELTA_MAX):
            logger.warning(f"Delta {observed_delta} outside bounds [{self.DELTA_MIN}, {self.DELTA_MAX}]")
            return QualityFlag.INVALID

        # Sanity check IV bounds (if provided)
        if iv is not None:
            if not (self.IV_MIN <= iv <= self.IV_MAX):
                logger.warning(f"IV {iv} outside bounds [{self.IV_MIN}, {self.IV_MAX}]")
                return QualityFlag.INVALID

        # Compute delta drift
        drift = abs(abs(observed_delta) - target_delta)

        # Apply tiered tolerances
        if delta_source == DeltaSource.TIER_1_MARKET:
            if drift <= self.TIER_1_OK_DRIFT:
                return QualityFlag.OK
            elif drift <= self.TIER_1_DEGRADED_DRIFT:
                return QualityFlag.DEGRADED
            else:
                return QualityFlag.INVALID
        else:
            # TIER_2_SURFACE or TIER_3_IBKR
            if drift <= self.TIER_23_OK_DRIFT:
                return QualityFlag.OK
            elif drift <= self.TIER_23_DEGRADED_DRIFT:
                return QualityFlag.DEGRADED
            else:
                return QualityFlag.INVALID

    def compute_spread_pct(self, bid: float, ask: float) -> Optional[float]:
        """Compute spread percentage from bid/ask."""
        if bid <= 0 or ask <= 0 or ask < bid:
            return None
        mid = (bid + ask) / 2
        if mid <= 0:
            return None
        return (ask - bid) / mid

    def is_execution_eligible(
        self,
        delta_source: DeltaSource,
        quote_state: QuoteState,
        quality_flag: QualityFlag
    ) -> bool:
        """
        Check if data is eligible for trade execution.

        Requires:
        - TIER_1 market-derived delta
        - LIVE_TRADABLE quote state
        - OK quality flag
        """
        return (
            delta_source == DeltaSource.TIER_1_MARKET and
            quote_state == QuoteState.LIVE_TRADABLE and
            quality_flag == QualityFlag.OK
        )

    def validate_for_assignment(
        self,
        delta_source: DeltaSource,
        target_delta: float,
        observed_delta: float
    ) -> bool:
        """
        Check if a strike is valid for assignment (within tolerance).

        Uses looser tolerances than quality_flag:
        - TIER_1: max 0.10 drift (same as DEGRADED threshold)
        - TIER_2/3: max 0.15 drift (same as DEGRADED threshold)
        """
        drift = abs(abs(observed_delta) - target_delta)

        if delta_source == DeltaSource.TIER_1_MARKET:
            return drift <= self.TIER_1_DEGRADED_DRIFT
        else:
            return drift <= self.TIER_23_DEGRADED_DRIFT
