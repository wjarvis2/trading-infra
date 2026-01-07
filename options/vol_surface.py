"""
Vol Surface for institutional-grade IV interpolation.

Thin, gated vol surface that fits a simple spline to liquid strikes.
Outputs are always TIER_2 (analytics only, never execution-eligible).

Key principles:
- Liquidity-gated: only fit when enough liquid strikes available
- Simple interpolation: no SVI/SABR sophistication yet
- Fail closed: return None if gate not met, don't fake confidence
- Ephemeral: rebuilt on each probe, not persisted
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional

import numpy as np
from scipy import interpolate

from .strike_prober import StrikeWithDelta
from .quality_enums import DeltaSource, QuoteState

logger = logging.getLogger(__name__)


@dataclass
class SurfaceFitResult:
    """Result of vol surface fitting."""
    success: bool
    n_points: int
    strike_range: tuple[float, float]  # (min, max)
    iv_range: tuple[float, float]      # (min, max)
    atm_iv: Optional[float]
    residual_std: Optional[float]      # Standard deviation of residuals
    coverage_pct: float                # % of strike range covered
    fit_timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_usable(self) -> bool:
        """Check if fit is usable for interpolation."""
        return self.success and self.n_points >= 5


class VolSurface:
    """
    Vol surface for IV interpolation across strikes.

    Usage:
        surface = VolSurface(expiry='20260215', right='C', atm_strike=75.0)

        # Check if we can fit
        if surface.can_fit(liquid_strikes):
            surface.fit(liquid_strikes)

        # Interpolate IV for a strike
        iv = surface.interpolate_iv(72.5)

        # Get delta for a strike
        delta = surface.interpolate_delta(72.5, underlying_price=75.0)
    """

    # =====================================================================
    # Liquidity gates
    # =====================================================================
    PREFERRED_LIQUID_STRIKES = 7     # Ideal minimum
    MIN_LIQUID_STRIKES = 5           # Absolute minimum
    MIN_STRIKES_EACH_SIDE = 2        # Must have 2+ below AND 2+ above ATM

    # =====================================================================
    # Interpolation settings
    # =====================================================================
    SPLINE_DEGREE = 3                # Cubic spline
    EXTRAPOLATE_BUFFER = 0.10        # Allow 10% extrapolation beyond fitted range

    def __init__(
        self,
        expiry: str,
        right: str,
        atm_strike: float,
        underlying_price: Optional[float] = None
    ):
        """
        Initialize vol surface.

        Args:
            expiry: Option expiry in YYYYMMDD format
            right: 'C' for calls or 'P' for puts
            atm_strike: ATM strike price
            underlying_price: Current underlying price (for delta calculation)
        """
        self.expiry = expiry
        self.right = right
        self.atm_strike = atm_strike
        self.underlying_price = underlying_price

        # Fitted surface (None until fit() called)
        self._spline: Optional[interpolate.UnivariateSpline] = None
        self._strike_min: float = 0.0
        self._strike_max: float = 0.0
        self._fit_result: Optional[SurfaceFitResult] = None

        # Parse expiry for Greeks calculation
        self._expiry_date = date(int(expiry[:4]), int(expiry[4:6]), int(expiry[6:8]))

    def can_fit(self, liquid_strikes: List[StrikeWithDelta]) -> bool:
        """
        Check if liquidity gate is met for surface fitting.

        Requires:
        - At least MIN_LIQUID_STRIKES liquid strikes
        - At least MIN_STRIKES_EACH_SIDE below ATM
        - At least MIN_STRIKES_EACH_SIDE above ATM
        """
        if not liquid_strikes:
            return False

        n = len(liquid_strikes)
        if n < self.MIN_LIQUID_STRIKES:
            logger.debug(f"Vol surface gate: only {n} strikes, need {self.MIN_LIQUID_STRIKES}")
            return False

        below_atm = sum(1 for s in liquid_strikes if s.strike < self.atm_strike)
        above_atm = sum(1 for s in liquid_strikes if s.strike > self.atm_strike)

        if below_atm < self.MIN_STRIKES_EACH_SIDE:
            logger.debug(f"Vol surface gate: only {below_atm} strikes below ATM, need {self.MIN_STRIKES_EACH_SIDE}")
            return False

        if above_atm < self.MIN_STRIKES_EACH_SIDE:
            logger.debug(f"Vol surface gate: only {above_atm} strikes above ATM, need {self.MIN_STRIKES_EACH_SIDE}")
            return False

        return True

    def fit(self, liquid_strikes: List[StrikeWithDelta]) -> SurfaceFitResult:
        """
        Fit spline to liquid strikes' implied volatilities.

        Args:
            liquid_strikes: List of strikes with valid IV data

        Returns:
            SurfaceFitResult with fit quality metrics
        """
        if not self.can_fit(liquid_strikes):
            self._fit_result = SurfaceFitResult(
                success=False,
                n_points=len(liquid_strikes),
                strike_range=(0, 0),
                iv_range=(0, 0),
                atm_iv=None,
                residual_std=None,
                coverage_pct=0.0,
            )
            return self._fit_result

        # Extract strike/IV pairs, filtering for valid IVs
        valid_points = [
            (s.strike, s.iv)
            for s in liquid_strikes
            if s.iv is not None and s.iv > 0.01 and s.iv < 5.0
        ]

        if len(valid_points) < self.MIN_LIQUID_STRIKES:
            logger.warning(f"Only {len(valid_points)} valid IV points after filtering")
            self._fit_result = SurfaceFitResult(
                success=False,
                n_points=len(valid_points),
                strike_range=(0, 0),
                iv_range=(0, 0),
                atm_iv=None,
                residual_std=None,
                coverage_pct=0.0,
            )
            return self._fit_result

        # Sort by strike
        valid_points.sort(key=lambda x: x[0])
        strikes = np.array([p[0] for p in valid_points])
        ivs = np.array([p[1] for p in valid_points])

        # Record strike range
        self._strike_min = float(strikes.min())
        self._strike_max = float(strikes.max())

        try:
            # Fit cubic spline with smoothing
            # s=0 would be exact interpolation; small s allows smoothing
            smoothing = len(strikes) * 0.001  # Very light smoothing
            self._spline = interpolate.UnivariateSpline(
                strikes, ivs, k=min(self.SPLINE_DEGREE, len(strikes) - 1), s=smoothing
            )

            # Compute fit quality metrics
            fitted_ivs = self._spline(strikes)
            residuals = ivs - fitted_ivs
            residual_std = float(np.std(residuals))

            # Interpolate ATM IV
            atm_iv = None
            if self._strike_min <= self.atm_strike <= self._strike_max:
                atm_iv = float(self._spline(self.atm_strike))

            # Compute coverage (what % of typical strike range do we cover)
            # Typical range: ATM ± 20% (so 40% total range)
            typical_range = self.atm_strike * 0.40
            actual_range = self._strike_max - self._strike_min
            coverage_pct = min(1.0, actual_range / typical_range)

            self._fit_result = SurfaceFitResult(
                success=True,
                n_points=len(strikes),
                strike_range=(self._strike_min, self._strike_max),
                iv_range=(float(ivs.min()), float(ivs.max())),
                atm_iv=atm_iv,
                residual_std=residual_std,
                coverage_pct=coverage_pct,
            )

            atm_iv_str = f"{atm_iv:.3f}" if atm_iv is not None else "N/A"
            logger.info(
                f"Vol surface fit: {len(strikes)} points, "
                f"strikes [{self._strike_min:.1f}, {self._strike_max:.1f}], "
                f"ATM IV={atm_iv_str}, "
                f"coverage={coverage_pct:.1%}"
            )

        except Exception as e:
            logger.error(f"Vol surface fit failed: {e}")
            self._spline = None
            self._fit_result = SurfaceFitResult(
                success=False,
                n_points=len(strikes),
                strike_range=(self._strike_min, self._strike_max),
                iv_range=(float(ivs.min()), float(ivs.max())),
                atm_iv=None,
                residual_std=None,
                coverage_pct=0.0,
            )

        return self._fit_result

    def interpolate_iv(self, strike: float) -> Optional[float]:
        """
        Get IV for arbitrary strike via spline interpolation.

        Returns None if:
        - Surface not fitted
        - Strike outside interpolation range (with buffer)
        """
        if self._spline is None:
            return None

        # Check strike is within range (with small extrapolation buffer)
        buffer = (self._strike_max - self._strike_min) * self.EXTRAPOLATE_BUFFER
        if strike < self._strike_min - buffer or strike > self._strike_max + buffer:
            return None

        try:
            iv = float(self._spline(strike))
            # Sanity check
            if iv <= 0.01 or iv > 5.0:
                return None
            return iv
        except Exception:
            return None

    def interpolate_delta(
        self,
        strike: float,
        underlying_price: Optional[float] = None
    ) -> Optional[float]:
        """
        Get delta for arbitrary strike via interpolated IV + Black-Scholes.

        This is TIER_2 data - never execution-eligible.

        Returns None if:
        - IV interpolation fails
        - Greeks calculation fails
        """
        iv = self.interpolate_iv(strike)
        if iv is None:
            return None

        price = underlying_price or self.underlying_price
        if price is None:
            return None

        try:
            # Use QuantLib for delta calculation with known IV
            from .greeks_engine import GreeksEngine

            engine = GreeksEngine()
            greeks = engine.calc_greeks_from_iv(
                iv=iv,
                underlying_price=price,
                strike=strike,
                expiry=self._expiry_date,
                right=self.right,
            )

            if greeks and greeks.delta is not None:
                return greeks.delta

        except Exception as e:
            logger.debug(f"Delta interpolation failed for strike {strike}: {e}")

        return None

    @property
    def is_fitted(self) -> bool:
        """Check if surface is fitted and usable."""
        return self._spline is not None and self._fit_result is not None and self._fit_result.success

    @property
    def fit_quality(self) -> Optional[SurfaceFitResult]:
        """Get fit quality metrics."""
        return self._fit_result

    def get_liquid_strikes(
        self,
        all_strikes: List[StrikeWithDelta]
    ) -> List[StrikeWithDelta]:
        """
        Filter to liquid strikes only (TIER_1 eligible).

        A strike is liquid if it has:
        - Valid bid/ask (not MODEL_ONLY or MISSING)
        - Valid IV
        """
        return [
            s for s in all_strikes
            if s.is_liquid and s.iv is not None and s.iv > 0.01
        ]


def build_surface_for_expiry(
    probe_result,
    expiry: str,
    right: str,
    atm_price: float
) -> Optional[VolSurface]:
    """
    Convenience function to build vol surface from probe result.

    Returns None if liquidity gate not met.
    """
    candidates = probe_result.calls if right == 'C' else probe_result.puts

    surface = VolSurface(expiry=expiry, right=right, atm_strike=atm_price)
    liquid = surface.get_liquid_strikes(candidates)

    if not surface.can_fit(liquid):
        logger.debug(f"Cannot fit vol surface for {expiry} {right}: insufficient liquidity")
        return None

    surface.fit(liquid)

    if not surface.is_fitted:
        logger.warning(f"Vol surface fit failed for {expiry} {right}")
        return None

    return surface
