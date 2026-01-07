"""
Greeks Engine for energy futures options.

Computes implied volatility and Greeks using QuantLib Black-76 model.
Black-76 is the appropriate model for options on futures (no cost of carry).

Key features:
- IV calculation via Newton-Raphson root finding
- Full Greeks: delta, gamma, theta, vega
- Handles edge cases (deep ITM/OTM, near expiry)
- Batch processing for efficiency
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import QuantLib as ql

logger = logging.getLogger(__name__)


@dataclass
class Greeks:
    """Container for option Greeks."""
    iv: float          # Implied volatility (annualized, e.g., 0.25 = 25%)
    delta: float       # Option delta (-1 to 1)
    gamma: float       # Gamma (sensitivity of delta)
    theta: float       # Theta (time decay per day, negative for long options)
    vega: float        # Vega (sensitivity to 1% IV change)

    def __repr__(self) -> str:
        return (
            f"Greeks(iv={self.iv:.4f}, delta={self.delta:.4f}, "
            f"gamma={self.gamma:.6f}, theta={self.theta:.4f}, vega={self.vega:.4f})"
        )


class GreeksEngine:
    """
    Computes Greeks for futures options using QuantLib Black-76.

    Usage:
        engine = GreeksEngine()

        greeks = engine.calc_greeks(
            option_price=2.50,
            underlying_price=75.00,
            strike=74.00,
            expiry=date(2025, 3, 15),
            right='C'
        )

        if greeks:
            print(f"Delta: {greeks.delta:.4f}")
    """

    # IV solver bounds
    IV_MIN = 0.001    # 0.1% - floor for IV
    IV_MAX = 5.0      # 500% - ceiling for IV
    IV_INITIAL = 0.30 # 30% - starting guess

    # Numerical tolerances
    IV_TOLERANCE = 1e-6
    IV_MAX_ITERATIONS = 100

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize Greeks engine.

        Args:
            risk_free_rate: Risk-free rate for discounting (default 0 for futures)
        """
        self.risk_free_rate = risk_free_rate
        self._calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        self._day_count = ql.Actual365Fixed()

    def calc_greeks(
        self,
        option_price: float,
        underlying_price: float,
        strike: float,
        expiry: date,
        right: str,
        valuation_date: Optional[date] = None
    ) -> Optional[Greeks]:
        """
        Calculate IV and Greeks for a futures option.

        Args:
            option_price: Current option mid price
            underlying_price: Current futures price
            strike: Option strike price
            expiry: Option expiration date
            right: 'C' for call, 'P' for put
            valuation_date: Date for valuation (default: today)

        Returns:
            Greeks dataclass or None if calculation fails
        """
        if valuation_date is None:
            valuation_date = date.today()

        # Validate inputs
        if option_price <= 0:
            logger.debug(f"Invalid option price: {option_price}")
            return None

        if underlying_price <= 0:
            logger.debug(f"Invalid underlying price: {underlying_price}")
            return None

        if strike <= 0:
            logger.debug(f"Invalid strike: {strike}")
            return None

        # Calculate time to expiry
        tte_years = self._calc_tte(valuation_date, expiry)
        if tte_years <= 0:
            logger.debug(f"Option expired or expiring today: {expiry}")
            return None

        # Check for intrinsic value violations
        intrinsic = self._intrinsic_value(underlying_price, strike, right)
        if option_price < intrinsic * 0.99:  # Allow 1% tolerance
            logger.debug(
                f"Option price {option_price:.4f} below intrinsic {intrinsic:.4f}"
            )
            return None

        try:
            # Set up QuantLib environment
            ql_valuation = ql.Date(valuation_date.day, valuation_date.month, valuation_date.year)
            ql.Settings.instance().evaluationDate = ql_valuation

            ql_expiry = ql.Date(expiry.day, expiry.month, expiry.year)

            # Option type
            option_type = ql.Option.Call if right.upper() == 'C' else ql.Option.Put

            # Build the option
            payoff = ql.PlainVanillaPayoff(option_type, strike)
            exercise = ql.EuropeanExercise(ql_expiry)
            option = ql.VanillaOption(payoff, exercise)

            # Market quotes
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(underlying_price))
            rate_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(ql_valuation, self.risk_free_rate, self._day_count)
            )

            # Implied volatility calculation
            iv = self._calc_iv(
                option, option_price, underlying_price, rate_handle, ql_valuation
            )

            if iv is None:
                return None

            # Set up Black-76 process with computed IV
            vol_handle = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(ql_valuation, self._calendar, iv, self._day_count)
            )

            process = ql.BlackProcess(spot_handle, rate_handle, vol_handle)

            # Price the option to get Greeks
            engine = ql.AnalyticEuropeanEngine(process)
            option.setPricingEngine(engine)

            # Extract Greeks
            delta = option.delta()
            gamma = option.gamma()
            theta = option.theta() / 365.0  # Convert to daily
            vega = option.vega() / 100.0    # Convert to per 1% IV move

            return Greeks(
                iv=iv,
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega
            )

        except Exception as e:
            logger.warning(
                f"Greeks calculation failed: {e} "
                f"(price={option_price}, underlying={underlying_price}, "
                f"strike={strike}, expiry={expiry}, right={right})"
            )
            return None

    def calc_greeks_from_iv(
        self,
        iv: float,
        underlying_price: float,
        strike: float,
        expiry: date,
        right: str,
        valuation_date: Optional[date] = None
    ) -> Optional[Greeks]:
        """
        Calculate Greeks given a known IV (skip IV solver).

        Useful when IV is already known from another source.
        """
        if valuation_date is None:
            valuation_date = date.today()

        if iv <= 0 or underlying_price <= 0 or strike <= 0:
            return None

        tte_years = self._calc_tte(valuation_date, expiry)
        if tte_years <= 0:
            return None

        try:
            ql_valuation = ql.Date(valuation_date.day, valuation_date.month, valuation_date.year)
            ql.Settings.instance().evaluationDate = ql_valuation

            ql_expiry = ql.Date(expiry.day, expiry.month, expiry.year)

            option_type = ql.Option.Call if right.upper() == 'C' else ql.Option.Put

            payoff = ql.PlainVanillaPayoff(option_type, strike)
            exercise = ql.EuropeanExercise(ql_expiry)
            option = ql.VanillaOption(payoff, exercise)

            spot_handle = ql.QuoteHandle(ql.SimpleQuote(underlying_price))
            rate_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(ql_valuation, self.risk_free_rate, self._day_count)
            )
            vol_handle = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(ql_valuation, self._calendar, iv, self._day_count)
            )

            process = ql.BlackProcess(spot_handle, rate_handle, vol_handle)
            engine = ql.AnalyticEuropeanEngine(process)
            option.setPricingEngine(engine)

            return Greeks(
                iv=iv,
                delta=option.delta(),
                gamma=option.gamma(),
                theta=option.theta() / 365.0,
                vega=option.vega() / 100.0
            )

        except Exception as e:
            logger.warning(f"Greeks from IV calculation failed: {e}")
            return None

    def calc_theoretical_price(
        self,
        iv: float,
        underlying_price: float,
        strike: float,
        expiry: date,
        right: str,
        valuation_date: Optional[date] = None
    ) -> Optional[float]:
        """
        Calculate theoretical option price given IV.

        Useful for checking if market price is reasonable.
        """
        if valuation_date is None:
            valuation_date = date.today()

        if iv <= 0 or underlying_price <= 0 or strike <= 0:
            return None

        tte_years = self._calc_tte(valuation_date, expiry)
        if tte_years <= 0:
            return None

        try:
            ql_valuation = ql.Date(valuation_date.day, valuation_date.month, valuation_date.year)
            ql.Settings.instance().evaluationDate = ql_valuation

            ql_expiry = ql.Date(expiry.day, expiry.month, expiry.year)

            option_type = ql.Option.Call if right.upper() == 'C' else ql.Option.Put

            payoff = ql.PlainVanillaPayoff(option_type, strike)
            exercise = ql.EuropeanExercise(ql_expiry)
            option = ql.VanillaOption(payoff, exercise)

            spot_handle = ql.QuoteHandle(ql.SimpleQuote(underlying_price))
            rate_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(ql_valuation, self.risk_free_rate, self._day_count)
            )
            vol_handle = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(ql_valuation, self._calendar, iv, self._day_count)
            )

            process = ql.BlackProcess(spot_handle, rate_handle, vol_handle)
            engine = ql.AnalyticEuropeanEngine(process)
            option.setPricingEngine(engine)

            return option.NPV()

        except Exception as e:
            logger.warning(f"Theoretical price calculation failed: {e}")
            return None

    def _calc_iv(
        self,
        option: ql.VanillaOption,
        target_price: float,
        underlying_price: float,
        rate_handle: ql.YieldTermStructureHandle,
        valuation_date: ql.Date
    ) -> Optional[float]:
        """
        Calculate implied volatility using QuantLib's solver.
        """
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(underlying_price))

        # Try QuantLib's built-in IV solver first
        try:
            # Create a process with initial guess
            vol_handle = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(valuation_date, self._calendar, self.IV_INITIAL, self._day_count)
            )
            process = ql.BlackProcess(spot_handle, rate_handle, vol_handle)
            engine = ql.AnalyticEuropeanEngine(process)
            option.setPricingEngine(engine)

            # Use impliedVolatility method
            iv = option.impliedVolatility(
                target_price,
                process,
                self.IV_TOLERANCE,
                self.IV_MAX_ITERATIONS,
                self.IV_MIN,
                self.IV_MAX
            )

            if self.IV_MIN < iv < self.IV_MAX:
                return iv

        except Exception as e:
            logger.debug(f"QuantLib IV solver failed: {e}, trying bisection")

        # Fallback to manual bisection
        return self._bisection_iv(
            option, target_price, underlying_price, rate_handle, valuation_date
        )

    def _bisection_iv(
        self,
        option: ql.VanillaOption,
        target_price: float,
        underlying_price: float,
        rate_handle: ql.YieldTermStructureHandle,
        valuation_date: ql.Date
    ) -> Optional[float]:
        """
        Bisection fallback for IV calculation.
        """
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(underlying_price))

        low = self.IV_MIN
        high = self.IV_MAX

        for _ in range(self.IV_MAX_ITERATIONS):
            mid = (low + high) / 2

            vol_handle = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(valuation_date, self._calendar, mid, self._day_count)
            )
            process = ql.BlackProcess(spot_handle, rate_handle, vol_handle)
            engine = ql.AnalyticEuropeanEngine(process)
            option.setPricingEngine(engine)

            try:
                calc_price = option.NPV()
            except:
                return None

            if abs(calc_price - target_price) < self.IV_TOLERANCE:
                return mid

            if calc_price > target_price:
                high = mid
            else:
                low = mid

            if high - low < self.IV_TOLERANCE:
                return mid

        logger.debug(f"Bisection IV did not converge after {self.IV_MAX_ITERATIONS} iterations")
        return None

    def _calc_tte(self, valuation_date: date, expiry: date) -> float:
        """Calculate time to expiry in years."""
        delta = expiry - valuation_date
        return delta.days / 365.0

    def _intrinsic_value(self, underlying: float, strike: float, right: str) -> float:
        """Calculate intrinsic value of option."""
        if right.upper() == 'C':
            return max(0, underlying - strike)
        else:
            return max(0, strike - underlying)


class GreeksBatchProcessor:
    """
    Batch processor for computing Greeks on multiple options.

    More efficient than individual calls when processing many options.
    """

    def __init__(self, engine: Optional[GreeksEngine] = None):
        self.engine = engine or GreeksEngine()

    def process_batch(
        self,
        options: list[dict],
        valuation_date: Optional[date] = None
    ) -> dict[int, Optional[Greeks]]:
        """
        Process a batch of options.

        Args:
            options: List of dicts with keys:
                - option_id: Unique identifier
                - option_price: Current price
                - underlying_price: Futures price
                - strike: Strike price
                - expiry: Expiration date
                - right: 'C' or 'P'
            valuation_date: Valuation date (default: today)

        Returns:
            Dict mapping option_id to Greeks (or None if failed)
        """
        results = {}

        for opt in options:
            option_id = opt['option_id']

            greeks = self.engine.calc_greeks(
                option_price=opt['option_price'],
                underlying_price=opt['underlying_price'],
                strike=opt['strike'],
                expiry=opt['expiry'],
                right=opt['right'],
                valuation_date=valuation_date
            )

            results[option_id] = greeks

        return results
