"""
Spread Execution Simulators.

Provides executors for spread trading:
- SyntheticSpreadExecutor: Coordinates two legs with atomic fills
- NativeSpreadExecutor: Treats spread as single instrument

Both implement SpreadExecutorProtocol for interchangeable use.

See research/decisions/003_execution_architecture.md for design rationale.

"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Literal, Optional, Union, Any
import pandas as pd

from common_lib.execution.product_spec import ProductSpec
from common_lib.execution.executor_config import ExecutorConfig
from common_lib.execution.cost_models import SlippageModel, CommissionModel


# Explicit execution style type
ExecutionStyle = Literal["aggressive", "conservative"]


# =============================================================================
# Exceptions
# =============================================================================


class MissingLegDataError(ValueError):
    """
    Raised when leg price data is missing from bar.

    This is a hard failure - no silent fallbacks allowed.
    Missing leg data indicates upstream data pipeline issues
    that would create phantom alpha if silently estimated.
    """

    def __init__(
        self,
        leg: str,
        field: str,
        bar_ts: Optional[Any] = None,
    ):
        self.leg = leg
        self.field = field
        self.bar_ts = bar_ts
        ts_str = f" at {bar_ts}" if bar_ts else ""
        super().__init__(
            f"Missing {leg} leg data: '{field}' not in bar{ts_str}. "
            "Check data pipeline - no silent fallbacks allowed."
        )


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class FillResult:
    """
    Result of simulated fill execution.

    Matches the existing FillResult from strategies/spread_regime/backtest/execution.py
    to ensure behavioral equivalence.

    Attributes
    ----------
    fill_spread : float
        Actual spread fill price
    fill_front : float
        Front leg fill price
    fill_back : float
        Back leg fill price
    slippage_spread : float
        Slippage on spread (can be + or -)
    slippage_dollars : float
        Total slippage in dollars
    commission : float
        Total commission
    execution_quality : str
        Quality assessment: 'good', 'average', 'poor'
    notes : str
        Execution notes
    """

    fill_spread: float
    fill_front: float
    fill_back: float
    slippage_spread: float
    slippage_dollars: float
    commission: float
    execution_quality: Literal["good", "average", "poor"]
    notes: str


# =============================================================================
# Protocol Definition
# =============================================================================


class SpreadExecutorProtocol(Protocol):
    """
    Protocol for spread execution simulators.

    Both SyntheticSpreadExecutor and NativeSpreadExecutor implement this.
    """

    def execute_entry(
        self,
        direction: int,
        bar: pd.Series,
        strategy: str,
        quantity: int = 1,
        execution_style: Optional[ExecutionStyle] = None,
    ) -> FillResult:
        """Simulate entry fill."""
        ...

    def execute_exit(
        self,
        direction: int,
        bar: pd.Series,
        exit_reason: str,
        quantity: int = 1,
    ) -> FillResult:
        """Simulate exit fill."""
        ...


# =============================================================================
# Synthetic Spread Executor
# =============================================================================


class SyntheticSpreadExecutor:
    """
    Simulate spread order execution with realistic fills.

    Coordinates two legs (front - back) with atomic fills.
    Uses pluggable slippage and commission models.

    This is the replacement for strategies/spread_regime/backtest/execution.py
    SpreadExecutor, with the same behavioral semantics but:
    - Product properties from ProductSpec (not hardcoded)
    - Behavior from ExecutorConfig (explicit, not defaults)
    - Cost models injected (testable in isolation)

    Parameters
    ----------
    product : ProductSpec
        Product specification (multiplier, tick_size, etc.)
    config : ExecutorConfig
        Behavioral configuration
    slippage_model : SlippageModel
        Pluggable slippage calculator
    commission_model : CommissionModel
        Pluggable commission calculator

    Examples
    --------
    >>> from common_lib.execution.product_spec import CL_SPEC
    >>> from common_lib.execution.executor_config import DEFAULT_CONFIG
    >>> from common_lib.execution.cost_models import (
    ...     BarRangeSlippageModel, PerContractCommissionModel
    ... )
    >>> executor = SyntheticSpreadExecutor(
    ...     product=CL_SPEC,
    ...     config=DEFAULT_CONFIG,
    ...     slippage_model=BarRangeSlippageModel.from_config(DEFAULT_CONFIG),
    ...     commission_model=PerContractCommissionModel(),
    ... )
    >>> fill = executor.execute_entry(direction=-1, bar=bar, strategy='MR')
    """

    def __init__(
        self,
        product: ProductSpec,
        config: ExecutorConfig,
        slippage_model: SlippageModel,
        commission_model: CommissionModel,
    ):
        self.product = product
        self.config = config
        self.slippage_model = slippage_model
        self.commission_model = commission_model

    def _assess_quality(self, slippage: float, bar: pd.Series) -> str:
        """Assess execution quality based on slippage relative to bar range."""
        bar_range = bar["high"] - bar["low"]
        if bar_range <= 0:
            return "average"

        relative_slip = abs(slippage) / bar_range

        if relative_slip < 0.1:
            return "good"
        elif relative_slip < 0.3:
            return "average"
        else:
            return "poor"

    def _get_leg_prices(self, bar: pd.Series, spread_close: float) -> tuple[float, float]:
        """
        Extract front/back leg prices from bar.

        Returns (front_close, back_close).

        Raises
        ------
        MissingLegDataError
            If leg prices are not present in bar. No silent fallbacks.
        """
        front_close = bar.get("front_close")
        back_close = bar.get("back_close")

        if front_close is None:
            raise MissingLegDataError(
                leg="front",
                field="front_close",
                bar_ts=bar.get("ts"),
            )
        if back_close is None:
            raise MissingLegDataError(
                leg="back",
                field="back_close",
                bar_ts=bar.get("ts"),
            )

        return float(front_close), float(back_close)

    def execute_entry(
        self,
        direction: int,
        bar: pd.Series,
        strategy: str,
        quantity: int = 1,
        execution_style: Optional[ExecutionStyle] = None,
    ) -> FillResult:
        """
        Simulate entry fill.

        Entry fills assume:
        - We enter at bar close with some slippage
        - Long entries pay slightly more (buy at ask)
        - Short entries receive slightly less (sell at bid)

        Parameters
        ----------
        direction : int
            1 for LONG, -1 for SHORT
        bar : pd.Series
            OHLCV bar with close, optionally front_close, back_close
        strategy : str
            Strategy identifier (used for notes, not for slippage inference)
        quantity : int
            Number of spreads
        execution_style : "aggressive" | "conservative", optional
            Explicit execution style. If None, defaults to conservative.

        Returns
        -------
        FillResult
            Simulated fill details
        """
        # Base spread price
        spread_close = bar["close"]

        # Determine if aggressive from explicit execution_style
        # No more strategy-name inference - caller must set execution_style explicitly
        is_aggressive = execution_style == "aggressive" if execution_style else False

        # Calculate slippage using injected model
        slippage_spread = self.slippage_model(
            direction=direction,
            bar=bar,
            product=self.product,
            is_aggressive=is_aggressive,
        )

        # Fill price: entry slippage is adverse (costs money)
        fill_spread = spread_close + slippage_spread

        # Leg prices
        front_close, back_close = self._get_leg_prices(bar, spread_close)

        # Split slippage between legs (approximate)
        half_slip = slippage_spread / 2
        if direction == 1:  # Long spread: buy front, sell back
            fill_front = front_close + half_slip
            fill_back = back_close - half_slip
        else:  # Short spread: sell front, buy back
            fill_front = front_close - half_slip
            fill_back = back_close + half_slip

        # Commission using injected model
        commission = self.commission_model(
            quantity=quantity,
            product=self.product,
            is_spread=True,
        )

        # Slippage in dollars
        slippage_dollars = abs(slippage_spread) * quantity * self.product.multiplier

        # Assess quality
        quality = self._assess_quality(slippage_spread, bar)

        return FillResult(
            fill_spread=fill_spread,
            fill_front=fill_front,
            fill_back=fill_back,
            slippage_spread=slippage_spread,
            slippage_dollars=slippage_dollars,
            commission=commission,
            execution_quality=quality,
            notes=f"Entry {strategy} {'LONG' if direction == 1 else 'SHORT'} @ bar close + slippage",
        )

    def execute_exit(
        self,
        direction: int,
        bar: pd.Series,
        exit_reason: str,
        quantity: int = 1,
    ) -> FillResult:
        """
        Simulate exit fill.

        Exit fills assume:
        - We exit at bar close with some slippage
        - Exiting long means selling (get bid, which is lower)
        - Exiting short means buying back (pay ask, which is higher)

        Parameters
        ----------
        direction : int
            Original position direction (1 for LONG, -1 for SHORT)
        bar : pd.Series
            OHLCV bar
        exit_reason : str
            Reason for exit ('mr_target', 'max_hold', 'stop_loss', etc.)
        quantity : int
            Number of spreads

        Returns
        -------
        FillResult
            Simulated fill details
        """
        # Base spread price
        spread_close = bar["close"]

        # Exit direction is opposite of position direction
        exit_direction = -direction

        # Exits are usually less aggressive (not chasing momentum)
        slippage_spread = self.slippage_model(
            direction=exit_direction,
            bar=bar,
            product=self.product,
            is_aggressive=False,
        )

        # Fill price
        fill_spread = spread_close + slippage_spread

        # Leg prices
        front_close, back_close = self._get_leg_prices(bar, spread_close)

        # Split slippage between legs
        half_slip = slippage_spread / 2
        fill_front = front_close + half_slip
        fill_back = back_close - half_slip

        # Commission
        commission = self.commission_model(
            quantity=quantity,
            product=self.product,
            is_spread=True,
        )

        # Slippage in dollars
        slippage_dollars = abs(slippage_spread) * quantity * self.product.multiplier

        # Assess quality
        quality = self._assess_quality(slippage_spread, bar)

        return FillResult(
            fill_spread=fill_spread,
            fill_front=fill_front,
            fill_back=fill_back,
            slippage_spread=slippage_spread,
            slippage_dollars=slippage_dollars,
            commission=commission,
            execution_quality=quality,
            notes=f"Exit ({exit_reason}) @ bar close + slippage",
        )


# =============================================================================
# Native Spread Executor
# =============================================================================


class NativeSpreadExecutor:
    """
    Execute spread as a single native instrument.

    For exchange-listed spreads (e.g., CME calendar spreads) that have
    their own order book and market data.

    Uses the spread bar directly - no leg coordination needed.
    Sim-complete; IBKR wiring deferred to live integration.

    Parameters
    ----------
    product : ProductSpec
        Product specification
    config : ExecutorConfig
        Behavioral configuration
    slippage_model : SlippageModel
        Pluggable slippage calculator
    commission_model : CommissionModel
        Pluggable commission calculator
    """

    def __init__(
        self,
        product: ProductSpec,
        config: ExecutorConfig,
        slippage_model: SlippageModel,
        commission_model: CommissionModel,
    ):
        self.product = product
        self.config = config
        self.slippage_model = slippage_model
        self.commission_model = commission_model

    def _assess_quality(self, slippage: float, bar: pd.Series) -> str:
        """Assess execution quality based on slippage relative to bar range."""
        bar_range = bar["high"] - bar["low"]
        if bar_range <= 0:
            return "average"

        relative_slip = abs(slippage) / bar_range

        if relative_slip < 0.1:
            return "good"
        elif relative_slip < 0.3:
            return "average"
        else:
            return "poor"

    def execute_entry(
        self,
        direction: int,
        bar: pd.Series,
        strategy: str,
        quantity: int = 1,
        execution_style: Optional[ExecutionStyle] = None,
    ) -> FillResult:
        """
        Simulate entry fill for native spread.

        Native spreads fill as a single instrument - no leg coordination.

        Parameters
        ----------
        direction : int
            1 for LONG, -1 for SHORT
        bar : pd.Series
            OHLCV bar
        strategy : str
            Strategy identifier (used for notes, not for slippage inference)
        quantity : int
            Number of spreads
        execution_style : "aggressive" | "conservative", optional
            Explicit execution style. If None, defaults to conservative.
        """
        spread_close = bar["close"]

        # Determine if aggressive from explicit execution_style
        # No more strategy-name inference - caller must set execution_style explicitly
        is_aggressive = execution_style == "aggressive" if execution_style else False

        slippage_spread = self.slippage_model(
            direction=direction,
            bar=bar,
            product=self.product,
            is_aggressive=is_aggressive,
        )

        fill_spread = spread_close + slippage_spread

        # For native spreads, leg prices are still required for FillResult
        # No silent fallbacks - missing data is a pipeline issue
        front_close = bar.get("front_close")
        back_close = bar.get("back_close")

        if front_close is None:
            raise MissingLegDataError(
                leg="front",
                field="front_close",
                bar_ts=bar.get("ts"),
            )
        if back_close is None:
            raise MissingLegDataError(
                leg="back",
                field="back_close",
                bar_ts=bar.get("ts"),
            )

        # Native spread = single instrument, but still 2 legs for commission
        commission = self.commission_model(
            quantity=quantity,
            product=self.product,
            is_spread=True,
        )

        slippage_dollars = abs(slippage_spread) * quantity * self.product.multiplier
        quality = self._assess_quality(slippage_spread, bar)

        return FillResult(
            fill_spread=fill_spread,
            fill_front=float(front_close),
            fill_back=float(back_close),
            slippage_spread=slippage_spread,
            slippage_dollars=slippage_dollars,
            commission=commission,
            execution_quality=quality,
            notes=f"Native spread entry {strategy} {'LONG' if direction == 1 else 'SHORT'}",
        )

    def execute_exit(
        self,
        direction: int,
        bar: pd.Series,
        exit_reason: str,
        quantity: int = 1,
    ) -> FillResult:
        """
        Simulate exit fill for native spread.
        """
        spread_close = bar["close"]
        exit_direction = -direction

        slippage_spread = self.slippage_model(
            direction=exit_direction,
            bar=bar,
            product=self.product,
            is_aggressive=False,
        )

        fill_spread = spread_close + slippage_spread

        # Leg prices required - no silent fallbacks
        front_close = bar.get("front_close")
        back_close = bar.get("back_close")

        if front_close is None:
            raise MissingLegDataError(
                leg="front",
                field="front_close",
                bar_ts=bar.get("ts"),
            )
        if back_close is None:
            raise MissingLegDataError(
                leg="back",
                field="back_close",
                bar_ts=bar.get("ts"),
            )

        commission = self.commission_model(
            quantity=quantity,
            product=self.product,
            is_spread=True,
        )

        slippage_dollars = abs(slippage_spread) * quantity * self.product.multiplier
        quality = self._assess_quality(slippage_spread, bar)

        return FillResult(
            fill_spread=fill_spread,
            fill_front=float(front_close),
            fill_back=float(back_close),
            slippage_spread=slippage_spread,
            slippage_dollars=slippage_dollars,
            commission=commission,
            execution_quality=quality,
            notes=f"Native spread exit ({exit_reason})",
        )
