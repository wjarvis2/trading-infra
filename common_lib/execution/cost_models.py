"""
Pluggable Cost Models for Execution Simulation.

Defines Protocol interfaces and implementations for slippage and commission.
These are injected into executors - no hidden defaults inside executor code.

See research/decisions/003_execution_architecture.md for design rationale.

"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import pandas as pd

from common_lib.execution.product_spec import ProductSpec
from common_lib.execution.executor_config import ExecutorConfig


# =============================================================================
# Protocol Definitions
# =============================================================================


@runtime_checkable
class SlippageModel(Protocol):
    """
    Protocol for slippage calculation.

    Slippage models take bar data and return slippage in price units.
    The sign convention: positive slippage is adverse (costs money).
    """

    def __call__(
        self,
        direction: int,
        bar: pd.Series,
        product: ProductSpec,
        is_aggressive: bool = False,
    ) -> float:
        """
        Calculate slippage for an order.

        Parameters
        ----------
        direction : int
            Order direction: +1 for buy, -1 for sell
        bar : pd.Series
            OHLCV bar with at least 'high', 'low', 'close'
        product : ProductSpec
            Product specification
        is_aggressive : bool
            True for aggressive/momentum orders (higher slippage)

        Returns
        -------
        float
            Slippage in price units.
            Positive = adverse (buying costs more, selling receives less)
            The sign is already adjusted for direction.
        """
        ...


@runtime_checkable
class CommissionModel(Protocol):
    """
    Protocol for commission calculation.

    Commission models return total commission in dollars for a trade.
    """

    def __call__(
        self,
        quantity: int,
        product: ProductSpec,
        is_spread: bool = False,
    ) -> float:
        """
        Calculate commission for a trade.

        Parameters
        ----------
        quantity : int
            Number of contracts (absolute value used)
        product : ProductSpec
            Product specification
        is_spread : bool
            True if this is a spread trade (may have 2 legs)

        Returns
        -------
        float
            Total commission in dollars
        """
        ...


# =============================================================================
# Slippage Model Implementations
# =============================================================================


@dataclass
class BarRangeSlippageModel:
    """
    Slippage model based on bar range (volatility proxy).

    This matches the behavior of the existing SpreadExecutor.estimate_slippage().

    Formula:
        base_slippage = bar_range * slippage_vol_mult
        bid_ask_half = mid_price * half_spread_bps / 10000
        slippage = (base_slippage + bid_ask_half) * [aggressive_mult if aggressive]
        return slippage * direction

    Attributes
    ----------
    slippage_vol_mult : float
        Slippage as fraction of bar range (default 0.5)
    half_spread_bps : float
        Half bid-ask spread in basis points (default 2.0)
    aggressive_slippage_mult : float
        Multiplier for aggressive orders (default 1.5)
    """

    slippage_vol_mult: float = 0.5
    half_spread_bps: float = 2.0
    aggressive_slippage_mult: float = 1.5

    def __call__(
        self,
        direction: int,
        bar: pd.Series,
        product: ProductSpec,
        is_aggressive: bool = False,
    ) -> float:
        """Calculate slippage based on bar range."""
        # Bar range as volatility proxy
        bar_range = bar["high"] - bar["low"]

        # Base slippage from volatility
        base_slippage = bar_range * self.slippage_vol_mult

        # Bid-ask component
        mid_price = (bar["high"] + bar["low"]) / 2
        bid_ask_half = mid_price * self.half_spread_bps / 10000

        # Total slippage estimate
        slippage = base_slippage + bid_ask_half

        # Extra slippage for aggressive signals
        if is_aggressive:
            slippage *= self.aggressive_slippage_mult

        # Slippage is always adverse: buying costs more, selling gets less
        return slippage * direction

    @classmethod
    def from_config(cls, config: ExecutorConfig) -> "BarRangeSlippageModel":
        """Create model from ExecutorConfig."""
        return cls(
            slippage_vol_mult=config.slippage_vol_mult,
            half_spread_bps=config.half_spread_bps,
            aggressive_slippage_mult=config.aggressive_slippage_mult,
        )


@dataclass
class FixedSlippageModel:
    """
    Fixed slippage model - constant slippage per trade.

    Attributes
    ----------
    slippage_ticks : float
        Slippage in ticks (multiplied by tick_size from ProductSpec)
    """

    slippage_ticks: float = 1.0

    def __call__(
        self,
        direction: int,
        bar: pd.Series,
        product: ProductSpec,
        is_aggressive: bool = False,
    ) -> float:
        """Calculate fixed slippage in price units."""
        slippage = self.slippage_ticks * product.tick_size
        return slippage * direction

    @classmethod
    def from_config(cls, config: ExecutorConfig) -> "FixedSlippageModel":
        """Create model from ExecutorConfig."""
        return cls(slippage_ticks=config.fixed_slippage_ticks)


@dataclass
class ZeroSlippageModel:
    """Zero slippage model for idealized simulations."""

    def __call__(
        self,
        direction: int,
        bar: pd.Series,
        product: ProductSpec,
        is_aggressive: bool = False,
    ) -> float:
        """Return zero slippage."""
        return 0.0


# =============================================================================
# Commission Model Implementations
# =============================================================================


@dataclass
class PerContractCommissionModel:
    """
    Per-contract commission model.

    Uses commission from ProductSpec, optionally with spread handling.

    Attributes
    ----------
    spread_legs : int
        Number of legs for spread trades (default 2)
    """

    spread_legs: int = 2

    def __call__(
        self,
        quantity: int,
        product: ProductSpec,
        is_spread: bool = False,
    ) -> float:
        """Calculate commission based on ProductSpec."""
        legs = self.spread_legs if is_spread else 1
        return product.total_commission(abs(quantity)) * legs


@dataclass
class FixedCommissionModel:
    """
    Fixed commission per contract, ignoring ProductSpec.

    Useful for testing or when you want to override ProductSpec.

    Attributes
    ----------
    commission_per_contract : float
        Commission per contract per leg
    """

    commission_per_contract: float = 2.50

    def __call__(
        self,
        quantity: int,
        product: ProductSpec,
        is_spread: bool = False,
    ) -> float:
        """Calculate fixed commission."""
        legs = 2 if is_spread else 1
        return self.commission_per_contract * abs(quantity) * legs


@dataclass
class ZeroCommissionModel:
    """Zero commission model for gross P&L calculations."""

    def __call__(
        self,
        quantity: int,
        product: ProductSpec,
        is_spread: bool = False,
    ) -> float:
        """Return zero commission."""
        return 0.0


# =============================================================================
# Factory Functions
# =============================================================================


def create_slippage_model(config: ExecutorConfig) -> SlippageModel:
    """
    Create slippage model from ExecutorConfig.

    Parameters
    ----------
    config : ExecutorConfig
        Executor configuration specifying slippage_model type

    Returns
    -------
    SlippageModel
        Configured slippage model instance
    """
    if config.slippage_model == "bar_range":
        return BarRangeSlippageModel.from_config(config)
    elif config.slippage_model == "fixed":
        return FixedSlippageModel.from_config(config)
    elif config.slippage_model == "atr":
        # ATR model would need historical ATR - defer for now
        raise NotImplementedError("ATR slippage model not yet implemented")
    else:
        raise ValueError(f"Unknown slippage model: {config.slippage_model}")


def create_commission_model(use_product_spec: bool = True) -> CommissionModel:
    """
    Create commission model.

    Parameters
    ----------
    use_product_spec : bool
        If True, use PerContractCommissionModel (reads from ProductSpec)
        If False, use zero commission

    Returns
    -------
    CommissionModel
        Commission model instance
    """
    if use_product_spec:
        return PerContractCommissionModel()
    else:
        return ZeroCommissionModel()
