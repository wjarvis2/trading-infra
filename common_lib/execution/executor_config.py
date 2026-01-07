"""
Executor Configuration - Behavioral Settings Only.

Defines HOW execution is simulated, not WHAT is being traded.
Product-specific properties belong in ProductSpec.

See research/decisions/003_execution_architecture.md for design rationale.

"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ExecutorConfig:
    """
    Behavioral configuration for execution simulation.

    This controls HOW fills are simulated, not product properties.
    All fields have explicit values - no hidden defaults that drift.

    Attributes
    ----------
    slippage_model : str
        How to model slippage:
        - "fixed": Fixed slippage per trade
        - "bar_range": Slippage proportional to bar high-low range
        - "atr": Slippage proportional to ATR
    fill_model : str
        Price basis for fills:
        - "close": Fill at bar close + slippage
        - "vwap": Fill at estimated VWAP + slippage
        - "midpoint": Fill at (high+low)/2 + slippage
    atomic_fills : bool
        For spreads: require both legs fill or neither
    allow_partial_fills : bool
        Whether to allow partial quantity fills
    latency_ms : int
        Simulated latency in milliseconds (for realistic timing)

    Slippage Model Parameters
    -------------------------
    half_spread_bps : float
        Half bid-ask spread in basis points (for bar_range model)
    slippage_vol_mult : float
        Slippage as multiple of bar range (for bar_range model)
    aggressive_slippage_mult : float
        Extra multiplier for aggressive/momentum orders
    fixed_slippage_ticks : float
        Fixed slippage in ticks (for fixed model)

    Examples
    --------
    >>> config = ExecutorConfig(
    ...     slippage_model="bar_range",
    ...     fill_model="close",
    ...     half_spread_bps=2.0,
    ...     slippage_vol_mult=0.5,
    ... )
    """

    # Core behavior
    slippage_model: Literal["fixed", "bar_range", "atr"] = "bar_range"
    fill_model: Literal["close", "vwap", "midpoint"] = "close"
    atomic_fills: bool = True
    allow_partial_fills: bool = False
    latency_ms: int = 0

    # Slippage model parameters (bar_range model)
    half_spread_bps: float = 2.0        # Matches DEFAULT_HALF_SPREAD_BPS
    slippage_vol_mult: float = 0.5      # Matches DEFAULT_SLIPPAGE_VOL_MULT
    aggressive_slippage_mult: float = 1.5  # Extra slippage for momentum

    # Slippage model parameters (fixed model)
    fixed_slippage_ticks: float = 1.0   # For fixed slippage model

    def __post_init__(self):
        if self.half_spread_bps < 0:
            raise ValueError("half_spread_bps cannot be negative")
        if self.slippage_vol_mult < 0:
            raise ValueError("slippage_vol_mult cannot be negative")
        if self.aggressive_slippage_mult < 1.0:
            raise ValueError("aggressive_slippage_mult must be >= 1.0")
        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")


# =============================================================================
# Standard Configurations
# =============================================================================
# Named configurations for common use cases.

# Default config matching existing execution.py behavior
DEFAULT_CONFIG = ExecutorConfig(
    slippage_model="bar_range",
    fill_model="close",
    atomic_fills=True,
    allow_partial_fills=False,
    latency_ms=0,
    half_spread_bps=2.0,
    slippage_vol_mult=0.5,
    aggressive_slippage_mult=1.5,
)

# Conservative config with higher slippage assumptions
CONSERVATIVE_CONFIG = ExecutorConfig(
    slippage_model="bar_range",
    fill_model="close",
    atomic_fills=True,
    allow_partial_fills=False,
    latency_ms=50,
    half_spread_bps=3.0,
    slippage_vol_mult=0.75,
    aggressive_slippage_mult=2.0,
)

# Optimistic config for liquid markets
OPTIMISTIC_CONFIG = ExecutorConfig(
    slippage_model="bar_range",
    fill_model="close",
    atomic_fills=True,
    allow_partial_fills=False,
    latency_ms=0,
    half_spread_bps=1.0,
    slippage_vol_mult=0.25,
    aggressive_slippage_mult=1.25,
)

# Fixed slippage config for simple simulations
FIXED_SLIPPAGE_CONFIG = ExecutorConfig(
    slippage_model="fixed",
    fill_model="close",
    atomic_fills=True,
    allow_partial_fills=False,
    latency_ms=0,
    fixed_slippage_ticks=1.0,
)
