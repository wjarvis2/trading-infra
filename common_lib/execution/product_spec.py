"""
Product Specifications for Execution.

Defines per-instrument/venue properties that are independent of
execution behavior. These are "what you trade" properties.

See research/decisions/003_execution_architecture.md for design rationale.

"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ProductSpec:
    """
    Specification of a tradable product's properties.

    These are venue/instrument properties, NOT behavioral settings.
    Behavioral settings belong in ExecutorConfig.

    Attributes
    ----------
    root : str
        Product root symbol (e.g., "CL", "RB", "HO", "LC")
    multiplier : float
        Contract multiplier (dollars per point)
    tick_size : float
        Minimum price increment
    commission_per_contract : float
        Commission per contract per side (entry or exit)
    exchange_fee : float
        Exchange/clearing fees per contract per side
    currency : str
        Contract currency (default "USD")

    Examples
    --------
    >>> cl_spec = ProductSpec(
    ...     root="CL",
    ...     multiplier=1000.0,
    ...     tick_size=0.01,
    ...     commission_per_contract=2.50,
    ... )
    >>> cl_spec.point_value  # 1 point move = $1000
    1000.0
    >>> cl_spec.tick_value  # 1 tick move = $10
    10.0
    """

    root: str
    multiplier: float
    tick_size: float
    commission_per_contract: float
    exchange_fee: float = 0.0
    currency: str = "USD"

    def __post_init__(self):
        if self.multiplier <= 0:
            raise ValueError(f"multiplier must be positive, got {self.multiplier}")
        if self.tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {self.tick_size}")
        if self.commission_per_contract < 0:
            raise ValueError(f"commission_per_contract cannot be negative")

    @property
    def point_value(self) -> float:
        """Dollar value of a 1-point price move."""
        return self.multiplier

    @property
    def tick_value(self) -> float:
        """Dollar value of a 1-tick price move."""
        return self.tick_size * self.multiplier

    def round_to_tick(self, price: float) -> float:
        """Round price to nearest valid tick."""
        return round(price / self.tick_size) * self.tick_size

    def total_commission(self, quantity: int, round_trip: bool = False) -> float:
        """
        Calculate total commission for a trade.

        Parameters
        ----------
        quantity : int
            Number of contracts
        round_trip : bool
            If True, calculate for entry + exit (2 sides)

        Returns
        -------
        float
            Total commission in dollars
        """
        sides = 2 if round_trip else 1
        per_contract = self.commission_per_contract + self.exchange_fee
        return per_contract * abs(quantity) * sides


class ProductCatalog:
    """
    Registry of known product specifications.

    Provides lookup by root symbol with explicit registration.
    No hidden defaults - products must be registered before use.

    Examples
    --------
    >>> catalog = ProductCatalog()
    >>> catalog.register(ProductSpec(root="CL", multiplier=1000, ...))
    >>> cl = catalog.get("CL")
    """

    def __init__(self):
        self._products: Dict[str, ProductSpec] = {}

    def register(self, spec: ProductSpec) -> None:
        """Register a product specification."""
        if spec.root in self._products:
            raise ValueError(f"Product {spec.root} already registered")
        self._products[spec.root] = spec

    def get(self, root: str) -> ProductSpec:
        """
        Get product specification by root symbol.

        Raises KeyError if product not registered.
        """
        if root not in self._products:
            raise KeyError(
                f"Product '{root}' not registered. "
                f"Available: {list(self._products.keys())}"
            )
        return self._products[root]

    def get_optional(self, root: str) -> Optional[ProductSpec]:
        """Get product specification, returning None if not found."""
        return self._products.get(root)

    def __contains__(self, root: str) -> bool:
        return root in self._products

    def list_products(self) -> list[str]:
        """List all registered product roots."""
        return list(self._products.keys())


# =============================================================================
# Standard Product Definitions
# =============================================================================
# These match the hardcoded values in the existing execution.py
# to ensure behavioral equivalence.

# Crude Oil (NYMEX)
CL_SPEC = ProductSpec(
    root="CL",
    multiplier=1000.0,      # $1000 per point
    tick_size=0.01,         # 1 cent
    commission_per_contract=2.50,  # Matches COMMISSION_PER_LEG in old code
    exchange_fee=0.0,       # Can add if needed
)

# RBOB Gasoline (NYMEX)
RB_SPEC = ProductSpec(
    root="RB",
    multiplier=42000.0,     # $42,000 per point (42,000 gallons)
    tick_size=0.0001,       # 0.01 cents per gallon
    commission_per_contract=2.50,
)

# Heating Oil / ULSD (NYMEX)
HO_SPEC = ProductSpec(
    root="HO",
    multiplier=42000.0,     # $42,000 per point
    tick_size=0.0001,       # 0.01 cents per gallon
    commission_per_contract=2.50,
)

# Lithium Carbonate (GFEX) - for cross-asset strategy
LC_SPEC = ProductSpec(
    root="LC",
    multiplier=1.0,         # 1 ton per contract, price in CNY/ton
    tick_size=50.0,         # 50 CNY
    commission_per_contract=0.0,  # TBD based on broker
    currency="CNY",
)


def create_default_catalog() -> ProductCatalog:
    """
    Create a ProductCatalog with standard energy products.

    Returns catalog with CL, RB, HO, LC registered.
    """
    catalog = ProductCatalog()
    catalog.register(CL_SPEC)
    catalog.register(RB_SPEC)
    catalog.register(HO_SPEC)
    catalog.register(LC_SPEC)
    return catalog
