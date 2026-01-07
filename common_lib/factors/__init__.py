"""
Factor calculation modules - stateless calculators for various financial factors.

All calculators in this module must be:
- Stateless (no instance variables that change)
- Pure functions (no side effects, no I/O)
- Testable with fixture data
- Single source of truth for each calculation
"""

from .pca_calculator import PCACalculator
from .carry_pca_calculator import CarryPCACalculator

__all__ = ["PCACalculator", "CarryPCACalculator"]