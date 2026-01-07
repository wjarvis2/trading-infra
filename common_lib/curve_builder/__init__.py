"""
Curve Builder Package - Ensures SQL/Python parity for curve calculations.

This package provides:
- Versioned SQL queries for curve calculations
- Python implementations that match SQL exactly
- SHA-256 validation to ensure SQL files haven't changed
- Property-based testing for edge cases
- Alembic migrations for database schema changes
"""

from .python.curve_calculator import CurveCalculator
from .python.validator import validate_sql_python_parity

__version__ = "1.0.0"
__all__ = ["CurveCalculator", "validate_sql_python_parity"]