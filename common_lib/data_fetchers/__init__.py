"""
Data fetcher modules - unified interfaces for all data access.

All data fetching logic must be centralized here to ensure:
- Separation of concerns (I/O separate from business logic)
- Single source of truth for data access patterns
- Consistent error handling and logging
- Easy testing with mock data sources
"""

from .curve_fetchers import CurveFetcher
from .bar_fetchers import BarFetcher, dispose_engines, BarFetchError, DataQualityResult
from .pca_fetcher import PCAFetcher, PCAMetadataFetcher, PCAFactorData
from .inventory_fetcher import InventoryFetcher
from .spread_fetcher import SpreadFetcher, SpreadConfig, SpreadLeg
from .cot_fetcher import COTFetcher, COTSnapshot

__all__ = [
    "CurveFetcher",
    "BarFetcher",
    "dispose_engines",
    "BarFetchError",
    "DataQualityResult",
    "PCAFetcher",
    "PCAMetadataFetcher",
    "PCAFactorData",
    "InventoryFetcher",
    "SpreadFetcher",
    "SpreadConfig",
    "SpreadLeg",
    "COTFetcher",
    "COTSnapshot",
]