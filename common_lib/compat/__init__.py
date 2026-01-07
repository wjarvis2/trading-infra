"""
Compatibility Layer.

Provides adapters for bridging different component interfaces during
the platform upgrade. These adapters enable gradual migration from
old patterns to new contracts.

Main components:
- DataFrameToDict: Adapt DataFrame-returning DataFeeds to Dict format
- DictToDataFrame: Adapt Dict-returning DataFeeds to DataFrame format
- adapt_feed: Factory function for automatic adaptation
- validate_feed_contract: Validate DataFeed contract compliance
- validate_executor_contract: Validate Executor contract compliance

"""

from .adapters import (
    # Format conversion utilities
    dataframe_to_dict,
    dict_to_dataframe,
    # DataFeed adapters
    DataFrameToDict,
    DictToDataFrame,
    adapt_feed,
    # Enricher adapters
    adapt_enricher_for_dict,
    # Validation helpers
    validate_feed_contract,
    validate_executor_contract,
)

__all__ = [
    "dataframe_to_dict",
    "dict_to_dataframe",
    "DataFrameToDict",
    "DictToDataFrame",
    "adapt_feed",
    "adapt_enricher_for_dict",
    "validate_feed_contract",
    "validate_executor_contract",
]
