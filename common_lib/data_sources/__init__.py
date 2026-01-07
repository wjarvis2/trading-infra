"""
Data sources - normalized wrappers around fetchers with explicit semantics.

Each source:
- Wraps a fetcher (I/O layer)
- Normalizes output format
- Declares key columns for joins
- Provides iteration/timestamp interface

This layer sits between fetchers (raw I/O) and the DataFeed (composition).

"""

from .base import DataSource, SourceMetadata
from .spread_source import SpreadSource
from .cot_source import COTSource
from .contract_meta_source import ContractMetaSource

__all__ = [
    "DataSource",
    "SourceMetadata",
    "SpreadSource",
    "COTSource",
    "ContractMetaSource",
]
