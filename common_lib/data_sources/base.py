"""
Base classes for data sources.

A DataSource wraps a fetcher and normalizes its output for composition.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable
from datetime import date, datetime
import pandas as pd


@dataclass
class SourceMetadata:
    """
    Metadata describing a data source's schema and semantics.

    Used by ComposableDataFeed to understand how to join sources.
    """
    name: str                           # Source name (e.g., "spread_prices")
    key_columns: List[str]              # Columns that identify a row (e.g., ["spread_id"])
    timestamp_column: str               # Column containing timestamps
    frequency: str                      # Data frequency ("5s", "1h", "1d", "weekly")
    value_columns: List[str]            # Non-key, non-timestamp columns
    description: str = ""


@runtime_checkable
class DataSource(Protocol):
    """
    Protocol for data sources.

    A data source provides:
    - metadata about its schema
    - a method to load data for a date range
    """

    @property
    def metadata(self) -> SourceMetadata:
        """Get source metadata."""
        ...

    def load(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Load data for the specified date range.

        Returns a DataFrame with columns matching metadata.
        """
        ...


class BaseDataSource(ABC):
    """
    Abstract base class for data sources.

    Provides common functionality for all sources.
    """

    @property
    @abstractmethod
    def metadata(self) -> SourceMetadata:
        """Get source metadata."""
        pass

    @abstractmethod
    def load(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Load data for the specified date range."""
        pass

    def validate_output(self, df: pd.DataFrame) -> None:
        """
        Validate that output matches declared metadata.

        Raises ValueError if validation fails.
        """
        meta = self.metadata

        # Check required columns exist
        required = set(meta.key_columns + [meta.timestamp_column] + meta.value_columns)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Source '{meta.name}' missing columns: {missing}. "
                f"Has: {list(df.columns)}"
            )

        # Check timestamp column is datetime-like
        ts_col = df[meta.timestamp_column]
        if not pd.api.types.is_datetime64_any_dtype(ts_col):
            raise ValueError(
                f"Source '{meta.name}' timestamp column '{meta.timestamp_column}' "
                f"is not datetime type. Got: {ts_col.dtype}"
            )
