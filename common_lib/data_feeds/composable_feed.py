"""
Composable Data Feed - multi-source composition with joins and enrichment.

This is the core composition layer that:
- Loads data from multiple sources
- Joins them with configurable join operations
- Applies enrichers (feature engineering)
- Implements DataFeedProtocol for GenericRunner

"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import (
    Callable, Dict, Iterator, List, Optional, Protocol, Tuple, Union
)
import pandas as pd
import numpy as np
import logging

from common_lib.data_sources.base import DataSource

logger = logging.getLogger(__name__)


# =============================================================================
# Join Specifications
# =============================================================================


@dataclass
class JoinSpec:
    """Base class for join specifications."""
    left_source: str    # Name of left source (or "result" for chained joins)
    right_source: str   # Name of right source
    name: str = ""      # Optional name for this join


@dataclass
class AsofJoin(JoinSpec):
    """
    As-of join specification.

    Joins right source to left source using pandas merge_asof.
    For each row in left, finds the most recent row in right where
    right.timestamp <= left.timestamp.

    Common use case: Join weekly COT data to intraday bars.

    Parameters
    ----------
    left_source : str
        Name of left source
    right_source : str
        Name of right source
    left_on : str
        Timestamp column in left source
    right_on : str
        Timestamp column in right source
    by : list[str], optional
        Columns to match exactly before asof merge
    tolerance : timedelta, optional
        Maximum time difference for match
    direction : str
        'backward' (default), 'forward', or 'nearest'
    suffixes : tuple[str, str]
        Suffixes for overlapping columns
    """
    left_on: str = "ts"
    right_on: str = "available_at"
    by: Optional[List[str]] = None
    tolerance: Optional[timedelta] = None
    direction: str = "backward"
    suffixes: Tuple[str, str] = ("", "_cot")


@dataclass
class RangeJoin(JoinSpec):
    """
    Range join specification.

    Joins right source to left source where left.timestamp falls within
    a range defined by right source columns.

    Common use case: Join contract metadata valid for date ranges.

    Parameters
    ----------
    left_source : str
        Name of left source
    right_source : str
        Name of right source
    left_on : str
        Timestamp column in left source
    range_start : str
        Start of range column in right source
    range_end : str
        End of range column in right source
    by : list[str], optional
        Columns to match exactly
    """
    left_on: str = "ts"
    range_start: str = "effective_start"
    range_end: str = "effective_end"
    by: Optional[List[str]] = None


# =============================================================================
# Enricher Specification
# =============================================================================


# Type for enricher functions: DataFrame -> DataFrame
EnricherFunc = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class EnricherSpec:
    """
    Specification for a feature enricher.

    Parameters
    ----------
    name : str
        Name of this enricher (for logging)
    func : callable
        Function that takes DataFrame and returns enriched DataFrame
    """
    name: str
    func: EnricherFunc


# =============================================================================
# Composable Data Feed
# =============================================================================


class ComposableDataFeed:
    """
    Composable data feed that joins multiple sources.

    Implements DataFeedProtocol for use with GenericRunner.

    Example
    -------
    >>> feed = ComposableDataFeed(
    ...     sources={
    ...         "bars": spread_source,
    ...         "cot": cot_source,
    ...         "meta": contract_meta_source,
    ...     },
    ...     primary_source="bars",
    ...     joins=[
    ...         AsofJoin("bars", "cot", left_on="ts", right_on="available_at"),
    ...     ],
    ...     enrichers=[
    ...         EnricherSpec("zscore", zscore_enricher),
    ...     ],
    ... )
    >>> feed.load(date(2024, 6, 1), date(2024, 6, 30))
    >>> for ts in feed.iter_timestamps():
    ...     bars = feed.get_bars(ts)
    """

    def __init__(
        self,
        sources: Dict[str, DataSource],
        primary_source: str,
        joins: Optional[List[JoinSpec]] = None,
        enrichers: Optional[List[EnricherSpec]] = None,
        group_by: str = "instrument_id",  # Canonical: instrument_id
    ):
        """
        Initialize the composable data feed.

        Parameters
        ----------
        sources : dict[str, DataSource]
            Named data sources to compose
        primary_source : str
            Name of the primary source (determines timestamps)
        joins : list[JoinSpec], optional
            Join specifications to apply
        enrichers : list[EnricherSpec], optional
            Enrichers to apply after joins
        group_by : str
            Column to group by when serving bars (default "spread_id")
        """
        self.sources = sources
        self.primary_source = primary_source
        self.joins = joins or []
        self.enrichers = enrichers or []
        self.group_by = group_by

        # Validate primary source exists
        if primary_source not in sources:
            raise ValueError(
                f"Primary source '{primary_source}' not in sources: "
                f"{list(sources.keys())}"
            )

        # State
        self._data: Optional[pd.DataFrame] = None
        self._timestamps: List[datetime] = []
        self._loaded = False

    def load(
        self,
        start_date: date,
        end_date: date,
    ) -> "ComposableDataFeed":
        """
        Load and compose data for the specified date range.

        Parameters
        ----------
        start_date : date
            Start date (inclusive)
        end_date : date
            End date (inclusive)

        Returns
        -------
        ComposableDataFeed
            Self for chaining
        """
        logger.info(f"Loading ComposableDataFeed from {start_date} to {end_date}")

        # Load all sources
        source_data = {}
        for name, source in self.sources.items():
            try:
                df = source.load(start_date, end_date)
                source_data[name] = df
                logger.info(f"  Loaded {name}: {len(df)} rows")
            except Exception as e:
                logger.error(f"  Failed to load {name}: {e}")
                raise

        # Start with primary source
        result = source_data[self.primary_source].copy()

        # Apply joins
        for join_spec in self.joins:
            result = self._apply_join(result, source_data, join_spec)

        # Apply enrichers
        for enricher in self.enrichers:
            logger.info(f"  Applying enricher: {enricher.name}")
            result = enricher.func(result)

        self._data = result
        self._timestamps = sorted(result['ts'].unique())
        self._loaded = True

        logger.info(
            f"ComposableDataFeed ready: {len(result)} rows, "
            f"{len(self._timestamps)} timestamps"
        )

        return self

    def _apply_join(
        self,
        left: pd.DataFrame,
        source_data: Dict[str, pd.DataFrame],
        join_spec: JoinSpec,
    ) -> pd.DataFrame:
        """Apply a join specification."""
        right = source_data.get(join_spec.right_source)
        if right is None:
            raise ValueError(f"Join source '{join_spec.right_source}' not found")

        if isinstance(join_spec, AsofJoin):
            return self._apply_asof_join(left, right, join_spec)
        elif isinstance(join_spec, RangeJoin):
            return self._apply_range_join(left, right, join_spec)
        else:
            raise ValueError(f"Unknown join type: {type(join_spec)}")

    def _apply_asof_join(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        spec: AsofJoin,
    ) -> pd.DataFrame:
        """Apply an as-of join."""
        logger.info(
            f"  Applying asof join: {spec.left_source} <- {spec.right_source}"
        )

        # Handle empty right side - just add NaN columns
        if right.empty:
            logger.warning(
                f"  Right side empty in asof join, adding NaN columns"
            )
            result = left.copy()
            # Add right columns with NaN values, respecting suffixes
            right_suffix = spec.suffixes[1] if spec.suffixes else ""
            for col in right.columns:
                if col != spec.right_on:  # Don't add the join key
                    new_col = f"{col}{right_suffix}" if col in left.columns else col
                    result[new_col] = np.nan
            return result

        # Ensure both sides are sorted by their timestamp columns
        left = left.sort_values(spec.left_on)
        right = right.sort_values(spec.right_on)

        # Perform asof merge
        if spec.by:
            result = pd.merge_asof(
                left,
                right,
                left_on=spec.left_on,
                right_on=spec.right_on,
                by=spec.by,
                direction=spec.direction,
                tolerance=spec.tolerance,
                suffixes=spec.suffixes,
            )
        else:
            result = pd.merge_asof(
                left,
                right,
                left_on=spec.left_on,
                right_on=spec.right_on,
                direction=spec.direction,
                tolerance=spec.tolerance,
                suffixes=spec.suffixes,
            )

        return result

    def _apply_range_join(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        spec: RangeJoin,
    ) -> pd.DataFrame:
        """Apply a range join."""
        logger.info(
            f"  Applying range join: {spec.left_source} <- {spec.right_source}"
        )

        # For range joins, we need to find right rows where
        # left.ts >= right.range_start AND left.ts <= right.range_end

        # Simple approach: cross join then filter
        # More efficient approach would use interval index

        # Add temporary key for cross join
        left = left.copy()
        right = right.copy()
        left['_join_key'] = 1
        right['_join_key'] = 1

        # Cross join
        merged = left.merge(right, on='_join_key', suffixes=('', '_right'))
        merged = merged.drop(columns=['_join_key'])

        # Filter by range
        mask = (
            (merged[spec.left_on] >= merged[spec.range_start]) &
            (merged[spec.left_on] <= merged[spec.range_end])
        )

        # Apply additional by columns if specified
        if spec.by:
            for col in spec.by:
                if col in left.columns and f"{col}_right" in merged.columns:
                    mask &= merged[col] == merged[f"{col}_right"]

        result = merged[mask].copy()

        # Clean up range columns
        result = result.drop(columns=[spec.range_start, spec.range_end], errors='ignore')

        return result

    # =========================================================================
    # DataFeedProtocol Implementation
    # =========================================================================

    @property
    def instruments(self) -> List[str]:
        """Get list of instrument IDs (spread_ids)."""
        if not self._loaded:
            raise RuntimeError("Call load() before accessing instruments")
        return list(self._data[self.group_by].unique())

    def iter_timestamps(self) -> Iterator[datetime]:
        """Iterate over all timestamps in the feed."""
        if not self._loaded:
            raise RuntimeError("Call load() before iterating")
        return iter(self._timestamps)

    def get_bars(self, timestamp: datetime) -> Dict[str, pd.Series]:
        """
        Get bars for all instruments at a timestamp.

        Returns a dict mapping instrument_id to a Series containing
        all columns for that instrument at that timestamp.
        """
        if not self._loaded:
            raise RuntimeError("Call load() before getting bars")

        # Filter to this timestamp
        mask = self._data['ts'] == timestamp
        rows = self._data[mask]

        bars = {}
        for _, row in rows.iterrows():
            instrument_id = row[self.group_by]
            bars[instrument_id] = row

        return bars

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def to_dataframe(self) -> pd.DataFrame:
        """Get the full composed DataFrame."""
        if not self._loaded:
            raise RuntimeError("Call load() before accessing data")
        return self._data.copy()

    def get_data_for_instrument(self, instrument_id: str) -> pd.DataFrame:
        """Get all data for a specific instrument."""
        if not self._loaded:
            raise RuntimeError("Call load() before accessing data")
        return self._data[self._data[self.group_by] == instrument_id].copy()

    @property
    def columns(self) -> List[str]:
        """Get list of columns in the composed data."""
        if not self._loaded:
            raise RuntimeError("Call load() before accessing columns")
        return list(self._data.columns)
