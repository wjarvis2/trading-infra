"""
Compatibility Adapters.

Adapters for bridging different component interfaces during transition.
These enable gradual migration from old patterns to new contracts.

VERSION: 1.0.0 (2026-01-03)

Adapters:
- DataFrameToDict: Wrap DataFeeds that return DataFrame to return Dict[str, pd.Series]
- DictToDataFrame: Wrap DataFeeds that return Dict to return DataFrame
- EnricherAdapter: Adapt enricher functions to work with Dict bar format

"""

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Iterator, List, Optional, Protocol, Union
import pandas as pd
import logging

from common_lib.interfaces.contracts import (
    DataFeedContract,
    DATAFEED_REQUIRED_FIELDS,
    validate_datafeed_output,
    ContractViolation,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Bar Format Utilities
# =============================================================================


def dataframe_to_dict(
    df: pd.DataFrame,
    instrument_col: str = "instrument_id",
) -> Dict[str, pd.Series]:
    """
    Convert DataFrame of bars to Dict[str, pd.Series].

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with one row per instrument
    instrument_col : str
        Column containing instrument ID

    Returns
    -------
    Dict[str, pd.Series]
        Bars by instrument_id
    """
    if df.empty:
        return {}

    bars = {}
    for _, row in df.iterrows():
        instrument_id = str(row[instrument_col])
        bars[instrument_id] = row.drop(instrument_col) if instrument_col in row.index else row

    return bars


def dict_to_dataframe(
    bars: Dict[str, pd.Series],
    instrument_col: str = "instrument_id",
) -> pd.DataFrame:
    """
    Convert Dict[str, pd.Series] to DataFrame of bars.

    Parameters
    ----------
    bars : Dict[str, pd.Series]
        Bars by instrument_id
    instrument_col : str
        Column name for instrument ID

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per instrument
    """
    if not bars:
        return pd.DataFrame()

    rows = []
    for instrument_id, bar in bars.items():
        row = bar.to_dict()
        row[instrument_col] = instrument_id
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# DataFeed Adapters
# =============================================================================


class DataFrameFeedProtocol(Protocol):
    """Protocol for feeds that return DataFrame from get_bars()."""

    def iter_timestamps(self) -> Iterator[datetime]:
        ...

    def get_bars(self, timestamp: datetime) -> pd.DataFrame:
        ...

    @property
    def instruments(self) -> List[str]:
        ...


@dataclass
class DataFrameToDict:
    """
    Adapter that wraps a DataFrame-returning DataFeed to return Dict[str, pd.Series].

    Use this to adapt older DataFeed implementations that return DataFrame
    to the new contract that returns Dict.

    Parameters
    ----------
    feed : DataFrameFeedProtocol
        DataFeed that returns DataFrame from get_bars()
    instrument_col : str
        Column containing instrument ID in the DataFrame
    validate : bool
        Whether to validate output against contract

    Examples
    --------
    >>> old_feed = OldDataFeed(...)  # returns DataFrame
    >>> adapted_feed = DataFrameToDict(old_feed)
    >>> bars = adapted_feed.get_bars(timestamp)  # returns Dict[str, pd.Series]
    """

    feed: DataFrameFeedProtocol
    instrument_col: str = "instrument_id"
    validate: bool = True

    def iter_timestamps(self) -> Iterator[datetime]:
        """Delegate to underlying feed."""
        return self.feed.iter_timestamps()

    def get_bars(self, timestamp: datetime) -> Dict[str, pd.Series]:
        """
        Get bars and convert DataFrame to Dict.

        Parameters
        ----------
        timestamp : datetime
            Timestamp to query

        Returns
        -------
        Dict[str, pd.Series]
            Bars by instrument_id
        """
        df = self.feed.get_bars(timestamp)

        # Handle None or empty
        if df is None or df.empty:
            return {}

        bars = dataframe_to_dict(df, self.instrument_col)

        if self.validate:
            validate_datafeed_output(bars, f"DataFrameToDict at {timestamp}")

        return bars

    @property
    def instruments(self) -> List[str]:
        """Delegate to underlying feed."""
        return self.feed.instruments


@dataclass
class DictToDataFrame:
    """
    Adapter that wraps a Dict-returning DataFeed to return DataFrame.

    Use this for components that expect DataFrame format.

    Parameters
    ----------
    feed : DataFeedContract
        DataFeed that returns Dict[str, pd.Series] from get_bars()
    instrument_col : str
        Column name for instrument ID in the output DataFrame

    Examples
    --------
    >>> new_feed = ComposableDataFeed(...)  # returns Dict[str, pd.Series]
    >>> adapted_feed = DictToDataFrame(new_feed)
    >>> bars = adapted_feed.get_bars(timestamp)  # returns DataFrame
    """

    feed: DataFeedContract
    instrument_col: str = "instrument_id"

    def iter_timestamps(self) -> Iterator[datetime]:
        """Delegate to underlying feed."""
        return self.feed.iter_timestamps()

    def get_bars(self, timestamp: datetime) -> pd.DataFrame:
        """
        Get bars and convert Dict to DataFrame.

        Parameters
        ----------
        timestamp : datetime
            Timestamp to query

        Returns
        -------
        pd.DataFrame
            Bars with one row per instrument
        """
        bars = self.feed.get_bars(timestamp)
        return dict_to_dataframe(bars, self.instrument_col)

    @property
    def instruments(self) -> List[str]:
        """Delegate to underlying feed."""
        return self.feed.instruments


# =============================================================================
# Enricher Adapters
# =============================================================================

# Type alias for enricher functions
EnricherFunc = Callable[[pd.DataFrame], pd.DataFrame]


def adapt_enricher_for_dict(
    enricher: EnricherFunc,
    instrument_col: str = "instrument_id",
) -> Callable[[Dict[str, pd.Series]], Dict[str, pd.Series]]:
    """
    Adapt a DataFrame enricher to work with Dict[str, pd.Series].

    Parameters
    ----------
    enricher : EnricherFunc
        Function that enriches a DataFrame
    instrument_col : str
        Column name for instrument ID

    Returns
    -------
    Callable[[Dict[str, pd.Series]], Dict[str, pd.Series]]
        Adapted enricher

    Examples
    --------
    >>> df_enricher = create_zscore_enricher(column="close", window=20)
    >>> dict_enricher = adapt_enricher_for_dict(df_enricher)
    >>> enriched_bars = dict_enricher(bars)
    """

    def adapted(bars: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        if not bars:
            return bars

        # Convert to DataFrame, enrich, convert back
        df = dict_to_dataframe(bars, instrument_col)
        enriched_df = enricher(df)
        return dataframe_to_dict(enriched_df, instrument_col)

    return adapted


# =============================================================================
# Factory Functions
# =============================================================================


def detect_feed_format(feed) -> str:
    """
    Detect whether a feed returns DataFrame or Dict format.

    Uses isinstance checks on return type annotation or explicit attribute.
    Never calls get_bars() to probe - no side effects.

    Parameters
    ----------
    feed : DataFeed
        Feed to check

    Returns
    -------
    str
        "dataframe" or "dict" or "unknown"
    """
    # Check for explicit format marker (preferred)
    if hasattr(feed, 'output_format'):
        return feed.output_format

    # Check type annotations
    get_bars = getattr(feed, 'get_bars', None)
    if get_bars is not None:
        annotations = getattr(get_bars, '__annotations__', {})
        return_type = annotations.get('return')
        if return_type is not None:
            type_str = str(return_type)
            if 'DataFrame' in type_str:
                return "dataframe"
            if 'Dict' in type_str or 'dict' in type_str:
                return "dict"

    return "unknown"


def adapt_feed(
    feed,
    target_format: str = "dict",
    instrument_col: str = "instrument_id",
    validate: bool = True,
    source_format: Optional[str] = None,
) -> Union[DataFrameToDict, DictToDataFrame]:
    """
    Adapt a feed to the target format.

    Parameters
    ----------
    feed : DataFeed
        Feed to adapt (either DataFrame or Dict format)
    target_format : str
        Target format: "dict" or "dataframe"
    instrument_col : str
        Column/key for instrument ID
    validate : bool
        Whether to validate output
    source_format : str, optional
        Source format ("dataframe" or "dict"). If not provided,
        attempts to detect from type annotations.

    Returns
    -------
    Adapted feed

    Raises
    ------
    ValueError
        If source format cannot be determined or is incompatible

    Examples
    --------
    >>> # Adapt a DataFrame feed to Dict format for GenericRunner
    >>> adapted = adapt_feed(df_feed, target_format="dict", source_format="dataframe")
    """
    # Detect source format if not provided
    if source_format is None:
        source_format = detect_feed_format(feed)

    if source_format == "unknown":
        raise ValueError(
            "Cannot determine feed format. Please specify source_format='dataframe' or 'dict'. "
            "Alternatively, add an 'output_format' attribute to your feed class."
        )

    # No-op if already in target format
    if source_format == target_format:
        logger.debug(f"Feed already in {target_format} format, returning unchanged")
        return feed

    # Validate transition
    if target_format == "dict" and source_format == "dataframe":
        return DataFrameToDict(
            feed=feed,
            instrument_col=instrument_col,
            validate=validate,
        )
    elif target_format == "dataframe" and source_format == "dict":
        return DictToDataFrame(
            feed=feed,
            instrument_col=instrument_col,
        )
    else:
        raise ValueError(
            f"Cannot adapt from {source_format} to {target_format}. "
            f"Valid transitions: dataframe→dict, dict→dataframe"
        )


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_feed_contract(
    feed,
    sample_timestamps: int = 5,
) -> List[str]:
    """
    Validate that a feed conforms to the DataFeedContract.

    Parameters
    ----------
    feed : DataFeed
        Feed to validate
    sample_timestamps : int
        Number of timestamps to sample for validation

    Returns
    -------
    List[str]
        List of validation errors (empty if valid)

    Examples
    --------
    >>> errors = validate_feed_contract(my_feed)
    >>> if errors:
    ...     print(f"Feed validation failed: {errors}")
    """
    errors = []

    # Check required methods
    if not hasattr(feed, 'iter_timestamps'):
        errors.append("Missing iter_timestamps() method")
    if not hasattr(feed, 'get_bars'):
        errors.append("Missing get_bars() method")
    if not hasattr(feed, 'instruments'):
        errors.append("Missing instruments property")

    if errors:
        return errors

    # Sample timestamps and validate output
    try:
        timestamps = list(feed.iter_timestamps())
        if not timestamps:
            errors.append("iter_timestamps() returned empty iterator")
            return errors

        # Check monotonicity
        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i - 1]:
                errors.append(
                    f"Timestamps not monotonic: {timestamps[i - 1]} >= {timestamps[i]}"
                )
                break

        # Sample and validate bars
        sample_ts = timestamps[:sample_timestamps]
        for ts in sample_ts:
            bars = feed.get_bars(ts)

            # Check format (should be Dict[str, pd.Series])
            if not isinstance(bars, dict):
                errors.append(
                    f"get_bars({ts}) returned {type(bars).__name__}, expected dict"
                )
                continue

            # Validate each bar
            try:
                validate_datafeed_output(bars, f"at {ts}")
            except ContractViolation as e:
                errors.append(str(e))

    except Exception as e:
        errors.append(f"Validation failed with exception: {e}")

    return errors


def validate_executor_contract(executor) -> List[str]:
    """
    Validate that an executor conforms to the ExecutorContract.

    Parameters
    ----------
    executor : Executor
        Executor to validate

    Returns
    -------
    List[str]
        List of validation errors (empty if valid)
    """
    errors = []

    if not hasattr(executor, 'execute'):
        errors.append("Missing execute() method")
    if not hasattr(executor, 'mark_to_market'):
        errors.append("Missing mark_to_market() method")

    return errors
