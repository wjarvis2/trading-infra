"""
COT data source - wraps COTFetcher with normalized output.

Provides CFTC Commitments of Traders positioning data.

"""

from datetime import date
from typing import List, Optional
import pandas as pd
import logging

from common_lib.data_fetchers import COTFetcher
from .base import BaseDataSource, SourceMetadata

logger = logging.getLogger(__name__)


class COTSource(BaseDataSource):
    """
    Data source for CFTC COT positioning data.

    Wraps COTFetcher and normalizes output for composition.

    Output columns:
    - available_at: timestamp when data became available (for asof join)
    - report_date: actual report date (Tuesday position date)
    - market_id: market identifier (e.g., "WTI")
    - mm_net: money manager net as % of OI
    - mm_net_pct: percentile rank (0-100)

    Example
    -------
    >>> source = COTSource(
    ...     fetcher=COTFetcher(engine),
    ...     markets=["WTI"],
    ... )
    >>> df = source.load(date(2024, 1, 1), date(2024, 6, 30))
    """

    def __init__(
        self,
        fetcher: COTFetcher,
        markets: Optional[List[str]] = None,
    ):
        """
        Initialize the COT source.

        Parameters
        ----------
        fetcher : COTFetcher
            Underlying COT fetcher
        markets : list[str], optional
            List of markets to load (default: ["WTI"])
        """
        self.fetcher = fetcher
        self.markets = markets or ["WTI"]

    @property
    def metadata(self) -> SourceMetadata:
        """Get source metadata."""
        return SourceMetadata(
            name="cot_positioning",
            key_columns=["market_id"],
            timestamp_column="available_at",  # Use availability for asof joins
            frequency="weekly",
            value_columns=["cot_report_date", "mm_net", "mm_net_pct"],  # Canonical: cot_report_date
            description="CFTC COT money manager positioning",
        )

    def load(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Load COT data for the specified date range.

        Parameters
        ----------
        start_date : date
            Start date (inclusive)
        end_date : date
            End date (inclusive)

        Returns
        -------
        pd.DataFrame
            COT data with normalized columns
        """
        frames = []

        for market in self.markets:
            try:
                df = self.fetcher.fetch_cot_timeseries(
                    market_id=market,
                    start_date=start_date,
                    end_date=end_date,
                )

                if not df.empty:
                    frames.append(df)

            except Exception as e:
                logger.warning(f"Failed to load COT for {market}: {e}")
                continue

        if not frames:
            logger.warning(
                f"No COT data loaded for {self.markets} "
                f"between {start_date} and {end_date}"
            )
            return pd.DataFrame(columns=[
                'available_at', 'cot_report_date', 'market_id', 'mm_net', 'mm_net_pct'
            ])

        result = pd.concat(frames, ignore_index=True)

        # Ensure available_at is datetime (may be timezone-aware)
        # Convert to naive UTC for consistency in joins
        if result['available_at'].dt.tz is not None:
            result['available_at'] = result['available_at'].dt.tz_convert('UTC').dt.tz_localize(None)

        # Rename to canonical field names
        result = result.rename(columns={'report_date': 'cot_report_date'})

        # Sort for consistent output
        result = result.sort_values(['available_at', 'market_id']).reset_index(drop=True)

        # Select and order columns (canonical names)
        result = result[[
            'available_at', 'cot_report_date', 'market_id', 'mm_net', 'mm_net_pct'
        ]]

        self.validate_output(result)

        logger.info(
            f"Loaded {len(result)} COT reports for {self.markets} "
            f"({result['available_at'].min()} to {result['available_at'].max()})"
        )

        return result
