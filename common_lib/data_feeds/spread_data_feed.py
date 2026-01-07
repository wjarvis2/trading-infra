"""
Spread Data Feed for GenericRunner.

Wraps existing validated components:
- SpreadDataLoader (Data Layer) - loads raw 5s bars
- HistoricalCacheBuilder (Feature Layer) - precomputes z-score, percentile, momentum

Implements DataFeedProtocol for use with GenericRunner.

Design Philosophy:
- Features precomputed at session start (no inline computation)
- Uses validated cache_builder logic (no duplication)
- Minimal wrapper, maximum reuse

"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterator, List, Optional
import pandas as pd
import numpy as np
import logging

from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


@dataclass
class SpreadDataFeedConfig:
    """Configuration for SpreadDataFeed."""

    # Date range
    start_date: date
    end_date: date

    # Bar configuration
    bar_frequency: str = "5min"  # Resample to 5-min bars

    # Spreads to trade (subset of all spreads)
    traded_spreads: Optional[List[str]] = None

    # Warm-up: require minimum history for features
    min_history_days: int = 200  # ~80% of 252-day percentile window


class SpreadDataFeed:
    """
    Data feed providing enriched spread bars for backtesting.

    Implements DataFeedProtocol for GenericRunner.

    Features per bar (canonical names):
    - close: spread price
    - z_score: 60-day rolling z-score
    - pct_rank: 252-day percentile rank
    - mom: 10-day momentum
    - mom_pct: momentum percentile
    - front_close, back_close: leg prices
    - front_symbol, back_symbol: contract symbols
    - front_expiry: front contract expiry date
    - rolling_mean, rolling_std: z-score components

    Example
    -------
    >>> config = SpreadDataFeedConfig(
    ...     start_date=date(2024, 6, 1),
    ...     end_date=date(2024, 6, 30),
    ... )
    >>> feed = SpreadDataFeed(engine, config)
    >>> feed.load()
    >>> for ts in feed.iter_timestamps():
    ...     bars = feed.get_bars(ts)
    """

    def __init__(
        self,
        engine: Engine,
        config: SpreadDataFeedConfig,
    ):
        """
        Initialize the spread data feed.

        Parameters
        ----------
        engine : Engine
            SQLAlchemy database engine
        config : SpreadDataFeedConfig
            Feed configuration
        """
        self.engine = engine
        self.config = config

        # Lazy imports to avoid circular dependencies
        from strategies.spread_regime.backtest.data_loader import (
            SpreadDataLoader, DEFAULT_SPREADS, TRADED_SPREAD_NAMES
        )
        from strategies.spread_regime.backtest.cache_builder import (
            HistoricalCacheBuilder
        )

        # Initialize data and feature components
        self.data_loader = SpreadDataLoader(engine=engine)
        self.cache_builder = HistoricalCacheBuilder(engine=engine)

        # Determine which spreads to serve
        if config.traded_spreads:
            self._instruments = config.traded_spreads
        else:
            self._instruments = list(TRADED_SPREAD_NAMES)

        # State
        self._data: Optional[pd.DataFrame] = None
        self._timestamps: List[datetime] = []
        self._loaded = False

    def load(self) -> "SpreadDataFeed":
        """
        Load and prepare data for the configured date range.

        Returns
        -------
        SpreadDataFeed
            Self for chaining
        """
        logger.info(
            f"Loading SpreadDataFeed from {self.config.start_date} "
            f"to {self.config.end_date}"
        )

        all_bars = []
        current_date = self.config.start_date
        n_sessions = 0

        while current_date <= self.config.end_date:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue

            try:
                session_bars = self._load_session(current_date)
                if not session_bars.empty:
                    all_bars.append(session_bars)
                    n_sessions += 1
            except Exception as e:
                logger.warning(f"Failed to load session {current_date}: {e}")

            current_date += timedelta(days=1)

        if not all_bars:
            logger.warning("No data loaded")
            self._data = pd.DataFrame()
            self._timestamps = []
            self._loaded = True
            return self

        # Combine all sessions
        self._data = pd.concat(all_bars, ignore_index=True)
        self._data = self._data.sort_values('ts').reset_index(drop=True)

        # Extract unique timestamps
        self._timestamps = sorted(self._data['ts'].unique())

        self._loaded = True

        logger.info(
            f"SpreadDataFeed ready: {len(self._data)} rows, "
            f"{len(self._timestamps)} timestamps, {n_sessions} sessions"
        )

        return self

    def _load_session(self, session_date: date) -> pd.DataFrame:
        """
        Load a single session with enriched features.

        Uses HistoricalCacheBuilder to precompute features at session start,
        then updates incrementally with each bar.
        """
        # Build feature cache at session start (252-day lookback)
        cache = self.cache_builder.build_cache(session_date)

        if not cache.entries:
            logger.debug(f"No cache entries for {session_date}")
            return pd.DataFrame()

        # Load 5-min spread bars for the session
        spread_bars = self.data_loader.load_session_spreads_5min(session_date)

        if not spread_bars:
            logger.debug(f"No spread bars for {session_date}")
            return pd.DataFrame()

        # Build enriched rows
        rows = []

        for spread_name in self._instruments:
            if spread_name not in spread_bars or spread_bars[spread_name].empty:
                continue

            bars_df = spread_bars[spread_name]

            # Check if this spread has cache entry
            if spread_name not in cache.entries:
                logger.debug(f"No cache for {spread_name} on {session_date}")
                continue

            # Process each bar, updating cache incrementally
            for _, bar in bars_df.iterrows():
                # Update cache with this bar
                self.cache_builder.update_cache_with_bar(cache, spread_name, bar)

                # Get updated cache entry
                entry = cache.entries[spread_name]

                # Build enriched row with canonical column names
                # Include OHLC for execution cost models
                row = {
                    'ts': bar['ts'],
                    'instrument_id': spread_name,  # Canonical: instrument_id
                    'open': bar.get('open', entry.current_spread),
                    'high': bar.get('high', entry.current_spread),
                    'low': bar.get('low', entry.current_spread),
                    'close': entry.current_spread,
                    'z_score': entry.z_score,
                    'pct_rank': entry.spread_pct,
                    'mom': entry.spread_mom,
                    'mom_pct': entry.spread_mom_pct,
                    'rolling_mean': entry.rolling_mean,
                    'rolling_std': entry.rolling_std,
                    'front_close': entry.front_close,
                    'back_close': entry.back_close,
                    'front_symbol': entry.front_symbol,
                    'back_symbol': entry.back_symbol,
                    'front_expiry': entry.front_expiry,
                }
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        logger.debug(f"Session {session_date}: {len(df)} enriched bars")

        return df

    # =========================================================================
    # DataFeedProtocol Implementation
    # =========================================================================

    @property
    def instruments(self) -> List[str]:
        """Get list of instrument IDs (spread names)."""
        if not self._loaded:
            raise RuntimeError("Call load() before accessing instruments")
        return self._instruments

    def iter_timestamps(self) -> Iterator[datetime]:
        """Iterate over all timestamps in the feed."""
        if not self._loaded:
            raise RuntimeError("Call load() before iterating")
        return iter(self._timestamps)

    def get_bars(self, timestamp: datetime) -> Dict[str, pd.Series]:
        """
        Get bars for all instruments at a timestamp.

        Parameters
        ----------
        timestamp : datetime
            The timestamp to get bars for

        Returns
        -------
        dict[str, pd.Series]
            Dictionary mapping spread_name to bar data
        """
        if not self._loaded:
            raise RuntimeError("Call load() before getting bars")

        # Filter to this timestamp
        mask = self._data['ts'] == timestamp
        rows = self._data[mask]

        bars = {}
        for _, row in rows.iterrows():
            instrument_id = row['instrument_id']
            bars[instrument_id] = row

        return bars

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def to_dataframe(self) -> pd.DataFrame:
        """Get the full enriched DataFrame."""
        if not self._loaded:
            raise RuntimeError("Call load() before accessing data")
        return self._data.copy()

    @property
    def n_bars(self) -> int:
        """Total number of bars loaded."""
        if not self._loaded:
            return 0
        return len(self._data)


def create_spread_feed(
    start_date: date,
    end_date: date,
    engine: Optional[Engine] = None,
    traded_spreads: Optional[List[str]] = None,
) -> SpreadDataFeed:
    """
    Factory function to create a SpreadDataFeed.

    Convenience function for common use case.

    Parameters
    ----------
    start_date : date
        Start date
    end_date : date
        End date
    engine : Engine, optional
        Database engine (created from environment if not provided)
    traded_spreads : list[str], optional
        Spreads to include (default: TRADED_SPREAD_NAMES)

    Returns
    -------
    SpreadDataFeed
        Configured and loaded data feed
    """
    if engine is None:
        from strategies.spread_regime.backtest.data_loader import get_engine
        engine = get_engine()

    config = SpreadDataFeedConfig(
        start_date=start_date,
        end_date=end_date,
        traded_spreads=traded_spreads,
    )

    feed = SpreadDataFeed(engine, config)
    feed.load()

    return feed
