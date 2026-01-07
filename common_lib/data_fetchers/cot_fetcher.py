"""
COT (Commitments of Traders) data fetcher.

Provides unified access to CFTC COT positioning data with proper
publication timing to prevent look-ahead bias.

Key Publication Timing Rules:
- CFTC releases COT reports on Friday at 3:30 PM ET
- Report reflects positions as of Tuesday close
- For backtesting, we conservatively assume Friday 4 PM ET availability
- Signals before Friday 4 PM use the previous week's report

Migrated from: strategies/spread_regime/backtest/cot_loader.py
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import date, datetime, timedelta, time
from dataclasses import dataclass
from sqlalchemy import text
from sqlalchemy.engine import Engine
import logging
import pytz

logger = logging.getLogger(__name__)


# COT publication timing (Eastern Time)
COT_PUBLICATION_DAY = 4        # Friday = 4 (Monday = 0)
COT_PUBLICATION_HOUR = 16      # 4 PM ET (conservative, actual is 3:30 PM)

# Percentile lookback for COT (weekly data)
DEFAULT_PERCENTILE_LOOKBACK = 52   # 52 weeks = 1 year


@dataclass
class COTSnapshot:
    """
    Snapshot of COT data available at a point in time.

    Represents the most recent COT report that would have been
    publicly available at the query timestamp.
    """
    market_id: str          # Market identifier (e.g., "WTI", "RBOB")
    mm_net: float           # Money manager net as % of OI
    mm_net_pct: float       # Percentile rank (0-100)
    report_date: date       # Date of the COT report (Tuesday position date)
    available_at: datetime  # When this report became available (Friday 4 PM ET)

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame construction."""
        return {
            "market_id": self.market_id,
            "mm_net": self.mm_net,
            "mm_net_pct": self.mm_net_pct,
            "report_date": self.report_date,
            "available_at": self.available_at,
        }


class COTFetcher:
    """
    Fetcher for CFTC Commitments of Traders (COT) data.

    Handles publication timing to ensure no look-ahead bias in backtesting.
    The CFTC releases weekly COT reports on Friday at 3:30 PM ET,
    reflecting positions as of Tuesday.

    Example
    -------
    >>> fetcher = COTFetcher(engine)
    >>> # Get COT data available on a Monday morning
    >>> snapshot = fetcher.get_available_cot("WTI", datetime(2024, 6, 17, 10, 0))
    >>> print(f"mm_net: {snapshot.mm_net:.2f}, percentile: {snapshot.mm_net_pct:.1f}")

    >>> # Get timeseries for backtest period
    >>> df = fetcher.fetch_cot_timeseries(
    ...     market_id="WTI",
    ...     start_date=date(2024, 1, 1),
    ...     end_date=date(2024, 6, 30),
    ... )
    """

    # Supported markets and their view names
    MARKET_VIEWS = {
        "WTI": "model.v_cot_wti",
        "RBOB": "model.v_cot_rbob",
        "HO": "model.v_cot_ho",
    }

    def __init__(
        self,
        engine: Engine,
        percentile_lookback: int = DEFAULT_PERCENTILE_LOOKBACK,
    ):
        """
        Initialize the COT fetcher.

        Parameters
        ----------
        engine : Engine
            SQLAlchemy engine for database access
        percentile_lookback : int
            Number of weeks for percentile calculation (default 52)
        """
        self.engine = engine
        self.percentile_lookback = percentile_lookback
        self._cache: Dict[str, pd.DataFrame] = {}
        self._et = pytz.timezone('America/New_York')

    def fetch_raw_cot(
        self,
        market_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Fetch raw COT data from database.

        Parameters
        ----------
        market_id : str
            Market identifier (WTI, RBOB, HO)
        start_date : date, optional
            Start date filter
        end_date : date, optional
            End date filter

        Returns
        -------
        pd.DataFrame
            Raw COT data with columns: report_date, mm_net, available_at

        Raises
        ------
        ValueError
            If market_id is not supported or no data found
        """
        if market_id not in self.MARKET_VIEWS:
            raise ValueError(
                f"Unknown market: {market_id}. "
                f"Supported: {list(self.MARKET_VIEWS.keys())}"
            )

        view_name = self.MARKET_VIEWS[market_id]
        cache_key = f"{market_id}_{start_date}_{end_date}"

        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        # Build query
        query = f"""
            SELECT
                report_date,
                money_mgr_net_pct_oi as mm_net
            FROM {view_name}
            WHERE 1=1
        """
        params = {}

        if start_date:
            query += " AND report_date >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND report_date <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY report_date"

        df = pd.read_sql(
            text(query),
            self.engine,
            params=params,
            parse_dates=['report_date']
        )

        if df.empty:
            raise ValueError(f"No COT data found for {market_id}")

        # Compute availability timestamp for each report
        df['available_at'] = df['report_date'].apply(
            self._compute_availability_time
        )

        self._cache[cache_key] = df
        logger.info(
            f"Loaded {len(df)} COT reports for {market_id} "
            f"from {df['report_date'].min()} to {df['report_date'].max()}"
        )

        return df.copy()

    def _compute_availability_time(self, report_date: date) -> datetime:
        """
        Compute when a COT report became available.

        COT is released on Friday for Tuesday's positions.
        Report_date is Tuesday, so we add 3 days to get Friday.
        """
        if isinstance(report_date, pd.Timestamp):
            report_date = report_date.date()

        # report_date is Tuesday, Friday = Tuesday + 3 days
        friday = report_date + timedelta(days=3)

        available_local = self._et.localize(
            datetime.combine(friday, time(COT_PUBLICATION_HOUR, 0))
        )
        return available_local

    def get_available_cot(
        self,
        market_id: str,
        as_of: datetime,
    ) -> Optional[COTSnapshot]:
        """
        Get the most recent COT report available at a given time.

        Parameters
        ----------
        market_id : str
            Market identifier (WTI, RBOB, HO)
        as_of : datetime
            The timestamp to check (timezone-aware or assumed ET)

        Returns
        -------
        COTSnapshot or None
            The most recent available COT data, or None if no data available

        Example
        -------
        >>> # Monday 10 AM ET - uses previous Friday's report
        >>> snapshot = fetcher.get_available_cot("WTI", datetime(2024, 6, 17, 10, 0))

        >>> # Friday 3 PM ET - still uses previous week (new report at 4 PM)
        >>> snapshot = fetcher.get_available_cot("WTI", datetime(2024, 6, 21, 15, 0))

        >>> # Friday 5 PM ET - uses today's report
        >>> snapshot = fetcher.get_available_cot("WTI", datetime(2024, 6, 21, 17, 0))
        """
        # Ensure timezone-aware
        if as_of.tzinfo is None:
            as_of = self._et.localize(as_of)
        else:
            as_of = as_of.astimezone(self._et)

        # Load all COT data
        df = self.fetch_raw_cot(market_id)

        # Filter to reports available before as_of
        available = df[df['available_at'] <= as_of].copy()

        if available.empty:
            logger.warning(f"No COT data available for {market_id} as of {as_of}")
            return None

        # Get most recent available report
        latest = available.iloc[-1]

        # Compute percentile using only data available at that time
        mm_net_pct = self._compute_percentile(
            available['mm_net'],
            latest['mm_net'],
        )

        report_date = latest['report_date']
        if isinstance(report_date, pd.Timestamp):
            report_date = report_date.date()

        return COTSnapshot(
            market_id=market_id,
            mm_net=float(latest['mm_net']),
            mm_net_pct=mm_net_pct,
            report_date=report_date,
            available_at=latest['available_at'],
        )

    def _compute_percentile(
        self,
        series: pd.Series,
        value: float,
    ) -> float:
        """
        Compute percentile rank using rolling window.

        Uses only the last `percentile_lookback` observations.
        """
        window = series.tail(self.percentile_lookback)

        if len(window) < 2:
            return 50.0  # Default to median if insufficient data

        pct = (window <= value).sum() / len(window) * 100
        return float(pct)

    def fetch_cot_timeseries(
        self,
        market_id: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Get COT data timeseries with percentiles.

        Returns a DataFrame with one row per COT report in the date range,
        including computed percentile at that point in time (no look-ahead).

        Parameters
        ----------
        market_id : str
            Market identifier (WTI, RBOB, HO)
        start_date : date
            Start date (inclusive)
        end_date : date
            End date (inclusive)

        Returns
        -------
        pd.DataFrame
            Columns: report_date, mm_net, mm_net_pct, available_at
        """
        df = self.fetch_raw_cot(market_id)

        # Filter date range
        mask = (
            (df['report_date'] >= pd.Timestamp(start_date)) &
            (df['report_date'] <= pd.Timestamp(end_date))
        )
        filtered = df[mask].copy()

        if filtered.empty:
            return pd.DataFrame(
                columns=['report_date', 'mm_net', 'mm_net_pct', 'available_at']
            )

        # Compute rolling percentiles (using only data available at each point)
        percentiles = []
        for idx, row in filtered.iterrows():
            available = df[df['available_at'] <= row['available_at']]
            pct = self._compute_percentile(available['mm_net'], row['mm_net'])
            percentiles.append(pct)

        filtered['mm_net_pct'] = percentiles
        filtered['market_id'] = market_id

        return filtered[[
            'report_date', 'market_id', 'mm_net', 'mm_net_pct', 'available_at'
        ]].reset_index(drop=True)

    def get_session_cot(
        self,
        market_id: str,
        session_date: date,
    ) -> Optional[COTSnapshot]:
        """
        Get COT data available for a trading session's premarket.

        Trading sessions start at 6 PM ET the day before.
        Returns the COT data that would be available at premarket time (5:50 PM ET).

        Parameters
        ----------
        market_id : str
            Market identifier (WTI, RBOB, HO)
        session_date : date
            The trading session date

        Returns
        -------
        COTSnapshot or None
            COT data available at premarket
        """
        # Premarket is 5:50 PM ET the day before
        premarket_time = self._et.localize(datetime.combine(
            session_date - timedelta(days=1),
            time(17, 50)
        ))

        return self.get_available_cot(market_id, premarket_time)

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()
