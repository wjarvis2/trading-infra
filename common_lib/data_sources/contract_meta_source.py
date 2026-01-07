"""
Contract metadata source - provides contract expiry and roll information.

Used for roll detection and position management.

"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .base import BaseDataSource, SourceMetadata

logger = logging.getLogger(__name__)


class ContractMetaSource(BaseDataSource):
    """
    Data source for contract metadata (expiry dates, symbols).

    Provides daily snapshots of which contracts map to which positions
    and their days to expiry. Used for roll detection.

    Output columns:
    - ts: date (for range/asof join)
    - root: product root (e.g., "CL")
    - position: continuous position (1=front, 2=second, etc.)
    - symbol: contract symbol (e.g., "CLZ24")
    - expiry: contract expiry date
    - days_to_expiry: days until expiry

    Example
    -------
    >>> source = ContractMetaSource(
    ...     engine=engine,
    ...     roots=["CL"],
    ...     positions=[1, 2, 3, 4, 5, 6],
    ... )
    >>> df = source.load(date(2024, 6, 1), date(2024, 6, 30))
    """

    def __init__(
        self,
        engine: Engine,
        roots: Optional[List[str]] = None,
        positions: Optional[List[int]] = None,
    ):
        """
        Initialize the contract metadata source.

        Parameters
        ----------
        engine : Engine
            SQLAlchemy engine for database access
        roots : list[str], optional
            Product roots to load (default: ["CL"])
        positions : list[int], optional
            Continuous positions to load (default: [1, 2, 3, 4, 5, 6])
        """
        self.engine = engine
        self.roots = roots or ["CL"]
        self.positions = positions or [1, 2, 3, 4, 5, 6]

    @property
    def metadata(self) -> SourceMetadata:
        """Get source metadata."""
        return SourceMetadata(
            name="contract_meta",
            key_columns=["root", "position"],
            timestamp_column="ts",
            frequency="1d",
            value_columns=["symbol", "front_expiry", "days_to_expiry"],  # Canonical: front_expiry
            description="Contract metadata with expiry dates",
        )

    def load(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Load contract metadata for the specified date range.

        Queries the continuous contract mapping to get which contract
        was at each position on each day.

        Parameters
        ----------
        start_date : date
            Start date (inclusive)
        end_date : date
            End date (inclusive)

        Returns
        -------
        pd.DataFrame
            Contract metadata with normalized columns
        """
        frames = []

        for root in self.roots:
            try:
                df = self._load_root_metadata(root, start_date, end_date)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {root}: {e}")
                continue

        if not frames:
            logger.warning(
                f"No contract metadata loaded for {self.roots} "
                f"between {start_date} and {end_date}"
            )
            return pd.DataFrame(columns=[
                'ts', 'root', 'position', 'symbol', 'front_expiry', 'days_to_expiry'
            ])

        result = pd.concat(frames, ignore_index=True)

        # Sort for consistent output
        result = result.sort_values(['ts', 'root', 'position']).reset_index(drop=True)

        self.validate_output(result)

        logger.info(
            f"Loaded {len(result)} contract metadata rows for {self.roots} "
            f"({result['ts'].min()} to {result['ts'].max()})"
        )

        return result

    def _load_root_metadata(
        self,
        root: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Load metadata for a single root."""
        # Query to get daily contract mappings
        query = text("""
            WITH daily_mappings AS (
                SELECT DISTINCT ON (bucket::date, continuous_position)
                    bucket::date as session_date,
                    continuous_position,
                    instrument_id as symbol
                FROM market_data.v_continuous_contracts
                WHERE root = :root
                  AND continuous_position = ANY(:positions)
                  AND bucket::date >= :start_date
                  AND bucket::date <= :end_date
                ORDER BY bucket::date, continuous_position, bucket DESC
            )
            SELECT
                dm.session_date,
                dm.continuous_position,
                dm.symbol,
                fc.expiry
            FROM daily_mappings dm
            JOIN market_data.futures_contracts fc ON dm.symbol = fc.instrument_id
            ORDER BY dm.session_date, dm.continuous_position
        """)

        df = pd.read_sql(
            query,
            self.engine,
            params={
                "root": root,
                "positions": self.positions,
                "start_date": start_date,
                "end_date": end_date,
            },
            parse_dates=['session_date', 'expiry'],
        )

        if df.empty:
            return df

        # Rename to canonical field names
        df = df.rename(columns={
            'session_date': 'ts',
            'continuous_position': 'position',
            'expiry': 'front_expiry',  # Canonical: front_expiry
        })
        df['root'] = root

        # Calculate days to expiry
        df['days_to_expiry'] = (df['front_expiry'] - df['ts']).dt.days

        # Convert ts to datetime for consistency
        df['ts'] = pd.to_datetime(df['ts'])

        # Select columns (canonical names)
        return df[['ts', 'root', 'position', 'symbol', 'front_expiry', 'days_to_expiry']]

    def get_contract_at_position(
        self,
        root: str,
        position: int,
        as_of: date,
    ) -> Optional[Dict]:
        """
        Get the contract at a specific position on a specific date.

        Convenience method for single-point lookups.

        Parameters
        ----------
        root : str
            Product root (e.g., "CL")
        position : int
            Continuous position (1=front, etc.)
        as_of : date
            Date to query

        Returns
        -------
        dict or None
            Contract info with keys: symbol, expiry, days_to_expiry
        """
        df = self.load(as_of, as_of)

        mask = (df['root'] == root) & (df['position'] == position)
        matched = df[mask]

        if matched.empty:
            return None

        row = matched.iloc[0]
        return {
            'symbol': row['symbol'],
            'front_expiry': row['front_expiry'],  # Canonical: front_expiry
            'days_to_expiry': row['days_to_expiry'],
        }
