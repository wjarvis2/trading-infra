"""
Spread data source - wraps SpreadFetcher with normalized output.

Provides spread price data for calendar spreads.

"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional
import pandas as pd
import logging

from common_lib.data_fetchers import SpreadFetcher, SpreadConfig, SpreadLeg
from .base import BaseDataSource, SourceMetadata

logger = logging.getLogger(__name__)


# Default CL calendar spreads
DEFAULT_CL_SPREADS = [
    SpreadConfig(
        name=f"CL{i}_CL{i+1}",
        spread_type="calendar",
        leg1=SpreadLeg(root="CL", position=i),
        leg2=SpreadLeg(root="CL", position=i+1),
    )
    for i in range(1, 7)  # CL1_CL2 through CL6_CL7
]


class SpreadSource(BaseDataSource):
    """
    Data source for calendar spread OHLC bars.

    Wraps SpreadFetcher and normalizes output for composition.

    Output columns (canonical names):
    - ts: timestamp (index column for joins)
    - instrument_id: spread identifier (e.g., "CL1_CL2")
    - open, high, low, close: spread OHLC (derived from leg OHLC)
    - front_close: front leg close price
    - back_close: back leg close price
    - front_symbol: front leg contract symbol
    - back_symbol: back leg contract symbol
    - volume: minimum volume of both legs

    Note: high/low are conservative envelope (assume favorable intrabar timing
    between legs). Suitable for slippage model bar-range proxy.

    Example
    -------
    >>> source = SpreadSource(
    ...     fetcher=SpreadFetcher(connection_string),
    ...     spreads=["CL1_CL2", "CL2_CL3"],
    ... )
    >>> df = source.load(date(2024, 6, 1), date(2024, 6, 30))
    """

    def __init__(
        self,
        fetcher: SpreadFetcher,
        spreads: Optional[List[str]] = None,
        spread_configs: Optional[List[SpreadConfig]] = None,
        bar_freq: str = "5s",
        trading_hours_only: bool = True,
    ):
        """
        Initialize the spread source.

        Parameters
        ----------
        fetcher : SpreadFetcher
            Underlying spread fetcher
        spreads : list[str], optional
            List of spread names to load (e.g., ["CL1_CL2", "CL2_CL3"]).
            If None, uses spread_configs or defaults.
        spread_configs : list[SpreadConfig], optional
            Explicit spread configurations. Overrides `spreads` if provided.
        bar_freq : str
            Bar frequency (default "5s")
        trading_hours_only : bool
            Filter to trading hours (default True)
        """
        self.fetcher = fetcher
        self.bar_freq = bar_freq
        self.trading_hours_only = trading_hours_only

        # Resolve spread configs
        if spread_configs:
            self.spread_configs = spread_configs
        elif spreads:
            self.spread_configs = self._build_configs_from_names(spreads)
        else:
            self.spread_configs = DEFAULT_CL_SPREADS[:5]  # Default: CL1-CL2 through CL5-CL6

        self._spread_names = [c.name for c in self.spread_configs]

    def _build_configs_from_names(self, names: List[str]) -> List[SpreadConfig]:
        """Build SpreadConfig objects from spread names like 'CL1_CL2'."""
        configs = []
        for name in names:
            parts = name.split("_")
            if len(parts) != 2:
                raise ValueError(f"Invalid spread name: {name}. Expected format: 'CL1_CL2'")

            root1 = ''.join(c for c in parts[0] if c.isalpha())
            pos1 = int(''.join(c for c in parts[0] if c.isdigit()))
            root2 = ''.join(c for c in parts[1] if c.isalpha())
            pos2 = int(''.join(c for c in parts[1] if c.isdigit()))

            if root1 != root2:
                raise ValueError(f"Inter-commodity spreads not supported: {name}")

            configs.append(SpreadConfig(
                name=name,
                spread_type="calendar",
                leg1=SpreadLeg(root=root1, position=pos1),
                leg2=SpreadLeg(root=root2, position=pos2),
            ))

        return configs

    @property
    def metadata(self) -> SourceMetadata:
        """Get source metadata."""
        return SourceMetadata(
            name="spread_prices",
            key_columns=["instrument_id"],  # Canonical name
            timestamp_column="ts",
            frequency=self.bar_freq,
            value_columns=[
                "open", "high", "low", "close",  # Full OHLC for slippage model
                "front_close", "back_close",
                "front_symbol", "back_symbol", "volume"
            ],
            description="Calendar spread OHLC bars from front8 leg data",
        )

    def load(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Load spread data for the specified date range.

        Parameters
        ----------
        start_date : date
            Start date (inclusive)
        end_date : date
            End date (inclusive)

        Returns
        -------
        pd.DataFrame
            Spread data with normalized columns
        """
        frames = []

        for config in self.spread_configs:
            try:
                df = self.fetcher.fetch_spread_front8(
                    config=config,
                    start_time=datetime.combine(start_date, datetime.min.time()),
                    end_time=datetime.combine(end_date, datetime.max.time()),
                    trading_hours_only=self.trading_hours_only,
                )

                # Normalize to canonical field names
                # Fetcher now outputs canonical OHLC names directly
                df = df.reset_index()  # ts becomes column
                df = df.rename(columns={
                    "leg1_close": "front_close",
                    "leg2_close": "back_close",
                    "leg1_symbol": "front_symbol",
                    "leg2_symbol": "back_symbol",
                })
                df["instrument_id"] = config.name  # Canonical: instrument_id

                frames.append(df)

            except Exception as e:
                logger.warning(f"Failed to load {config.name}: {e}")
                continue

        if not frames:
            raise ValueError(
                f"No spread data loaded for {self._spread_names} "
                f"between {start_date} and {end_date}"
            )

        result = pd.concat(frames, ignore_index=True)

        # Ensure ts is datetime
        if not pd.api.types.is_datetime64_any_dtype(result['ts']):
            result['ts'] = pd.to_datetime(result['ts'])

        # Sort for consistent output (using canonical field name)
        result = result.sort_values(['ts', 'instrument_id']).reset_index(drop=True)

        # Select and order columns (canonical names with full OHLC)
        result = result[[
            'ts', 'instrument_id',
            'open', 'high', 'low', 'close',
            'front_close', 'back_close',
            'front_symbol', 'back_symbol', 'volume'
        ]]

        self.validate_output(result)

        logger.info(
            f"Loaded {len(result)} spread bars for {len(self._spread_names)} spreads "
            f"({result['ts'].min()} to {result['ts'].max()})"
        )

        return result

    @property
    def spread_names(self) -> List[str]:
        """Get list of spread names this source provides."""
        return self._spread_names
