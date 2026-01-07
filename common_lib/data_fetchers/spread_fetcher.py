"""
Spread data fetcher for calendar and crack spreads.

This module provides functionality to fetch and calculate various spread types,
including calendar spreads (same product, different months) and crack spreads
(refinery margin between crude and products).

CRITICAL: No synthetic data - all calculations from real market data only.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
import yaml

from common_lib.data_fetchers.bar_fetchers import BarFetcher

logger = logging.getLogger(__name__)


@dataclass
class SpreadLeg:
    """Definition of a single spread leg."""
    root: str
    position: int  # 1=front, 2=second, etc.
    multiplier: float = 1.0


@dataclass
class SpreadConfig:
    """
    Generic spread configuration - supports any spread type.

    Can be loaded from YAML or constructed programmatically.

    Examples
    --------
    >>> # CL M1-M2 calendar spread
    >>> config = SpreadConfig(
    ...     name="CL_M1_M2",
    ...     spread_type="calendar",
    ...     leg1=SpreadLeg(root="CL", position=1),
    ...     leg2=SpreadLeg(root="CL", position=2),
    ...     ratio=1.0
    ... )

    >>> # RB-CL crack spread
    >>> config = SpreadConfig(
    ...     name="RB_CL_crack",
    ...     spread_type="crack",
    ...     leg1=SpreadLeg(root="RB", position=1, multiplier=42.0),
    ...     leg2=SpreadLeg(root="CL", position=1),
    ...     ratio=1.0
    ... )
    """
    name: str
    spread_type: str  # 'calendar', 'crack', 'inter_commodity'
    leg1: SpreadLeg
    leg2: SpreadLeg
    ratio: float = 1.0  # spread = leg1 * leg1.mult - ratio * leg2 * leg2.mult
    tick_size: float = 0.01
    tick_value: float = 10.0
    description: str = ""

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SpreadConfig":
        """Load spread config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        spread = data["spread"]
        return cls(
            name=spread["name"],
            spread_type=spread["type"],
            leg1=SpreadLeg(
                root=spread["leg1"]["root"],
                position=spread["leg1"]["month_offset"],
                multiplier=spread["leg1"].get("multiplier", 1.0),
            ),
            leg2=SpreadLeg(
                root=spread["leg2"]["root"],
                position=spread["leg2"]["month_offset"],
                multiplier=spread["leg2"].get("multiplier", 1.0),
            ),
            ratio=spread.get("ratio", 1.0),
            tick_size=spread.get("tick_size", 0.01),
            tick_value=spread.get("tick_value", 10.0),
            description=spread.get("description", ""),
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "spread_type": self.spread_type,
            "leg1": {"root": self.leg1.root, "position": self.leg1.position, "multiplier": self.leg1.multiplier},
            "leg2": {"root": self.leg2.root, "position": self.leg2.position, "multiplier": self.leg2.multiplier},
            "ratio": self.ratio,
            "tick_size": self.tick_size,
            "tick_value": self.tick_value,
        }


@dataclass
class CrackSpreadConfig:
    """Configuration for a crack spread."""
    name: str
    crude_ratio: int
    gasoline_ratio: int
    distillate_ratio: int
    description: str
    
    @property
    def total_crude_barrels(self) -> int:
        """Total barrels of crude input."""
        return self.crude_ratio
    
    @property
    def total_product_barrels(self) -> int:
        """Total barrels of product output."""
        return self.gasoline_ratio + self.distillate_ratio


# Standard crack spread configurations
CRACK_SPREADS = {
    '321': CrackSpreadConfig(
        name='321_crack',
        crude_ratio=3,
        gasoline_ratio=2,
        distillate_ratio=1,
        description='3:2:1 Gulf Coast crack spread'
    ),
    '532': CrackSpreadConfig(
        name='532_crack',
        crude_ratio=5,
        gasoline_ratio=3,
        distillate_ratio=2,
        description='5:3:2 complex refinery crack'
    ),
    '211': CrackSpreadConfig(
        name='211_crack',
        crude_ratio=2,
        gasoline_ratio=1,
        distillate_ratio=1,
        description='2:1:1 coking crack spread'
    )
}


class SpreadFetcher:
    """
    Fetcher for spread data including calendar and crack spreads.
    
    This class handles fetching individual contract data and calculating
    spreads with proper unit conversions and error handling.
    """
    
    GALLONS_PER_BARREL = 42.0
    
    def __init__(self, connection_string: str):
        """
        Initialize spread fetcher.
        
        Args:
            connection_string: Database connection string
        """
        self.bar_fetcher = BarFetcher(connection_string)
        self.connection_string = connection_string
        
    def fetch_calendar_spread(
        self,
        front_contract: str,
        back_contract: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = '15s'
    ) -> pd.DataFrame:
        """
        Fetch calendar spread prices (front - back).
        
        Args:
            front_contract: Front month contract (e.g., 'CL1')
            back_contract: Back month contract (e.g., 'CL2')
            start_date: Start of data range
            end_date: End of data range
            frequency: Bar frequency (default 15s)
            
        Returns:
            DataFrame with columns: timestamp, spread_price, front_price, back_price
            
        Raises:
            ValueError: If data is missing or invalid
        """
        try:
            # Fetch front and back contract data
            front_bars = self.bar_fetcher.fetch_continuous_bars(
                root=front_contract[:2],  # Extract root (e.g., 'CL' from 'CL1')
                positions=[int(front_contract[2:])],  # Extract position (e.g., 1 from 'CL1')
                start_date=start_date,
                end_date=end_date,
                freq=frequency
            )
            
            back_bars = self.bar_fetcher.fetch_continuous_bars(
                root=back_contract[:2],
                positions=[int(back_contract[2:])],
                start_date=start_date,
                end_date=end_date,
                freq=frequency
            )
            
            if front_bars.empty or back_bars.empty:
                raise ValueError(f"No data found for {front_contract} or {back_contract}")
            
            # Align data on timestamps
            front_close = front_bars['close'].unstack(level='position')[int(front_contract[2:])]
            back_close = back_bars['close'].unstack(level='position')[int(back_contract[2:])]
            
            # Create spread DataFrame
            spread_df = pd.DataFrame({
                'front_price': front_close,
                'back_price': back_close
            })
            
            # Calculate spread (can be negative)
            spread_df['spread_price'] = spread_df['front_price'] - spread_df['back_price']
            
            # Drop rows with missing data
            spread_df = spread_df.dropna()
            
            if spread_df.empty:
                raise ValueError(f"No overlapping data for {front_contract} and {back_contract}")
            
            # Reset index to have timestamp as column
            spread_df = spread_df.reset_index()
            spread_df = spread_df.rename(columns={'bucket': 'timestamp'})
            
            logger.info(f"Fetched {len(spread_df)} calendar spread bars for {front_contract}-{back_contract}")
            
            return spread_df
            
        except Exception as e:
            logger.error(f"Error fetching calendar spread: {e}")
            raise
    
    def fetch_crack_spread(
        self,
        crude_contract: str,
        gasoline_contract: str,
        distillate_contract: str,
        crack_type: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = '15s'
    ) -> pd.DataFrame:
        """
        Fetch crack spread data with proper unit conversions.
        
        Crack spread = (Product revenues) - (Crude costs)
        Products are quoted in $/gallon, crude in $/barrel
        
        Args:
            crude_contract: Crude oil contract (e.g., 'CL1')
            gasoline_contract: RBOB gasoline contract (e.g., 'RB1')
            distillate_contract: Heating oil/ULSD contract (e.g., 'HO1')
            crack_type: Type of crack spread ('321', '532', '211')
            start_date: Start of data range
            end_date: End of data range
            frequency: Bar frequency (default 15s)
            
        Returns:
            DataFrame with columns: timestamp, crack_value, crude_price, 
                                  gasoline_price, distillate_price, components
                                  
        Raises:
            ValueError: If crack_type is invalid or data is missing
        """
        if crack_type not in CRACK_SPREADS:
            raise ValueError(f"Invalid crack type: {crack_type}. Must be one of {list(CRACK_SPREADS.keys())}")
        
        config = CRACK_SPREADS[crack_type]
        
        try:
            # Fetch all three components
            crude_bars = self._fetch_contract_data(crude_contract, start_date, end_date, frequency)
            gasoline_bars = self._fetch_contract_data(gasoline_contract, start_date, end_date, frequency)
            distillate_bars = self._fetch_contract_data(distillate_contract, start_date, end_date, frequency)
            
            # Align all three series
            prices_df = pd.DataFrame({
                'crude': crude_bars,
                'gasoline': gasoline_bars,
                'distillate': distillate_bars
            })
            
            # Drop rows with any missing data
            prices_df = prices_df.dropna()
            
            if prices_df.empty:
                raise ValueError(f"No overlapping data for crack spread components")
            
            # Calculate crack spread value
            # Products in $/gallon need conversion to $/barrel
            product_revenue = (
                config.gasoline_ratio * prices_df['gasoline'] * self.GALLONS_PER_BARREL +
                config.distillate_ratio * prices_df['distillate'] * self.GALLONS_PER_BARREL
            )
            
            crude_cost = config.crude_ratio * prices_df['crude']
            
            crack_value = product_revenue - crude_cost
            
            # Create result DataFrame
            result_df = pd.DataFrame({
                'timestamp': prices_df.index,
                'crack_value': crack_value.values,
                'crude_price': prices_df['crude'].values,
                'gasoline_price': prices_df['gasoline'].values,
                'distillate_price': prices_df['distillate'].values,
                'crude_cost': crude_cost.values,
                'product_revenue': product_revenue.values,
                'crack_type': crack_type
            })
            
            # Add per-barrel crack for easier comparison
            result_df['crack_per_barrel'] = result_df['crack_value'] / config.crude_ratio
            
            logger.info(f"Fetched {len(result_df)} crack spread bars for {crack_type}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error fetching crack spread: {e}")
            raise
    
    def _fetch_contract_data(
        self,
        contract: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str
    ) -> pd.Series:
        """
        Fetch close prices for a single contract.
        
        Args:
            contract: Contract symbol (e.g., 'CL1')
            start_date: Start of data range
            end_date: End of data range
            frequency: Bar frequency
            
        Returns:
            Series of close prices indexed by timestamp
            
        Raises:
            ValueError: If no data found
        """
        root = contract[:2]
        position = int(contract[2:])
        
        bars = self.bar_fetcher.fetch_continuous_bars(
            root=root,
            positions=[position],
            start_date=start_date,
            end_date=end_date,
            freq=frequency
        )
        
        if bars.empty:
            raise ValueError(f"No data found for {contract}")
        
        # Extract close prices
        close_prices = bars['close'].unstack(level='position')[position]
        
        return close_prices
    
    def calculate_crack_statistics(
        self,
        crack_df: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate rolling statistics for crack spread.
        
        Args:
            crack_df: DataFrame with crack spread data
            window: Rolling window size (default 20)
            
        Returns:
            DataFrame with added statistical columns
        """
        if 'crack_value' not in crack_df.columns:
            raise ValueError("crack_df must contain 'crack_value' column")
        
        # Calculate rolling statistics
        crack_df['crack_mean'] = crack_df['crack_value'].rolling(window=window).mean()
        crack_df['crack_std'] = crack_df['crack_value'].rolling(window=window).std()
        
        # Calculate z-score
        crack_df['crack_zscore'] = (
            (crack_df['crack_value'] - crack_df['crack_mean']) / 
            crack_df['crack_std']
        )
        
        # Calculate percentile rank
        crack_df['crack_percentile'] = crack_df['crack_value'].rolling(window=window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
        )

        return crack_df

    def fetch_spread_front8(
        self,
        config: SpreadConfig,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        trading_hours_only: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch any spread from bars_5s_front8 table using SpreadConfig.

        This is the primary method for loading spread data for intraday strategies.
        Supports calendar spreads, crack spreads, and inter-commodity spreads.

        Parameters
        ----------
        config : SpreadConfig
            Spread configuration defining legs and ratio
        start_time : str or datetime
            Start timestamp
        end_time : str or datetime
            End timestamp
        trading_hours_only : bool
            If True, filter to CME trading hours

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
                - ts: timestamp (index)
                - spread: calculated spread price
                - leg1_close: leg1 close price
                - leg2_close: leg2 close price
                - leg1_symbol: leg1 contract symbol
                - leg2_symbol: leg2 contract symbol
                - volume: min volume of both legs

        Examples
        --------
        >>> config = SpreadConfig.from_yaml("config/spreads/cl_m1_m2.yaml")
        >>> df = fetcher.fetch_spread_front8(config, "2024-01-01", "2024-01-31")
        """
        # Fetch leg1 data
        leg1_df = self.bar_fetcher.fetch_front8_bars(
            root=config.leg1.root,
            positions=[config.leg1.position],
            start_time=start_time,
            end_time=end_time,
            trading_hours_only=trading_hours_only,
        )

        # Fetch leg2 data
        leg2_df = self.bar_fetcher.fetch_front8_bars(
            root=config.leg2.root,
            positions=[config.leg2.position],
            start_time=start_time,
            end_time=end_time,
            trading_hours_only=trading_hours_only,
        )

        # Reset index to align on timestamp
        leg1_df = leg1_df.reset_index()
        leg2_df = leg2_df.reset_index()

        # Merge on timestamp with full OHLC - inner join drops unaligned bars
        merge_cols = ['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        merged = pd.merge(
            leg1_df[merge_cols],
            leg2_df[merge_cols],
            on='ts',
            suffixes=('_leg1', '_leg2'),
            how='inner'
        )

        if merged.empty:
            raise ValueError(
                f"No overlapping data for spread {config.name} "
                f"between {start_time} and {end_time}"
            )

        # Calculate spread OHLC: spread = leg1 * mult1 - ratio * leg2 * mult2
        # high/low are conservative envelope (assume favorable intrabar timing)
        mult1 = config.leg1.multiplier
        mult2 = config.leg2.multiplier * config.ratio

        spread_open = merged['open_leg1'] * mult1 - merged['open_leg2'] * mult2
        spread_high = merged['high_leg1'] * mult1 - merged['low_leg2'] * mult2
        spread_low = merged['low_leg1'] * mult1 - merged['high_leg2'] * mult2
        spread_close = merged['close_leg1'] * mult1 - merged['close_leg2'] * mult2

        # Canonical OHLC column names
        result = pd.DataFrame({
            'ts': merged['ts'],
            'open': spread_open,
            'high': spread_high,
            'low': spread_low,
            'close': spread_close,
            'leg1_close': merged['close_leg1'],
            'leg2_close': merged['close_leg2'],
            'leg1_symbol': merged['symbol_leg1'],
            'leg2_symbol': merged['symbol_leg2'],
            'volume': merged[['volume_leg1', 'volume_leg2']].min(axis=1),
        })

        result.set_index('ts', inplace=True)

        # Add metadata
        result.attrs['spread_config'] = config
        result.attrs['tick_value'] = config.tick_value

        logger.info(
            f"Fetched {len(result)} bars for {config.name} spread "
            f"({result.index.min()} to {result.index.max()})"
        )

        return result

    def calculate_spread_zscore(
        self,
        spread_df: pd.DataFrame,
        window: int = 2880,  # 4 hours at 5s bars
        col: str = 'spread'
    ) -> pd.DataFrame:
        """
        Calculate rolling z-score for spread data.

        Parameters
        ----------
        spread_df : pd.DataFrame
            DataFrame with spread column
        window : int
            Rolling window size in bars (default: 2880 = 4 hours at 5s)
        col : str
            Column name for spread values

        Returns
        -------
        pd.DataFrame
            Original DataFrame with added columns: spread_mean, spread_std, zscore
        """
        if col not in spread_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

        df = spread_df.copy()
        df['spread_mean'] = df[col].rolling(window=window).mean()
        df['spread_std'] = df[col].rolling(window=window).std()
        df['zscore'] = (df[col] - df['spread_mean']) / df['spread_std']

        return df