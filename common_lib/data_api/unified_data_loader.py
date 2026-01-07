"""
Unified Data Loader for Trading System
Pure orchestration layer that delegates to fetchers
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Protocol
from datetime import datetime, timedelta
import pandas as pd
import re
from abc import abstractmethod


class NoDataError(Exception):
    """Raised when requested data is not available"""
    pass


class LoaderConfigError(Exception):
    """Raised when loader configuration is invalid"""
    pass


@dataclass
class LoadedData:
    """
    Typed return container for loaded data
    Replaces ambiguous dict returns
    """
    bars: Optional[pd.DataFrame] = None
    inventory: Optional[pd.DataFrame] = None
    curves: Optional[pd.DataFrame] = None
    spreads: Optional[pd.DataFrame] = None
    factors: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, pd.DataFrame]:
        """Convert to dictionary for compatibility"""
        result = {}
        if self.bars is not None:
            result['bars'] = self.bars
        if self.inventory is not None:
            result['inventory'] = self.inventory
        if self.curves is not None:
            result['curves'] = self.curves
        if self.spreads is not None:
            result['spreads'] = self.spreads
        if self.factors is not None:
            result['factors'] = self.factors
        return result
    
    def has_data(self) -> bool:
        """Check if any data is loaded"""
        return any([
            self.bars is not None,
            self.inventory is not None,
            self.curves is not None,
            self.spreads is not None,
            self.factors is not None
        ])


class IFetcher(Protocol):
    """Protocol for all data fetchers"""
    
    @abstractmethod
    def fetch(self, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """Fetch data for date range"""
        pass


class UnifiedDataLoader:
    """
    Pure orchestration layer for data loading
    No data manipulation, only coordination
    """
    
    # Regex for continuous contract parsing (CL1-CL12, not month codes like CLZ5)
    # Only matches 1-12 for continuous contracts, not month/year contracts
    # Checks that the root doesn't end with a month code letter
    CONTINUOUS_CONTRACT_PATTERN = re.compile(r'^([A-Z]{2,}?)(1[0-2]|[1-9])$')
    # Month codes that indicate specific contracts, not continuous
    MONTH_CODES = set('FGHJKMNQUVXZ')
    
    def __init__(self, fetchers: Dict[str, IFetcher]):
        """
        Initialize with dependency-injected fetchers
        
        Args:
            fetchers: Dictionary of fetcher instances by type
                     e.g., {'bar': BarFetcher(), 'spread': SpreadFetcher()}
        """
        if not fetchers:
            raise LoaderConfigError("At least one fetcher must be provided")
        self.fetchers = fetchers
        
    def load_for_backtest(self,
                         start_date: datetime,
                         end_date: datetime,
                         contracts: List[str],
                         warmup_bars: int = 240,
                         bar_type: str = '15s',
                         load_inventory: bool = False,
                         load_curves: bool = False,
                         load_spreads: bool = False,
                         load_factors: bool = False,
                         spread_configs: Optional[List[Dict]] = None) -> LoadedData:
        """
        Load all data needed for backtesting
        
        Args:
            start_date: Start of backtest period (UTC)
            end_date: End of backtest period (UTC)
            contracts: List of contracts to load (e.g., ['CL1', 'CL2', 'RB1'])
            warmup_bars: Number of bars before start_date for indicator warmup
            bar_type: Bar frequency ('15s', '1m', '5m', etc.)
            load_inventory: Whether to load fundamental inventory data
            load_curves: Whether to load forward curves
            load_spreads: Whether to load spread data
            load_factors: Whether to load PCA/factor data
            spread_configs: Configuration for spread calculations
            
        Returns:
            LoadedData object with all requested data
            
        Raises:
            NoDataError: If required data cannot be loaded
            LoaderConfigError: If configuration is invalid
        """
        # Validate inputs
        if not contracts:
            raise LoaderConfigError("At least one contract must be specified")
            
        if end_date <= start_date:
            raise LoaderConfigError("end_date must be after start_date")
            
        # Parse contracts with robust regex
        parsed_contracts = self._parse_contracts(contracts)
        
        # Calculate actual fetch start for warmup
        fetch_start = self._calculate_warmup_start(start_date, warmup_bars, bar_type)
        
        result = LoadedData()
        result.metadata['start_date'] = start_date
        result.metadata['end_date'] = end_date
        result.metadata['fetch_start'] = fetch_start
        result.metadata['contracts'] = contracts
        result.metadata['parsed_contracts'] = parsed_contracts
        result.metadata['bar_type'] = bar_type
        result.metadata['warmup_bars'] = warmup_bars
        
        # Load bars (primary data)
        if 'bar' in self.fetchers:
            try:
                bars_list = []
                for contract in contracts:
                    contract_bars = self.fetchers['bar'].fetch(
                        start_date=fetch_start,
                        end_date=end_date,
                        contracts=[contract],
                        bar_type=bar_type
                    )
                    if contract_bars is not None and not contract_bars.empty:
                        # Add symbol column if not present
                        if 'symbol' not in contract_bars.columns:
                            contract_bars['symbol'] = contract
                        bars_list.append(contract_bars)
                
                if bars_list:
                    # Concatenate preserving contract identity
                    result.bars = pd.concat(bars_list, ignore_index=False)
                else:
                    raise NoDataError(f"No bar data available for contracts {contracts}")
                    
            except Exception as e:
                raise NoDataError(f"Failed to load bar data: {e}")
        else:
            if not load_inventory:  # Bars are optional only if loading other data
                raise LoaderConfigError("Bar fetcher not provided but bars requested")
                
        # Load inventory/fundamental data
        if load_inventory and 'inventory' in self.fetchers:
            try:
                result.inventory = self.fetchers['inventory'].fetch(
                    start_date=fetch_start,
                    end_date=end_date
                )
            except Exception as e:
                # Inventory is often optional, log but don't fail
                result.metadata['inventory_error'] = str(e)
                
        # Load forward curves
        if load_curves and 'curve' in self.fetchers:
            try:
                # Curves might need contract roots
                roots = list(set(p['root'] for p in parsed_contracts))
                result.curves = self.fetchers['curve'].fetch(
                    start_date=fetch_start,
                    end_date=end_date,
                    roots=roots
                )
            except Exception as e:
                result.metadata['curve_error'] = str(e)
                
        # Load spread data
        if load_spreads and 'spread' in self.fetchers:
            try:
                if spread_configs:
                    result.spreads = self.fetchers['spread'].fetch(
                        start_date=fetch_start,
                        end_date=end_date,
                        spread_configs=spread_configs
                    )
                else:
                    # Default spread configs based on contracts
                    result.spreads = self._load_default_spreads(
                        fetch_start, end_date, parsed_contracts
                    )
            except Exception as e:
                result.metadata['spread_error'] = str(e)
                
        # Load factor/PCA data
        if load_factors and 'factor' in self.fetchers:
            try:
                result.factors = self.fetchers['factor'].fetch(
                    start_date=fetch_start,
                    end_date=end_date
                )
            except Exception as e:
                result.metadata['factor_error'] = str(e)
                
        # Validate we got something
        if not result.has_data():
            raise NoDataError("No data could be loaded for specified parameters")
            
        return result
    
    def _parse_contracts(self, contracts: List[str]) -> List[Dict[str, Any]]:
        """
        Parse contract symbols with robust regex
        Supports CL1, CL10, RB12, etc.
        
        Returns:
            List of dicts with 'root' and 'position' keys
        """
        parsed = []
        for contract in contracts:
            contract_upper = contract.upper()
            match = self.CONTINUOUS_CONTRACT_PATTERN.match(contract_upper)
            
            # Check if it's a continuous contract (not a month code)
            if match:
                root, position = match.groups()
                # Reject if the last letter of root is a month code (e.g., CLZ5)
                if len(root) > 2 and root[-1] in self.MONTH_CODES:
                    # This is a month/year contract, not continuous
                    parsed.append({
                        'symbol': contract_upper,
                        'root': contract_upper,
                        'position': None
                    })
                else:
                    # Valid continuous contract
                    parsed.append({
                        'symbol': contract_upper,
                        'root': root,
                        'position': int(position)
                    })
            else:
                # Try to handle special cases (e.g., specific month contracts)
                # For now, treat as opaque symbol
                parsed.append({
                    'symbol': contract_upper,
                    'root': contract_upper,
                    'position': None
                })
        return parsed
    
    def _calculate_warmup_start(self, 
                               start_date: datetime,
                               warmup_bars: int,
                               bar_type: str) -> datetime:
        """
        Calculate actual start date including warmup period
        Accounts for market hours and weekends
        """
        if warmup_bars <= 0:
            return start_date
            
        # Parse bar type
        if bar_type.endswith('s'):
            seconds = int(bar_type[:-1])
            bar_duration = timedelta(seconds=seconds)
        elif bar_type.endswith('m'):
            minutes = int(bar_type[:-1]) 
            bar_duration = timedelta(minutes=minutes)
        elif bar_type.endswith('h'):
            hours = int(bar_type[:-1])
            bar_duration = timedelta(hours=hours)
        else:
            # Default to 15 seconds
            bar_duration = timedelta(seconds=15)
            
        # Calculate raw offset
        raw_offset = bar_duration * warmup_bars
        
        # Add buffer for weekends/holidays (crude approximation)
        # Assume market is open 23 hours/day, 5 days/week
        market_hours_per_week = 23 * 5
        total_hours_per_week = 24 * 7
        market_ratio = market_hours_per_week / total_hours_per_week
        
        # Adjust for market hours
        adjusted_offset = raw_offset / market_ratio
        
        # Add some extra buffer
        buffer = timedelta(days=2)
        
        return start_date - adjusted_offset - buffer
    
    def _load_default_spreads(self,
                             start_date: datetime,
                             end_date: datetime,
                             parsed_contracts: List[Dict]) -> Optional[pd.DataFrame]:
        """
        Load default spreads based on available contracts
        E.g., calendar spreads for same root
        """
        if 'spread' not in self.fetchers:
            return None
            
        spread_configs = []
        
        # Group by root
        roots = {}
        for pc in parsed_contracts:
            root = pc['root']
            if root not in roots:
                roots[root] = []
            if pc['position'] is not None:
                roots[root].append(pc['position'])
                
        # Create calendar spreads for each root
        for root, positions in roots.items():
            positions = sorted(positions)
            
            # Adjacent month spreads
            for i in range(len(positions) - 1):
                spread_configs.append({
                    'type': 'calendar',
                    'leg1': f"{root}{positions[i]}",
                    'leg2': f"{root}{positions[i+1]}",
                    'ratio': [1, -1]
                })
                
        if spread_configs:
            return self.fetchers['spread'].fetch(
                start_date=start_date,
                end_date=end_date,
                spread_configs=spread_configs
            )
        
        return None
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate loader configuration
        
        Returns:
            True if valid
            
        Raises:
            LoaderConfigError: If invalid
        """
        required = ['start_date', 'end_date', 'contracts']
        missing = [k for k in required if k not in config]
        
        if missing:
            raise LoaderConfigError(f"Missing required config keys: {missing}")
            
        # Validate date types
        if not isinstance(config['start_date'], datetime):
            raise LoaderConfigError("start_date must be datetime")
        if not isinstance(config['end_date'], datetime):
            raise LoaderConfigError("end_date must be datetime")
            
        # Validate contracts
        if not isinstance(config['contracts'], list) or not config['contracts']:
            raise LoaderConfigError("contracts must be non-empty list")
            
        return True