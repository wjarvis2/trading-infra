"""
Data Aligner for Multi-Frequency Data
Handles alignment of different data types to a canonical time grid
"""

from typing import Optional, Dict, Any, Literal, Union
import pandas as pd
import numpy as np
from datetime import timedelta


class AlignmentError(Exception):
    """Raised when data alignment fails"""
    pass


class DataAligner:
    """Handles multi-frequency data alignment with clear policies"""
    
    def __init__(self,
                 policy: Literal['inner', 'left', 'right', 'outer'] = 'left',
                 tolerance: Optional[Union[str, timedelta]] = '1min',
                 ffill_limit: int = 5,
                 bfill_limit: int = 0):
        """
        Initialize DataAligner with alignment policies
        
        Args:
            policy: Join policy for alignment ('inner', 'left', 'right', 'outer')
                   'left' keeps all bar timestamps (recommended)
            tolerance: Maximum time difference for matching across frequencies
            ffill_limit: Maximum forward fill periods for NaN handling
            bfill_limit: Maximum backward fill periods (0 = no backfill to avoid lookahead)
        """
        self.policy = policy
        self.tolerance = pd.Timedelta(tolerance) if isinstance(tolerance, str) else tolerance
        self.ffill_limit = ffill_limit
        self.bfill_limit = bfill_limit
        
    def align_to_bars(self,
                      bars: pd.DataFrame,
                      inventory: Optional[pd.DataFrame] = None,
                      curves: Optional[pd.DataFrame] = None,
                      spreads: Optional[pd.DataFrame] = None,
                      factors: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        Align all data types to the canonical bar time grid
        
        Args:
            bars: Bar data with MultiIndex (timestamp, symbol) - defines canonical grid
            inventory: Fundamental data to align (lower frequency)
            curves: Forward curve data to align  
            spreads: Spread data to align
            factors: Factor/PCA data to align
            
        Returns:
            Dictionary with all aligned dataframes
            
        Raises:
            AlignmentError: If alignment fails
        """
        if bars is None or bars.empty:
            raise AlignmentError("Bars cannot be empty - they define the canonical time grid")
            
        result = {'bars': bars}
        
        # Get unique timestamps from bars (canonical grid)
        if isinstance(bars.index, pd.MultiIndex):
            canonical_timestamps = bars.index.get_level_values('timestamp').unique().sort_values()
        else:
            canonical_timestamps = bars.index.unique().sort_values()
            
        # Align inventory data (typically daily/weekly frequency)
        if inventory is not None and not inventory.empty:
            result['inventory'] = self._align_lower_frequency(
                inventory, canonical_timestamps, 'inventory'
            )
            
        # Align curve data  
        if curves is not None and not curves.empty:
            result['curves'] = self._align_curves(curves, canonical_timestamps)
            
        # Align spread data (should be same frequency as bars)
        if spreads is not None and not spreads.empty:
            result['spreads'] = self._align_same_frequency(
                spreads, canonical_timestamps, 'spreads'
            )
            
        # Align factor data
        if factors is not None and not factors.empty:
            result['factors'] = self._align_same_frequency(
                factors, canonical_timestamps, 'factors'
            )
            
        return result
    
    def _align_lower_frequency(self,
                               df: pd.DataFrame,
                               target_timestamps: pd.DatetimeIndex,
                               data_type: str) -> pd.DataFrame:
        """
        Align lower frequency data (e.g., daily inventory) to higher frequency grid
        Uses forward fill with limits to avoid excessive staleness
        """
        # Handle MultiIndex case
        if isinstance(df.index, pd.MultiIndex):
            # For inventory with (date, series_code) index
            aligned_dfs = []
            
            # Get series codes
            series_codes = df.index.get_level_values(1).unique()
            
            for series_code in series_codes:
                series_data = df.xs(series_code, level=1)
                
                # Reindex to target timestamps with forward fill
                aligned = series_data.reindex(target_timestamps, method=None)
                
                # Forward fill with limit (no backfill to avoid lookahead)
                if self.ffill_limit > 0:
                    aligned = aligned.ffill(limit=self.ffill_limit)
                    
                # Add series_code back
                aligned['series_code'] = series_code
                aligned_dfs.append(aligned)
                
            # Combine all series
            if aligned_dfs:
                result = pd.concat(aligned_dfs, ignore_index=False)
                result = result.reset_index()
                result = result.rename(columns={'index': 'timestamp'})
                result = result.set_index(['timestamp', 'series_code'])
            else:
                result = pd.DataFrame()
            
        else:
            # Simple index case
            result = df.reindex(target_timestamps, method=None)
            
            # Forward fill with limit
            if self.ffill_limit > 0:
                result = result.ffill(limit=self.ffill_limit)
                
        # Check for excessive NaNs after alignment
        if not result.empty:
            nan_ratio = result.isna().sum().sum() / result.size
            if nan_ratio > 0.8:  # More lenient for sparse data
                raise AlignmentError(
                    f"{data_type} alignment resulted in {nan_ratio:.1%} NaN values. "
                    f"Consider increasing ffill_limit or checking data availability."
                )
            
        return result
    
    def _align_curves(self, 
                     curves: pd.DataFrame,
                     target_timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Align forward curve data
        Curves need special handling to preserve contract structure
        """
        if isinstance(curves.index, pd.MultiIndex):
            # Curves with (date, contract) index
            aligned_dfs = []
            
            # Get unique contracts
            contracts = curves.index.get_level_values(1).unique()
            
            for contract in contracts:
                contract_data = curves.xs(contract, level=1)
                
                # Reindex to target timestamps
                aligned = contract_data.reindex(target_timestamps, method=None)
                
                # Forward fill prices (curves change less frequently)
                if self.ffill_limit > 0:
                    aligned = aligned.ffill(limit=self.ffill_limit * 2)  # More aggressive for curves
                    
                # Add contract back
                aligned['contract'] = contract
                aligned_dfs.append(aligned)
                
            # Combine all contracts
            result = pd.concat(aligned_dfs, ignore_index=False)
            result = result.reset_index().set_index(['timestamp', 'contract'])
            result.index.names = ['timestamp', 'contract']
            
        else:
            # Fallback to simple alignment
            result = curves.reindex(target_timestamps, method=None)
            if self.ffill_limit > 0:
                result = result.ffill(limit=self.ffill_limit)
                
        return result
    
    def _align_same_frequency(self,
                              df: pd.DataFrame,
                              target_timestamps: pd.DatetimeIndex,
                              data_type: str) -> pd.DataFrame:
        """
        Align same-frequency data with tolerance matching
        Used for spreads and factors that should match bar frequency
        """
        # For same frequency, we expect near-perfect alignment
        if isinstance(df.index, pd.MultiIndex):
            df_timestamps = df.index.get_level_values(0).unique()
        else:
            df_timestamps = df.index
            
        # Check timestamp alignment with tolerance
        if self.tolerance is not None:
            # Find closest matches within tolerance
            aligned = self._match_with_tolerance(df, target_timestamps)
        else:
            # Exact matching
            if isinstance(df.index, pd.MultiIndex):
                # Need to handle MultiIndex carefully
                aligned = df.reindex(target_timestamps, level=0)
            else:
                aligned = df.reindex(target_timestamps)
                
        # Minimal forward fill for small gaps
        if self.ffill_limit > 0 and aligned.isna().any().any():
            aligned = aligned.ffill(limit=min(self.ffill_limit, 2))
            
        # Check alignment quality
        nan_ratio = aligned.isna().sum().sum() / aligned.size
        if nan_ratio > 0.1:  # More strict for same-frequency data
            raise AlignmentError(
                f"{data_type} alignment resulted in {nan_ratio:.1%} NaN values. "
                f"Data may be on different time grid than bars."
            )
            
        return aligned
    
    def _match_with_tolerance(self,
                             df: pd.DataFrame,
                             target_timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Match timestamps with tolerance for small discrepancies
        Useful when data sources have slight timestamp differences
        """
        # This is a simplified implementation
        # In production, would use pd.merge_asof or similar
        
        if isinstance(df.index, pd.MultiIndex):
            # For MultiIndex, this is more complex
            # Simplified: just reindex for now
            return df.reindex(target_timestamps, level=0)
        else:
            # For simple index, use merge_asof logic
            df_reset = df.reset_index()
            target_df = pd.DataFrame(index=target_timestamps)
            target_df = target_df.reset_index()
            
            # Merge with tolerance
            merged = pd.merge_asof(
                target_df,
                df_reset,
                left_on='index',
                right_on=df_reset.columns[0],
                tolerance=self.tolerance,
                direction='backward'  # Avoid lookahead
            )
            
            merged = merged.set_index('index')
            merged = merged.drop(columns=[df_reset.columns[0]])
            
            return merged
    
    def validate_alignment(self, aligned_data: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate that all dataframes are properly aligned
        
        Returns:
            True if validation passes
            
        Raises:
            AlignmentError: If validation fails
        """
        if 'bars' not in aligned_data:
            raise AlignmentError("Bars must be present in aligned data")
            
        bars = aligned_data['bars']
        
        # Get canonical timestamps
        if isinstance(bars.index, pd.MultiIndex):
            canonical_timestamps = bars.index.get_level_values('timestamp').unique()
        else:
            canonical_timestamps = bars.index
            
        # Check each dataframe
        for name, df in aligned_data.items():
            if name == 'bars' or df is None:
                continue
                
            # Check timestamp alignment
            if isinstance(df.index, pd.MultiIndex):
                df_timestamps = df.index.get_level_values(0).unique()
            else:
                df_timestamps = df.index
                
            # All timestamps should be in canonical set
            extra_timestamps = set(df_timestamps) - set(canonical_timestamps)
            if extra_timestamps:
                raise AlignmentError(
                    f"{name} has timestamps not in canonical grid: {list(extra_timestamps)[:5]}"
                )
                
        return True