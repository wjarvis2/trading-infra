"""
Python implementation of curve calculations matching SQL logic exactly.
This ensures consistency between SQL queries and Python code.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class CurveCalculator:
    """
    Python implementation of curve calculations that exactly matches SQL logic.
    Used for backtesting and validation of SQL results.
    """
    
    def __init__(self, lookback_days: int = 2):
        """
        Initialize curve calculator.
        
        Args:
            lookback_days: Number of days to look back (default 2 to match SQL)
        """
        self.lookback_days = lookback_days
    
    def calculate_curve_mid_5s(
        self,
        bar_5s_df: pd.DataFrame,
        continuous_contracts_df: pd.DataFrame,
        reference_time: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Python implementation of curve_mid_5s SQL logic.
        
        This exactly replicates the logic in build_curve_mid_5s.sql:
        1. Join bars with continuous contract mapping
        2. Calculate mid prices as (high + low) / 2
        3. Pivot to wide format with CL1-CL8 columns
        4. Filter for minimum contracts and volume
        
        Args:
            bar_5s_df: DataFrame with columns [ts, instrument_id, high, low, volume]
            continuous_contracts_df: DataFrame with columns [bucket, continuous_position, instrument_id]
            reference_time: Reference timestamp for lookback calculation
            
        Returns:
            DataFrame matching curve_mid_5s structure
        """
        # Step 1: Filter for lookback period
        start_time = reference_time - timedelta(days=self.lookback_days)
        
        # Filter continuous contracts for recent mappings
        contract_mapping = (
            continuous_contracts_df[
                (continuous_contracts_df['bucket'] >= start_time) &
                (continuous_contracts_df['continuous_position'] <= 8)
            ]
            .sort_values(['continuous_position', 'bucket'], ascending=[True, False])
            .drop_duplicates('continuous_position', keep='first')
        )
        
        # Step 2: Join bars with contract mapping
        bars_with_position = pd.merge(
            bar_5s_df[bar_5s_df['ts'] >= start_time],
            contract_mapping[['continuous_position', 'instrument_id']],
            on='instrument_id',
            how='inner'
        )
        
        # Step 3: Calculate mid prices
        bars_with_position['mid_price'] = (
            bars_with_position['high'] + bars_with_position['low']
        ) / 2.0
        
        # Step 4: Pivot to wide format
        pivot_df = (
            bars_with_position
            .pivot_table(
                index='ts',
                columns='continuous_position',
                values=['mid_price', 'volume'],
                aggfunc={'mid_price': 'first', 'volume': 'sum'}
            )
        )
        
        # Flatten column names
        pivot_df.columns = [f'{col[0]}_{col[1]}' if col[0] == 'volume' else f'cl{col[1]}' 
                           for col in pivot_df.columns]
        
        # Rename columns to match SQL output
        result_df = pd.DataFrame(index=pivot_df.index)
        
        # Add CL1-CL8 columns
        for i in range(1, 9):
            col_name = f'cl{i}'
            if f'mid_price_{i}' in pivot_df.columns:
                result_df[col_name] = pivot_df[f'mid_price_{i}']
            else:
                result_df[col_name] = np.nan
        
        # Add volume and contract count
        volume_cols = [col for col in pivot_df.columns if col.startswith('volume_')]
        if volume_cols:
            result_df['total_volume'] = pivot_df[volume_cols].sum(axis=1)
        else:
            result_df['total_volume'] = 0
            
        result_df['contracts_available'] = result_df[['cl1', 'cl2', 'cl3', 'cl4', 
                                                      'cl5', 'cl6', 'cl7', 'cl8']].notna().sum(axis=1)
        
        # Step 5: Apply filters matching SQL WHERE clause
        # Filter out rows with volume = 0 (matching SQL: AND volume > 0)
        result_df = result_df[result_df['total_volume'] > 0]
        
        # Filter for minimum contracts (matching SQL: HAVING COUNT >= 4)
        result_df = result_df[result_df['contracts_available'] >= 4]
        
        # Reset index to have ts as column
        result_df = result_df.reset_index()
        
        # Sort by ts descending to match SQL
        result_df = result_df.sort_values('ts', ascending=False)
        
        return result_df
    
    def calculate_spot_adjustment(
        self,
        curve_df: pd.DataFrame,
        expiry_map: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Apply spot adjustment to curve prices.
        
        This matches the logic in models/spot_price.py:
        1. Calculate carry rate from CL1-CL2 spread
        2. Discount CL1 back to spot
        3. Adjust all contracts relative to spot
        
        Args:
            curve_df: DataFrame with CL1-CL8 prices
            expiry_map: Dict mapping contract names to days to expiry
            
        Returns:
            DataFrame with spot-adjusted prices
        """
        result_df = curve_df.copy()
        
        # Calculate spot price for each row
        spot_prices = []
        
        for idx, row in curve_df.iterrows():
            if pd.notna(row.get('cl1')) and pd.notna(row.get('cl2')):
                # Get days to expiry
                cl1_days = expiry_map.get('cl1', 21)  # Default 21 days
                cl2_days = expiry_map.get('cl2', 52)  # Default 52 days
                
                # Calculate time periods in years
                time_to_cl1 = cl1_days / 365.25
                time_between = (cl2_days - cl1_days) / 365.25
                
                # Calculate carry rate
                if row['cl2'] > 0 and row['cl1'] > 0 and time_between > 0:
                    carry_rate = np.log(row['cl2'] / row['cl1']) / time_between
                    
                    # Discount to spot
                    spot = row['cl1'] * np.exp(-carry_rate * time_to_cl1)
                else:
                    # Fallback: use simple carry adjustment
                    carry = (row['cl2'] - row['cl1']) * 0.4
                    spot = row['cl1'] - carry
                
                spot_prices.append(spot)
            else:
                spot_prices.append(np.nan)
        
        result_df['spot'] = spot_prices
        
        # Adjust all contracts relative to spot
        for col in ['cl1', 'cl2', 'cl3', 'cl4', 'cl5', 'cl6', 'cl7', 'cl8']:
            if col in result_df.columns:
                result_df[f'{col}_adjusted'] = result_df[col] - result_df['spot']
        
        return result_df
    
    def validate_curve_data(
        self,
        curve_df: pd.DataFrame,
        min_contracts: int = 4,
        max_spread: float = 10.0,
        max_daily_move: float = 10.0
    ) -> Tuple[bool, float, str]:
        """
        Validate curve data quality.
        
        Args:
            curve_df: DataFrame with curve prices
            min_contracts: Minimum required contracts
            max_spread: Maximum allowed spread between contracts
            max_daily_move: Maximum allowed daily price move
            
        Returns:
            Tuple of (is_valid, quality_score, message)
        """
        # Check minimum contracts
        contracts_available = curve_df[['cl1', 'cl2', 'cl3', 'cl4', 
                                       'cl5', 'cl6', 'cl7', 'cl8']].notna().sum(axis=1).min()
        
        if contracts_available < min_contracts:
            return False, 0.0, f"Insufficient contracts: {contracts_available} < {min_contracts}"
        
        # Check spreads
        max_spread_observed = 0
        for i in range(1, 8):
            col1, col2 = f'cl{i}', f'cl{i+1}'
            if col1 in curve_df.columns and col2 in curve_df.columns:
                spread = (curve_df[col2] - curve_df[col1]).abs().max()
                max_spread_observed = max(max_spread_observed, spread)
        
        if max_spread_observed > max_spread:
            return False, 0.5, f"Excessive spread: {max_spread_observed:.2f} > {max_spread}"
        
        # Check daily moves
        price_cols = ['cl1', 'cl2', 'cl3', 'cl4', 'cl5', 'cl6', 'cl7', 'cl8']
        for col in price_cols:
            if col in curve_df.columns:
                daily_move = curve_df[col].diff().abs().max()
                if daily_move > max_daily_move:
                    return False, 0.7, f"Excessive daily move in {col}: {daily_move:.2f}"
        
        # Calculate quality score
        quality_score = min(1.0, contracts_available / 8.0)
        
        return True, quality_score, "Validation passed"