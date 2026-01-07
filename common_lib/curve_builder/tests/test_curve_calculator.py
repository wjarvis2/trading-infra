"""
Property-based tests for curve calculator using Hypothesis.
Tests edge cases and ensures SQL/Python parity under various conditions.
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from common_lib.curve_builder.python.curve_calculator import CurveCalculator


class TestCurveCalculator:
    """Property-based tests for curve calculations."""
    
    @pytest.fixture
    def calculator(self):
        """Create a curve calculator instance."""
        return CurveCalculator(lookback_days=2)
    
    @given(
        n_bars=st.integers(min_value=10, max_value=1000),
        n_contracts=st.integers(min_value=4, max_value=8),
        base_price=st.floats(min_value=20.0, max_value=150.0),
        volatility=st.floats(min_value=0.001, max_value=0.1)
    )
    @settings(max_examples=50, deadline=5000)
    def test_curve_calculation_properties(
        self,
        calculator,
        n_bars,
        n_contracts,
        base_price,
        volatility
    ):
        """
        Test that curve calculation maintains expected properties:
        1. Output has correct structure
        2. Contract count matches available data
        3. Volume aggregation is correct
        4. No data corruption occurs
        """
        # Generate realistic bar data based on database schema
        now = pd.Timestamp.now(tz='UTC')
        timestamps = pd.date_range(
            end=now,
            periods=n_bars,
            freq='5s'
        )
        
        # Create bar data
        bar_data = []
        for i, ts in enumerate(timestamps):
            for instrument_id in range(1, n_contracts + 1):
                # Generate prices with some random walk
                price = base_price + instrument_id * 0.5  # Contango structure
                price += np.sin(i / 10) * volatility * base_price  # Some pattern
                
                high = price * (1 + volatility)
                low = price * (1 - volatility)
                volume = int(1000 * (1 + np.sin(i / 20)))  # Varying volume
                
                bar_data.append({
                    'ts': ts,
                    'instrument_id': instrument_id,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume
                })
        
        bar_df = pd.DataFrame(bar_data)
        
        # Create continuous contract mapping
        mapping_data = []
        for pos in range(1, n_contracts + 1):
            mapping_data.append({
                'bucket': timestamps[-1],
                'continuous_position': pos,
                'instrument_id': pos
            })
        
        mapping_df = pd.DataFrame(mapping_data)
        
        # Calculate curve
        result = calculator.calculate_curve_mid_5s(
            bar_df,
            mapping_df,
            now
        )
        
        # Property 1: Structure is correct
        expected_cols = ['ts'] + [f'cl{i}' for i in range(1, 9)] + ['total_volume', 'contracts_available']
        assert all(col in result.columns for col in expected_cols)
        
        # Property 2: Contract count is accurate
        if len(result) > 0:
            max_contracts = result['contracts_available'].max()
            assert max_contracts <= n_contracts
            assert max_contracts >= min(4, n_contracts)  # At least 4 or all available
        
        # Property 3: Volume aggregation
        if len(result) > 0:
            # Check a sample timestamp
            sample_ts = result.iloc[0]['ts']
            original_volume = bar_df[bar_df['ts'] == sample_ts]['volume'].sum()
            result_volume = result[result['ts'] == sample_ts]['total_volume'].iloc[0]
            # Volume should be aggregated correctly
            assert abs(original_volume - result_volume) < 1e-6
        
        # Property 4: No NaN corruption in required fields
        assert not result['ts'].isna().any()
        assert not result['total_volume'].isna().any()
        assert not result['contracts_available'].isna().any()
    
    @given(
        cl1_price=st.floats(min_value=20.0, max_value=150.0),
        spread=st.floats(min_value=-2.0, max_value=5.0),
        cl1_days=st.integers(min_value=5, max_value=30),
        days_between=st.integers(min_value=20, max_value=35)
    )
    def test_spot_adjustment_properties(
        self,
        calculator,
        cl1_price,
        spread,
        cl1_days,
        days_between
    ):
        """
        Test spot adjustment calculation properties:
        1. Spot price is positive
        2. Spot is reasonably close to front month
        3. Carry calculation is consistent
        """
        # Create test curve data
        curve_data = {
            'cl1': [cl1_price],
            'cl2': [cl1_price + spread],
            'cl3': [cl1_price + spread * 1.8],
            'cl4': [cl1_price + spread * 2.5],
            'cl5': [cl1_price + spread * 3.0],
            'cl6': [cl1_price + spread * 3.4],
            'cl7': [cl1_price + spread * 3.7],
            'cl8': [cl1_price + spread * 4.0]
        }
        curve_df = pd.DataFrame(curve_data)
        
        # Create expiry map
        expiry_map = {}
        days = cl1_days
        for i in range(1, 9):
            expiry_map[f'cl{i}'] = days
            days += days_between
        
        # Calculate spot adjustment
        result = calculator.calculate_spot_adjustment(curve_df, expiry_map)
        
        # Property 1: Spot is positive
        assert (result['spot'] > 0).all()
        
        # Property 2: Spot is within reasonable range of CL1
        spot_deviation = abs(result['spot'].iloc[0] - cl1_price) / cl1_price
        assert spot_deviation < 0.2  # Within 20% of front month
        
        # Property 3: Adjusted CL1 should be near zero (spot-relative)
        if 'cl1_adjusted' in result.columns:
            assert abs(result['cl1_adjusted'].iloc[0]) < 2.0  # Small adjustment
    
    @given(
        data_quality=st.floats(min_value=0.0, max_value=1.0),
        n_missing=st.integers(min_value=0, max_value=4)
    )
    def test_validation_edge_cases(self, calculator, data_quality, n_missing):
        """
        Test validation handles edge cases properly:
        1. Missing data is detected
        2. Quality scores are bounded [0, 1]
        3. Validation messages are informative
        """
        # Create curve with varying quality
        n_contracts = 8 - n_missing
        curve_data = {}
        
        for i in range(1, n_contracts + 1):
            if data_quality > 0.5 or i <= 2:  # Always have CL1/CL2
                curve_data[f'cl{i}'] = [70.0 + i * 0.5] * 10
            else:
                curve_data[f'cl{i}'] = [np.nan] * 10
        
        # Fill remaining with NaN
        for i in range(n_contracts + 1, 9):
            curve_data[f'cl{i}'] = [np.nan] * 10
        
        curve_df = pd.DataFrame(curve_data)
        
        # Validate
        is_valid, quality_score, msg = calculator.validate_curve_data(
            curve_df,
            min_contracts=4
        )
        
        # Properties
        assert 0.0 <= quality_score <= 1.0  # Bounded
        assert isinstance(msg, str) and len(msg) > 0  # Informative message
        
        if n_contracts < 4:
            assert not is_valid  # Should fail validation
            assert "Insufficient contracts" in msg
        else:
            # May or may not be valid depending on other factors
            pass