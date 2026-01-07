"""
SQL/Python parity validator with SHA-256 checksums.
Ensures SQL files haven't changed and Python implementations match.
"""

import hashlib
import os
import yaml
import pandas as pd
import psycopg2
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import logging
import numpy as np

from .curve_calculator import CurveCalculator

logger = logging.getLogger(__name__)


class ParityValidator:
    """Validates that SQL and Python implementations produce identical results."""
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize validator.
        
        Args:
            project_root: Root directory of trading system project
        """
        if project_root is None:
            # Find project root by looking for setup.py
            current_dir = Path(__file__).parent
            while current_dir.parent != current_dir:
                if (current_dir / 'setup.py').exists():
                    project_root = str(current_dir)
                    break
                current_dir = current_dir.parent
            else:
                raise ValueError("Could not find project root")
        
        self.project_root = Path(project_root)
        self.registry_path = self.project_root / 'common_lib' / 'curve_builder' / 'sql' / 'registry.yaml'
        self.registry = self._load_registry()
        self.calculator = CurveCalculator()
    
    def _load_registry(self) -> Dict:
        """Load SQL query registry."""
        with open(self.registry_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _save_registry(self):
        """Save updated registry with hashes."""
        with open(self.registry_path, 'w') as f:
            yaml.safe_dump(self.registry, f, default_flow_style=False)
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        full_path = self.project_root / file_path
        
        with open(full_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def validate_sql_hashes(self) -> Tuple[bool, List[str]]:
        """
        Validate that SQL files haven't changed.
        
        Returns:
            Tuple of (all_valid, list_of_changed_queries)
        """
        changed_queries = []
        
        for query_name, query_info in self.registry['queries'].items():
            if query_info['path'] is None:
                continue
                
            current_hash = self.calculate_file_hash(query_info['path'])
            
            if query_info['sha256'] is None:
                # First time - store the hash
                query_info['sha256'] = current_hash
                query_info['last_validated'] = datetime.now().isoformat()
                logger.info(f"Stored initial hash for {query_name}: {current_hash[:8]}...")
            elif query_info['sha256'] != current_hash:
                # Hash mismatch - SQL has changed
                changed_queries.append(query_name)
                logger.error(
                    f"SQL file changed for {query_name}! "
                    f"Expected: {query_info['sha256'][:8]}..., "
                    f"Got: {current_hash[:8]}..."
                )
        
        self._save_registry()
        
        return len(changed_queries) == 0, changed_queries
    
    def validate_curve_mid_5s_parity(
        self,
        db_conn_str: str,
        sample_time: pd.Timestamp,
        tolerance: float = 1e-6
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that Python implementation matches SQL for curve_mid_5s.
        
        Args:
            db_conn_str: Database connection string
            sample_time: Time to test at
            tolerance: Numerical tolerance for float comparison
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Get SQL results
            with psycopg2.connect(db_conn_str) as conn:
                # Get curve data from materialized view
                sql_curve_query = """
                    SELECT ts, cl1, cl2, cl3, cl4, cl5, cl6, cl7, cl8, 
                           total_volume, contracts_available
                    FROM market_data.curve_mid_5s
                    WHERE ts >= %s - INTERVAL '1 hour'
                      AND ts <= %s
                    ORDER BY ts
                """
                sql_curve_df = pd.read_sql(
                    sql_curve_query,
                    conn,
                    params=(sample_time, sample_time),
                    parse_dates=['ts']
                )
                
                # Get raw bar data for Python calculation
                bar_query = """
                    SELECT ts, instrument_id, high, low, close, volume
                    FROM market_data.bars_5s
                    WHERE ts >= %s - INTERVAL '2 days'
                      AND ts <= %s
                """
                bar_df = pd.read_sql(
                    bar_query,
                    conn,
                    params=(sample_time, sample_time),
                    parse_dates=['ts']
                )
                
                # Get continuous contract mapping
                mapping_query = """
                    SELECT bucket, continuous_position, instrument_id
                    FROM market_data.v_continuous_contracts_5s
                    WHERE bucket >= %s - INTERVAL '2 days'
                      AND bucket <= %s
                """
                mapping_df = pd.read_sql(
                    mapping_query,
                    conn,
                    params=(sample_time, sample_time),
                    parse_dates=['bucket']
                )
            
            # Calculate using Python
            python_curve_df = self.calculator.calculate_curve_mid_5s(
                bar_df,
                mapping_df,
                sample_time
            )
            
            # Compare results
            if len(sql_curve_df) == 0 and len(python_curve_df) == 0:
                return True, None
            
            if len(sql_curve_df) != len(python_curve_df):
                return False, f"Row count mismatch: SQL={len(sql_curve_df)}, Python={len(python_curve_df)}"
            
            # Merge on timestamp for comparison
            merged = pd.merge(
                sql_curve_df,
                python_curve_df,
                on='ts',
                suffixes=('_sql', '_py')
            )
            
            # Check each column
            for col in ['cl1', 'cl2', 'cl3', 'cl4', 'cl5', 'cl6', 'cl7', 'cl8', 'total_volume']:
                sql_col = f'{col}_sql' if col != 'total_volume' else 'total_volume_sql'
                py_col = f'{col}_py' if col != 'total_volume' else 'total_volume_py'
                
                if sql_col in merged.columns and py_col in merged.columns:
                    # Handle NaN comparison
                    sql_vals = merged[sql_col].fillna(-999999)
                    py_vals = merged[py_col].fillna(-999999)
                    
                    max_diff = (sql_vals - py_vals).abs().max()
                    if max_diff > tolerance:
                        return False, f"Column {col} differs by {max_diff:.6f} (tolerance={tolerance})"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def validate_spot_adjustment_parity(
        self,
        db_conn_str: str,
        sample_time: pd.Timestamp,
        tolerance: float = 1e-6
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate spot adjustment calculation matches between implementations.
        
        Args:
            db_conn_str: Database connection string
            sample_time: Time to test at
            tolerance: Numerical tolerance
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Import the canonical spot calculation
            import sys
            sys.path.append(str(self.project_root))
            from common_lib.pricing import calculate_spot_price
            
            # Get real curve data from database
            with psycopg2.connect(db_conn_str) as conn:
                curve_query = """
                    SELECT ts, cl1, cl2, cl3, cl4, cl5, cl6, cl7, cl8
                    FROM market_data.curve_mid_5s
                    WHERE ts >= %s - INTERVAL '1 hour'
                      AND ts <= %s
                      AND cl1 IS NOT NULL 
                      AND cl2 IS NOT NULL
                    ORDER BY ts DESC
                    LIMIT 10
                """
                test_data = pd.read_sql(
                    curve_query,
                    conn,
                    params=(sample_time, sample_time),
                    parse_dates=['ts']
                )
                
                if len(test_data) == 0:
                    return False, "No curve data available for spot validation"
                
                # Get actual expiry data
                expiry_query = """
                    WITH latest_mapping AS (
                        SELECT DISTINCT ON (v.continuous_position)
                            v.continuous_position,
                            v.instrument_id,
                            c.expiry,
                            v.bucket
                        FROM market_data.v_continuous_contracts_5s v
                        JOIN market_data.futures_contracts c ON v.instrument_id = c.instrument_id
                        WHERE v.bucket::date = %s::date
                        ORDER BY v.continuous_position, v.bucket DESC
                    )
                    SELECT 
                        lm.continuous_position,
                        (lm.expiry - %s::date)::int as days_to_expiry
                    FROM latest_mapping lm
                    WHERE continuous_position <= 8
                    ORDER BY continuous_position
                """
                expiry_df = pd.read_sql(
                    expiry_query,
                    conn,
                    params=(sample_time, sample_time)
                )
                
                # Build expiry map from real data
                expiry_map = {}
                for _, row in expiry_df.iterrows():
                    pos = int(row['continuous_position'])
                    expiry_map[f'cl{pos}'] = float(row['days_to_expiry'])
            
            # Calculate using curve_calculator
            python_adjusted = self.calculator.calculate_spot_adjustment(
                test_data,
                expiry_map
            )
            
            # Calculate using canonical spot_price module
            canonical_spots = []
            for _, row in test_data.iterrows():
                if pd.notna(row.get('cl1')) and pd.notna(row.get('cl2')):
                    spot = calculate_spot_price(
                        cl1_price=row['cl1'],
                        cl2_price=row['cl2'],
                        cl1_days_to_expiry=expiry_map.get('cl1', 21),
                        cl2_days_to_expiry=expiry_map.get('cl2', 52)
                    )
                    canonical_spots.append(spot)
                else:
                    canonical_spots.append(np.nan)
            
            # Compare spot calculations
            python_spots = python_adjusted['spot'].values
            canonical_spots = pd.Series(canonical_spots).values
            
            # Handle NaN comparison
            mask = ~(pd.isna(python_spots) | pd.isna(canonical_spots))
            if mask.sum() > 0:
                max_diff = np.abs(python_spots[mask] - canonical_spots[mask]).max()
                if max_diff > tolerance:
                    return False, f"Spot calculation differs by {max_diff:.6f}"
            
            return True, None
            
        except Exception as e:
            return False, f"Spot adjustment validation error: {str(e)}"


def validate_sql_python_parity(
    db_conn_str: str,
    sample_time: Optional[pd.Timestamp] = None
) -> bool:
    """
    Main validation function to check SQL/Python parity.
    
    Args:
        db_conn_str: Database connection string
        sample_time: Time to test at (default: now)
        
    Returns:
        True if all validations pass
    """
    if sample_time is None:
        sample_time = pd.Timestamp.now(tz='UTC')
    
    validator = ParityValidator()
    
    # Check SQL hashes
    logger.info("Checking SQL file hashes...")
    hashes_valid, changed_queries = validator.validate_sql_hashes()
    if not hashes_valid:
        logger.error(f"SQL files have changed: {changed_queries}")
        logger.error("Python implementations may need to be updated!")
        return False
    
    # Validate curve_mid_5s
    logger.info("Validating curve_mid_5s SQL/Python parity...")
    curve_valid, curve_error = validator.validate_curve_mid_5s_parity(
        db_conn_str,
        sample_time
    )
    if not curve_valid:
        logger.error(f"Curve validation failed: {curve_error}")
        return False
    
    # Validate spot adjustment using real data
    logger.info("Validating spot adjustment parity with real data...")
    spot_valid, spot_error = validator.validate_spot_adjustment_parity(
        db_conn_str,
        sample_time
    )
    if not spot_valid:
        logger.error(f"Spot adjustment validation failed: {spot_error}")
        return False
    
    logger.info("All SQL/Python parity validations passed!")
    return True