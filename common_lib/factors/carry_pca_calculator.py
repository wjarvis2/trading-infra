"""
Carry curve PCA calculator for term structure analysis.

This module calculates PCA factors from carry curves (not price curves).
Carry curves represent the term structure of carry rates relative to spot,
which is fundamentally different from PCA on price curves.

Design principles:
- Separate from price PCA as the calculations are fundamentally different
- Uses canonical spot price calculation to build carry curves
- Pure functions with no I/O operations
- Single source of truth for carry-based factor analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

from common_lib.pricing import calculate_spot_price, validate_spot_calculation
from common_lib.data_fetchers import CurveFetcher


class CarryPCACalculator:
    """
    PCA calculator specifically for carry curve analysis.
    
    This calculator:
    1. Fetches continuous contract data (CL1, CL2, etc.)
    2. Computes spot price using canonical calculation
    3. Calculates carry rates for each contract relative to spot
    4. Performs PCA on the carry curve (not prices)
    
    This is different from price PCA which analyzes the shape of the futures curve.
    Carry PCA analyzes the term structure of expected returns.
    """
    
    def __init__(self, n_components: int = 3, n_contracts: int = 8):
        """
        Initialize carry PCA calculator.
        
        Args:
            n_components: Number of principal components to compute (default: 3)
            n_contracts: Number of contracts to analyze (default: 8 for CL1-CL8)
        """
        self.n_components = n_components
        self.n_contracts = n_contracts
    
    def compute_carry_curve_from_continuous(
        self,
        curve_df: pd.DataFrame,
        expiry_map: Dict[str, float],
        use_spot_relative: bool = True
    ) -> pd.DataFrame:
        """
        Compute annualized carry curve from continuous contract data.
        
        Args:
            curve_df: DataFrame with continuous contract columns (cl1, cl2, ..., cl8)
                     as returned by CurveFetcher
            expiry_map: Dict mapping contract names to days to expiry
                       e.g., {'cl1': 21.5, 'cl2': 52.5, ...}
            use_spot_relative: If True, calculate carry relative to spot price
                             If False, use simple adjacent contract carry
            
        Returns:
            DataFrame of annualized carry rates
            
        Raises:
            ValueError: If insufficient data for calculations
        """
        # Standardize column names to uppercase for consistency
        price_data = pd.DataFrame(index=curve_df.index)
        for i in range(1, self.n_contracts + 1):
            lower_col = f'cl{i}'
            upper_col = f'CL{i}'
            if lower_col in curve_df.columns:
                price_data[upper_col] = curve_df[lower_col]
            elif upper_col in curve_df.columns:
                price_data[upper_col] = curve_df[upper_col]
        
        # Convert expiry map to uppercase keys
        expiry_days_dict = {}
        for i in range(1, self.n_contracts + 1):
            lower_key = f'cl{i}'
            upper_key = f'CL{i}'
            if lower_key in expiry_map:
                # Create a Series with the same index as price_data
                expiry_days_dict[upper_key] = pd.Series(
                    expiry_map[lower_key], 
                    index=price_data.index
                )
            elif upper_key in expiry_map:
                expiry_days_dict[upper_key] = pd.Series(
                    expiry_map[upper_key], 
                    index=price_data.index
                )
        
        if use_spot_relative:
            return self._compute_spot_relative_carry(price_data, expiry_days_dict)
        else:
            return self._compute_adjacent_carry(price_data, expiry_days_dict)
    
    def _compute_spot_relative_carry(
        self,
        prices: pd.DataFrame,
        expiry_days: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Compute carry rates relative to spot price.
        
        This is the theoretically correct approach that accounts for
        the true cost of carry from spot to each contract.
        """
        # First calculate spot price using canonical method
        spot_prices = pd.Series(index=prices.index, dtype=float)
        
        if 'CL1' not in prices.columns or 'CL2' not in prices.columns:
            raise ValueError("CL1 and CL2 required for spot calculation")
        
        for idx in prices.index:
            try:
                spot = calculate_spot_price(
                    cl1_price=float(prices.loc[idx, 'CL1']),
                    cl2_price=float(prices.loc[idx, 'CL2']),
                    cl1_days_to_expiry=float(expiry_days['CL1'].loc[idx]),
                    cl2_days_to_expiry=float(expiry_days['CL2'].loc[idx])
                )
                
                # Validate spot
                is_valid, _ = validate_spot_calculation(spot, prices.loc[idx, 'CL1'])
                if is_valid:
                    spot_prices.loc[idx] = spot
                else:
                    spot_prices.loc[idx] = np.nan
                    
            except (ValueError, KeyError) as e:
                spot_prices.loc[idx] = np.nan
        
        # Calculate carry rates for each contract relative to spot
        carry_curve = pd.DataFrame(index=prices.index)
        
        for i in range(1, self.n_contracts + 1):
            col = f'CL{i}'
            if col in prices.columns and col in expiry_days:
                contract_price = prices[col]
                time_to_expiry = expiry_days[col] / 365.25  # Convert to years
                
                # Valid data mask
                valid_mask = (
                    (time_to_expiry > 0.01) & 
                    (contract_price > 0) & 
                    (spot_prices > 0) &
                    ~spot_prices.isna()
                )
                
                # Calculate annualized carry rate: ln(F/S) / T
                carry_rate = pd.Series(np.nan, index=prices.index)
                if valid_mask.any():
                    carry_rate[valid_mask] = np.log(
                        contract_price[valid_mask] / spot_prices[valid_mask]
                    ) / time_to_expiry[valid_mask]
                
                carry_curve[f'Carry_CL{i}'] = carry_rate
        
        # Remove infinite values
        carry_curve = carry_curve.replace([np.inf, -np.inf], np.nan)
        
        return carry_curve
    
    def _compute_adjacent_carry(
        self,
        prices: pd.DataFrame,
        expiry_days: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Compute simplified carry between adjacent contracts.
        
        This is a simpler approach that doesn't require spot calculation
        but may be less theoretically accurate.
        """
        carry_curve = pd.DataFrame(index=prices.index)
        
        for i in range(1, self.n_contracts):
            front_col = f'CL{i}'
            back_col = f'CL{i+1}'
            
            if (front_col in prices.columns and back_col in prices.columns and
                front_col in expiry_days and back_col in expiry_days):
                
                # Time between contracts in years
                time_diff = (expiry_days[back_col] - expiry_days[front_col]) / 365.25
                
                # Valid data mask
                valid_mask = (
                    (time_diff > 0.01) &
                    (prices[front_col] > 0) &
                    (prices[back_col] > 0)
                )
                
                # Annualized carry: ln(F2/F1) / (T2-T1)
                carry = pd.Series(np.nan, index=prices.index)
                if valid_mask.any():
                    carry[valid_mask] = np.log(
                        prices[back_col][valid_mask] / prices[front_col][valid_mask]
                    ) / time_diff[valid_mask]
                
                carry_curve[f'carry_{i}_{i+1}'] = carry
        
        return carry_curve
    
    def calculate_from_continuous(
        self,
        curve_df: pd.DataFrame,
        expiry_map: Dict[str, float],
        window_end: Optional[pd.Timestamp] = None,
        min_valid_columns: int = 3,
        required_variance_explained: float = 0.90,
        use_spot_relative: bool = True
    ) -> Dict:
        """
        Calculate PCA factors from continuous contract curve data.
        
        This is the main entry point when using data from CurveFetcher.
        
        Args:
            curve_df: DataFrame with continuous contract data (cl1, cl2, etc.)
            expiry_map: Dict mapping contract names to days to expiry
            window_end: End timestamp (default: last timestamp in data)
            min_valid_columns: Minimum carry series required for PCA
            required_variance_explained: Minimum cumulative variance threshold
            use_spot_relative: Whether to use spot-relative carry calculation
            
        Returns:
            Dict containing:
                - factors: DataFrame with PCA factors
                - loadings: Component loadings matrix
                - variance_explained: Variance explained by each component
                - carry_curve: The computed carry curve
                - timestamp: Calculation timestamp
                - metadata: Additional calculation metadata
        """
        # Compute carry curve from continuous contract data
        carry_curve = self.compute_carry_curve_from_continuous(
            curve_df, expiry_map, use_spot_relative
        )
        
        # Remove columns with too many NaNs
        valid_cols = carry_curve.columns[
            carry_curve.notna().sum() > len(carry_curve) * 0.5
        ]
        
        if len(valid_cols) < min_valid_columns:
            raise ValueError(
                f"Insufficient valid carry series: {len(valid_cols)} < {min_valid_columns}"
            )
        
        # Clean data for PCA
        clean_data = carry_curve[valid_cols].dropna()
        
        if len(clean_data) < 10:
            raise ValueError(f"Insufficient rows after cleaning: {len(clean_data)}")
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_data)
        
        # Perform PCA
        try:
            pca = PCA(n_components=min(self.n_components, len(valid_cols)))
            factors = pca.fit_transform(scaled_data)
            
            # Check variance explained
            cumsum_var = np.cumsum(pca.explained_variance_ratio_)
            if cumsum_var[-1] < required_variance_explained:
                warnings.warn(
                    f"PCA explains only {cumsum_var[-1]:.2%} variance, "
                    f"less than required {required_variance_explained:.2%}"
                )
                
        except Exception as e:
            raise RuntimeError(f"PCA calculation failed: {str(e)}")
        
        # Create factor DataFrame
        factor_df = pd.DataFrame(
            factors,
            index=clean_data.index,
            columns=[f"PC{i+1}" for i in range(factors.shape[1])]
        )
        
        # Add interpretations for carry PCA
        # PC1: Overall carry level (parallel shifts in carry curve)
        # PC2: Carry slope (term structure steepening/flattening)
        # PC3: Carry curvature (butterfly patterns)
        factor_df["level"] = factor_df["PC1"]
        if factors.shape[1] >= 2:
            factor_df["slope"] = factor_df["PC2"]
        if factors.shape[1] >= 3:
            factor_df["curvature"] = factor_df["PC3"]
        
        # Ensure consistent signs
        loadings = pca.components_.T
        loadings, factors, sign_flips = self.ensure_consistent_signs(loadings, factors)
        
        # Update factor_df with sign-corrected values
        for i in range(factors.shape[1]):
            factor_df[f"PC{i+1}"] = factors[:, i]
        factor_df["level"] = factor_df["PC1"]
        if factors.shape[1] >= 2:
            factor_df["slope"] = factor_df["PC2"]
        if factors.shape[1] >= 3:
            factor_df["curvature"] = factor_df["PC3"]
        
        # Prepare results
        if window_end is None:
            window_end = clean_data.index[-1]
        
        # Truncate window_end to seconds for consistency
        if hasattr(window_end, 'floor'):
            window_end = window_end.floor('S')
            
        results = {
            "factors": factor_df,
            "loadings": pd.DataFrame(
                loadings,
                index=valid_cols,
                columns=[f"PC{i+1}" for i in range(loadings.shape[1])]
            ),
            "variance_explained": pca.explained_variance_ratio_,
            "cumulative_variance_explained": cumsum_var,
            "carry_curve": carry_curve,
            "timestamp": window_end,
            "metadata": {
                "n_samples": len(clean_data),
                "carry_columns_used": list(valid_cols),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "spot_relative_carry": use_spot_relative,
                "sign_flips": sign_flips
            }
        }
        
        return results
    
    def ensure_consistent_signs(
        self,
        loadings: np.ndarray,
        scores: np.ndarray,
        reference_loadings: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Ensure consistent factor signs for carry PCA interpretation.
        
        For carry curves:
        - PC1 (level): Should load positively on average (higher carry overall)
        - PC2 (slope): Should be negative for front, positive for back (contango)
        - PC3 (curvature): Should show typical butterfly pattern
        """
        n_components = loadings.shape[1]
        sign_flips = []
        
        for i in range(n_components):
            flip_sign = False
            
            if reference_loadings is not None and i < reference_loadings.shape[1]:
                # Compare to reference
                corr = np.corrcoef(loadings[:, i], reference_loadings[:, i])[0, 1]
                flip_sign = corr < 0
            else:
                # Use economic interpretation for carry curves
                if i == 0:  # PC1 (carry level)
                    # Average loading should be positive (higher carry)
                    flip_sign = np.mean(loadings[:, i]) < 0
                elif i == 1:  # PC2 (carry slope)
                    # Should show contango pattern (increasing with maturity)
                    n_contracts = len(loadings[:, i])
                    if n_contracts >= 2:
                        # Compare front vs back average
                        front_avg = np.mean(loadings[:n_contracts//2, i])
                        back_avg = np.mean(loadings[n_contracts//2:, i])
                        flip_sign = (back_avg - front_avg) < 0
                elif i == 2:  # PC3 (carry curvature)
                    # Butterfly: positive ends, negative middle
                    n_contracts = len(loadings[:, i])
                    if n_contracts >= 3:
                        ends_avg = (loadings[0, i] + loadings[-1, i]) / 2
                        middle_avg = loadings[n_contracts//2, i]
                        flip_sign = (ends_avg - middle_avg) < 0
            
            if flip_sign:
                loadings[:, i] *= -1
                scores[:, i] *= -1
                sign_flips.append(i)
        
        return loadings, scores, sign_flips
    
    def calculate_with_fetcher(
        self,
        curve_fetcher: CurveFetcher,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        **kwargs
    ) -> Dict:
        """
        Calculate carry PCA using a CurveFetcher for data access.
        
        This method handles the data fetching and then delegates to
        calculate_from_continuous for the actual calculations.
        
        Args:
            curve_fetcher: CurveFetcher instance for data access
            start_time: Start time for data fetch
            end_time: End time for data fetch
            **kwargs: Additional arguments for calculate_from_continuous
            
        Returns:
            Dict with PCA results
        """
        # Fetch curve data
        curve_df = curve_fetcher.fetch_from_db(start_time, end_time)
        
        # Fetch expiry data for the end date
        expiry_map = curve_fetcher.fetch_expiry_data(end_time)
        
        # Calculate PCA
        return self.calculate_from_continuous(
            curve_df, 
            expiry_map,
            window_end=end_time,
            **kwargs
        )