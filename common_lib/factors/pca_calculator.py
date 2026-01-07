"""
Stateless PCA calculator for term structure factor analysis.

This module contains the SINGLE authoritative implementation of PCA factor calculations.
All PCA calculations across the codebase MUST use this implementation.

Design principles:
- Pure functions only - no database queries or file I/O
- Accept data as parameters, return results
- Comprehensive error handling with descriptive messages
- Fully testable with fixture data
- Reuse existing common_lib components where available
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

from common_lib.pricing import calculate_spot_price, validate_spot_calculation
from common_lib.curve_builder.python.curve_calculator import CurveCalculator


class PCACalculator:
    """
    Stateless PCA calculator for term structure analysis.
    
    This calculator performs PCA on futures curve data to extract
    level, slope, and curvature factors.
    """
    
    def __init__(self, n_components: int = 3):
        """
        Initialize PCA calculator.
        
        Args:
            n_components: Number of principal components to compute (default: 3)
        """
        self.n_components = n_components
        self.curve_calculator = CurveCalculator()
        
        # Optional pre-fitted components for transform-only operations
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.scale_ = None
        self.n_samples_seen_ = None
    
    def calculate(
        self,
        curve_df: pd.DataFrame,
        window_end: pd.Timestamp,
        expiry_map: Dict[str, float],
        window_bars: int = 720,
        bar_freq: str = "5s",
        min_contracts: int = 8,
        max_contracts: int = 12,
        required_variance_explained: float = 0.95,
        apply_spot_adj: bool = True,
        max_spread: float = 10.0,
        max_daily_move: float = 10.0
    ) -> Dict:
        """
        Calculate PCA factors from curve data.
        
        Args:
            curve_df: DataFrame with curve data (columns: cl1, cl2, ..., index: timestamps)
            window_end: End timestamp for the calculation window
            expiry_map: Dict mapping contract names to days to expiry
            window_bars: Number of bars to include in the window (default: 720)
            bar_freq: Bar frequency string (default: "5s")
            min_contracts: Minimum number of contracts required (default: 8)
            max_contracts: Maximum number of contracts to use (default: 12)
            required_variance_explained: Minimum variance explained threshold (default: 0.95)
            apply_spot_adj: Whether to apply spot adjustment (default: True)
            max_spread: Maximum allowed spread for validation (default: 10.0)
            max_daily_move: Maximum allowed daily move for validation (default: 10.0)
            
        Returns:
            Dict containing:
                - factors: DataFrame with PCA factors (level, slope, curvature)
                - loadings: Component loadings matrix
                - variance_explained: Variance explained by each component
                - contracts_used: List of contracts used in the calculation
                - timestamp: Calculation timestamp
                - metadata: Additional calculation metadata
                
        Raises:
            ValueError: If insufficient data or contracts
            RuntimeError: If PCA calculation fails
        """
        if curve_df.empty:
            raise ValueError("Empty curve data provided")
            
        # Calculate window start
        # Convert bar_freq to proper format for Timedelta
        if bar_freq == '5s':
            window_start = window_end - pd.Timedelta(seconds=window_bars * 5)
        elif bar_freq == '1D':
            window_start = window_end - pd.Timedelta(days=window_bars)
        elif bar_freq == '15s':
            window_start = window_end - pd.Timedelta(seconds=window_bars * 15)
        elif bar_freq == '30s':
            window_start = window_end - pd.Timedelta(seconds=window_bars * 30)
        elif bar_freq == '1m':
            window_start = window_end - pd.Timedelta(minutes=window_bars)
        else:
            # Fallback - try to parse the frequency
            window_start = window_end - pd.Timedelta(window_bars * pd.Timedelta(bar_freq))
        
        # Truncate timestamps to seconds for consistent matching
        # This prevents millisecond/microsecond mismatches
        if hasattr(curve_df.index, 'floor'):
            curve_df_truncated = curve_df.copy()
            curve_df_truncated.index = curve_df_truncated.index.floor('S')
            window_start_truncated = pd.Timestamp(window_start).floor('S')
            window_end_truncated = pd.Timestamp(window_end).floor('S')
        else:
            curve_df_truncated = curve_df
            window_start_truncated = window_start
            window_end_truncated = window_end
        
        # Filter data to window
        window_data = curve_df_truncated[
            (curve_df_truncated.index > window_start_truncated) & 
            (curve_df_truncated.index <= window_end_truncated)
        ]
        
        if window_data.empty:
            raise ValueError(f"No data found in window {window_start} to {window_end}")
            
        # Validate curve data quality using existing validator
        is_valid, quality_score, msg = self.curve_calculator.validate_curve_data(
            window_data,
            min_contracts=min_contracts,
            max_spread=max_spread,
            max_daily_move=max_daily_move
        )
        
        if not is_valid:
            raise ValueError(f"Curve data validation failed: {msg} (quality_score={quality_score:.3f})")
            
        # Apply spot adjustment if requested
        if apply_spot_adj:
            # Use the improved spot adjustment that calls canonical calculate_spot_price
            window_data = self._apply_spot_adjustment_canonical(window_data, expiry_map)
            # Use adjusted columns for PCA
            contract_cols = [f'cl{i}_adjusted' for i in range(1, 9) 
                           if f'cl{i}_adjusted' in window_data.columns]
        else:
            contract_cols = [f'cl{i}' for i in range(1, 9) 
                           if f'cl{i}' in window_data.columns]
            
        # Select contracts (remove any with too many NaNs)
        valid_contracts = []
        for col in contract_cols:
            if window_data[col].notna().sum() > len(window_data) * 0.8:
                valid_contracts.append(col)
        
        if len(valid_contracts) < min_contracts:
            raise ValueError(
                f"Insufficient contracts: {len(valid_contracts)} < {min_contracts} required"
            )
            
        # Use up to max_contracts
        contracts_to_use = valid_contracts[:max_contracts]
        
        # Prepare data for PCA
        pca_data = window_data[contracts_to_use].copy()
        
        # Forward fill then drop remaining NaNs
        pca_data = pca_data.ffill().dropna()
        
        if len(pca_data) < 10:  # Minimum rows for meaningful PCA
            raise ValueError(f"Insufficient data rows after cleaning: {len(pca_data)}")
            
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_data)
        
        # Perform PCA
        try:
            pca = PCA(n_components=self.n_components)
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
            index=pca_data.index,
            columns=[f"PC{i+1}" for i in range(self.n_components)]
        )
        
        # Add factor interpretations
        # PC1: Level (parallel shifts in curve)
        # PC2: Slope (steepening/flattening)
        # PC3: Curvature (butterfly moves)
        factor_df["level"] = factor_df["PC1"]
        if self.n_components >= 2:
            factor_df["slope"] = factor_df["PC2"]
        if self.n_components >= 3:
            factor_df["curvature"] = factor_df["PC3"]
            
        # Ensure consistent factor signs
        loadings = pca.components_.T
        scores = factors
        loadings, scores, sign_flips = self.ensure_consistent_signs(loadings, scores)
        
        # Update factor_df with sign-corrected scores
        for i in range(self.n_components):
            factor_df[f"PC{i+1}"] = scores[:, i]
        # Update interpretations
        factor_df["level"] = factor_df["PC1"]
        if self.n_components >= 2:
            factor_df["slope"] = factor_df["PC2"]
        if self.n_components >= 3:
            factor_df["curvature"] = factor_df["PC3"]
            
        # Prepare results
        results = {
            "factors": factor_df,
            "loadings": pd.DataFrame(
                loadings,
                index=contracts_to_use,
                columns=[f"PC{i+1}" for i in range(self.n_components)]
            ),
            "variance_explained": pca.explained_variance_ratio_,
            "cumulative_variance_explained": cumsum_var,
            "contracts_used": contracts_to_use,
            "timestamp": window_end,
            "quality_score": quality_score,
            "sign_flips": sign_flips,
            "metadata": {
                "window_bars": window_bars,
                "bar_freq": bar_freq,
                "n_samples": len(pca_data),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "spot_adjustment_applied": apply_spot_adj
            }
        }
        
        # Add spot prices if calculated
        if apply_spot_adj and 'spot' in window_data.columns:
            results["spot_prices"] = window_data['spot'].dropna()
        
        return results
    
    def _apply_spot_adjustment_canonical(
        self,
        curve_df: pd.DataFrame,
        expiry_map: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Apply spot adjustment using the canonical calculate_spot_price function.
        
        This replaces the duplicate implementation in curve_calculator to ensure
        single source of truth for spot calculations.
        
        Args:
            curve_df: DataFrame with curve prices
            expiry_map: Dict mapping contract names to days to expiry
            
        Returns:
            DataFrame with spot-adjusted prices
        """
        result_df = curve_df.copy()
        
        # Calculate spot price for each row using canonical function
        spot_prices = []
        
        for idx, row in curve_df.iterrows():
            if pd.notna(row.get('cl1')) and pd.notna(row.get('cl2')):
                try:
                    spot = calculate_spot_price(
                        cl1_price=row['cl1'],
                        cl2_price=row['cl2'],
                        cl1_days_to_expiry=expiry_map.get('cl1', 21),
                        cl2_days_to_expiry=expiry_map.get('cl2', 52)
                    )
                    
                    # Validate using canonical validator
                    is_valid, error_msg = validate_spot_calculation(spot, row['cl1'])
                    if not is_valid:
                        warnings.warn(f"Spot validation failed at {idx}: {error_msg}")
                        # Use fallback calculation
                        carry = (row['cl2'] - row['cl1']) * 0.4
                        spot = row['cl1'] - carry
                    
                    spot_prices.append(spot)
                except Exception as e:
                    warnings.warn(f"Failed to calculate spot for {idx}: {e}")
                    # Fallback
                    carry = (row['cl2'] - row['cl1']) * 0.4
                    spot_prices.append(row['cl1'] - carry)
            else:
                spot_prices.append(np.nan)
        
        result_df['spot'] = spot_prices
        
        # Adjust all contracts relative to spot (forward curve)
        for col in ['cl1', 'cl2', 'cl3', 'cl4', 'cl5', 'cl6', 'cl7', 'cl8']:
            if col in result_df.columns:
                result_df[f'{col}_adjusted'] = result_df[col] - result_df['spot']
        
        return result_df
    
    def calculate_rolling(
        self,
        curve_df: pd.DataFrame,
        timestamps: List[pd.Timestamp],
        expiry_map: Dict[str, float],
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate PCA factors for multiple timestamps (rolling window).
        
        Args:
            curve_df: DataFrame with curve data
            timestamps: List of timestamps to calculate factors for
            expiry_map: Dict mapping contract names to days to expiry
            **kwargs: Additional arguments passed to calculate()
            
        Returns:
            DataFrame with factors for all timestamps
        """
        all_factors = []
        
        for ts in timestamps:
            try:
                result = self.calculate(curve_df, ts, expiry_map, **kwargs)
                # Get the last factor values for this timestamp
                latest_factors = result["factors"].iloc[-1:].copy()
                latest_factors.index = [ts]
                all_factors.append(latest_factors)
            except (ValueError, RuntimeError) as e:
                warnings.warn(f"Failed to calculate PCA for {ts}: {str(e)}")
                continue
                
        if not all_factors:
            raise ValueError("Failed to calculate PCA for any timestamps")
            
        return pd.concat(all_factors)
    
    def ensure_consistent_signs(
        self,
        loadings: np.ndarray,
        scores: np.ndarray,
        reference_loadings: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Ensure consistent factor signs across calculations.
        
        PCA components can have arbitrary signs. This function ensures
        consistency by comparing to reference loadings or using
        economic interpretation.
        
        Args:
            loadings: Component loadings matrix
            scores: Factor scores matrix
            reference_loadings: Reference loadings for sign comparison
            
        Returns:
            Tuple of (adjusted_loadings, adjusted_scores, sign_flips)
        """
        n_components = min(loadings.shape[1], self.n_components)
        sign_flips = []
        
        for i in range(n_components):
            flip_sign = False
            
            if reference_loadings is not None and i < reference_loadings.shape[1]:
                # Compare to reference
                corr = np.corrcoef(loadings[:, i], reference_loadings[:, i])[0, 1]
                flip_sign = corr < 0
            else:
                # Use economic interpretation
                if i == 0:  # PC1 (level)
                    # Level factor should load positively on average
                    flip_sign = np.mean(loadings[:, i]) < 0
                elif i == 1:  # PC2 (slope)
                    # Slope should be negative for front, positive for back
                    n_contracts = len(loadings[:, i])
                    front_idx = min(3, n_contracts)
                    back_idx = max(0, n_contracts - 3)
                    front_avg = np.mean(loadings[:front_idx, i])
                    back_avg = np.mean(loadings[back_idx:, i])
                    flip_sign = (back_avg - front_avg) < 0
                elif i == 2:  # PC3 (curvature)
                    # Curvature should be positive at ends, negative in middle
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
    
    def validate_factors(
        self,
        factors: pd.DataFrame,
        tolerance: float = 0.01
    ) -> Tuple[bool, List[str]]:
        """
        Validate calculated factors for reasonableness.
        
        Args:
            factors: DataFrame with calculated factors
            tolerance: Tolerance for validation checks
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for NaNs
        if factors.isna().any().any():
            issues.append("Factors contain NaN values")
            
        # Check for extreme values (> 5 std dev)
        for col in factors.columns:
            if col.startswith("PC") or col in ["level", "slope", "curvature"]:
                std = factors[col].std()
                mean = factors[col].mean()
                if std > 0:  # Avoid division by zero
                    extreme = (np.abs(factors[col] - mean) > 5 * std).any()
                    if extreme:
                        issues.append(f"{col} contains extreme values")
                    
        # Check for low variance
        for col in factors.columns:
            if col.startswith("PC") or col in ["level", "slope", "curvature"]:
                if factors[col].std() < tolerance:
                    issues.append(f"{col} has suspiciously low variance")
                    
        # Check correlations between factors (should be uncorrelated)
        factor_cols = [col for col in factors.columns if col.startswith("PC")]
        if len(factor_cols) > 1:
            corr_matrix = factors[factor_cols].corr()
            # Check off-diagonal elements
            for i in range(len(factor_cols)):
                for j in range(i+1, len(factor_cols)):
                    if abs(corr_matrix.iloc[i, j]) > 0.3:
                        issues.append(
                            f"{factor_cols[i]} and {factor_cols[j]} are highly correlated "
                            f"({corr_matrix.iloc[i, j]:.2f})"
                        )
                    
        is_valid = len(issues) == 0
        return is_valid, issues
    
    @classmethod
    def from_sklearn(cls, sklearn_pca, scaler=None, n_components=None):
        """
        Create PCACalculator from sklearn PCA object.
        
        This allows using pre-fitted models from IncrementalPCA or other
        sklearn PCA implementations.
        
        Args:
            sklearn_pca: Fitted sklearn PCA object
            scaler: Optional StandardScaler used for preprocessing
            n_components: Number of components (default: from sklearn_pca)
            
        Returns:
            PCACalculator instance with pre-fitted parameters
        """
        instance = cls(n_components=n_components or sklearn_pca.n_components_)
        
        # Transfer fitted parameters
        instance.components_ = sklearn_pca.components_
        instance.explained_variance_ratio_ = sklearn_pca.explained_variance_ratio_
        instance.mean_ = getattr(sklearn_pca, 'mean_', None)
        instance.n_samples_seen_ = getattr(sklearn_pca, 'n_samples_seen_', 0)
        
        # Transfer scaler parameters if provided
        if scaler is not None:
            instance.scale_ = scaler.scale_
            instance.mean_ = scaler.mean_
        
        return instance
    
    def transform_with_prefitted(
        self,
        curve_df: pd.DataFrame,
        contracts: List[str],
        ensure_signs: bool = True
    ) -> pd.DataFrame:
        """
        Transform data using pre-fitted PCA parameters.
        
        This method is used when PCA has been fitted elsewhere (e.g., using
        IncrementalPCA on historical data) and we want to apply the same
        transformation.
        
        Args:
            curve_df: DataFrame with curve data
            contracts: List of contract columns to use
            ensure_signs: Whether to ensure consistent factor signs
            
        Returns:
            DataFrame with PCA factors
            
        Raises:
            ValueError: If no pre-fitted parameters available
        """
        if self.components_ is None:
            raise ValueError("No pre-fitted PCA parameters available. Use from_sklearn() first.")
        
        # Extract and prepare data
        pca_data = curve_df[contracts].copy()
        
        # Forward fill then drop remaining NaNs
        pca_data = pca_data.ffill().dropna()
        
        if len(pca_data) == 0:
            raise ValueError("No valid data after cleaning")
        
        # Standardize using stored parameters
        if self.mean_ is not None and self.scale_ is not None:
            scaled_data = (pca_data.values - self.mean_) / self.scale_
        else:
            # Fallback to standardizing with current data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(pca_data.values)
        
        # Transform using stored components
        scores = scaled_data @ self.components_.T
        
        # Ensure consistent signs if requested
        if ensure_signs:
            loadings = self.components_.T
            loadings, scores, _ = self.ensure_consistent_signs(loadings, scores)
        
        # Create factor DataFrame
        factor_df = pd.DataFrame(
            scores,
            index=pca_data.index,
            columns=[f"PC{i+1}" for i in range(self.n_components)]
        )
        
        # Add interpretations
        factor_df["level"] = factor_df["PC1"]
        if self.n_components >= 2:
            factor_df["slope"] = factor_df["PC2"]
        if self.n_components >= 3:
            factor_df["curvature"] = factor_df["PC3"]
        
        return factor_df
    
    def save(self, filepath: str):
        """Save PCA model parameters to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n_components': self.n_components,
                'components_': self.components_,
                'explained_variance_ratio_': self.explained_variance_ratio_,
                'mean_': self.mean_,
                'scale_': self.scale_,
                'n_samples_seen_': self.n_samples_seen_
            }, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Load PCA model parameters from file."""
        import pickle
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        instance = cls(n_components=params['n_components'])
        instance.components_ = params['components_']
        instance.explained_variance_ratio_ = params['explained_variance_ratio_']
        instance.mean_ = params.get('mean_')
        instance.scale_ = params.get('scale_')
        instance.n_samples_seen_ = params.get('n_samples_seen_', 0)
        
        return instance
    
    @staticmethod
    def calculate_pca_speed_index(
        pca_factors: pd.DataFrame,
        lookback_window: int = 20,
        components: List[str] = None
    ) -> pd.Series:
        """
        Calculate PCA Speed Index measuring volatility of curve structure.
        
        The PCA Speed Index quantifies how quickly the term structure is changing
        by measuring the sum of absolute changes in principal components.
        Formula: speed = |PC1_t - PC1_{t-1}| + |PC2_t - PC2_{t-1}| + ...
        
        Args:
            pca_factors: DataFrame with PCA factors (must have PC1, PC2, etc. columns)
            lookback_window: Window for median normalization (default: 20)
            components: List of component names to use (default: ['PC1', 'PC2'])
            
        Returns:
            Series with PCA Speed Index values (normalized by trailing median)
            
        Raises:
            ValueError: If required components not found in DataFrame
        """
        if components is None:
            components = ['PC1', 'PC2']
        
        # Validate components exist
        missing = [c for c in components if c not in pca_factors.columns]
        if missing:
            raise ValueError(f"Components not found in DataFrame: {missing}")
        
        # Calculate absolute changes for each component
        speed_components = []
        for component in components:
            abs_change = pca_factors[component].diff().abs()
            speed_components.append(abs_change)
        
        # Sum absolute changes
        raw_speed = pd.concat(speed_components, axis=1).sum(axis=1)
        
        # Normalize by trailing median
        median_speed = raw_speed.rolling(window=lookback_window, min_periods=1).median()
        
        # Avoid division by zero
        speed_index = raw_speed.copy()
        mask = median_speed > 0
        speed_index[mask] = raw_speed[mask] / median_speed[mask]
        
        # Fill NaN values with 1.0 (neutral speed)
        speed_index = speed_index.fillna(1.0)
        
        return speed_index
    
    @staticmethod
    def calculate_pca_speed_from_curve_data(
        curve_df: pd.DataFrame,
        window_bars: int = 240,
        lookback_window: int = 20,
        n_components: int = 3,
        apply_spot_adj: bool = True
    ) -> pd.Series:
        """
        Calculate PCA Speed Index directly from curve data.
        
        Convenience method that combines PCA calculation and speed index
        calculation for use in strategies that don't have pre-computed PCA factors.
        
        Args:
            curve_df: DataFrame with curve data (columns: cl1, cl2, ...)
            window_bars: Rolling window for PCA calculation (default: 240)
            lookback_window: Window for speed median normalization (default: 20)
            n_components: Number of PCA components (default: 3)
            apply_spot_adj: Whether to apply spot adjustment (default: True)
            
        Returns:
            Series with PCA Speed Index values
        """
        # Initialize calculator
        calculator = PCACalculator(n_components=n_components)
        
        # Calculate rolling PCA
        speed_values = []
        timestamps = []
        
        # Need at least window_bars of data
        if len(curve_df) < window_bars:
            return pd.Series(dtype=float)
        
        # Calculate PCA for rolling windows
        for i in range(window_bars, len(curve_df)):
            window_end = curve_df.index[i]
            window_start_idx = i - window_bars + 1
            window_data = curve_df.iloc[window_start_idx:i+1]
            
            try:
                # Simple expiry map (can be enhanced)
                expiry_map = {f'cl{j}': j*30 for j in range(1, 9)}
                
                result = calculator.calculate(
                    curve_df=window_data,
                    window_end=window_end,
                    expiry_map=expiry_map,
                    window_bars=window_bars,
                    apply_spot_adj=apply_spot_adj
                )
                
                factors = result['factors']
                if not factors.empty:
                    # Store the last factor values
                    speed_values.append(factors.iloc[-1])
                    timestamps.append(window_end)
                    
            except Exception:
                # Skip windows with errors
                continue
        
        if not speed_values:
            return pd.Series(dtype=float)
        
        # Create DataFrame from collected factors
        pca_factors = pd.DataFrame(speed_values, index=timestamps)
        
        # Calculate speed index
        speed_index = PCACalculator.calculate_pca_speed_index(
            pca_factors,
            lookback_window=lookback_window,
            components=['PC1', 'PC2'] if n_components >= 2 else ['PC1']
        )
        
        return speed_index