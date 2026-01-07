"""
Canonical spot price calculation for WTI crude oil futures

This module provides the single source of truth for spot price calculations
across the trading system. It uses the theoretically correct approach of
discounting the front month contract back to immediate delivery using the
carry rate implied by the front-to-second month spread.

Critical: This replaces all other spot price calculations in the codebase
to ensure consistency and accuracy.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def calculate_spot_price(
    cl1_price: float,
    cl2_price: float,
    cl1_days_to_expiry: float,
    cl2_days_to_expiry: float
) -> float:
    """
    Calculate theoretical spot price from front two futures contracts.
    
    This is the canonical implementation based on cost-of-carry theory:
    - Spot price is the price for immediate delivery
    - Front month (CL1) trades at spot + carry cost to delivery
    - Carry rate is implied by the CL1-CL2 spread
    
    Args:
        cl1_price: Front month futures price
        cl2_price: Second month futures price
        cl1_days_to_expiry: Days until CL1 expiry
        cl2_days_to_expiry: Days until CL2 expiry
        
    Returns:
        Theoretical spot price
        
    Raises:
        ValueError: If inputs are invalid (negative prices, invalid expiries)
    """
    # Validate inputs
    if cl1_price <= 0 or cl2_price <= 0:
        raise ValueError(f"Invalid prices: CL1={cl1_price}, CL2={cl2_price}")
    
    if cl1_days_to_expiry <= 0 or cl2_days_to_expiry <= cl1_days_to_expiry:
        raise ValueError(f"Invalid expiries: CL1={cl1_days_to_expiry}, CL2={cl2_days_to_expiry}")
    
    # Calculate time periods in years
    time_to_cl1 = cl1_days_to_expiry / 365.25
    time_between_contracts = (cl2_days_to_expiry - cl1_days_to_expiry) / 365.25
    
    # Calculate implied carry rate from CL1-CL2 spread
    # carry_rate = ln(F2/F1) / (T2-T1)
    carry_rate = np.log(cl2_price / cl1_price) / time_between_contracts
    
    # Discount CL1 back to spot using the carry rate
    # S = F1 * exp(-carry_rate * T1)
    spot_price = cl1_price * np.exp(-carry_rate * time_to_cl1)
    
    # Log calculation details for debugging
    logger.debug(
        f"Spot calculation: CL1=${cl1_price:.2f} ({cl1_days_to_expiry:.0f}d), "
        f"CL2=${cl2_price:.2f} ({cl2_days_to_expiry:.0f}d), "
        f"carry={carry_rate:.4f}, spot=${spot_price:.2f}"
    )
    
    return spot_price


def calculate_spot_price_series(
    futures_df: pd.DataFrame,
    cl1_col: str = 'cl1_price',
    cl2_col: str = 'cl2_price',
    cl1_expiry_col: str = 'cl1_days_to_expiry',
    cl2_expiry_col: str = 'cl2_days_to_expiry'
) -> pd.Series:
    """
    Calculate spot price for a time series of futures data.
    
    Args:
        futures_df: DataFrame with futures prices and expiry data
        cl1_col: Column name for CL1 prices
        cl2_col: Column name for CL2 prices
        cl1_expiry_col: Column name for CL1 days to expiry
        cl2_expiry_col: Column name for CL2 days to expiry
        
    Returns:
        Series of spot prices with same index as input
    """
    spot_prices = []
    
    for idx, row in futures_df.iterrows():
        try:
            spot = calculate_spot_price(
                cl1_price=row[cl1_col],
                cl2_price=row[cl2_col],
                cl1_days_to_expiry=row[cl1_expiry_col],
                cl2_days_to_expiry=row[cl2_expiry_col]
            )
            spot_prices.append(spot)
        except ValueError as e:
            logger.warning(f"Failed to calculate spot for {idx}: {e}")
            spot_prices.append(np.nan)
    
    return pd.Series(spot_prices, index=futures_df.index, name='spot_price')


def get_wti_expiry_date(year: int, month: int) -> datetime:
    """
    Calculate WTI futures expiry date using CME rules.
    
    WTI expires 3 business days before the 25th calendar day of the month
    prior to the delivery month. If the 25th is not a business day, use
    the prior business day.
    
    Args:
        year: Delivery year
        month: Delivery month (1-12)
        
    Returns:
        Expiry date (last trading date)
    """
    # Get month prior to delivery
    if month == 1:
        prior_month = 12
        prior_year = year - 1
    else:
        prior_month = month - 1
        prior_year = year
    
    # Start from 25th of prior month
    from pandas.tseries.offsets import BDay
    base_date = pd.Timestamp(prior_year, prior_month, 25)
    
    # If 25th is not a business day, go to prior business day
    if not pd.bdate_range(base_date, base_date).size:
        base_date = base_date - BDay(1)
    
    # Go back 3 business days
    expiry_date = base_date - BDay(3)
    
    return expiry_date.to_pydatetime()


def calculate_days_to_expiry(
    current_date: Union[datetime, pd.Timestamp],
    contract_code: str
) -> float:
    """
    Calculate days to expiry for a WTI contract.
    
    Args:
        current_date: Current date
        contract_code: Contract code (e.g., 'CLK25' for May 2025)
        
    Returns:
        Days to expiry (can be negative if expired)
    """
    # Extract month and year from contract code
    month_codes = {
        'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
        'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
    }
    
    if len(contract_code) < 5 or contract_code[:2] != 'CL':
        raise ValueError(f"Invalid contract code: {contract_code}")
    
    month_code = contract_code[2]
    year = 2000 + int(contract_code[3:5])
    
    if month_code not in month_codes:
        raise ValueError(f"Invalid month code in {contract_code}")
    
    month = month_codes[month_code]
    
    # Get expiry date
    expiry_date = get_wti_expiry_date(year, month)
    
    # Calculate days
    current_date = pd.Timestamp(current_date)
    days_to_expiry = (expiry_date - current_date).days
    
    return days_to_expiry


def validate_spot_calculation(
    spot_price: float,
    cl1_price: float,
    threshold: float = 0.10
) -> Tuple[bool, Optional[str]]:
    """
    Validate that calculated spot price is reasonable.
    
    Args:
        spot_price: Calculated spot price
        cl1_price: Front month price
        threshold: Maximum allowed deviation (default 10%)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if spot_price <= 0:
        return False, f"Spot price is non-positive: {spot_price}"
    
    # Check if spot is within reasonable range of front month
    deviation = abs(spot_price - cl1_price) / cl1_price
    if deviation > threshold:
        return False, f"Spot price ${spot_price:.2f} deviates {deviation:.1%} from CL1 ${cl1_price:.2f}"
    
    # Spot should generally be less than front month in contango
    # and greater than front month in backwardation
    # But this is market-dependent, so we just warn
    if spot_price > cl1_price * 1.05:
        logger.warning(f"Spot ${spot_price:.2f} is significantly above CL1 ${cl1_price:.2f} - strong backwardation")
    elif spot_price < cl1_price * 0.95:
        logger.warning(f"Spot ${spot_price:.2f} is significantly below CL1 ${cl1_price:.2f} - strong contango")
    
    return True, None


# Convenience function for backward compatibility
def get_spot_price(
    futures_strip: pd.DataFrame,
    date_col: str = 'expiry',
    price_col: str = 'price',
    reference_date: Optional[Union[str, datetime, pd.Timestamp]] = None
) -> float:
    """
    Calculate spot price from a futures strip DataFrame.
    
    This is a convenience wrapper for the canonical calculation,
    designed to work with existing code that passes a futures strip.
    
    Args:
        futures_strip: DataFrame with at least 2 rows, sorted by expiry
        date_col: Column containing expiry dates
        price_col: Column containing prices
        reference_date: Date to calculate from (for backtesting). If None, uses today.
        
    Returns:
        Calculated spot price
        
    Raises:
        ValueError: If insufficient data or calculation fails
    """
    if len(futures_strip) < 2:
        raise ValueError("Need at least 2 contracts to calculate spot price")
    
    # Get front two contracts
    cl1 = futures_strip.iloc[0]
    cl2 = futures_strip.iloc[1]
    
    # Calculate days to expiry from reference date
    if reference_date is None:
        ref_date = pd.Timestamp.now()
    else:
        ref_date = pd.Timestamp(reference_date)
    
    cl1_days = (pd.Timestamp(cl1[date_col]) - ref_date).days
    cl2_days = (pd.Timestamp(cl2[date_col]) - ref_date).days
    
    # Calculate spot
    spot = calculate_spot_price(
        cl1_price=cl1[price_col],
        cl2_price=cl2[price_col],
        cl1_days_to_expiry=cl1_days,
        cl2_days_to_expiry=cl2_days
    )
    
    # Validate
    is_valid, error_msg = validate_spot_calculation(spot, cl1[price_col])
    if not is_valid:
        logger.warning(f"Spot calculation validation failed: {error_msg}")
    
    return spot