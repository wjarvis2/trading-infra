"""
Microstructure analysis and curve momentum calculations.
"""

import pandas as pd
from .technical import atr


def detect_microstructure_breaks(
    df: pd.DataFrame,
    lookback_bars: int = 20,
    volume_threshold: float = 2.0,
    price_threshold: float = 2.0
) -> pd.DataFrame:
    """
    Detect microstructure breaks (unusual volume/price moves).
    
    Parameters
    ----------
    df : pd.DataFrame
        Intraday OHLCV data
    lookback_bars : int
        Number of bars for rolling statistics
    volume_threshold : float
        Threshold for volume spike (in standard deviations)
    price_threshold : float
        Threshold for price move (in ATR units)
        
    Returns
    -------
    pd.DataFrame
        Original data with break indicators
    """
    # Calculate rolling statistics
    df['volume_mean'] = df['volume'].rolling(lookback_bars).mean()
    df['volume_std'] = df['volume'].rolling(lookback_bars).std()
    df['volume_zscore'] = (df['volume'] - df['volume_mean']) / df['volume_std']
    
    # Calculate ATR for price move normalization
    df['atr'] = pd.DataFrame({
        'atr': atr(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            window=lookback_bars
        )
    }, index=df.index)['atr']
    
    # Price change in ATR units
    df['price_change'] = df['close'].diff().abs()
    df['price_change_atr'] = df['price_change'] / df['atr']
    
    # Identify breaks
    df['volume_break'] = df['volume_zscore'].abs() > volume_threshold
    df['price_break'] = df['price_change_atr'] > price_threshold
    df['microstructure_break'] = df['volume_break'] | df['price_break']
    
    # Clean up intermediate columns
    df = df.drop(columns=['volume_mean', 'volume_std', 'price_change'])
    
    return df


def calculate_curve_momentum(
    curve_df: pd.DataFrame,
    lookback_bars: int = 60,
    contracts: list = None
) -> pd.DataFrame:
    """
    Calculate momentum metrics for futures curve.
    
    Parameters
    ----------
    curve_df : pd.DataFrame
        DataFrame with cl1, cl2, ..., cl8 columns
    lookback_bars : int
        Lookback period for momentum calculation
    contracts : list
        List of contracts to analyze (default: ['cl1', 'cl2', 'cl3', 'cl4'])
        
    Returns
    -------
    pd.DataFrame
        Momentum metrics for each contract
    """
    if contracts is None:
        contracts = ['cl1', 'cl2', 'cl3', 'cl4']
    
    result = pd.DataFrame(index=curve_df.index)
    
    for contract in contracts:
        if contract in curve_df.columns:
            # Simple momentum (rate of change)
            result[f'{contract}_momentum'] = (
                curve_df[contract] / curve_df[contract].shift(lookback_bars) - 1
            ) * 100
            
            # Normalized momentum (z-score of returns)
            returns = curve_df[contract].pct_change()
            result[f'{contract}_momentum_zscore'] = (
                returns.rolling(lookback_bars).mean() / 
                returns.rolling(lookback_bars).std()
            )
    
    # Cross-contract momentum spread
    if 'cl1' in contracts and 'cl2' in contracts:
        result['front_back_momentum_spread'] = (
            result['cl1_momentum'] - result['cl2_momentum']
        )
    
    return result