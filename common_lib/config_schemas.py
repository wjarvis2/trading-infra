"""
Configuration schema validation using Pydantic.

Provides validated configuration models for all strategies and components.
Ensures type safety and catches configuration errors at load time.

Created: 2025-08-06
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime


class SpreadDefinition(BaseModel):
    """Definition of a spread instrument."""
    front_contract: str = Field(..., description="Front month contract (e.g., 'CL1')")
    back_contract: str = Field(..., description="Back month contract (e.g., 'CL2')")
    direction: Literal['long', 'short'] = Field('long', description="Spread direction")
    weight: float = Field(1.0, description="Position weight in portfolio")
    
    @validator('front_contract', 'back_contract')
    def validate_contract_format(cls, v):
        """Ensure contract follows expected format."""
        if not v or not v[0].isalpha():
            raise ValueError(f"Invalid contract format: {v}")
        return v
    
    @validator('weight')
    def validate_weight(cls, v):
        """Ensure weight is positive."""
        if v <= 0:
            raise ValueError(f"Weight must be positive, got {v}")
        return v


class SeasonalPatternConfig(BaseModel):
    """Configuration for a seasonal pattern."""
    months: List[int] = Field(..., description="Active months (1-12)")
    direction: Literal['long', 'short'] = Field(..., description="Pattern direction")
    base_adjustment: float = Field(..., description="Base adjustment factor")
    confidence: float = Field(0.7, ge=0, le=1, description="Pattern confidence (0-1)")
    smooth_transitions: bool = Field(True, description="Use smooth transitions between periods")
    transition_days: int = Field(5, ge=0, description="Days for transition smoothing")
    
    @validator('months')
    def validate_months(cls, v):
        """Ensure months are valid."""
        for month in v:
            if month < 1 or month > 12:
                raise ValueError(f"Invalid month: {month}. Must be 1-12.")
        return v
    
    @validator('base_adjustment')
    def validate_adjustment(cls, v):
        """Ensure adjustment is reasonable."""
        if abs(v) > 10:
            raise ValueError(f"Base adjustment too large: {v}. Should be < 10.")
        return v


class RiskParameters(BaseModel):
    """Risk management parameters."""
    max_position_size: int = Field(10, ge=1, description="Maximum contracts per position")
    max_portfolio_risk: float = Field(0.02, gt=0, le=0.1, description="Max portfolio risk per trade")
    stop_loss_atr_multiple: float = Field(2.0, gt=0, description="Stop loss in ATR multiples")
    position_sizing_method: Literal['fixed', 'volatility', 'kelly'] = Field('volatility')
    use_regime_scaling: bool = Field(True, description="Scale positions by volatility regime")
    max_correlation: float = Field(0.7, ge=0, le=1, description="Max correlation between positions")
    
    @validator('max_portfolio_risk')
    def validate_risk(cls, v):
        """Ensure risk is reasonable."""
        if v > 0.05:
            raise ValueError(f"Portfolio risk {v} too high. Should be <= 5%.")
        return v


class SignalParameters(BaseModel):
    """Signal generation parameters."""
    entry_threshold: float = Field(1.5, gt=0, description="Entry threshold in std deviations")
    exit_threshold: float = Field(0.5, gt=0, description="Exit threshold in std deviations")
    min_holding_period: int = Field(24, ge=0, description="Minimum holding period in bars")
    max_holding_period: int = Field(480, ge=0, description="Maximum holding period in bars")
    use_inventory_signal: bool = Field(True, description="Use inventory deviations")
    inventory_weight: float = Field(0.3, ge=0, le=1, description="Weight of inventory signal")
    
    @validator('exit_threshold')
    def validate_thresholds(cls, v, values):
        """Ensure exit threshold is less than entry threshold."""
        if 'entry_threshold' in values and v >= values['entry_threshold']:
            raise ValueError(f"Exit threshold {v} must be less than entry threshold {values['entry_threshold']}")
        return v


class ExitParameters(BaseModel):
    """Exit management parameters."""
    use_time_exit: bool = Field(True, description="Use time-based exits")
    use_profit_target: bool = Field(True, description="Use profit targets")
    profit_target_atr: float = Field(3.0, gt=0, description="Profit target in ATR multiples")
    use_trailing_stop: bool = Field(False, description="Use trailing stops")
    trailing_stop_atr: float = Field(1.5, gt=0, description="Trailing stop in ATR multiples")
    seasonal_flip_exit: bool = Field(True, description="Exit on seasonal pattern reversal")


class SeasonalityParameters(BaseModel):
    """Seasonality calculation parameters."""
    lookback_years: int = Field(5, ge=1, le=20, description="Years of history for patterns")
    min_pattern_confidence: float = Field(0.6, ge=0, le=1, description="Minimum pattern confidence")
    use_dynamic_patterns: bool = Field(True, description="Update patterns dynamically")
    pattern_update_frequency: int = Field(60, ge=1, description="Bars between pattern updates")


class SeasonalSpreadConfig(BaseModel):
    """Complete configuration for seasonal spread strategy."""
    strategy_name: str = Field("SeasonalSpread", description="Strategy identifier")
    version: str = Field("1.0.0", description="Strategy version")
    
    # Sub-configurations
    spreads: List[SpreadDefinition] = Field(..., description="Spread definitions")
    patterns: Dict[str, SeasonalPatternConfig] = Field(..., description="Seasonal patterns")
    risk: RiskParameters = Field(default_factory=RiskParameters)
    signal: SignalParameters = Field(default_factory=SignalParameters)
    exit: ExitParameters = Field(default_factory=ExitParameters)
    seasonality: SeasonalityParameters = Field(default_factory=SeasonalityParameters)
    
    # Execution settings
    bar_frequency: str = Field("15s", description="Bar frequency for backtesting")
    warmup_bars: int = Field(240, ge=0, description="Warmup bars before trading")
    commission_per_contract: float = Field(2.0, ge=0, description="Commission per contract")
    slippage_mode: Literal['fixed', 'dynamic'] = Field('dynamic', description="Slippage model")
    
    @validator('spreads')
    def validate_spreads(cls, v):
        """Ensure at least one spread is defined."""
        if not v:
            raise ValueError("At least one spread must be defined")
        return v
    
    @validator('patterns')
    def validate_patterns(cls, v):
        """Ensure at least one pattern is defined."""
        if not v:
            raise ValueError("At least one pattern must be defined")
        return v
    
    @validator('bar_frequency')
    def validate_frequency(cls, v):
        """Ensure frequency is valid."""
        from common_lib.constants import BARS_PER_DAY
        if v not in BARS_PER_DAY:
            raise ValueError(f"Invalid bar frequency: {v}. Must be one of {list(BARS_PER_DAY.keys())}")
        return v
    
    class Config:
        """Pydantic config."""
        extra = 'forbid'  # Don't allow extra fields
        validate_assignment = True  # Validate on assignment


def validate_strategy_config(config_dict: Dict[str, Any], strategy_type: str = 'seasonal_spread') -> BaseModel:
    """
    Validate a strategy configuration dictionary.
    
    Args:
        config_dict: Raw configuration dictionary
        strategy_type: Type of strategy to validate
        
    Returns:
        Validated configuration model
        
    Raises:
        ValidationError: If configuration is invalid
    """
    if strategy_type == 'seasonal_spread':
        # Extract parameters section if nested
        if 'parameters' in config_dict:
            # Flatten the structure for validation
            validated_config = {
                'strategy_name': config_dict.get('name', 'SeasonalSpread'),
                'version': config_dict.get('version', '1.0.0'),
                'spreads': config_dict['parameters'].get('spreads', []),
                'patterns': config_dict['parameters'].get('patterns', {}),
                'risk': config_dict['parameters'].get('risk', {}),
                'signal': config_dict['parameters'].get('signal', {}),
                'exit': config_dict['parameters'].get('exit', {}),
                'seasonality': config_dict['parameters'].get('seasonality', {}),
                'bar_frequency': config_dict.get('bar_frequency', '15s'),
                'warmup_bars': config_dict.get('warmup_bars', 240),
                'commission_per_contract': config_dict.get('commission_per_contract', 2.0),
                'slippage_mode': config_dict.get('slippage_mode', 'dynamic'),
            }
            return SeasonalSpreadConfig(**validated_config)
        else:
            return SeasonalSpreadConfig(**config_dict)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")