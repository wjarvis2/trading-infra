"""
Signal Mining data models.

All models are frozen dataclasses for immutability.
"""

from common_lib.signal_mining.models.hypothesis_spec import (
    HypothesisSpec,
    VariantSpec,
    SamplingClock,
    TargetType,
    Direction,
    NormalizationType,
)
from common_lib.signal_mining.models.scoring_result import (
    MetricBundle,
    ScoringResult,
)
from common_lib.signal_mining.models.outcome import (
    OutcomeTag,
    ClassificationResult,
    ExpansionMove,
)

__all__ = [
    'HypothesisSpec',
    'VariantSpec',
    'SamplingClock',
    'TargetType',
    'Direction',
    'NormalizationType',
    'MetricBundle',
    'ScoringResult',
    'OutcomeTag',
    'ClassificationResult',
    'ExpansionMove',
]
