"""
Signal Mining Flywheel

A self-propelling research engine for systematic commodities signal discovery.

Key components:
- models: Data structures for hypotheses, scores, and outcomes
- scorers: IC, rank-IC, and decay profile calculations
- state: Regime state vector with hysteresis
- classifier: Automated outcome classification
- expander: Linkage-driven hypothesis expansion
- registry: Promoted signal storage and drift monitoring
- orchestrator: Main flywheel loop
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
    # Hypothesis types
    'HypothesisSpec',
    'VariantSpec',
    'SamplingClock',
    'TargetType',
    'Direction',
    'NormalizationType',
    # Scoring types
    'MetricBundle',
    'ScoringResult',
    # Outcome types
    'OutcomeTag',
    'ClassificationResult',
    'ExpansionMove',
]
