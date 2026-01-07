"""
Outcome classification data models.

Contains diagnostic tags for signal validation outcomes and
expansion moves for linkage-driven hypothesis generation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class OutcomeTag(Enum):
    """
    Diagnostic classification for signal validation outcomes.

    Each tag indicates WHY a signal succeeded or failed,
    enabling targeted linkage expansion.

    Tags:
    - SUCCESS: Strong IC, stable across time, interpretable economic thesis
    - STATE_CONDITIONAL: Works only in specific regimes
    - EXPRESSION_MISMATCH: Signal captures pressure but wrong instrument
    - TIMING_MISMATCH: Signal has predictive content but at different horizon
    - PROXY_FAILURE: Observable doesn't capture the intended economic pressure
    - REDUNDANT: High correlation with existing promoted signal
    - TRUE_FAILURE: No predictive content found
    """
    SUCCESS = "success"
    STATE_CONDITIONAL = "state_conditional"
    EXPRESSION_MISMATCH = "expression_mismatch"
    TIMING_MISMATCH = "timing_mismatch"
    PROXY_FAILURE = "proxy_failure"
    REDUNDANT = "redundant"
    TRUE_FAILURE = "true_failure"


class ExpansionMove(Enum):
    """
    Canonical expansion moves for linkage-driven hypothesis generation.

    These moves are applied to classified outcomes to generate
    new hypotheses for testing.

    MVP moves (implement first):
    - SHIFT_EXPRESSION: flat -> spreads
    - SHIFT_HORIZON: lead/lag timing
    - CONDITION_ON_REGIME: add regime filter

    Phase 2+ moves:
    - SWAP_OBSERVABLE: different proxy for same pressure
    - WALK_ADJACENT: move to adjacent market
    - DECOMPOSE_SIGNAL: break down successful signal
    - GENERALIZE_ASSET: apply to other assets
    """
    # MVP moves
    SHIFT_EXPRESSION = "shift_expression"
    SHIFT_HORIZON = "shift_horizon"
    CONDITION_ON_REGIME = "condition_on_regime"

    # Phase 2+ moves
    SWAP_OBSERVABLE = "swap_observable"
    WALK_ADJACENT = "walk_adjacent"
    DECOMPOSE_SIGNAL = "decompose_signal"
    GENERALIZE_ASSET = "generalize_asset"


# Mapping from outcome tag to suggested expansion moves
OUTCOME_TO_MOVES: Dict[OutcomeTag, List[ExpansionMove]] = {
    OutcomeTag.SUCCESS: [
        ExpansionMove.DECOMPOSE_SIGNAL,
        ExpansionMove.GENERALIZE_ASSET,
    ],
    OutcomeTag.STATE_CONDITIONAL: [
        ExpansionMove.CONDITION_ON_REGIME,
    ],
    OutcomeTag.EXPRESSION_MISMATCH: [
        ExpansionMove.SHIFT_EXPRESSION,
    ],
    OutcomeTag.TIMING_MISMATCH: [
        ExpansionMove.SHIFT_HORIZON,
    ],
    OutcomeTag.PROXY_FAILURE: [
        ExpansionMove.SWAP_OBSERVABLE,
    ],
    OutcomeTag.REDUNDANT: [],  # No expansion needed
    OutcomeTag.TRUE_FAILURE: [
        ExpansionMove.WALK_ADJACENT,  # Try related market
    ],
}


@dataclass
class ClassificationResult:
    """
    Result of outcome classification.

    Contains the diagnostic tag plus context for specific outcomes.

    Attributes
    ----------
    variant_id : int
        Database ID of the variant
    outcome_tag : OutcomeTag
        Diagnostic classification
    confidence : float
        0-1 confidence in classification
    primary_reason : str
        Main reason for classification
    secondary_reasons : list
        Additional supporting reasons
    effective_regimes : dict, optional
        When signal works (for STATE_CONDITIONAL)
    ineffective_regimes : dict, optional
        When signal fails (for STATE_CONDITIONAL)
    optimal_horizon : int, optional
        Best horizon found (for TIMING_MISMATCH)
    current_horizon : int, optional
        Tested horizon (for TIMING_MISMATCH)
    suggested_expression : str, optional
        Better expression (for EXPRESSION_MISMATCH)
    correlated_signal : str, optional
        Name of correlated signal (for REDUNDANT)
    requires_human_review : bool
        Whether human review is needed (typically for SUCCESS)

    Methods
    -------
    suggested_expansion_moves()
        Return list of expansion moves based on outcome tag
    """
    variant_id: int
    outcome_tag: OutcomeTag
    confidence: float

    primary_reason: str
    secondary_reasons: List[str] = field(default_factory=list)

    # Context for STATE_CONDITIONAL
    effective_regimes: Optional[Dict[str, Any]] = None
    ineffective_regimes: Optional[Dict[str, Any]] = None

    # Context for TIMING_MISMATCH
    optimal_horizon: Optional[int] = None
    current_horizon: Optional[int] = None

    # Context for EXPRESSION_MISMATCH
    suggested_expression: Optional[str] = None

    # Context for REDUNDANT
    correlated_signal: Optional[str] = None

    # Human review
    requires_human_review: bool = False

    def suggested_expansion_moves(self) -> List[ExpansionMove]:
        """
        Get suggested expansion moves based on outcome tag.

        Returns
        -------
        list[ExpansionMove]
            List of expansion moves to try, may be empty
        """
        return OUTCOME_TO_MOVES.get(self.outcome_tag, [])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for database storage."""
        return {
            'variant_id': self.variant_id,
            'outcome_tag': self.outcome_tag.value,
            'confidence': self.confidence,
            'primary_reason': self.primary_reason,
            'secondary_reasons': self.secondary_reasons,
            'effective_regimes': self.effective_regimes,
            'ineffective_regimes': self.ineffective_regimes,
            'optimal_horizon_days': self.optimal_horizon,
            'current_horizon_days': self.current_horizon,
            'suggested_expression': self.suggested_expression,
            'most_correlated_signal': self.correlated_signal,
            'requires_human_review': self.requires_human_review,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClassificationResult':
        """Create from dictionary (database row)."""
        return cls(
            variant_id=data['variant_id'],
            outcome_tag=OutcomeTag(data['outcome_tag']),
            confidence=float(data['confidence']),
            primary_reason=data['primary_reason'],
            secondary_reasons=data.get('secondary_reasons') or [],
            effective_regimes=data.get('effective_regimes'),
            ineffective_regimes=data.get('ineffective_regimes'),
            optimal_horizon=data.get('optimal_horizon_days'),
            current_horizon=data.get('current_horizon_days'),
            suggested_expression=data.get('suggested_expression'),
            correlated_signal=data.get('most_correlated_signal'),
            requires_human_review=data.get('requires_human_review', False),
        )

    def __repr__(self) -> str:
        review = " [REVIEW]" if self.requires_human_review else ""
        return (
            f"ClassificationResult(variant={self.variant_id}, "
            f"tag={self.outcome_tag.value}, "
            f"conf={self.confidence:.2f}{review})"
        )
