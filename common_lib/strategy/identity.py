"""
Generic strategy identity canonicalization utility.

Provides a factory for creating strategy ID canonicalizers. Each strategy
defines its own alias table and canonical IDs - this module is strategy-agnostic.

Usage
-----
>>> from common_lib.strategy.identity import create_canonicalizer, create_validator
>>>
>>> # Strategy defines its own aliases and valid IDs
>>> ALIASES = {"spread_regime_mr": "MR", "mr": "MR"}
>>> VALID_IDS = {"MR", "MO"}
>>> canonicalize = create_canonicalizer(ALIASES)
>>> assert_valid = create_validator(VALID_IDS, ALIASES, "spread_regime")
>>>
>>> canonicalize("spread_regime_mr")  # -> "MR"
>>> canonicalize("unknown")           # -> "unknown"
>>>
>>> assert_valid("MR")                # passes
>>> assert_valid("unknown")           # raises InvalidStrategyIdError

"""

from typing import Callable, Dict, Optional, Set


class InvalidStrategyIdError(ValueError):
    """Raised when a strategy ID is not in the allowed set."""

    def __init__(self, strategy_id: str, valid_ids: Set[str], context: str = ""):
        self.strategy_id = strategy_id
        self.valid_ids = valid_ids
        self.context = context
        msg = f"Invalid strategy ID '{strategy_id}'"
        if context:
            msg += f" in {context}"
        msg += f". Valid IDs: {sorted(valid_ids)}"
        super().__init__(msg)


def create_canonicalizer(
    aliases: Dict[str, str],
    case_insensitive: bool = True,
) -> Callable[[Optional[str]], str]:
    """
    Create a canonicalization function from an alias table.

    Parameters
    ----------
    aliases : dict[str, str]
        Mapping from alias -> canonical ID.
        If case_insensitive=True, keys should be lowercase.
    case_insensitive : bool, default True
        If True, lookup uses .lower() on input.

    Returns
    -------
    Callable[[str | None], str]
        Function that canonicalizes strategy IDs.
        Returns "" for None/empty/whitespace.
        Returns trimmed original if no alias match.

    Examples
    --------
    >>> aliases = {"mr": "MR", "spread_regime_mr": "MR"}
    >>> canon = create_canonicalizer(aliases)
    >>> canon("spread_regime_mr")
    'MR'
    >>> canon("  MR  ")
    'MR'
    >>> canon(None)
    ''
    >>> canon("unknown")
    'unknown'
    """
    def canonicalize(strategy: Optional[str]) -> str:
        if strategy is None:
            return ""

        trimmed = strategy.strip()
        if not trimmed:
            return ""

        key = trimmed.lower() if case_insensitive else trimmed
        return aliases.get(key, trimmed)

    return canonicalize


def create_validator(
    valid_ids: Set[str],
    aliases: Dict[str, str],
    context: str = "",
    case_insensitive: bool = True,
) -> Callable[[Optional[str]], str]:
    """
    Create a validation function that asserts strategy ID is valid.

    Canonicalizes the input first, then checks if it's in the valid set.

    Parameters
    ----------
    valid_ids : set[str]
        Set of canonical strategy IDs that are allowed.
    aliases : dict[str, str]
        Alias mapping (same as create_canonicalizer).
    context : str, optional
        Context for error messages (e.g., "spread_regime").
    case_insensitive : bool, default True
        If True, lookup uses .lower() on input.

    Returns
    -------
    Callable[[str | None], str]
        Function that validates and returns canonical ID.
        Raises InvalidStrategyIdError if not valid.

    Examples
    --------
    >>> valid = {"MR", "MO"}
    >>> aliases = {"mr": "MR", "spread_regime_mr": "MR"}
    >>> validate = create_validator(valid, aliases, "position_creation")
    >>> validate("mr")
    'MR'
    >>> validate("unknown")
    InvalidStrategyIdError: Invalid strategy ID 'unknown' in position_creation
    """
    canonicalize = create_canonicalizer(aliases, case_insensitive)

    def assert_valid(strategy: Optional[str]) -> str:
        canonical = canonicalize(strategy)

        if not canonical:
            raise InvalidStrategyIdError("", valid_ids, context)

        if canonical not in valid_ids:
            raise InvalidStrategyIdError(canonical, valid_ids, context)

        return canonical

    return assert_valid
