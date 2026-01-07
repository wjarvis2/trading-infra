"""
Versioned Interface Contracts.

These are the "hard" interfaces that define the contracts between components.
Changes require a version bump and follow the deprecation policy.

VERSION: 1.0.0 (2026-01-03)

Contract changes:
- 1.0.0: Initial versioned contracts

"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)
import pandas as pd

from .bar_schema import (
    REQUIRED_BAR_FIELDS,
    validate_bar_schema,
)


# =============================================================================
# Contract Version
# =============================================================================

CONTRACT_VERSION = "1.0.0"


# =============================================================================
# DataFeed Contract
# =============================================================================

# Required fields for bars from DataFeed (execution needs OHLC)
# These fields must be present in each bar (pd.Series)
DATAFEED_REQUIRED_FIELDS: FrozenSet[str] = frozenset({
    "open",
    "high",
    "low",
    "close",
})


@runtime_checkable
class DataFeedContract(Protocol):
    """
    Contract for data feeds used by GenericRunner.

    VERSION: 1.0.0

    Required methods:
    - iter_timestamps() -> Iterator[datetime]
    - get_bars(timestamp) -> Dict[str, pd.Series]
    - instruments property -> List[str]

    Required output schema (per bar Series):
    - open, high, low, close: float (OHLC prices)
    - volume: float (optional but recommended)

    Note: Returns Dict[str, pd.Series] where key is instrument_id,
    not a DataFrame. This aligns with executor expectations.
    """

    def iter_timestamps(self) -> Iterator[datetime]:
        """
        Iterate over all timestamps in the feed.

        Timestamps must be:
        - Monotonically increasing
        - Timezone-aware or consistently naive
        """
        ...

    def get_bars(self, timestamp: datetime) -> Dict[str, pd.Series]:
        """
        Get bars for all instruments at a timestamp.

        Parameters
        ----------
        timestamp : datetime
            Timestamp to query

        Returns
        -------
        Dict[str, pd.Series]
            Bars by instrument_id. Each Series contains OHLC fields.
            Empty dict if no data.
        """
        ...

    @property
    def instruments(self) -> List[str]:
        """List of instrument IDs available in the feed."""
        ...


def validate_datafeed_output(bars: Dict[str, pd.Series], context: str = "") -> None:
    """
    Validate DataFeed output conforms to contract.

    Parameters
    ----------
    bars : Dict[str, pd.Series]
        Output from get_bars()
    context : str
        Context for error messages

    Raises
    ------
    ContractViolation
        If output doesn't conform to contract
    """
    if not bars:
        return  # Empty is valid

    for instrument_id, bar in bars.items():
        missing = DATAFEED_REQUIRED_FIELDS - set(bar.index)
        if missing:
            raise ContractViolation(
                f"Bar for {instrument_id} missing required fields: {sorted(missing)}. "
                f"Available: {sorted(bar.index)}. Context: {context}"
            )


# =============================================================================
# Executor Contract
# =============================================================================


@runtime_checkable
class ExecutorContract(Protocol):
    """
    Contract for executors used by GenericRunner.

    VERSION: 1.0.0

    Required methods:
    - execute(signal, bars) -> List[ExecutionEvent]
    - mark_to_market(instrument_ids, bars) -> Dict[str, float]

    ExecutionEvent is either FillEvent or RejectEvent.
    Note: bars is Dict[str, pd.Series], matching DataFeed output.
    """

    def execute(
        self,
        signal: Any,  # Signal type (PositionSignal)
        bars: Dict[str, pd.Series],
    ) -> List[Any]:  # List[ExecutionEvent]
        """
        Execute a signal against current bars.

        Parameters
        ----------
        signal : Signal
            Position signal to execute
        bars : Dict[str, pd.Series]
            Current bars by instrument_id

        Returns
        -------
        list[ExecutionEvent]
            Fill events or reject events
        """
        ...

    def mark_to_market(
        self,
        instrument_ids: List[str],
        bars: Dict[str, pd.Series],
    ) -> Dict[str, float]:
        """
        Get current prices for mark-to-market.

        Parameters
        ----------
        instrument_ids : list[str]
            Instruments to price
        bars : Dict[str, pd.Series]
            Current bars by instrument_id

        Returns
        -------
        dict[str, float]
            Instrument ID to current price
        """
        ...


# =============================================================================
# RunResult Contract
# =============================================================================

# Required fields in RunResult
RUNRESULT_REQUIRED_FIELDS: FrozenSet[str] = frozenset({
    "total_return",
    "total_return_pct",
    "sharpe_ratio",
    "max_drawdown",
    "max_drawdown_pct",
    "total_trades",
    "win_rate",
    "equity_curve",
    "positions",
    "events",
    "initial_capital",
    "final_equity",
})


def validate_run_result(result: Any, context: str = "") -> None:
    """
    Validate RunResult conforms to contract.

    Parameters
    ----------
    result : RunResult
        Result to validate
    context : str
        Context for error messages

    Raises
    ------
    ContractViolation
        If result doesn't conform to contract
    """
    missing = []
    for field_name in RUNRESULT_REQUIRED_FIELDS:
        if not hasattr(result, field_name):
            missing.append(field_name)

    if missing:
        raise ContractViolation(
            f"RunResult missing required fields: {sorted(missing)}. "
            f"Context: {context}"
        )


# =============================================================================
# Contract Exceptions
# =============================================================================


class ContractViolation(Exception):
    """Raised when a component violates its contract."""

    def __init__(self, message: str, contract_version: str = CONTRACT_VERSION):
        self.contract_version = contract_version
        super().__init__(f"[Contract v{contract_version}] {message}")


# =============================================================================
# Contract Registry
# =============================================================================


@dataclass
class ContractInfo:
    """Information about a contract."""
    name: str
    version: str
    required_methods: Tuple[str, ...]
    required_fields: FrozenSet[str]
    description: str


CONTRACTS: Dict[str, ContractInfo] = {
    "DataFeed": ContractInfo(
        name="DataFeed",
        version="1.0.0",
        required_methods=("iter_timestamps", "get_bars", "instruments"),
        required_fields=DATAFEED_REQUIRED_FIELDS,
        description="Contract for data feeds providing OHLC bars",
    ),
    "Executor": ContractInfo(
        name="Executor",
        version="1.0.0",
        required_methods=("execute", "mark_to_market"),
        required_fields=frozenset(),
        description="Contract for order execution",
    ),
    "RunResult": ContractInfo(
        name="RunResult",
        version="1.0.0",
        required_methods=(),
        required_fields=RUNRESULT_REQUIRED_FIELDS,
        description="Contract for backtest/run results",
    ),
}


def get_contract_version(contract_name: str) -> str:
    """Get version of a named contract."""
    if contract_name not in CONTRACTS:
        raise ValueError(f"Unknown contract: {contract_name}")
    return CONTRACTS[contract_name].version
