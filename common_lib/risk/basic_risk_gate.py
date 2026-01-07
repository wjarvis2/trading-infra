"""
Basic Risk Gate Implementation.

Reference implementation of the RiskGate protocol with common risk checks:
- Position count limits
- Drawdown kill switches
- Position size limits
- Concentration limits

"""

from dataclasses import dataclass, replace
from typing import List, Optional

from common_lib.interfaces.signal import Signal, SignalIntent
from common_lib.interfaces.risk_gate import (
    RiskGate,
    RiskGateResult,
    approve,
    reject,
    modify,
)
from common_lib.portfolio.event_ledger import EventLedger, LedgerPosition


@dataclass
class BasicRiskGate:
    """
    Basic risk gate with configurable limits.

    Parameters
    ----------
    max_positions : int
        Maximum number of concurrent positions (default 5)
    max_drawdown_pct : float
        Maximum drawdown before blocking new entries (default 0.10 = 10%)
    max_position_size : float
        Maximum position size in contracts (default 5.0)
    max_strategy_concentration : int
        Maximum positions per strategy (default None = no limit)
    block_entries_on_drawdown : bool
        If True, block new entries during drawdown (default True)
    allow_exits_always : bool
        If True, never block CLOSE signals (default True)

    Examples
    --------
    >>> gate = BasicRiskGate(max_positions=3, max_drawdown_pct=0.05)
    >>> runner = GenericRunner(
    ...     strategy=my_strategy,
    ...     data_feed=my_feed,
    ...     executor=executor,
    ...     ledger=ledger,
    ...     risk_gate=gate,
    ... )
    """

    max_positions: int = 5
    max_drawdown_pct: float = 0.10
    max_position_size: float = 5.0
    max_strategy_concentration: Optional[int] = None
    block_entries_on_drawdown: bool = True
    allow_exits_always: bool = True

    def evaluate(
        self,
        signal: Signal,
        open_positions: List[LedgerPosition],
        equity: float,
        ledger: EventLedger,
    ) -> RiskGateResult:
        """
        Evaluate signal against risk constraints.

        Checks are applied in order:
        1. Always allow exits (if configured)
        2. Drawdown kill switch
        3. Position count limit
        4. Strategy concentration limit
        5. Position size limit (may modify signal)
        """
        # 1. Always allow exits
        if self.allow_exits_always and signal.intent == SignalIntent.CLOSE:
            return approve(signal)

        # 2. Drawdown kill switch
        if self.block_entries_on_drawdown and signal.intent == SignalIntent.OPEN:
            unrealized_pnl = ledger.get_total_unrealized_pnl()
            if equity > 0:
                drawdown_pct = -unrealized_pnl / equity if unrealized_pnl < 0 else 0.0
                if drawdown_pct >= self.max_drawdown_pct:
                    return reject(
                        f"Drawdown {drawdown_pct:.1%} >= limit {self.max_drawdown_pct:.1%}"
                    )

        # 3. Position count limit
        if signal.intent == SignalIntent.OPEN:
            if len(open_positions) >= self.max_positions:
                return reject(
                    f"Position count {len(open_positions)} >= limit {self.max_positions}"
                )

        # 4. Strategy concentration limit
        if self.max_strategy_concentration is not None and signal.intent == SignalIntent.OPEN:
            strategy_positions = [
                p for p in open_positions
                if p.metadata.get("strategy") == signal.strategy
            ]
            if len(strategy_positions) >= self.max_strategy_concentration:
                return reject(
                    f"Strategy '{signal.strategy}' has {len(strategy_positions)} "
                    f"positions >= limit {self.max_strategy_concentration}"
                )

        # 5. Position size limit (may modify)
        if abs(signal.target_position) > self.max_position_size:
            # Scale down to max allowed
            capped_position = (
                self.max_position_size
                if signal.target_position > 0
                else -self.max_position_size
            )
            # Create modified signal (Signal is frozen, use replace)
            modified_signal = Signal(
                timestamp=signal.timestamp,
                tradable=signal.tradable,
                intent=signal.intent,
                target_position=capped_position,
                strategy=signal.strategy,
                reason=signal.reason,
                roll_to=signal.roll_to,
                execution_style=signal.execution_style,
                metadata=signal.metadata,
                confidence=signal.confidence,
                magnitude=signal.magnitude,
                expires_at=signal.expires_at,
            )
            return modify(
                modified_signal,
                f"Position size {abs(signal.target_position)} capped to {self.max_position_size}",
            )

        # All checks passed
        return approve(signal)


@dataclass
class DrawdownKillSwitch:
    """
    Simple risk gate that only checks drawdown.

    Useful for adding a single kill switch without other limits.

    Parameters
    ----------
    max_drawdown_pct : float
        Maximum drawdown before blocking (default 0.15 = 15%)
    allow_exits : bool
        Always allow exit signals (default True)
    """

    max_drawdown_pct: float = 0.15
    allow_exits: bool = True

    def evaluate(
        self,
        signal: Signal,
        open_positions: List[LedgerPosition],
        equity: float,
        ledger: EventLedger,
    ) -> RiskGateResult:
        """Evaluate signal - only checks drawdown."""
        if self.allow_exits and signal.intent == SignalIntent.CLOSE:
            return approve(signal)

        unrealized_pnl = ledger.get_total_unrealized_pnl()
        if equity > 0 and unrealized_pnl < 0:
            drawdown_pct = -unrealized_pnl / equity
            if drawdown_pct >= self.max_drawdown_pct:
                return reject(
                    f"Kill switch: drawdown {drawdown_pct:.1%} >= {self.max_drawdown_pct:.1%}"
                )

        return approve(signal)


@dataclass
class PositionLimiter:
    """
    Simple risk gate that only limits position count.

    Parameters
    ----------
    max_positions : int
        Maximum number of concurrent positions
    """

    max_positions: int = 5

    def evaluate(
        self,
        signal: Signal,
        open_positions: List[LedgerPosition],
        equity: float,
        ledger: EventLedger,
    ) -> RiskGateResult:
        """Evaluate signal - only checks position count."""
        if signal.intent == SignalIntent.CLOSE:
            return approve(signal)

        if signal.intent == SignalIntent.OPEN:
            if len(open_positions) >= self.max_positions:
                return reject(
                    f"Position limit: {len(open_positions)} >= {self.max_positions}"
                )

        return approve(signal)
