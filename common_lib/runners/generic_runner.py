"""
Generic Backtest/Simulation Runner.

Orchestrates: DataFeed → Strategy → Executor → Ledger

Design principles:
- Strategy-agnostic: uses minimal StrategyProtocol
- DataFeed-agnostic: uses minimal DataFeedProtocol
- Event-driven: all state changes via EventLedger.apply()
- Composable: swap executor, ledger, or strategy implementations

"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Tuple
from datetime import datetime, date, timedelta
import pandas as pd
import logging

from common_lib.interfaces.signal import Signal
from common_lib.interfaces.execution_events import ExecutionEvent, FillEvent, RejectEvent
from common_lib.interfaces.risk_gate import RiskGate, RiskGateEvent, RiskGateResult
from common_lib.portfolio.event_ledger import EventLedger, LedgerPosition, create_ledger
from common_lib.execution.event_executor import EventExecutor, create_event_executor


logger = logging.getLogger(__name__)


# =============================================================================
# Protocols (minimal interfaces)
# =============================================================================


class StrategyProtocol(Protocol):
    """
    Minimal interface for strategies to integrate with GenericRunner.

    Only requires one method: generate_signals().
    Your strategy can have any additional methods/state it needs.
    """

    def generate_signals(
        self,
        timestamp: datetime,
        bars: Dict[str, pd.Series],
        open_positions: List[LedgerPosition],
    ) -> List[Signal]:
        """
        Generate signals for the current timestamp.

        Parameters
        ----------
        timestamp : datetime
            Current bar timestamp
        bars : dict[str, pd.Series]
            Current bars by instrument_id
        open_positions : list[LedgerPosition]
            Currently open positions

        Returns
        -------
        list[Signal]
            Signals to execute (can be empty)
        """
        ...


class DataFeedProtocol(Protocol):
    """
    Minimal interface for data feeds.

    Must provide iteration over timestamps and bar retrieval.
    """

    def iter_timestamps(self) -> Iterator[datetime]:
        """Iterate over all timestamps in the feed."""
        ...

    def get_bars(self, timestamp: datetime) -> Dict[str, pd.Series]:
        """Get bars for all instruments at a timestamp."""
        ...

    @property
    def instruments(self) -> List[str]:
        """List of instrument IDs in the feed."""
        ...


# =============================================================================
# Result Container
# =============================================================================


@dataclass
class EquityPoint:
    """Single point on the equity curve."""
    timestamp: datetime
    cash: float
    unrealized_pnl: float
    total_equity: float
    drawdown: float
    drawdown_pct: float


@dataclass
class RunResult:
    """
    Complete results from a runner execution.

    Contains all the data needed for analysis:
    - Summary metrics (Sharpe, returns, etc.)
    - Equity curve
    - Positions
    - Events (audit trail)
    """
    # Summary metrics
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float

    # Trade metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    # Granular data
    equity_curve: pd.DataFrame
    positions: List[LedgerPosition]
    events: List[ExecutionEvent]

    # Metadata
    start_time: datetime
    end_time: datetime
    initial_capital: float
    final_equity: float
    runtime_seconds: float

    # Risk gate tracking
    risk_gate_events: List[RiskGateEvent] = field(default_factory=list)

    # Strategy-specific metrics
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Equity Tracker
# =============================================================================


class EquityTracker:
    """
    Tracks equity curve during a run.

    Simple tracker that records cash + unrealized P&L at each update.
    """

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self._points: List[EquityPoint] = []
        self._peak_equity = initial_capital

    def record_trade_pnl(self, realized_pnl: float) -> None:
        """Update cash when a trade is closed."""
        self.cash += realized_pnl

    def update(
        self,
        timestamp: datetime,
        unrealized_pnl: float,
    ) -> None:
        """Record an equity point."""
        total_equity = self.cash + unrealized_pnl

        # Track peak and drawdown
        if total_equity > self._peak_equity:
            self._peak_equity = total_equity

        drawdown = self._peak_equity - total_equity
        drawdown_pct = drawdown / self._peak_equity if self._peak_equity > 0 else 0.0

        point = EquityPoint(
            timestamp=timestamp,
            cash=self.cash,
            unrealized_pnl=unrealized_pnl,
            total_equity=total_equity,
            drawdown=drawdown,
            drawdown_pct=drawdown_pct,
        )
        self._points.append(point)

    @property
    def current_equity(self) -> float:
        """Current total equity."""
        if self._points:
            return self._points[-1].total_equity
        return self.cash

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown in dollars."""
        if not self._points:
            return 0.0
        return max(p.drawdown for p in self._points)

    @property
    def max_drawdown_pct(self) -> float:
        """Maximum drawdown as percentage."""
        if not self._points:
            return 0.0
        return max(p.drawdown_pct for p in self._points)

    @property
    def total_return(self) -> float:
        """Total return as decimal (0.1 = 10%)."""
        if self.initial_capital == 0:
            return 0.0
        return (self.current_equity - self.initial_capital) / self.initial_capital

    def to_dataframe(self) -> pd.DataFrame:
        """Convert equity curve to DataFrame."""
        if not self._points:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "timestamp": p.timestamp,
                "cash": p.cash,
                "unrealized_pnl": p.unrealized_pnl,
                "total_equity": p.total_equity,
                "drawdown": p.drawdown,
                "drawdown_pct": p.drawdown_pct,
            }
            for p in self._points
        ])

    def compute_statistics(self) -> Dict[str, float]:
        """Compute equity curve statistics."""
        df = self.to_dataframe()
        if df.empty or len(df) < 2:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "total_return": 0.0,
            }

        # Daily returns (assuming intraday data)
        df["returns"] = df["total_equity"].pct_change()
        returns = df["returns"].dropna()

        if len(returns) == 0 or returns.std() == 0:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": self.max_drawdown,
                "max_drawdown_pct": self.max_drawdown_pct,
                "total_return": self.total_return,
            }

        # Annualization factor (assume 252 trading days, 78 5-min bars/day)
        # Adjust based on actual data frequency
        periods_per_year = 252 * 78  # ~19,656 for 5-min bars

        sharpe = (returns.mean() / returns.std()) * (periods_per_year ** 0.5)

        # Sortino (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            sortino = (returns.mean() / downside_std) * (periods_per_year ** 0.5) if downside_std > 0 else 0.0
        else:
            sortino = 0.0

        return {
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_return": self.total_return,
        }


# =============================================================================
# Generic Runner
# =============================================================================


@dataclass
class GenericRunner:
    """
    Generic backtest/simulation runner.

    Orchestrates the main loop:
    1. Get bars from data feed
    2. Update MTM on open positions
    3. Get signals from strategy
    4. Execute signals through executor
    5. Apply events to ledger
    6. Update equity curve

    Parameters
    ----------
    strategy : StrategyProtocol
        Strategy that generates signals
    data_feed : DataFeedProtocol
        Data feed that provides bars
    executor : EventExecutor
        Executor that simulates fills
    ledger : EventLedger
        Ledger that tracks positions
    initial_capital : float
        Starting capital (default 100,000)
    risk_gate : RiskGate, optional
        Risk gate for portfolio-level constraints. If provided, all signals
        are evaluated before execution. Can approve, modify, or reject.

    Examples
    --------
    >>> runner = GenericRunner(
    ...     strategy=my_strategy,
    ...     data_feed=my_feed,
    ...     executor=create_event_executor("CL"),
    ...     ledger=create_ledger("CL"),
    ... )
    >>> result = runner.run()
    >>> print(f"Sharpe: {result.sharpe_ratio:.2f}")
    """

    strategy: StrategyProtocol
    data_feed: DataFeedProtocol
    executor: EventExecutor
    ledger: EventLedger
    initial_capital: float = 100_000.0
    risk_gate: Optional[RiskGate] = None

    def run(
        self,
        verbose: bool = False,
        on_bar: Optional[Callable[[datetime, Dict[str, pd.Series]], None]] = None,
    ) -> RunResult:
        """
        Execute the backtest/simulation.

        Parameters
        ----------
        verbose : bool
            Print progress updates
        on_bar : callable, optional
            Hook called after each bar is processed

        Returns
        -------
        RunResult
            Complete results
        """
        import time
        start_time = time.time()

        # Initialize tracking
        equity = EquityTracker(self.initial_capital)
        risk_gate_events: List[RiskGateEvent] = []

        n_bars = 0
        n_signals = 0
        first_ts = None
        last_ts = None

        # Main loop
        for timestamp in self.data_feed.iter_timestamps():
            if first_ts is None:
                first_ts = timestamp
            last_ts = timestamp
            n_bars += 1

            # Get bars for this timestamp
            bars = self.data_feed.get_bars(timestamp)
            if bars is None or (hasattr(bars, 'empty') and bars.empty):
                continue

            # Update MTM on open positions
            prices = self.executor.mark_to_market(
                [p.instrument_id for p in self.ledger.get_open_positions()],
                bars,
            )
            self.ledger.mark_to_market(prices, timestamp)

            # Get signals from strategy
            open_positions = self.ledger.get_open_positions()
            signals = self.strategy.generate_signals(timestamp, bars, open_positions)

            # Execute signals
            for signal in signals:
                # Risk gate evaluation (if configured)
                if self.risk_gate is not None:
                    result = self.risk_gate.evaluate(
                        signal=signal,
                        open_positions=open_positions,
                        equity=equity.current_equity,
                        ledger=self.ledger,
                    )

                    # Track all risk gate decisions
                    risk_gate_events.append(RiskGateEvent(
                        timestamp=timestamp,
                        original_signal=signal,
                        result=result,
                    ))

                    if result.signal is None:
                        continue  # Rejected
                    signal = result.signal  # Use (possibly modified) signal

                events = self.executor.execute(signal, bars)

                for event in events:
                    position = self.ledger.apply(event)

                    # Track realized P&L on closes
                    if isinstance(event, FillEvent) and position and position.is_closed:
                        equity.record_trade_pnl(position.net_pnl)

                n_signals += 1

            # Update equity curve
            unrealized_pnl = self.ledger.get_total_unrealized_pnl()
            equity.update(timestamp, unrealized_pnl)

            # Optional callback
            if on_bar:
                on_bar(timestamp, bars)

            # Progress reporting
            if verbose and n_bars % 1000 == 0:
                logger.info(f"Processed {n_bars} bars, {n_signals} signals")

        runtime = time.time() - start_time

        if verbose:
            logger.info(f"Run complete: {n_bars} bars, {n_signals} signals in {runtime:.1f}s")

        # Build result
        return self._build_result(
            equity=equity,
            first_ts=first_ts,
            last_ts=last_ts,
            runtime=runtime,
            risk_gate_events=risk_gate_events,
        )

    def _build_result(
        self,
        equity: EquityTracker,
        first_ts: Optional[datetime],
        last_ts: Optional[datetime],
        runtime: float,
        risk_gate_events: List[RiskGateEvent],
    ) -> RunResult:
        """Build the result object."""
        stats = equity.compute_statistics()

        # Trade statistics
        closed_positions = self.ledger.get_closed_positions()
        wins = [p for p in closed_positions if p.net_pnl > 0]
        losses = [p for p in closed_positions if p.net_pnl <= 0]

        win_rate = len(wins) / len(closed_positions) if closed_positions else 0.0
        avg_win = sum(p.net_pnl for p in wins) / len(wins) if wins else 0.0
        avg_loss = sum(p.net_pnl for p in losses) / len(losses) if losses else 0.0

        total_wins = sum(p.net_pnl for p in wins)
        total_losses = abs(sum(p.net_pnl for p in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        return RunResult(
            total_return=equity.total_return,
            total_return_pct=equity.total_return * 100,
            sharpe_ratio=stats["sharpe_ratio"],
            sortino_ratio=stats["sortino_ratio"],
            max_drawdown=stats["max_drawdown"],
            max_drawdown_pct=stats["max_drawdown_pct"],
            total_trades=len(closed_positions),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            equity_curve=equity.to_dataframe(),
            positions=self.ledger.get_all_positions(),
            events=self.ledger.get_events(),
            start_time=first_ts or datetime.now(),
            end_time=last_ts or datetime.now(),
            initial_capital=self.initial_capital,
            final_equity=equity.current_equity,
            runtime_seconds=runtime,
            risk_gate_events=risk_gate_events,
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_runner(
    strategy: StrategyProtocol,
    data_feed: DataFeedProtocol,
    product: str = "CL",
    initial_capital: float = 100_000.0,
) -> GenericRunner:
    """
    Create a GenericRunner with product-specific defaults.

    Parameters
    ----------
    strategy : StrategyProtocol
        Strategy that generates signals
    data_feed : DataFeedProtocol
        Data feed that provides bars
    product : str
        Product code (CL, RB, HO, etc.)
    initial_capital : float
        Starting capital

    Returns
    -------
    GenericRunner
        Configured runner
    """
    executor = create_event_executor(product)
    ledger = create_ledger(product)

    return GenericRunner(
        strategy=strategy,
        data_feed=data_feed,
        executor=executor,
        ledger=ledger,
        initial_capital=initial_capital,
    )
