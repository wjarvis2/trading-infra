"""
Equity Tracking for Backtesting.

Tracks portfolio equity at configurable granularity including:
- Cash balance
- Unrealized P&L
- High water mark and drawdown

Migrated from strategies/spread_regime/backtest/equity.py
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
from dataclasses import dataclass
import logging

from common_lib.interfaces.protocols import HasUnrealizedPnl

logger = logging.getLogger(__name__)


@dataclass
class EquityPoint:
    """Single point on equity curve."""
    timestamp: datetime
    cash: float
    unrealized_pnl: float
    total_equity: float
    n_positions: int
    high_water_mark: float
    drawdown: float
    drawdown_pct: float


class EquityCurve:
    """
    Track equity at high granularity.

    Provides:
    - Cash balance tracking
    - Unrealized P&L from open positions
    - Drawdown calculations
    - High water mark tracking

    Example
    -------
    >>> curve = EquityCurve(initial_capital=100_000)
    >>> curve.update(timestamp, cash, open_positions)
    >>> df = curve.to_dataframe()
    >>> print(f"Max drawdown: {curve.max_drawdown:.2%}")
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        bars_per_day: int = 78,  # 5-min bars by default
        trading_days_per_year: int = 252,
    ):
        """
        Initialize equity curve.

        Parameters
        ----------
        initial_capital : float
            Starting capital
        bars_per_day : int
            Number of bars per trading day (for Sharpe calculation)
        trading_days_per_year : int
            Trading days per year (for annualization)
        """
        self.initial_capital = initial_capital
        self.bars_per_day = bars_per_day
        self.trading_days_per_year = trading_days_per_year
        self.cash = initial_capital
        self._points: List[EquityPoint] = []
        self._high_water_mark = initial_capital

    def update(
        self,
        timestamp: datetime,
        cash: float,
        open_positions: List[HasUnrealizedPnl],
    ) -> EquityPoint:
        """
        Record equity point.

        Parameters
        ----------
        timestamp : datetime
            Current timestamp
        cash : float
            Current cash balance
        open_positions : list
            Currently open positions (must have unrealized_pnl attribute)

        Returns
        -------
        EquityPoint
            The recorded point
        """
        self.cash = cash

        # Calculate unrealized P&L
        unrealized_pnl = sum(p.unrealized_pnl for p in open_positions)

        # Total equity
        total_equity = cash + unrealized_pnl

        # Update high water mark
        if total_equity > self._high_water_mark:
            self._high_water_mark = total_equity

        # Calculate drawdown
        drawdown = self._high_water_mark - total_equity
        drawdown_pct = drawdown / self._high_water_mark if self._high_water_mark > 0 else 0

        point = EquityPoint(
            timestamp=timestamp,
            cash=cash,
            unrealized_pnl=unrealized_pnl,
            total_equity=total_equity,
            n_positions=len(open_positions),
            high_water_mark=self._high_water_mark,
            drawdown=drawdown,
            drawdown_pct=drawdown_pct,
        )

        self._points.append(point)
        return point

    def record_trade_pnl(
        self,
        timestamp: datetime,
        pnl: float,
    ) -> None:
        """
        Record realized P&L from a closed trade.

        Parameters
        ----------
        timestamp : datetime
            Trade close time
        pnl : float
            Net P&L (after costs)
        """
        self.cash += pnl

    @property
    def current_equity(self) -> float:
        """Get current total equity."""
        if self._points:
            return self._points[-1].total_equity
        return self.cash

    @property
    def max_drawdown(self) -> float:
        """Get maximum drawdown percentage."""
        if not self._points:
            return 0.0
        return max(p.drawdown_pct for p in self._points)

    @property
    def max_drawdown_dollars(self) -> float:
        """Get maximum drawdown in dollars."""
        if not self._points:
            return 0.0
        return max(p.drawdown for p in self._points)

    @property
    def total_return(self) -> float:
        """Get total return percentage."""
        if not self._points:
            return 0.0
        final = self._points[-1].total_equity
        return (final - self.initial_capital) / self.initial_capital

    def to_dataframe(self) -> pd.DataFrame:
        """Convert equity curve to DataFrame."""
        if not self._points:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'timestamp': p.timestamp,
                'cash': p.cash,
                'unrealized_pnl': p.unrealized_pnl,
                'total_equity': p.total_equity,
                'n_positions': p.n_positions,
                'high_water_mark': p.high_water_mark,
                'drawdown': p.drawdown,
                'drawdown_pct': p.drawdown_pct,
            }
            for p in self._points
        ])

    def compute_statistics(self) -> Dict:
        """Compute equity curve statistics."""
        if not self._points:
            return {}

        df = self.to_dataframe()

        # Calculate returns for Sharpe
        df['equity_return'] = df['total_equity'].pct_change()

        # Annualization factor
        bars_per_year = self.bars_per_day * self.trading_days_per_year

        avg_return = df['equity_return'].mean()
        std_return = df['equity_return'].std()

        # Annualized Sharpe (no risk-free rate subtraction)
        sharpe = (avg_return / std_return * np.sqrt(bars_per_year)) if std_return > 0 else 0

        # Sortino (downside deviation)
        downside = df[df['equity_return'] < 0]['equity_return']
        downside_std = downside.std() if len(downside) > 0 else std_return
        sortino = (avg_return / downside_std * np.sqrt(bars_per_year)) if downside_std > 0 else 0

        # Calmar ratio
        calmar = self.total_return / self.max_drawdown if self.max_drawdown > 0 else 0

        return {
            'initial_capital': self.initial_capital,
            'final_equity': self.current_equity,
            'total_return': self.total_return,
            'total_return_pct': self.total_return * 100,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown * 100,
            'max_drawdown_dollars': self.max_drawdown_dollars,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'n_points': len(self._points),
        }

    def reset(self) -> None:
        """Reset equity curve to initial state."""
        self.cash = self.initial_capital
        self._points = []
        self._high_water_mark = self.initial_capital
