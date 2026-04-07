"""Portfolio management for backtesting."""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from futures.strategies.base import Position


@dataclass
class Trade:
    """Record of a completed trade."""

    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    side: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    commission: float


@dataclass
class Portfolio:
    """
    Portfolio state manager.

    Tracks positions, cash, and trade history.
    """

    initial_cash: float = 100_000.0
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    transaction_cost_pct: float = 0.001  # 0.1%
    slippage_pct: float = 0.0005  # 0.05%

    def __post_init__(self):
        """Initialize cash to initial value."""
        self.cash = self.initial_cash

    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_cash
        self.positions = {}
        self.trades = []

    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.cash + position_value

    @property
    def position_value(self) -> float:
        """Total value of all positions."""
        return sum(p.market_value for p in self.positions.values())

    def update_prices(self, prices: dict[str, float]):
        """Update current prices for all positions."""
        for ticker, price in prices.items():
            if ticker in self.positions:
                self.positions[ticker].current_price = price

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage to execution price."""
        if is_buy:
            return price * (1 + self.slippage_pct)
        return price * (1 - self.slippage_pct)

    def _calculate_commission(self, value: float) -> float:
        """Calculate transaction commission."""
        return value * self.transaction_cost_pct

    def buy(
        self,
        ticker: str,
        price: float,
        shares: Optional[float] = None,
        value: Optional[float] = None,
        date: Optional[pd.Timestamp] = None,
        max_holding_days: Optional[int] = None,
    ) -> bool:
        """
        Execute a buy order.

        Args:
            ticker: Stock ticker
            price: Current market price
            shares: Number of shares to buy (optional)
            value: Dollar value to buy (optional, used if shares not specified)
            date: Trade date

        Returns:
            True if order executed successfully
        """
        exec_price = self._apply_slippage(price, is_buy=True)

        if shares is None:
            if value is None:
                value = self.cash * 0.95  # Use 95% of cash by default
            shares = value / exec_price

        total_cost = shares * exec_price
        commission = self._calculate_commission(total_cost)
        total_with_commission = total_cost + commission

        if total_with_commission > self.cash:
            # Reduce shares to fit available cash
            available = self.cash / (exec_price * (1 + self.transaction_cost_pct))
            shares = available * 0.99  # Leave small buffer
            total_cost = shares * exec_price
            commission = self._calculate_commission(total_cost)
            total_with_commission = total_cost + commission

        if shares <= 0 or total_with_commission > self.cash:
            return False

        self.cash -= total_with_commission

        if ticker in self.positions:
            # Add to existing position (average up)
            pos = self.positions[ticker]
            total_shares = pos.shares + shares
            avg_price = (
                pos.shares * pos.entry_price + shares * exec_price
            ) / total_shares
            pos.shares = total_shares
            pos.entry_price = avg_price
            pos.current_price = price
            # Keep the stricter (shorter) holding period when averaging up
            if max_holding_days is not None:
                if pos.max_holding_days is None:
                    pos.max_holding_days = max_holding_days
                else:
                    pos.max_holding_days = min(pos.max_holding_days, max_holding_days)
        else:
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                entry_price=exec_price,
                entry_date=date or pd.Timestamp.now(),
                current_price=price,
                max_holding_days=max_holding_days,
            )

        return True

    def sell(
        self,
        ticker: str,
        price: float,
        shares: Optional[float] = None,
        date: Optional[pd.Timestamp] = None,
    ) -> bool:
        """
        Execute a sell order.

        Args:
            ticker: Stock ticker
            price: Current market price
            shares: Number of shares to sell (optional, sells all if not specified)
            date: Trade date

        Returns:
            True if order executed successfully
        """
        if ticker not in self.positions:
            return False

        pos = self.positions[ticker]
        shares_to_sell = shares if shares else pos.shares

        if shares_to_sell > pos.shares:
            shares_to_sell = pos.shares

        exec_price = self._apply_slippage(price, is_buy=False)
        proceeds = shares_to_sell * exec_price
        commission = self._calculate_commission(proceeds)
        net_proceeds = proceeds - commission

        # Record the trade
        entry_value = shares_to_sell * pos.entry_price
        pnl = net_proceeds - entry_value
        pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0

        trade = Trade(
            ticker=ticker,
            entry_date=pos.entry_date,
            exit_date=date or pd.Timestamp.now(),
            entry_price=pos.entry_price,
            exit_price=exec_price,
            shares=shares_to_sell,
            side="long",
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
        )
        self.trades.append(trade)

        self.cash += net_proceeds

        # Update or remove position
        remaining = pos.shares - shares_to_sell
        if remaining > 0.0001:  # Small threshold for floating point
            pos.shares = remaining
        else:
            del self.positions[ticker]

        return True

    def close_all(self, prices: dict[str, float], date: Optional[pd.Timestamp] = None):
        """Close all open positions."""
        for ticker in list(self.positions.keys()):
            if ticker in prices:
                self.sell(ticker, prices[ticker], date=date)

    def get_holdings(self) -> pd.DataFrame:
        """Get current holdings as DataFrame."""
        if not self.positions:
            return pd.DataFrame()

        records = []
        for ticker, pos in self.positions.items():
            records.append({
                "ticker": ticker,
                "shares": pos.shares,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "cost_basis": pos.cost_basis,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
            })

        return pd.DataFrame(records)

    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append({
                "ticker": t.ticker,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "side": t.side,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "commission": t.commission,
            })

        return pd.DataFrame(records)
