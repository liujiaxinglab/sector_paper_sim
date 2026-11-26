"""Deterministic daily portfolio accounting and execution for five accounts."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
import math
from typing import Any, Mapping, Sequence

from src.data_loader import MarketBundle, build_forward_adjusted_daily
from src.strategies import (
    GridConfig,
    GridState,
    SymbolIndicators,
    TurtleConfig,
    TurtleSignal,
    apply_split_to_grid,
    calc_notional,
    compute_symbol_indicators,
    evaluate_turtle_signal,
    initialize_grid,
    match_grid_bar,
    max_affordable_shares,
    trade_amounts,
)

ENGINE_SCHEMA_VERSION = "2.0.0"
ACCOUNT_IDS = ("A", "B1", "B2", "C", "D")
FIXED_REGIMES = {"B1": "TREND_UP", "B2": "RANGE_GRID", "C": "BUY_AND_HOLD", "D": "BENCHMARK"}
CASH_REGIMES = {"TRANSITION", "TRAP_OFF"}
SECTOR_COUNT = 6


@dataclass
class Position:
    shares: int = 0
    avg_cost: float = 0.0


@dataclass
class SymbolStrategyState:
    active_regime: str | None = None
    grid: GridState | None = None
    bh_initialized: bool = False
    pending_regime: str | None = None
    pending_liquidation: str | None = None
    pending_turtle: TurtleSignal | None = None
    pending_dividend: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value not in (None, False)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SymbolStrategyState:
        fields = dict(data)
        if "grid" in fields:
            fields["grid"] = GridState.from_dict(fields["grid"])
        if "pending_turtle" in fields:
            fields["pending_turtle"] = TurtleSignal(**fields["pending_turtle"])
        return cls(**fields)


@dataclass
class AccountState:
    account_id: str
    cash: float
    equity: float
    peak_equity: float
    positions: dict[str, Position]
    symbol_states: dict[str, SymbolStrategyState]
    nav: float = 1.0
    max_drawdown: float = 0.0
    next_trade_id: int = 1

    def get_shares(self, symbol: str) -> int:
        return self.positions[symbol].shares

    def to_dict(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "equity": self.equity,
            "nav": self.nav,
            "peak_equity": self.peak_equity,
            "max_drawdown": self.max_drawdown,
            "next_trade_id": self.next_trade_id,
            "positions": {symbol: asdict(pos) for symbol, pos in self.positions.items()},
            "symbol_states": {
                symbol: saved for symbol, state in self.symbol_states.items()
                if (saved := state.to_dict())
            },
        }


@dataclass(frozen=True)
class Bar:
    open: float
    high: float
    low: float
    close: float


@dataclass
class _AccountDay:
    account: AccountState
    date: str
    previous_cash: float
    previous_market_value: float
    bought_today: dict[str, int] = field(default_factory=dict)
    trades: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def sellable(self, symbol: str) -> int:
        return self.account.get_shares(symbol) - self.bought_today.get(symbol, 0)


class SimulationEngine:
    """Advance consecutive BS-01 sessions from the configured anchor."""

    def __init__(
        self, bundle: MarketBundle, initial_state: Mapping[str, Any] | None = None,
        *, trading_dates: Sequence[str] | None = None,
    ) -> None:
        self.bundle = bundle
        self.sectors = sorted(bundle.sectors)
        self.symbols = sorted(bundle.symbols)
        self.start_date = date.fromisoformat(bundle.config["start_date"]).isoformat()
        if bundle.regime_history.anchor_start_date > self.start_date:
            raise ValueError(f"Regime history does not cover start_date {self.start_date}")

        calendar = bundle.authoritative_trading_dates if trading_dates is None else trading_dates
        self._trading_dates = [day for day in calendar if day >= self.start_date]
        if not self._trading_dates or self._trading_dates[0] != self.start_date:
            raise ValueError(f"BS-01 calendar does not contain start_date {self.start_date}")
        if self._trading_dates != sorted(set(self._trading_dates)):
            raise ValueError("BS-01 trading dates must be unique and chronological")

        settings = bundle.config["account_defaults"]
        self.initial_capital = float(settings["initial_capital"])
        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0.0:
            raise ValueError("initial_capital must be finite and positive")
        if not math.isclose(
            settings["sector_max_weight"], 1 / SECTOR_COUNT, rel_tol=0, abs_tol=5e-8
        ):
            raise ValueError("sector_max_weight must represent 1/6")
        self.lot_size = settings["lot_size"]
        self.commission_rate = settings["commission_rate"]
        self.min_commission = settings["min_commission"]
        self.stamp_duty_rate = settings["stamp_duty_rate"]
        self.fee_rules = (self.commission_rate, self.min_commission, self.stamp_duty_rate)
        if type(self.lot_size) is not int or self.lot_size != 100 or self.fee_rules != (0.0002, 0.1, 0):
            raise ValueError("Trading requires 100-share lots, 0.0002 commission, a 0.10 minimum and zero duty")
        self.dividend_reinvest = settings["dividend_reinvest"]
        if not isinstance(self.dividend_reinvest, bool):
            raise ValueError("dividend_reinvest must be a boolean")
        self.turtle_config = TurtleConfig.from_mapping(bundle.config["strategies"]["turtle"])
        self.grid_config = GridConfig.from_mapping(bundle.config["strategies"]["grid"])

        self._bars: dict[str, dict[str, Bar]] = {}
        self._bar_dates: dict[str, list[str]] = {}
        self._split_factors: dict[str, dict[str, list[float]]] = {}
        self.last_known_closes: dict[str, float] = {}
        for symbol in self.symbols:
            bars = {}
            for row in bundle.bars[symbol].itertuples(index=False):
                if row.low > min(row.open, row.close) or row.high < max(row.open, row.close):
                    raise ValueError(f"Invalid OHLC range for {symbol} on {row.date}")
                bars[row.date] = Bar(row.open, row.high, row.low, row.close)
            self._bars[symbol] = bars
            self._bar_dates[symbol] = list(bars)
            index = bisect_left(self._bar_dates[symbol], self.start_date)
            if index:
                prior_date = self._bar_dates[symbol][index - 1]
                self.last_known_closes[symbol] = self._bars[symbol][prior_date].close
            self._split_factors[symbol] = {}
            for event in bundle.splits[symbol]:
                self._split_factors[symbol].setdefault(event["effective_date"], []).append(event["factor"])

        self.accounts = {
            account_id: self._fresh_account(
                account_id, [bundle.benchmark] if account_id == "D" else self.sectors
            )
            for account_id in ACCOUNT_IDS
        }
        self.last_processed_date: str | None = None
        self.trading_days_processed = 0
        self.daily_results: list[dict[str, Any]] = []
        if initial_state is not None:
            self._load_state(initial_state)

    def _fresh_account(self, account_id: str, symbols: Sequence[str]) -> AccountState:
        return AccountState(
            account_id=account_id,
            cash=self.initial_capital,
            equity=self.initial_capital,
            peak_equity=self.initial_capital,
            positions={symbol: Position() for symbol in symbols},
            symbol_states={symbol: SymbolStrategyState() for symbol in symbols},
        )

    @classmethod
    def from_state(
        cls, bundle: MarketBundle, state_dict: Mapping[str, Any],
        *, trading_dates: Sequence[str] | None = None,
    ) -> SimulationEngine:
        return cls(bundle, state_dict, trading_dates=trading_dates)

    def _load_state(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != ENGINE_SCHEMA_VERSION:
            raise ValueError("Incompatible state.json schema_version. Rebuild with --from-scratch.")
        if state["start_date"] != self.start_date or set(state["accounts"]) != set(ACCOUNT_IDS):
            raise ValueError("Checkpoint has an incompatible start date or account set")

        last_date = state["last_processed_date"]
        if last_date is not None:
            if last_date not in self._trading_dates:
                raise ValueError(f"Checkpoint date {last_date} is outside the loaded BS-01 sessions")
            self.trading_days_processed = self._trading_dates.index(last_date) + 1
        self.last_processed_date = last_date
        self.last_known_closes = dict(state["last_known_closes"])
        for account_id, account in self.accounts.items():
            saved = state["accounts"][account_id]
            if (
                set(saved["positions"]) != set(account.positions)
                or not set(saved["symbol_states"]).issubset(account.symbol_states)
            ):
                raise ValueError(f"Checkpoint universe does not match account {account_id}")
            for key in ("cash", "equity", "nav", "peak_equity", "max_drawdown", "next_trade_id"):
                setattr(account, key, saved[key])
            if (
                account.cash < 0
                or type(account.next_trade_id) is not int
                or account.next_trade_id < 1
                or account.nav != account.equity / self.initial_capital
            ):
                raise ValueError(f"Invalid checkpoint balances or trade counter for account {account_id}")
            for symbol, position in saved["positions"].items():
                if type(position["shares"]) is not int or position["shares"] < 0:
                    raise ValueError(f"Invalid checkpoint shares for {account_id} {symbol}")
                account.positions[symbol] = Position(**position)
            for symbol, strategy in saved["symbol_states"].items():
                account.symbol_states[symbol] = SymbolStrategyState.from_dict(strategy)

    def export_state(self) -> dict[str, Any]:
        """Return only the closing balances, order books and pending open actions."""
        return {
            "schema_version": ENGINE_SCHEMA_VERSION,
            "start_date": self.start_date,
            "last_processed_date": self.last_processed_date,
            "last_known_closes": dict(sorted(self.last_known_closes.items())),
            "accounts": {account_id: account.to_dict() for account_id, account in self.accounts.items()},
        }

    def _indicators(self, symbol: str, trade_date: str, cutoff: str) -> SymbolIndicators | None:
        end = bisect_left(self._bar_dates[symbol], trade_date)
        if end == 0:
            return None
        window = max(
            self.turtle_config.entry_lookback, self.turtle_config.exit_lookback,
            self.grid_config.atr_window,
        ) + 1
        history = self.bundle.bars[symbol].iloc[max(0, end - window):end]
        adjusted = build_forward_adjusted_daily(
            history,
            split_events=self.bundle.splits[symbol],
            dividend_events=self.bundle.dividends[symbol],
            cutoff_date=cutoff,
        )
        return compute_symbol_indicators(adjusted, self.turtle_config, self.grid_config.atr_window)

    def _market_value(self, account: AccountState, closes: Mapping[str, float]) -> float:
        try:
            return math.fsum(pos.shares * closes[sym] for sym, pos in account.positions.items() if pos.shares)
        except KeyError as exc:
            raise ValueError(f"No valid close for held symbol {exc.args[0]}") from exc

    def _trade(
        self, day: _AccountDay, symbol: str, regime: str, action: str, side: str,
        shares: int, price: float, *, trigger_price: float | None = None,
        grid_level: int | None = None, note: str = "",
    ) -> None:
        account = day.account
        position = account.positions[symbol]
        if shares <= 0 or price <= 0:
            raise ValueError(f"Invalid order for {account.account_id} {symbol} on {day.date}")

        notional, commission, duty = trade_amounts(shares, price, *self.fee_rules)
        if side == "BUY":
            if shares % self.lot_size:
                raise RuntimeError(f"Invalid buy lot size for {symbol}: {shares}")
            cash_flow = -(notional + commission + duty)
            shares_after = position.shares + shares
            avg_cost = (position.shares * position.avg_cost - cash_flow) / shares_after
        else:
            if shares > day.sellable(symbol):
                raise RuntimeError(f"T+1 sellable shares exceeded for {account.account_id} {symbol}")
            cash_flow = notional - commission - duty
            shares_after = position.shares - shares
            avg_cost = position.avg_cost if shares_after else 0.0

        cash_after = account.cash + cash_flow
        if cash_after < -1e-9 or shares_after < 0:
            raise RuntimeError(f"Negative balance on {account.account_id} {symbol} on {day.date}")
        account.cash = max(0.0, cash_after)
        position.shares = shares_after
        position.avg_cost = avg_cost
        if side == "BUY":
            day.bought_today[symbol] = day.bought_today.get(symbol, 0) + shares

        day.trades.append({
            "date": day.date, "account": account.account_id, "trade_id": account.next_trade_id,
            "symbol": symbol, "action": action, "side": side, "shares": shares, "price": price,
            "notional": notional, "commission": commission, "stamp_duty": duty,
            "net_cash_flow": cash_flow, "regime": regime, "trigger_price": trigger_price,
            "grid_level": grid_level, "note": note,
        })
        account.next_trade_id += 1

    def _buy_at_open(
        self, day: _AccountDay, symbol: str, regime: str, action: str, price: float,
        budget: float, *, trigger_price: float | None = None,
    ) -> bool:
        shares = max_affordable_shares(min(budget, day.account.cash), price, self.lot_size, *self.fee_rules)
        if shares == 0:
            return False
        self._trade(day, symbol, regime, action, "BUY", shares, price, trigger_price=trigger_price)
        return True

    def _liquidation(self, symbol: str, target: str) -> str | None:
        account = self.accounts["A"]
        state = account.symbol_states[symbol]
        if account.get_shares(symbol) and (target in CASH_REGIMES or target != state.active_regime):
            return "TRAP_OFF_LIQUIDATION" if target == "TRAP_OFF" else "REGIME_SWITCH_EXIT"
        return state.pending_liquidation

    def _opening_adjustment(self, symbol: str, trade_date: str) -> float:
        factor = math.prod(self._split_factors[symbol].get(trade_date, ()))
        cash = self.bundle.dividends[symbol].get(trade_date)
        previous = self.last_known_closes.get(symbol)
        if cash is not None and previous is not None:
            if cash >= previous:
                raise ValueError(f"Dividend on {trade_date} for {symbol} is incompatible with its prior close")
            factor *= (previous - cash) / previous
        return factor

    def _corporate_actions(
        self, day: _AccountDay, symbol: str, regime: str, bar: Bar | None,
        liquidation: str | None, adjustment: float,
    ) -> None:
        account = day.account
        position = account.positions[symbol]
        state = account.symbol_states[symbol]
        labels = {"date": day.date, "account": account.account_id, "symbol": symbol}
        for factor in self._split_factors[symbol].get(day.date, ()):
            if position.shares:
                previous_shares = position.shares
                position.shares = round(previous_shares / factor)
                position.avg_cost = position.avg_cost * factor if position.shares else 0.0
                day.events.append({
                    **labels, "event_type": "SPLIT", "factor": factor,
                    "old_shares": previous_shares, "new_shares": position.shares,
                })
            if state.grid is not None:
                state.grid = apply_split_to_grid(state.grid, factor)

        if state.pending_turtle is not None and adjustment != 1:
            intent = state.pending_turtle
            state.pending_turtle = replace(intent, trigger_price=intent.trigger_price * adjustment)

        cash_per_unit = self.bundle.dividends[symbol].get(day.date)
        if cash_per_unit is not None and position.shares:
            dividend = calc_notional(position.shares, cash_per_unit)
            account.cash += dividend
            day.events.append({
                **labels, "event_type": "DIVIDEND_CASH", "shares": position.shares,
                "cash_per_unit": cash_per_unit, "div_cash": dividend,
            })
            if self.dividend_reinvest:
                state.pending_dividend += dividend

        if liquidation or not self.dividend_reinvest:
            state.pending_dividend = 0.0
        if bar is not None and state.pending_dividend > 0:
            self._buy_at_open(
                day, symbol, regime, "DIVIDEND_REINVEST", bar.open,
                state.pending_dividend, trigger_price=bar.open,
            )
            state.pending_dividend = 0.0

    def _turtle_open(
        self, day: _AccountDay, symbol: str, state: SymbolStrategyState, bar: Bar | None,
        indicators: SymbolIndicators | None, adjustment: float, liquidation: str | None,
    ) -> None:
        # A suspended position due for a route exit is flat for the new strategy's signal.
        held = 0 if liquidation else day.account.get_shares(symbol)
        sellable = 0 if liquidation else day.sellable(symbol)
        intent = state.pending_turtle
        if intent is not None and ((intent.side == "BUY") != (held == 0)):
            intent = None
        if intent is None and indicators is not None:
            signal = evaluate_turtle_signal(indicators, held, sellable)
            if signal is not None:
                # Today's adjustment maps the reference price into spot space after the signal.
                intent = replace(signal, trigger_price=signal.trigger_price * adjustment)
        if bar is None:
            state.pending_turtle = intent
            return

        state.pending_turtle = None
        if intent is None:
            return
        if intent.side == "BUY":
            self._buy_at_open(
                day, symbol, "TREND_UP", "TURTLE_ENTRY_BUY", bar.open,
                day.account.equity / SECTOR_COUNT, trigger_price=intent.trigger_price,
            )
        elif sellable > 0:
            self._trade(
                day, symbol, "TREND_UP", "TURTLE_EXIT_SELL", "SELL", sellable, bar.open,
                trigger_price=intent.trigger_price,
            )

    def _opening_trades(
        self, day: _AccountDay, symbol: str, regime: str, bar: Bar | None,
        liquidation: str | None, indicators: SymbolIndicators | None, adjustment: float,
    ) -> None:
        account = day.account
        state = account.symbol_states[symbol]
        if account.account_id == "A":
            if bar is None:
                state.pending_regime = regime if regime != state.active_regime or liquidation else None
                state.pending_liquidation = liquidation
            else:
                if liquidation and account.get_shares(symbol):
                    self._trade(
                        day, symbol, regime, liquidation, "SELL", account.get_shares(symbol), bar.open,
                        trigger_price=bar.open, note=f"from={state.active_regime}",
                    )
                if liquidation or regime != state.active_regime:
                    state.grid = None
                state.active_regime = regime
                state.pending_regime = None
                state.pending_liquidation = None

        if regime == "TREND_UP":
            self._turtle_open(day, symbol, state, bar, indicators, adjustment, liquidation)
            return
        state.pending_turtle = None
        if bar is None:
            return

        sector_budget = account.equity / SECTOR_COUNT
        if (
            regime == "RANGE_GRID" and state.grid is None
            and indicators is not None and indicators.atr is not None
        ):
            grid, shares = initialize_grid(
                trade_date=day.date, open_price=bar.open,
                atr20=indicators.atr, previous_close=indicators.previous_close,
                sector_budget=sector_budget, available_cash=account.cash, grid_config=self.grid_config,
                lot_size=self.lot_size, commission_rate=self.commission_rate,
                min_commission=self.min_commission, stamp_duty_rate=self.stamp_duty_rate,
            )
            if shares:
                self._trade(
                    day, symbol, regime, "GRID_INIT_BUY", "BUY", shares, bar.open,
                    trigger_price=bar.open, grid_level=0, note=f"step={grid.step:.12g}",
                )
            state.grid = grid
        elif regime in ("BUY_AND_HOLD", "BENCHMARK") and not state.bh_initialized:
            budget = account.cash if regime == "BENCHMARK" else sector_budget
            state.bh_initialized = self._buy_at_open(
                day, symbol, regime, "BH_INIT_BUY", bar.open, budget, trigger_price=bar.open
            )

    def _match_grid(self, day: _AccountDay, symbol: str, bar: Bar) -> None:
        account = day.account
        state = account.symbol_states[symbol]
        if state.grid is None:
            return
        grid, fills = match_grid_bar(
            grid_state=state.grid, open_price=bar.open, high_price=bar.high,
            low_price=bar.low, close_price=bar.close, sellable_shares=day.sellable(symbol),
            available_cash=account.cash, lot_size=self.lot_size, commission_rate=self.commission_rate,
            min_commission=self.min_commission, stamp_duty_rate=self.stamp_duty_rate,
        )
        for fill in fills:
            counter = fill.counter_order
            self._trade(
                day, symbol, "RANGE_GRID", f"GRID_{fill.side}", fill.side, fill.shares, fill.exec_price,
                trigger_price=fill.trigger_price, grid_level=fill.level,
                note=f"counter={counter.side}@{counter.level}",
            )
        state.grid = grid

    def _close_account(self, day: _AccountDay) -> dict[str, Any]:
        account = day.account
        market_value = self._market_value(account, self.last_known_closes)
        equity = account.cash + market_value
        dividend_cash = math.fsum(event.get("div_cash", 0.0) for event in day.events)
        expected_cash_delta = dividend_cash + math.fsum(trade["net_cash_flow"] for trade in day.trades)
        cash_delta = account.cash - day.previous_cash
        equity_delta = equity - account.equity
        expected_equity_delta = market_value - day.previous_market_value + cash_delta
        if not abs(cash_delta - expected_cash_delta) < 1e-6:
            raise RuntimeError(f"Cash conservation violation on {account.account_id} {day.date}")
        if not abs(equity_delta - expected_equity_delta) < 1e-6:
            raise RuntimeError(f"Equity conservation violation on {account.account_id} {day.date}")

        nav = equity / self.initial_capital
        peak = max(account.peak_equity, equity)
        drawdown = equity / peak - 1.0
        daily_return = equity / account.equity - 1.0 if account.equity > 0 else 0.0
        account.equity = equity
        account.nav = nav
        account.peak_equity = peak
        account.max_drawdown = min(account.max_drawdown, drawdown)
        return {
            "cash": account.cash, "market_value": market_value, "equity": equity, "nav": nav,
            "daily_return": daily_return, "cum_return": nav - 1.0, "peak_equity": peak,
            "drawdown": drawdown, "max_drawdown": account.max_drawdown,
            "dividend_cash": dividend_cash, "trades_today": len(day.trades),
        }

    def _routed_positions(self, targets: Mapping[str, str]) -> dict[str, dict[str, Any]]:
        account = self.accounts["A"]
        positions = {}
        for symbol, target in targets.items():
            state = account.symbol_states[symbol]
            positions[symbol] = {
                "target_regime": target, "active_regime": state.active_regime,
                "pending_liquidation": state.pending_liquidation, "shares": account.get_shares(symbol),
                "grid_orders": len(state.grid.pending_orders) if state.grid is not None else 0,
            }
        return positions

    def step_day(self, trade_date: str) -> dict[str, Any]:
        index = self.trading_days_processed
        if index >= len(self._trading_dates) or trade_date != self._trading_dates[index]:
            raise ValueError(f"{trade_date} is not the next BS-01 trading date")
        if trade_date > self.bundle.metadata["corporate_actions_as_of"]:
            raise ValueError(f"Corporate-actions ledger does not cover {trade_date}")

        targets = {}
        for symbol in self.sectors:
            target = self.bundle.regime_history.regime_for(symbol, trade_date)
            targets[symbol] = "TRANSITION" if target == "UNRATED" else target
        liquidations = {symbol: self._liquidation(symbol, target) for symbol, target in targets.items()}
        bars = {symbol: self._bars[symbol].get(trade_date) for symbol in self.symbols}
        adjustments = {symbol: self._opening_adjustment(symbol, trade_date) for symbol in self.symbols}
        cutoff = (date.fromisoformat(trade_date) - timedelta(days=1)).isoformat()
        indicators = {symbol: self._indicators(symbol, trade_date, cutoff) for symbol in self.sectors}
        days = [
            _AccountDay(account, trade_date, account.cash, self._market_value(account, self.last_known_closes))
            for account in self.accounts.values()
        ]
        # Each step visits every account's symbols in code order before the next step begins.
        entries = [
            (day, symbol, targets[symbol] if day.account.account_id == "A" else FIXED_REGIMES[day.account.account_id],
             bars[symbol], liquidations[symbol] if day.account.account_id == "A" else None)
            for day in days
            for symbol in day.account.positions
        ]
        for day, symbol, regime, bar, liquidation in entries:
            self._corporate_actions(day, symbol, regime, bar, liquidation, adjustments[symbol])
        for day, symbol, regime, bar, liquidation in entries:
            self._opening_trades(day, symbol, regime, bar, liquidation, indicators.get(symbol), adjustments[symbol])
        for day, symbol, regime, bar, _ in entries:
            if regime == "RANGE_GRID" and bar is not None:
                self._match_grid(day, symbol, bar)

        self.last_known_closes.update({symbol: bar.close for symbol, bar in bars.items() if bar is not None})
        accounts = {day.account.account_id: self._close_account(day) for day in days}
        excess = {
            f"A_vs_{account_id}": accounts["A"]["cum_return"] - accounts[account_id]["cum_return"]
            for account_id in ACCOUNT_IDS[1:]
        }
        self.last_processed_date = trade_date
        self.trading_days_processed += 1
        missing = [symbol for symbol, bar in bars.items() if bar is None]
        lagged = [symbol for symbol in missing if trade_date > self._bar_dates[symbol][-1]]
        result = {
            "date": trade_date, "accounts": accounts, "excess_returns": excess,
            "suspended": [symbol for symbol in missing if symbol not in lagged], "lagged": lagged,
            "routed_positions": self._routed_positions(targets),
            "closes": dict(self.last_known_closes),
            "trades": [trade for day in days for trade in day.trades],
            "events": [event for day in days for event in day.events],
        }
        self.daily_results.append(result)
        return result

    def run(self, end_date: str | None = None) -> dict[str, Any]:
        end = end_date or self._trading_dates[-1]
        dates = [day for day in self._trading_dates[self.trading_days_processed:] if day <= end]
        if dates and dates[-1] > self.bundle.metadata["corporate_actions_as_of"]:
            raise ValueError(f"Corporate-actions ledger does not cover {dates[-1]}")
        for trade_date in dates:
            self.step_day(trade_date)
        return {"processed_dates": dates, "last_processed_date": self.last_processed_date}


def run_simulation(
    bundle: MarketBundle, end_date: str | None = None,
    initial_state: Mapping[str, Any] | None = None,
) -> SimulationEngine:
    engine = SimulationEngine(bundle, initial_state)
    engine.run(end_date=end_date)
    return engine
