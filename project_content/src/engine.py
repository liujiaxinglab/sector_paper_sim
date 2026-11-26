"""Deterministic 5-account daily simulation engine for sector_paper_sim.

Accounts simulated in parallel under identical cost & sizing conventions:
- `A`  (`routed`):          Sector ETFs driven by 4-state `regime_schedule` / `watchlist.json`
- `B1` (`pure_turtle`):     Sector ETFs running mechanical 55/20/ATR20 Turtle throughout
- `B2` (`pure_grid`):       Sector ETFs running mechanical ATR20-locked Grid throughout
- `C`  (`sector_equal_bh`): Equal-weight buy-and-hold across configured sector ETFs (1/N at start Open)
- `D`  (`benchmark_bh`):    Full-allocation buy-and-hold on benchmark ETF (`159330.SZ`) at start Open

Each trading day T executes strictly in 4 serial steps:
1. Pre-open:  Split share/grid adjustments & commission-free cash dividend reinvestment at Open
2. At open:   Account A regime routing / TRAP_OFF liquidation / mode switch exits; C/D start init
3. Intraday & close: Strategy matching (Turtle 55/20 intraday, Grid limit matching intraday,
              and Grid initial construction at Close on entry day)
4. After close: Valuation at raw Close, NAV / drawdown / excess return stats, & cash conservation checks

Supports lossless JSON-serializable state export (`export_state`) and incremental resumption
(`SimulationEngine.from_state`), guaranteeing identical results between full and segmented runs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal
import math
import re
from typing import Any, Mapping

from src.data_loader import ALLOWED_REGIMES, MarketBundle
from src.strategies import (
    GridConfig,
    GridState,
    TurtleConfig,
    apply_split_to_grid,
    calc_commission,
    calc_notional,
    calc_stamp_duty,
    compute_symbol_indicators,
    evaluate_turtle_bar,
    initialize_grid,
    match_grid_bar,
    max_affordable_shares,
)

ENGINE_SCHEMA_VERSION = "1.0.0"

ACCOUNT_IDS: tuple[str, ...] = ("A", "B1", "B2", "C", "D")

ACCOUNT_TYPES: dict[str, str] = {
    "A": "routed",
    "B1": "pure_turtle",
    "B2": "pure_grid",
    "C": "sector_equal_bh",
    "D": "benchmark_bh",
}

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalize_regime(regime: str) -> str:
    """Validate regime string and normalize UNRATED -> TRANSITION."""
    if not isinstance(regime, str):
        raise ValueError(f"Regime must be a string, got {regime!r}")
    cleaned = regime.strip()
    if cleaned not in ALLOWED_REGIMES:
        raise ValueError(
            f"Unsupported regime {regime!r}; allowed={sorted(ALLOWED_REGIMES)}"
        )
    if cleaned == "UNRATED":
        return "TRANSITION"
    return cleaned


def resolve_regime_for_date(
    trade_date: str,
    symbol: str,
    watchlist: Mapping[str, Mapping[str, Any]],
    regime_schedule: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Resolve `(raw_regime, effective_regime)` for `(trade_date, symbol)`.

    Supports:
    - `regime_schedule is None`: falls back to `watchlist[symbol]["current_regime"]`.
    - Date-keyed `regime_schedule` (`{"YYYY-MM-DD": {symbol: regime}}`): step-function
      lookup using the latest date `<= trade_date` that specifies `symbol`, falling
      back to `watchlist[symbol]["current_regime"]` if none precedes `trade_date`.
    - Symbol-keyed `regime_schedule` (`{symbol: regime}`): direct override.
    """
    raw_regime: str | None = None

    if regime_schedule:
        keys = list(regime_schedule.keys())
        is_date_keyed = all(
            isinstance(k, str) and bool(_ISO_DATE_RE.match(k)) for k in keys
        )
        if is_date_keyed:
            candidate_dates = sorted(d for d in keys if d <= trade_date)
            for sched_date in reversed(candidate_dates):
                day_map = regime_schedule[sched_date]
                if not isinstance(day_map, Mapping):
                    raise ValueError(
                        f"regime_schedule[{sched_date!r}] must be a symbol->regime mapping"
                    )
                if symbol in day_map:
                    raw_regime = str(day_map[symbol]).strip()
                    break
        else:
            if symbol in regime_schedule:
                val = regime_schedule[symbol]
                if not isinstance(val, str):
                    raise ValueError(
                        f"regime_schedule[{symbol!r}] must be a regime string, got {val!r}"
                    )
                raw_regime = val.strip()

    if raw_regime is None:
        if symbol not in watchlist:
            raise KeyError(f"Symbol {symbol!r} not found in watchlist")
        raw_regime = str(watchlist[symbol]["current_regime"]).strip()

    effective_regime = normalize_regime(raw_regime)
    if effective_regime == "BENCHMARK":
        raise ValueError(
            f"Sector symbol {symbol!r} cannot be routed with BENCHMARK regime"
        )
    return raw_regime, effective_regime


@dataclass
class Position:
    """Mutable holding state for one symbol inside an account."""

    shares: int = 0
    avg_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "shares": int(self.shares),
            "avg_cost": float(self.avg_cost),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Position":
        return cls(
            shares=int(data.get("shares", 0)),
            avg_cost=float(data.get("avg_cost", 0.0)),
        )


@dataclass
class SymbolStrategyState:
    """Per-symbol routing and strategy execution state within an account."""

    raw_regime: str | None = None
    effective_regime: str | None = None
    position_mode: str | None = None
    grid: GridState | None = None
    last_indicators: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_regime": self.raw_regime,
            "effective_regime": self.effective_regime,
            "position_mode": self.position_mode,
            "grid": self.grid.to_dict() if self.grid is not None else None,
            "last_indicators": (
                dict(self.last_indicators)
                if self.last_indicators is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SymbolStrategyState":
        raw_grid = data.get("grid")
        raw_ind = data.get("last_indicators")
        return cls(
            raw_regime=data.get("raw_regime"),
            effective_regime=data.get("effective_regime"),
            position_mode=data.get("position_mode"),
            grid=GridState.from_dict(raw_grid) if raw_grid is not None else None,
            last_indicators=dict(raw_ind) if raw_ind is not None else None,
        )


@dataclass
class AccountState:
    """Complete serializable ledger state for one virtual account."""

    account_id: str
    account_type: str
    initial_capital: float
    cash: float
    equity: float
    nav: float = 1.0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    initialized: bool = False
    positions: dict[str, Position] = field(default_factory=dict)
    symbol_states: dict[str, SymbolStrategyState] = field(default_factory=dict)
    trades: list[dict[str, Any]] = field(default_factory=list)
    corporate_actions: list[dict[str, Any]] = field(default_factory=list)
    daily_records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.peak_equity <= 0.0:
            self.peak_equity = float(self.initial_capital)

    def get_shares(self, symbol: str) -> int:
        pos = self.positions.get(symbol)
        return int(pos.shares) if pos is not None else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "account_type": self.account_type,
            "initial_capital": float(self.initial_capital),
            "cash": float(self.cash),
            "equity": float(self.equity),
            "nav": float(self.nav),
            "peak_equity": float(self.peak_equity),
            "max_drawdown": float(self.max_drawdown),
            "initialized": bool(self.initialized),
            "positions": {
                sym: self.positions[sym].to_dict()
                for sym in sorted(self.positions.keys())
            },
            "symbol_states": {
                sym: self.symbol_states[sym].to_dict()
                for sym in sorted(self.symbol_states.keys())
            },
            "trades": deepcopy(self.trades),
            "corporate_actions": deepcopy(self.corporate_actions),
            "daily_records": deepcopy(self.daily_records),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AccountState":
        return cls(
            account_id=str(data["account_id"]),
            account_type=str(data["account_type"]),
            initial_capital=float(data["initial_capital"]),
            cash=float(data["cash"]),
            equity=float(data["equity"]),
            nav=float(data["nav"]),
            peak_equity=float(data["peak_equity"]),
            max_drawdown=float(data["max_drawdown"]),
            initialized=bool(data["initialized"]),
            positions={
                str(sym): Position.from_dict(pos_data)
                for sym, pos_data in data.get("positions", {}).items()
            },
            symbol_states={
                str(sym): SymbolStrategyState.from_dict(st_data)
                for sym, st_data in data.get("symbol_states", {}).items()
            },
            trades=deepcopy(list(data.get("trades", []))),
            corporate_actions=deepcopy(list(data.get("corporate_actions", []))),
            daily_records=deepcopy(list(data.get("daily_records", []))),
        )


def _clean_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    val = float(value)
    return val if math.isfinite(val) else None


class SimulationEngine:
    """Deterministic daily 5-account simulation engine with lossless incremental state."""

    def __init__(
        self,
        bundle: MarketBundle,
        initial_state: Mapping[str, Any] | None = None,
    ) -> None:
        self.bundle = bundle
        self.config = bundle.config
        self.benchmark: str = bundle.benchmark
        self.sectors: list[str] = sorted(bundle.sectors)
        self.symbols: list[str] = [self.benchmark, *self.sectors]

        acct_cfg = self.config.get("account_defaults", {})
        self.initial_capital: float = float(acct_cfg.get("initial_capital", 100000.0))
        self.sector_max_weight: float = float(
            acct_cfg.get("sector_max_weight", 1.0 / max(1, len(self.sectors)))
        )
        self.lot_size: int = int(acct_cfg.get("lot_size", 100))
        self.commission_rate: float = float(acct_cfg.get("commission_rate", 0.0002))
        self.min_commission: float = float(acct_cfg.get("min_commission", 0.1))
        self.stamp_duty_rate: float = float(acct_cfg.get("stamp_duty_rate", 0.0))
        self.dividend_reinvest: bool = bool(acct_cfg.get("dividend_reinvest", True))

        strat_cfg = self.config.get("strategies", {})
        self.turtle_config = TurtleConfig.from_mapping(strat_cfg.get("turtle"))
        self.grid_config = GridConfig.from_mapping(strat_cfg.get("grid"))

        # Precompute fast date-indexed raw OHLC, split/dividend maps, and T-1 spot indicators.
        self._raw_bars_by_symbol: dict[str, dict[str, dict[str, float]]] = {}
        self._indicators_by_symbol: dict[str, dict[str, dict[str, float | None]]] = {}
        self._splits_by_symbol: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self._dividends_by_symbol: dict[str, dict[str, float]] = {}

        atr_window = max(self.turtle_config.atr_window, self.grid_config.atr_window)
        if self.turtle_config.atr_window != self.grid_config.atr_window:
            # Compute separate ATRs if ever configured differently; sim_config uses 20 for both.
            self._grid_indicators_by_symbol: dict[str, dict[str, dict[str, float | None]]] | None = {}
        else:
            self._grid_indicators_by_symbol = None

        for sym in self.symbols:
            raw_df = bundle.bars[sym]
            bar_map: dict[str, dict[str, float]] = {}
            for row in raw_df.itertuples(index=False):
                bar_map[str(row.date)] = {
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                }
            self._raw_bars_by_symbol[sym] = bar_map

            adj_df = bundle.adjusted_bars[sym]
            ind_df = compute_symbol_indicators(
                adj_df,
                entry_lookback=self.turtle_config.entry_lookback,
                exit_lookback=self.turtle_config.exit_lookback,
                atr_window=self.turtle_config.atr_window,
            )
            ind_map: dict[str, dict[str, float | None]] = {}
            for row in ind_df.itertuples(index=False):
                ind_map[str(row.date)] = {
                    "prev_55d_high": _clean_optional_float(row.prev_55d_high),
                    "prev_20d_low": _clean_optional_float(row.prev_20d_low),
                    "atr20": _clean_optional_float(row.atr20),
                }
            self._indicators_by_symbol[sym] = ind_map

            if self._grid_indicators_by_symbol is not None:
                g_ind_df = compute_symbol_indicators(
                    adj_df,
                    entry_lookback=self.turtle_config.entry_lookback,
                    exit_lookback=self.turtle_config.exit_lookback,
                    atr_window=self.grid_config.atr_window,
                )
                g_map: dict[str, dict[str, float | None]] = {}
                for row in g_ind_df.itertuples(index=False):
                    g_map[str(row.date)] = {
                        "atr20": _clean_optional_float(row.atr20),
                    }
                self._grid_indicators_by_symbol[sym] = g_map

            split_map: dict[str, list[dict[str, Any]]] = {}
            wl_entry = bundle.watchlist.get(sym, {})
            for ev in wl_entry.get("split_events", []):
                eff_date = str(ev["effective_date"])
                split_map.setdefault(eff_date, []).append(dict(ev))
            self._splits_by_symbol[sym] = split_map

            self._dividends_by_symbol[sym] = {
                str(dt): float(val)
                for dt, val in bundle.dividends.get(sym, {}).items()
            }

        self.start_date: str | None = None
        self.last_processed_date: str | None = None
        self.trading_days_processed: int = 0
        self.last_known_closes: dict[str, float] = {}
        self.excess_history: list[dict[str, Any]] = []
        self.accounts: dict[str, AccountState] = {}

        if initial_state is not None:
            self._load_state(initial_state)
        else:
            self._init_fresh_accounts()

        # Keep atr_window referenced for sanity
        self._atr_window = atr_window

    @classmethod
    def from_state(
        cls,
        bundle: MarketBundle,
        state_dict: Mapping[str, Any],
    ) -> "SimulationEngine":
        """Restore a SimulationEngine from a previously exported JSON-compatible state_dict."""
        return cls(bundle=bundle, initial_state=state_dict)

    def _account_symbols(self, account_id: str) -> list[str]:
        if account_id == "D":
            return [self.benchmark]
        return list(self.sectors)

    def _init_fresh_accounts(self) -> None:
        self.accounts = {}
        for account_id in ACCOUNT_IDS:
            syms = self._account_symbols(account_id)
            positions = {sym: Position() for sym in syms}
            symbol_states = {sym: SymbolStrategyState() for sym in syms}
            self.accounts[account_id] = AccountState(
                account_id=account_id,
                account_type=ACCOUNT_TYPES[account_id],
                initial_capital=self.initial_capital,
                cash=self.initial_capital,
                equity=self.initial_capital,
                nav=1.0,
                peak_equity=self.initial_capital,
                max_drawdown=0.0,
                initialized=False,
                positions=positions,
                symbol_states=symbol_states,
            )

    def _load_state(self, state_dict: Mapping[str, Any]) -> None:
        schema_ver = state_dict.get("schema_version")
        if schema_ver != ENGINE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported state schema_version {schema_ver!r}; expected {ENGINE_SCHEMA_VERSION!r}"
            )
        accounts_raw = state_dict.get("accounts")
        if not isinstance(accounts_raw, Mapping) or set(accounts_raw.keys()) != set(ACCOUNT_IDS):
            raise ValueError(
                f"state_dict['accounts'] must contain exactly {list(ACCOUNT_IDS)}"
            )

        self.start_date = (
            str(state_dict["start_date"])
            if state_dict.get("start_date") is not None
            else None
        )
        self.last_processed_date = (
            str(state_dict["last_processed_date"])
            if state_dict.get("last_processed_date") is not None
            else None
        )
        self.trading_days_processed = int(state_dict.get("trading_days_processed", 0))
        self.last_known_closes = {
            str(sym): float(val)
            for sym, val in state_dict.get("last_known_closes", {}).items()
        }
        self.excess_history = deepcopy(list(state_dict.get("excess_history", [])))
        self.accounts = {
            account_id: AccountState.from_dict(accounts_raw[account_id])
            for account_id in ACCOUNT_IDS
        }

    def export_state(self) -> dict[str, Any]:
        """Export full JSON-serializable state snapshot (lossless for incremental resume)."""
        return {
            "schema_version": ENGINE_SCHEMA_VERSION,
            "start_date": self.start_date,
            "last_processed_date": self.last_processed_date,
            "trading_days_processed": int(self.trading_days_processed),
            "config_snapshot": {
                "account_defaults": {
                    "initial_capital": float(self.initial_capital),
                    "sector_max_weight": float(self.sector_max_weight),
                    "lot_size": int(self.lot_size),
                    "commission_rate": float(self.commission_rate),
                    "min_commission": float(self.min_commission),
                    "stamp_duty_rate": float(self.stamp_duty_rate),
                    "dividend_reinvest": bool(self.dividend_reinvest),
                },
                "strategies": {
                    "turtle": {
                        "entry_lookback": int(self.turtle_config.entry_lookback),
                        "exit_lookback": int(self.turtle_config.exit_lookback),
                        "atr_window": int(self.turtle_config.atr_window),
                        "risk_fraction": float(self.turtle_config.risk_fraction),
                    },
                    "grid": {
                        "step_mode": str(self.grid_config.step_mode),
                        "atr_window": int(self.grid_config.atr_window),
                        "step_min": float(self.grid_config.step_min),
                        "step_max": float(self.grid_config.step_max),
                        "grid_levels_each_side": int(self.grid_config.grid_levels_each_side),
                        "total_slices": int(self.grid_config.total_slices),
                        "initial_slices": int(self.grid_config.initial_slices),
                    },
                },
                "universe": {
                    "benchmark": self.benchmark,
                    "sectors": list(self.sectors),
                },
            },
            "last_known_closes": {
                sym: float(self.last_known_closes[sym])
                for sym in sorted(self.last_known_closes.keys())
            },
            "excess_history": deepcopy(self.excess_history),
            "accounts": {
                account_id: self.accounts[account_id].to_dict()
                for account_id in ACCOUNT_IDS
            },
        }

    def _get_bar(self, symbol: str, trade_date: str) -> dict[str, float] | None:
        return self._raw_bars_by_symbol[symbol].get(trade_date)

    def _get_indicators(self, symbol: str, trade_date: str) -> dict[str, float | None]:
        base = self._indicators_by_symbol[symbol].get(trade_date)
        if base is None:
            return {"prev_55d_high": None, "prev_20d_low": None, "atr20": None}
        return dict(base)

    def _get_grid_atr20(self, symbol: str, trade_date: str) -> float | None:
        if self._grid_indicators_by_symbol is not None:
            rec = self._grid_indicators_by_symbol[symbol].get(trade_date)
            return rec["atr20"] if rec is not None else None
        return self._get_indicators(symbol, trade_date)["atr20"]

    def _get_close_price(self, symbol: str, trade_date: str) -> float:
        bar = self._get_bar(symbol, trade_date)
        if bar is not None:
            return float(bar["close"])
        if symbol in self.last_known_closes:
            return float(self.last_known_closes[symbol])
        raise ValueError(
            f"Missing close price for {symbol} on {trade_date} and no prior known close"
        )

    def _execute_buy(
        self,
        *,
        account: AccountState,
        trade_date: str,
        symbol: str,
        action: str,
        shares: int,
        price: float,
        regime: str | None,
        trades_today: list[dict[str, Any]],
        bought_today: dict[str, int],
        commission_free: bool = False,
        trigger_price: float | None = None,
        grid_level: int | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        if shares <= 0:
            raise ValueError(f"Buy shares must be positive, got {shares}")
        notional = calc_notional(shares, price)
        commission = (
            0.0
            if commission_free
            else calc_commission(notional, self.commission_rate, self.min_commission)
        )
        stamp_duty = (
            0.0
            if commission_free
            else calc_stamp_duty(notional, self.stamp_duty_rate)
        )
        total_cost = notional + commission + stamp_duty
        if total_cost > account.cash + 1e-9:
            raise RuntimeError(
                f"Cash overdraft blocked on {account.account_id} {symbol} {trade_date}: "
                f"total_cost={total_cost:.6f} > cash={account.cash:.6f}"
            )

        account.cash -= total_cost
        if -1e-9 <= account.cash < 0.0:
            account.cash = 0.0

        pos = account.positions[symbol]
        old_shares = pos.shares
        new_shares = old_shares + int(shares)
        pos.avg_cost = float(
            (old_shares * pos.avg_cost + total_cost) / float(new_shares)
        )
        pos.shares = new_shares
        bought_today[symbol] = bought_today.get(symbol, 0) + int(shares)

        trade = {
            "trade_id": len(account.trades) + 1,
            "date": trade_date,
            "account": account.account_id,
            "symbol": symbol,
            "action": action,
            "side": "BUY",
            "shares": int(shares),
            "price": float(price),
            "notional": float(notional),
            "commission": float(commission),
            "stamp_duty": float(stamp_duty),
            "net_cash_flow": float(-total_cost),
            "regime": regime,
            "trigger_price": float(trigger_price) if trigger_price is not None else None,
            "grid_level": int(grid_level) if grid_level is not None else None,
            "note": note,
        }
        account.trades.append(trade)
        trades_today.append(trade)
        return trade

    def _execute_sell(
        self,
        *,
        account: AccountState,
        trade_date: str,
        symbol: str,
        action: str,
        shares: int,
        price: float,
        regime: str | None,
        trades_today: list[dict[str, Any]],
        trigger_price: float | None = None,
        grid_level: int | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        pos = account.positions[symbol]
        if shares <= 0 or shares > pos.shares:
            raise ValueError(
                f"Invalid sell shares={shares} for {account.account_id} {symbol} "
                f"(held={pos.shares})"
            )
        notional = calc_notional(shares, price)
        commission = calc_commission(notional, self.commission_rate, self.min_commission)
        stamp_duty = calc_stamp_duty(notional, self.stamp_duty_rate)
        net_proceeds = notional - commission - stamp_duty

        account.cash += net_proceeds
        pos.shares -= int(shares)
        if pos.shares == 0:
            pos.avg_cost = 0.0

        trade = {
            "trade_id": len(account.trades) + 1,
            "date": trade_date,
            "account": account.account_id,
            "symbol": symbol,
            "action": action,
            "side": "SELL",
            "shares": int(shares),
            "price": float(price),
            "notional": float(notional),
            "commission": float(commission),
            "stamp_duty": float(stamp_duty),
            "net_cash_flow": float(net_proceeds),
            "regime": regime,
            "trigger_price": float(trigger_price) if trigger_price is not None else None,
            "grid_level": int(grid_level) if grid_level is not None else None,
            "note": note,
        }
        account.trades.append(trade)
        trades_today.append(trade)
        return trade

    def _will_liquidate_at_open_account_a(
        self,
        account: AccountState,
        symbol: str,
        target_effective_regime: str,
    ) -> bool:
        pos = account.positions[symbol]
        if pos.shares <= 0:
            return False
        sym_state = account.symbol_states[symbol]
        prev_regime = sym_state.effective_regime
        if target_effective_regime == "TRAP_OFF":
            return True
        if target_effective_regime == "TREND_UP" and (
            prev_regime == "RANGE_GRID" or sym_state.position_mode == "RANGE_GRID"
        ):
            return True
        if target_effective_regime == "RANGE_GRID" and prev_regime != "RANGE_GRID":
            return True
        return False

    def _step_account(
        self,
        account: AccountState,
        trade_date: str,
        resolved_a_regimes: Mapping[str, tuple[str, str]],
    ) -> dict[str, Any]:
        acct_symbols = self._account_symbols(account.account_id)
        prev_equity = float(account.equity)
        cash_start_of_day = float(account.cash)
        prev_market_value = prev_equity - cash_start_of_day
        ref_equity = prev_equity

        trades_today: list[dict[str, Any]] = []
        bought_today: dict[str, int] = {}
        div_cash_today: float = 0.0

        # =====================================================================
        # Step 1: Pre-open — Share splits & cash dividend reinvestment at Open
        # =====================================================================
        for sym in acct_symbols:
            pos = account.positions[sym]
            sym_state = account.symbol_states[sym]
            bar = self._get_bar(sym, trade_date)

            # 1a. Split adjustments
            for split_ev in self._splits_by_symbol[sym].get(trade_date, []):
                factor = float(split_ev["factor"])
                if pos.shares > 0:
                    old_shares = pos.shares
                    pos.shares = int(round(pos.shares / factor))
                    pos.avg_cost = float(pos.avg_cost * factor)
                    account.corporate_actions.append(
                        {
                            "date": trade_date,
                            "account": account.account_id,
                            "symbol": sym,
                            "event_type": "SPLIT",
                            "factor": factor,
                            "old_shares": int(old_shares),
                            "new_shares": int(pos.shares),
                        }
                    )
                if sym_state.grid is not None:
                    sym_state.grid = apply_split_to_grid(sym_state.grid, factor)

            # 1b. Cash dividend & commission-free reinvestment at Open
            cash_per_unit = self._dividends_by_symbol[sym].get(trade_date)
            if cash_per_unit is not None and pos.shares > 0:
                div_cash = float(Decimal(int(pos.shares)) * Decimal(str(float(cash_per_unit))))
                account.cash += div_cash
                div_cash_today += div_cash
                account.corporate_actions.append(
                    {
                        "date": trade_date,
                        "account": account.account_id,
                        "symbol": sym,
                        "event_type": "DIVIDEND_CASH",
                        "shares": int(pos.shares),
                        "cash_per_unit": float(cash_per_unit),
                        "div_cash": float(div_cash),
                    }
                )

                skip_reinvest = False
                if account.account_id == "A":
                    _, target_eff = resolved_a_regimes[sym]
                    if self._will_liquidate_at_open_account_a(account, sym, target_eff):
                        skip_reinvest = True

                if (
                    self.dividend_reinvest
                    and not skip_reinvest
                    and bar is not None
                    and bar["open"] > 0.0
                ):
                    open_price = float(bar["open"])
                    reinvest_shares = max_affordable_shares(
                        budget=div_cash,
                        price=open_price,
                        lot_size=self.lot_size,
                        commission_free=True,
                    )
                    if reinvest_shares >= self.lot_size:
                        reinvest_notional = calc_notional(reinvest_shares, open_price)
                        residual = max(0.0, div_cash - reinvest_notional)
                        regime_tag = (
                            resolved_a_regimes[sym][1]
                            if account.account_id == "A"
                            else (
                                "TREND_UP"
                                if account.account_id == "B1"
                                else (
                                    "RANGE_GRID"
                                    if account.account_id == "B2"
                                    else "BUY_AND_HOLD"
                                )
                            )
                        )
                        self._execute_buy(
                            account=account,
                            trade_date=trade_date,
                            symbol=sym,
                            action="DIVIDEND_REINVEST",
                            shares=reinvest_shares,
                            price=open_price,
                            regime=regime_tag,
                            trades_today=trades_today,
                            bought_today=bought_today,
                            commission_free=True,
                            trigger_price=open_price,
                            note=f"div_cash={div_cash:.6f},residual={residual:.6f}",
                        )

        # =====================================================================
        # Step 2: At open — Account A routing switch & TRAP_OFF liquidation;
        #         Accounts C / D start-day Buy-and-Hold initialization
        # =====================================================================
        if account.account_id == "A":
            for sym in acct_symbols:
                pos = account.positions[sym]
                sym_state = account.symbol_states[sym]
                bar = self._get_bar(sym, trade_date)
                raw_regime, eff_regime = resolved_a_regimes[sym]
                prev_regime = sym_state.effective_regime
                sym_state.raw_regime = raw_regime

                if eff_regime == "TRAP_OFF":
                    sym_state.grid = None
                    if pos.shares > 0 and bar is not None:
                        self._execute_sell(
                            account=account,
                            trade_date=trade_date,
                            symbol=sym,
                            action="TRAP_OFF_LIQUIDATION",
                            shares=pos.shares,
                            price=float(bar["open"]),
                            regime=eff_regime,
                            trades_today=trades_today,
                            trigger_price=float(bar["open"]),
                            note=f"from={prev_regime}",
                        )
                    sym_state.position_mode = None
                    sym_state.effective_regime = "TRAP_OFF"

                elif eff_regime == "TRANSITION":
                    # Cancel pending grid orders, hold existing position without new trades
                    sym_state.grid = None
                    if pos.shares == 0:
                        sym_state.position_mode = None
                    sym_state.effective_regime = "TRANSITION"

                elif eff_regime == "TREND_UP":
                    switching_from_grid = (
                        prev_regime == "RANGE_GRID"
                        or (pos.shares > 0 and sym_state.position_mode == "RANGE_GRID")
                    )
                    if switching_from_grid:
                        sym_state.grid = None
                        if pos.shares > 0 and bar is not None:
                            self._execute_sell(
                                account=account,
                                trade_date=trade_date,
                                symbol=sym,
                                action="REGIME_SWITCH_EXIT",
                                shares=pos.shares,
                                price=float(bar["open"]),
                                regime=eff_regime,
                                trades_today=trades_today,
                                trigger_price=float(bar["open"]),
                                note=f"switch:{prev_regime}->TREND_UP",
                            )
                        sym_state.position_mode = None
                    elif sym_state.grid is not None:
                        sym_state.grid = None
                    sym_state.effective_regime = "TREND_UP"

                elif eff_regime == "RANGE_GRID":
                    if prev_regime != "RANGE_GRID":
                        sym_state.grid = None
                        if pos.shares > 0 and bar is not None:
                            self._execute_sell(
                                account=account,
                                trade_date=trade_date,
                                symbol=sym,
                                action="REGIME_SWITCH_EXIT",
                                shares=pos.shares,
                                price=float(bar["open"]),
                                regime=eff_regime,
                                trades_today=trades_today,
                                trigger_price=float(bar["open"]),
                                note=f"switch:{prev_regime}->RANGE_GRID",
                            )
                        sym_state.position_mode = None
                    sym_state.effective_regime = "RANGE_GRID"

        elif account.account_id == "B1":
            for sym in acct_symbols:
                sym_state = account.symbol_states[sym]
                sym_state.raw_regime = "TREND_UP"
                sym_state.effective_regime = "TREND_UP"

        elif account.account_id == "B2":
            for sym in acct_symbols:
                sym_state = account.symbol_states[sym]
                sym_state.raw_regime = "RANGE_GRID"
                sym_state.effective_regime = "RANGE_GRID"

        elif account.account_id == "C":
            if not account.initialized:
                per_sector_budget = account.initial_capital / float(len(acct_symbols))
                for sym in acct_symbols:
                    bar = self._get_bar(sym, trade_date)
                    sym_state = account.symbol_states[sym]
                    sym_state.raw_regime = "BUY_AND_HOLD"
                    sym_state.effective_regime = "BUY_AND_HOLD"
                    if bar is None:
                        continue
                    open_price = float(bar["open"])
                    target_budget = min(per_sector_budget, account.cash)
                    buy_shares = max_affordable_shares(
                        budget=target_budget,
                        price=open_price,
                        lot_size=self.lot_size,
                        commission_rate=self.commission_rate,
                        min_commission=self.min_commission,
                        stamp_duty_rate=self.stamp_duty_rate,
                    )
                    if buy_shares >= self.lot_size:
                        self._execute_buy(
                            account=account,
                            trade_date=trade_date,
                            symbol=sym,
                            action="BH_INIT_BUY",
                            shares=buy_shares,
                            price=open_price,
                            regime="BUY_AND_HOLD",
                            trades_today=trades_today,
                            bought_today=bought_today,
                            trigger_price=open_price,
                            note=f"equal_weight=1/{len(acct_symbols)}",
                        )
                        sym_state.position_mode = "BH"
                account.initialized = True

        elif account.account_id == "D":
            if not account.initialized:
                sym = self.benchmark
                bar = self._get_bar(sym, trade_date)
                sym_state = account.symbol_states[sym]
                sym_state.raw_regime = "BENCHMARK"
                sym_state.effective_regime = "BENCHMARK"
                if bar is not None:
                    open_price = float(bar["open"])
                    buy_shares = max_affordable_shares(
                        budget=account.cash,
                        price=open_price,
                        lot_size=self.lot_size,
                        commission_rate=self.commission_rate,
                        min_commission=self.min_commission,
                        stamp_duty_rate=self.stamp_duty_rate,
                    )
                    if buy_shares >= self.lot_size:
                        self._execute_buy(
                            account=account,
                            trade_date=trade_date,
                            symbol=sym,
                            action="BH_INIT_BUY",
                            shares=buy_shares,
                            price=open_price,
                            regime="BENCHMARK",
                            trades_today=trades_today,
                            bought_today=bought_today,
                            trigger_price=open_price,
                            note="benchmark_full_allocation",
                        )
                        sym_state.position_mode = "BH"
                account.initialized = True

        # =====================================================================
        # Step 3: Intraday & Close strategy matching (Accounts A, B1, B2)
        # =====================================================================
        if account.account_id in ("A", "B1", "B2"):
            # Step 3a: Intraday Turtle & active Grid matching
            for sym in acct_symbols:
                bar = self._get_bar(sym, trade_date)
                if bar is None:
                    continue
                pos = account.positions[sym]
                sym_state = account.symbol_states[sym]
                eff_regime = sym_state.effective_regime
                sellable_shares = max(0, pos.shares - bought_today.get(sym, 0))

                if eff_regime == "TREND_UP":
                    ind = self._get_indicators(sym, trade_date)
                    signal = evaluate_turtle_bar(
                        open_price=bar["open"],
                        high_price=bar["high"],
                        low_price=bar["low"],
                        prev_55d_high=ind["prev_55d_high"],
                        prev_20d_low=ind["prev_20d_low"],
                        atr20=ind["atr20"],
                        current_shares=pos.shares,
                        sellable_shares=sellable_shares,
                        account_equity=ref_equity,
                        available_cash=account.cash,
                        sector_max_weight=self.sector_max_weight,
                        lot_size=self.lot_size,
                        commission_rate=self.commission_rate,
                        min_commission=self.min_commission,
                        stamp_duty_rate=self.stamp_duty_rate,
                        risk_fraction=self.turtle_config.risk_fraction,
                    )
                    if signal.action == "BUY" and signal.exec_price is not None:
                        self._execute_buy(
                            account=account,
                            trade_date=trade_date,
                            symbol=sym,
                            action="TURTLE_ENTRY_BUY",
                            shares=signal.shares,
                            price=signal.exec_price,
                            regime="TREND_UP",
                            trades_today=trades_today,
                            bought_today=bought_today,
                            trigger_price=signal.trigger_price,
                            note=(
                                f"risk_sh={signal.risk_shares},"
                                f"cap_sh={signal.cap_shares},"
                                f"cash_sh={signal.cash_affordable_shares}"
                            ),
                        )
                        sym_state.position_mode = "TREND_UP"
                    elif signal.action == "SELL" and signal.exec_price is not None:
                        self._execute_sell(
                            account=account,
                            trade_date=trade_date,
                            symbol=sym,
                            action="TURTLE_EXIT_SELL",
                            shares=signal.shares,
                            price=signal.exec_price,
                            regime="TREND_UP",
                            trades_today=trades_today,
                            trigger_price=signal.trigger_price,
                        )
                        if pos.shares == 0:
                            sym_state.position_mode = None

                elif eff_regime == "RANGE_GRID" and sym_state.grid is not None:
                    next_grid, fills = match_grid_bar(
                        grid_state=sym_state.grid,
                        open_price=bar["open"],
                        high_price=bar["high"],
                        low_price=bar["low"],
                        sellable_shares=sellable_shares,
                        available_cash=account.cash,
                        lot_size=self.lot_size,
                        commission_rate=self.commission_rate,
                        min_commission=self.min_commission,
                        stamp_duty_rate=self.stamp_duty_rate,
                    )
                    sym_state.grid = next_grid
                    for fill in fills:
                        counter_note = (
                            f"counter={fill.counter_order.side}@{fill.counter_order.level}"
                            if fill.counter_order is not None
                            else ""
                        )
                        if fill.side == "BUY":
                            self._execute_buy(
                                account=account,
                                trade_date=trade_date,
                                symbol=sym,
                                action=fill.action,
                                shares=fill.shares,
                                price=fill.exec_price,
                                regime="RANGE_GRID",
                                trades_today=trades_today,
                                bought_today=bought_today,
                                trigger_price=fill.trigger_price,
                                grid_level=fill.level,
                                note=counter_note,
                            )
                            sym_state.position_mode = "RANGE_GRID"
                        else:
                            self._execute_sell(
                                account=account,
                                trade_date=trade_date,
                                symbol=sym,
                                action=fill.action,
                                shares=fill.shares,
                                price=fill.exec_price,
                                regime="RANGE_GRID",
                                trades_today=trades_today,
                                trigger_price=fill.trigger_price,
                                grid_level=fill.level,
                                note=counter_note,
                            )

            # Step 3b: Close-of-day Grid initialization for newly entered RANGE_GRID symbols
            for sym in acct_symbols:
                bar = self._get_bar(sym, trade_date)
                if bar is None:
                    continue
                sym_state = account.symbol_states[sym]
                if sym_state.effective_regime == "RANGE_GRID" and sym_state.grid is None:
                    grid_atr = self._get_grid_atr20(sym, trade_date)
                    new_grid, init_fill = initialize_grid(
                        trade_date=trade_date,
                        close_price=bar["close"],
                        atr20=grid_atr,
                        account_equity=ref_equity,
                        available_cash=account.cash,
                        sector_max_weight=self.sector_max_weight,
                        grid_config=self.grid_config,
                        lot_size=self.lot_size,
                        commission_rate=self.commission_rate,
                        min_commission=self.min_commission,
                        stamp_duty_rate=self.stamp_duty_rate,
                    )
                    if new_grid is not None:
                        sym_state.grid = new_grid
                        sym_state.position_mode = "RANGE_GRID"
                        if init_fill is not None:
                            self._execute_buy(
                                account=account,
                                trade_date=trade_date,
                                symbol=sym,
                                action=init_fill.action,
                                shares=init_fill.shares,
                                price=init_fill.exec_price,
                                regime="RANGE_GRID",
                                trades_today=trades_today,
                                bought_today=bought_today,
                                trigger_price=init_fill.trigger_price,
                                grid_level=0,
                                note=f"anchor={new_grid.anchor_price:.6f},step={new_grid.step:.6f}",
                            )

        # =====================================================================
        # Step 4: After close — Valuation, risk metrics & accounting conservation
        # =====================================================================
        market_value = 0.0
        positions_count = 0
        for sym in acct_symbols:
            pos = account.positions[sym]
            if pos.shares < 0:
                raise RuntimeError(
                    f"Negative share balance on {account.account_id} {sym} {trade_date}: {pos.shares}"
                )
            close_price = self._get_close_price(sym, trade_date)
            if pos.shares > 0:
                market_value += float(pos.shares) * close_price
                positions_count += 1

            ind = self._get_indicators(sym, trade_date)
            account.symbol_states[sym].last_indicators = {
                "as_of": trade_date,
                "prev_55d_high": ind["prev_55d_high"],
                "prev_20d_low": ind["prev_20d_low"],
                "atr20": ind["atr20"],
            }

        if account.cash < -1e-9:
            raise RuntimeError(
                f"Negative cash balance on {account.account_id} {trade_date}: {account.cash}"
            )
        if account.cash < 0.0:
            account.cash = 0.0

        # Conservation check 1: Cash flow identity
        expected_cash_delta = div_cash_today + sum(
            float(t["net_cash_flow"]) for t in trades_today
        )
        actual_cash_delta = account.cash - cash_start_of_day
        if abs(actual_cash_delta - expected_cash_delta) > 1e-6:
            raise RuntimeError(
                f"Cash conservation violation on {account.account_id} {trade_date}: "
                f"actual_delta={actual_cash_delta:.8f} vs expected={expected_cash_delta:.8f}"
            )

        equity = float(account.cash + market_value)

        # Conservation check 2: Equity identity
        expected_equity_delta = (market_value - prev_market_value) + expected_cash_delta
        actual_equity_delta = equity - prev_equity
        if abs(actual_equity_delta - expected_equity_delta) > 1e-6:
            raise RuntimeError(
                f"Equity conservation violation on {account.account_id} {trade_date}: "
                f"actual_delta={actual_equity_delta:.8f} vs expected={expected_equity_delta:.8f}"
            )

        nav = float(equity / account.initial_capital)
        daily_return = float(equity / prev_equity - 1.0) if prev_equity > 0.0 else 0.0
        cum_return = float(nav - 1.0)
        peak_equity = float(max(account.peak_equity, equity))
        drawdown = float(equity / peak_equity - 1.0) if peak_equity > 0.0 else 0.0
        if abs(drawdown) < 1e-15:
            drawdown = 0.0
        max_drawdown = float(min(account.max_drawdown, drawdown))

        account.equity = equity
        account.nav = nav
        account.peak_equity = peak_equity
        account.max_drawdown = max_drawdown

        record = {
            "date": trade_date,
            "cash": float(account.cash),
            "market_value": float(market_value),
            "equity": float(equity),
            "nav": float(nav),
            "daily_return": float(daily_return),
            "cum_return": float(cum_return),
            "peak_equity": float(peak_equity),
            "drawdown": float(drawdown),
            "max_drawdown": float(max_drawdown),
            "positions_count": int(positions_count),
            "trades_today": int(len(trades_today)),
        }
        account.daily_records.append(record)
        return record

    def step_day(
        self,
        trade_date: str,
        regime_schedule: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Advance all 5 accounts by one trading day strictly following the 4-step order."""
        if not isinstance(trade_date, str) or not _ISO_DATE_RE.match(trade_date):
            raise ValueError(f"trade_date must be YYYY-MM-DD, got {trade_date!r}")
        if (
            self.last_processed_date is not None
            and trade_date <= self.last_processed_date
        ):
            raise ValueError(
                f"trade_date {trade_date} is not strictly after "
                f"last_processed_date {self.last_processed_date}"
            )

        # Resolve Account A target regimes once for this day
        resolved_a_regimes: dict[str, tuple[str, str]] = {
            sym: resolve_regime_for_date(
                trade_date=trade_date,
                symbol=sym,
                watchlist=self.bundle.watchlist,
                regime_schedule=regime_schedule,
            )
            for sym in self.sectors
        }

        # Update last known closes for any symbol with a bar on trade_date
        for sym in self.symbols:
            bar = self._get_bar(sym, trade_date)
            if bar is not None:
                self.last_known_closes[sym] = float(bar["close"])

        day_records: dict[str, dict[str, Any]] = {}
        for account_id in ACCOUNT_IDS:
            day_records[account_id] = self._step_account(
                account=self.accounts[account_id],
                trade_date=trade_date,
                resolved_a_regimes=resolved_a_regimes,
            )

        a_cum = day_records["A"]["cum_return"]
        excess = {
            "date": trade_date,
            "A_vs_B1": float(a_cum - day_records["B1"]["cum_return"]),
            "A_vs_B2": float(a_cum - day_records["B2"]["cum_return"]),
            "A_vs_C": float(a_cum - day_records["C"]["cum_return"]),
            "A_vs_D": float(a_cum - day_records["D"]["cum_return"]),
        }
        self.excess_history.append(excess)

        if self.start_date is None:
            self.start_date = trade_date
        self.last_processed_date = trade_date
        self.trading_days_processed += 1

        return {
            "date": trade_date,
            "accounts": day_records,
            "excess_returns": excess,
        }

    def run(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        regime_schedule: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run or incrementally advance all 5 accounts over aligned common trading dates."""
        resolved_end = end_date if end_date is not None else self.bundle.common_end_date
        dates = [dt for dt in self.bundle.common_trading_dates if dt <= resolved_end]

        if self.last_processed_date is not None:
            dates = [dt for dt in dates if dt > self.last_processed_date]
            if start_date is not None and start_date > self.last_processed_date:
                dates = [dt for dt in dates if dt >= start_date]
        else:
            if start_date is not None:
                dates = [dt for dt in dates if dt >= start_date]

        processed_dates: list[str] = []
        for trade_date in dates:
            self.step_day(trade_date, regime_schedule=regime_schedule)
            processed_dates.append(trade_date)

        summary = self.get_summary()
        summary["processed_dates"] = processed_dates
        return summary

    def get_summary(self) -> dict[str, Any]:
        """Return current performance and excess-return summary for all 5 accounts."""
        accounts_summary: dict[str, dict[str, Any]] = {}
        for account_id in ACCOUNT_IDS:
            acct = self.accounts[account_id]
            market_value = float(acct.equity - acct.cash)
            accounts_summary[account_id] = {
                "account_id": acct.account_id,
                "account_type": acct.account_type,
                "cash": float(acct.cash),
                "market_value": market_value,
                "equity": float(acct.equity),
                "nav": float(acct.nav),
                "cum_return": float(acct.nav - 1.0),
                "max_drawdown": float(acct.max_drawdown),
                "trade_count": int(len(acct.trades)),
                "positions": {
                    sym: pos.to_dict()
                    for sym, pos in sorted(acct.positions.items())
                    if pos.shares > 0
                },
            }

        latest_excess = (
            dict(self.excess_history[-1])
            if self.excess_history
            else {
                "date": self.last_processed_date,
                "A_vs_B1": 0.0,
                "A_vs_B2": 0.0,
                "A_vs_C": 0.0,
                "A_vs_D": 0.0,
            }
        )
        return {
            "start_date": self.start_date,
            "last_processed_date": self.last_processed_date,
            "trading_days_processed": int(self.trading_days_processed),
            "accounts": accounts_summary,
            "excess_returns": latest_excess,
        }


def run_simulation(
    bundle: MarketBundle,
    start_date: str | None = None,
    end_date: str | None = None,
    regime_schedule: Mapping[str, Any] | None = None,
    initial_state: Mapping[str, Any] | None = None,
) -> SimulationEngine:
    """Convenience entry point to run or resume a simulation and return the engine."""
    engine = SimulationEngine(bundle=bundle, initial_state=initial_state)
    engine.run(
        start_date=start_date,
        end_date=end_date,
        regime_schedule=regime_schedule,
    )
    return engine
