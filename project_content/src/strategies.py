"""Pure deterministic strategy rules, fee math, and indicator precomputation.

This module contains zero file I/O and zero global mutable state.
- Signals & volatility indicators (`prev_55d_high`, `prev_20d_low`, `atr20`) are
  computed strictly from forward-adjusted bars shifted by 1 trading day (T-1 and
  earlier, zero lookahead) and mapped back into day T's raw spot price space via
  dividing by `adj_factor[T]`.
- Trade sizing, commission (`max(notional * rate, min_commission)`), lot-size
  flooring (100 shares), Turtle (55/20/ATR20), and Grid (ATR20-locked step)
  matching rules are implemented as pure deterministic functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _to_optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    val = float(value)
    return val if math.isfinite(val) else None


def calc_notional(shares: int, price: float) -> float:
    """Compute exact trade notional avoiding binary floating-point rounding drift."""
    if shares <= 0 or price <= 0.0:
        return 0.0
    return float(Decimal(int(shares)) * Decimal(str(float(price))))


def calc_commission(
    notional: float,
    commission_rate: float = 0.0002,
    min_commission: float = 0.1,
) -> float:
    """Compute ETF trading commission: max(notional * commission_rate, min_commission)."""
    if notional <= 0.0:
        return 0.0
    return float(max(notional * commission_rate, min_commission))


def calc_stamp_duty(notional: float, stamp_duty_rate: float = 0.0) -> float:
    """Compute stamp duty (0.0 for ETFs under sim_config.json)."""
    if notional <= 0.0 or stamp_duty_rate <= 0.0:
        return 0.0
    return float(notional * stamp_duty_rate)


def max_affordable_shares(
    budget: float,
    price: float,
    lot_size: int = 100,
    commission_rate: float = 0.0002,
    min_commission: float = 0.1,
    stamp_duty_rate: float = 0.0,
    commission_free: bool = False,
) -> int:
    """Return the largest multiple of `lot_size` affordable within `budget`.

    Guarantees `notional + commission + stamp_duty <= budget` and never produces
    a negative cash residual. Uses Decimal arithmetic to avoid IEEE-754 boundary
    under-allocation (e.g. budget=110.0, price=1.10, lot_size=100 -> 100 shares).
    """
    if (
        not math.isfinite(budget)
        or not math.isfinite(price)
        or budget <= 0.0
        or price <= 0.0
        or lot_size <= 0
    ):
        return 0

    d_budget = Decimal(str(float(budget)))
    d_price = Decimal(str(float(price)))
    d_lot = Decimal(int(lot_size))
    d_lot_cost = d_price * d_lot
    if d_lot_cost <= 0:
        return 0

    if commission_free:
        lots = int(d_budget // d_lot_cost)
        while lots > 0:
            shares = lots * lot_size
            d_notional = Decimal(shares) * d_price
            if d_notional <= d_budget and calc_notional(shares, price) <= budget + 1e-12:
                return int(shares)
            lots -= 1
        return 0

    d_comm_rate = Decimal(str(float(commission_rate)))
    d_stamp_rate = Decimal(str(float(stamp_duty_rate)))
    d_min_comm = Decimal(str(float(min_commission)))

    # Tight upper bound ignoring min_commission floor; actual cost >= proportional cost.
    est_lots = int(d_budget // (d_lot_cost * (Decimal("1") + d_comm_rate + d_stamp_rate)))
    lots = max(0, est_lots)
    while lots > 0:
        shares = lots * lot_size
        d_notional = Decimal(shares) * d_price
        d_comm = max(d_notional * d_comm_rate, d_min_comm)
        d_stamp = d_notional * d_stamp_rate
        if d_notional + d_comm + d_stamp <= d_budget:
            notional = calc_notional(shares, price)
            fee = calc_commission(notional, commission_rate, min_commission)
            stamp = calc_stamp_duty(notional, stamp_duty_rate)
            if notional + fee + stamp <= budget + 1e-12:
                return int(shares)
        lots -= 1
    return 0


def compute_symbol_indicators(
    adjusted_df: pd.DataFrame,
    entry_lookback: int = 55,
    exit_lookback: int = 20,
    atr_window: int = 20,
) -> pd.DataFrame:
    """Precompute T-1 shifted indicators on forward-adjusted bars, mapped to day-T spot space.

    Parameters
    ----------
    adjusted_df:
        Forward-adjusted OHLC DataFrame produced by
        `data_loader.build_forward_adjusted_daily`. Must contain columns
        `date`, `open`, `high`, `low`, `close` (and optionally `adj_factor`).
    entry_lookback:
        Donchian entry lookback window (default 55).
    exit_lookback:
        Donchian exit lookback window (default 20).
    atr_window:
        ATR simple moving average window (default 20).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns `date`, `prev_55d_high`, `prev_20d_low`, `atr20`,
        and `adj_factor`. Every indicator at row T strictly depends only on rows
        `<= T-1` and corporate actions effective on or before day T (zero lookahead).
        Dividing by `adj_factor[T]` cancels any corporate actions strictly after T
        so spot-space triggers match raw OHLC on day T identically across full and
        incremental runs.
    """
    if adjusted_df.empty:
        raise ValueError("Cannot compute indicators from an empty DataFrame")
    if entry_lookback <= 0 or exit_lookback <= 0 or atr_window <= 0:
        raise ValueError(
            f"Indicator windows must be positive: entry={entry_lookback}, "
            f"exit={exit_lookback}, atr={atr_window}"
        )

    dates = adjusted_df["date"].astype(str).reset_index(drop=True)
    adj_high = adjusted_df["high"].astype(float).reset_index(drop=True)
    adj_low = adjusted_df["low"].astype(float).reset_index(drop=True)
    adj_close = adjusted_df["close"].astype(float).reset_index(drop=True)
    if "adj_factor" in adjusted_df.columns:
        adj_factor = adjusted_df["adj_factor"].astype(float).reset_index(drop=True)
    else:
        adj_factor = pd.Series(np.ones(len(adjusted_df), dtype=float))

    if (adj_factor <= 0.0).any() or not np.isfinite(adj_factor.to_numpy(dtype=float)).all():
        raise ValueError("adjusted_df['adj_factor'] must contain finite positive values")

    prev_adj_close = adj_close.shift(1)
    tr_components = pd.concat(
        [
            adj_high - adj_low,
            (adj_high - prev_adj_close).abs(),
            (adj_low - prev_adj_close).abs(),
        ],
        axis=1,
    )
    tr_adj = tr_components.max(axis=1)

    prev_entry_high_adj = (
        adj_high.rolling(window=entry_lookback, min_periods=entry_lookback)
        .max()
        .shift(1)
    )
    prev_exit_low_adj = (
        adj_low.rolling(window=exit_lookback, min_periods=exit_lookback)
        .min()
        .shift(1)
    )
    atr_adj = (
        tr_adj.rolling(window=atr_window, min_periods=atr_window)
        .mean()
        .shift(1)
    )

    # Map into day-T spot space and round to 12 decimals to eliminate 1-ULP
    # `(x * f) / f` drift when future corporate actions shift `adj_factor`.
    prev_55d_high = (prev_entry_high_adj / adj_factor).round(12)
    prev_20d_low = (prev_exit_low_adj / adj_factor).round(12)
    atr20 = (atr_adj / adj_factor).round(12)

    return pd.DataFrame(
        {
            "date": dates,
            "prev_55d_high": prev_55d_high,
            "prev_20d_low": prev_20d_low,
            "atr20": atr20,
            "adj_factor": adj_factor,
        }
    )


@dataclass(frozen=True)
class TurtleConfig:
    """Configuration for the 55/20/ATR20 Turtle strategy."""

    entry_lookback: int = 55
    exit_lookback: int = 20
    atr_window: int = 20
    risk_fraction: float = 0.01

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "TurtleConfig":
        data = raw or {}
        return cls(
            entry_lookback=int(data.get("entry_lookback", 55)),
            exit_lookback=int(data.get("exit_lookback", 20)),
            atr_window=int(data.get("atr_window", 20)),
            risk_fraction=float(data.get("risk_fraction", 0.01)),
        )


@dataclass(frozen=True)
class TurtleSignal:
    """Result of evaluating a single bar under Turtle rules."""

    action: str  # "BUY" | "SELL" | "HOLD"
    exec_price: float | None = None
    trigger_price: float | None = None
    shares: int = 0
    reason: str = "HOLD"
    risk_shares: int = 0
    cap_shares: int = 0
    cash_affordable_shares: int = 0


def compute_turtle_risk_shares(
    account_equity: float,
    atr20: float | None,
    risk_fraction: float = 0.01,
    lot_size: int = 100,
) -> int:
    """Compute risk-parity share count: floor((account_equity * 0.01) / atr20 / lot_size) * lot_size."""
    if (
        atr20 is None
        or not math.isfinite(account_equity)
        or not math.isfinite(atr20)
        or account_equity <= 0.0
        or atr20 <= 0.0
        or risk_fraction <= 0.0
        or lot_size <= 0
    ):
        return 0
    d_risk_budget = Decimal(str(float(account_equity))) * Decimal(str(float(risk_fraction)))
    d_atr = Decimal(str(float(atr20)))
    d_raw_shares = d_risk_budget / d_atr
    lots = int(d_raw_shares // Decimal(int(lot_size)))
    return max(0, lots * lot_size)


def evaluate_turtle_bar(
    *,
    open_price: float,
    high_price: float,
    low_price: float,
    prev_55d_high: float | None,
    prev_20d_low: float | None,
    atr20: float | None,
    current_shares: int,
    sellable_shares: int,
    account_equity: float,
    available_cash: float,
    sector_max_weight: float,
    lot_size: int = 100,
    commission_rate: float = 0.0002,
    min_commission: float = 0.1,
    stamp_duty_rate: float = 0.0,
    risk_fraction: float = 0.01,
) -> TurtleSignal:
    """Evaluate a single raw daily OHLC bar for Turtle entry/exit and sizing.

    Rules:
    - If flat (`current_shares == 0`) and `high_price >= prev_55d_high`:
      `exec_price = max(open_price, prev_55d_high)`.
      `shares = min(risk_shares, cap_shares, cash_affordable_shares)`.
    - If holding (`current_shares > 0`, `sellable_shares > 0`) and `low_price <= prev_20d_low`:
      `exec_price = min(open_price, prev_20d_low)`.
      Exit full `sellable_shares`.
    """
    entry_trigger = _to_optional_finite_float(prev_55d_high)
    exit_trigger = _to_optional_finite_float(prev_20d_low)
    resolved_atr = _to_optional_finite_float(atr20)

    if current_shares == 0:
        if entry_trigger is None or resolved_atr is None or resolved_atr <= 0.0:
            return TurtleSignal(action="HOLD", reason="INSUFFICIENT_HISTORY")
        if high_price >= entry_trigger:
            exec_price = float(max(open_price, entry_trigger))
            cap_budget = account_equity * sector_max_weight
            risk_shares = compute_turtle_risk_shares(
                account_equity=account_equity,
                atr20=resolved_atr,
                risk_fraction=risk_fraction,
                lot_size=lot_size,
            )
            cap_shares = max_affordable_shares(
                budget=cap_budget,
                price=exec_price,
                lot_size=lot_size,
                commission_rate=commission_rate,
                min_commission=min_commission,
                stamp_duty_rate=stamp_duty_rate,
            )
            cash_shares = max_affordable_shares(
                budget=available_cash,
                price=exec_price,
                lot_size=lot_size,
                commission_rate=commission_rate,
                min_commission=min_commission,
                stamp_duty_rate=stamp_duty_rate,
            )
            shares = min(risk_shares, cap_shares, cash_shares)
            if shares >= lot_size:
                return TurtleSignal(
                    action="BUY",
                    exec_price=exec_price,
                    trigger_price=entry_trigger,
                    shares=shares,
                    reason="TURTLE_ENTRY_BUY",
                    risk_shares=risk_shares,
                    cap_shares=cap_shares,
                    cash_affordable_shares=cash_shares,
                )
            return TurtleSignal(
                action="HOLD",
                trigger_price=entry_trigger,
                reason="TURTLE_SIZE_ZERO",
                risk_shares=risk_shares,
                cap_shares=cap_shares,
                cash_affordable_shares=cash_shares,
            )
        return TurtleSignal(action="HOLD", trigger_price=entry_trigger, reason="HOLD")

    # Holding position -> evaluate 20-day low exit
    if sellable_shares <= 0:
        return TurtleSignal(action="HOLD", trigger_price=exit_trigger, reason="T_PLUS_1_LOCKED")
    if exit_trigger is None:
        return TurtleSignal(action="HOLD", reason="INSUFFICIENT_HISTORY")
    if low_price <= exit_trigger:
        exec_price = float(min(open_price, exit_trigger))
        return TurtleSignal(
            action="SELL",
            exec_price=exec_price,
            trigger_price=exit_trigger,
            shares=int(sellable_shares),
            reason="TURTLE_EXIT_SELL",
        )
    return TurtleSignal(action="HOLD", trigger_price=exit_trigger, reason="HOLD")


@dataclass(frozen=True)
class GridConfig:
    """Configuration for the ATR20-locked geometric Grid strategy."""

    step_mode: str = "atr20_locked"
    atr_window: int = 20
    step_min: float = 0.02
    step_max: float = 0.05
    grid_levels_each_side: int = 4
    total_slices: int = 5
    initial_slices: int = 2

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "GridConfig":
        data = raw or {}
        cfg = cls(
            step_mode=str(data.get("step_mode", "atr20_locked")),
            atr_window=int(data.get("atr_window", 20)),
            step_min=float(data.get("step_min", 0.02)),
            step_max=float(data.get("step_max", 0.05)),
            grid_levels_each_side=int(data.get("grid_levels_each_side", 4)),
            total_slices=int(data.get("total_slices", 5)),
            initial_slices=int(data.get("initial_slices", 2)),
        )
        if cfg.step_min <= 0.0 or cfg.step_max < cfg.step_min or cfg.step_max >= 1.0:
            raise ValueError(f"Invalid grid step bounds: [{cfg.step_min}, {cfg.step_max}]")
        if cfg.total_slices <= 0 or cfg.initial_slices < 0 or cfg.initial_slices > cfg.total_slices:
            raise ValueError(
                f"Invalid grid slice config: total={cfg.total_slices}, initial={cfg.initial_slices}"
            )
        return cfg


@dataclass(frozen=True)
class GridOrder:
    """A pending limit order in the Grid order book."""

    side: str  # "BUY" | "SELL"
    level: int
    price: float
    shares: int | None = None
    budget: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "level": int(self.level),
            "price": float(self.price),
            "shares": int(self.shares) if self.shares is not None else None,
            "budget": float(self.budget) if self.budget is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GridOrder":
        return cls(
            side=str(data["side"]),
            level=int(data["level"]),
            price=float(data["price"]),
            shares=int(data["shares"]) if data.get("shares") is not None else None,
            budget=float(data["budget"]) if data.get("budget") is not None else None,
        )


@dataclass(frozen=True)
class GridState:
    """Serializable state of an active Grid instance for one symbol."""

    anchor_price: float
    step: float
    slice_budget: float
    total_slices: int
    initial_slices: int
    initialized_date: str
    pending_orders: tuple[GridOrder, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_price": float(self.anchor_price),
            "step": float(self.step),
            "slice_budget": float(self.slice_budget),
            "total_slices": int(self.total_slices),
            "initial_slices": int(self.initial_slices),
            "initialized_date": str(self.initialized_date),
            "pending_orders": [order.to_dict() for order in self.pending_orders],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GridState":
        orders = tuple(
            GridOrder.from_dict(item) for item in data.get("pending_orders", [])
        )
        return cls(
            anchor_price=float(data["anchor_price"]),
            step=float(data["step"]),
            slice_budget=float(data["slice_budget"]),
            total_slices=int(data["total_slices"]),
            initial_slices=int(data["initial_slices"]),
            initialized_date=str(data["initialized_date"]),
            pending_orders=orders,
        )


@dataclass(frozen=True)
class GridFill:
    """Represents a single Grid order execution (initial build or intraday trigger)."""

    side: str  # "BUY" | "SELL"
    action: str  # "GRID_INIT_BUY" | "GRID_BUY" | "GRID_SELL"
    level: int
    trigger_price: float
    exec_price: float
    shares: int
    counter_order: GridOrder | None = None


def compute_grid_step(
    atr20: float,
    anchor_price: float,
    step_min: float = 0.02,
    step_max: float = 0.05,
) -> float:
    """Compute locked grid step: clip(ATR20 / P0, step_min, step_max)."""
    if not math.isfinite(anchor_price) or anchor_price <= 0.0:
        raise ValueError(f"anchor_price must be finite and > 0, got {anchor_price!r}")
    if not math.isfinite(atr20) or atr20 <= 0.0:
        raise ValueError(f"atr20 must be finite and > 0, got {atr20!r}")
    raw_step = atr20 / anchor_price
    return float(min(max(raw_step, step_min), step_max))


def grid_level_price(anchor_price: float, step: float, level: int) -> float:
    """Compute geometric grid price for integer level.

    - level == 0: P0
    - level < 0 (k = -level): P0 * (1 - s)^k
    - level > 0 (k = level): P0 * (1 + s)^k
    """
    if level == 0:
        return float(anchor_price)
    if level < 0:
        return float(anchor_price * ((1.0 - step) ** (-level)))
    return float(anchor_price * ((1.0 + step) ** level))


def initialize_grid(
    *,
    trade_date: str,
    close_price: float,
    atr20: float | None,
    account_equity: float,
    available_cash: float,
    sector_max_weight: float,
    grid_config: GridConfig,
    lot_size: int = 100,
    commission_rate: float = 0.0002,
    min_commission: float = 0.1,
    stamp_duty_rate: float = 0.0,
) -> tuple[GridState | None, GridFill | None]:
    """Initialize a Grid at `close_price` (P0) using T-1 `atr20`.

    Buys `initial_slices` (default 2 of 5) at `close_price` and places:
    - `total_slices - initial_slices` (3) buy orders below P0 at P0 * (1 - s)^k (k=1..3)
    - `initial_slices` (2) sell orders above P0 at P0 * (1 + s)^k (k=1..2)

    Returns `(None, None)` if `atr20` is unavailable/non-positive (strict no-lookahead).
    """
    resolved_atr = _to_optional_finite_float(atr20)
    if resolved_atr is None or resolved_atr <= 0.0:
        return None, None

    anchor_price = float(close_price)
    step = compute_grid_step(
        atr20=resolved_atr,
        anchor_price=anchor_price,
        step_min=grid_config.step_min,
        step_max=grid_config.step_max,
    )

    sector_budget = account_equity * sector_max_weight
    slice_budget = sector_budget / float(grid_config.total_slices)

    init_target_budget = min(slice_budget * float(grid_config.initial_slices), available_cash)
    init_shares = max_affordable_shares(
        budget=init_target_budget,
        price=anchor_price,
        lot_size=lot_size,
        commission_rate=commission_rate,
        min_commission=min_commission,
        stamp_duty_rate=stamp_duty_rate,
    )

    pending: list[GridOrder] = []

    buy_level_count = min(
        grid_config.total_slices - grid_config.initial_slices,
        grid_config.grid_levels_each_side,
    )
    for k in range(buy_level_count, 0, -1):
        level = -k
        pending.append(
            GridOrder(
                side="BUY",
                level=level,
                price=grid_level_price(anchor_price, step, level),
                shares=None,
                budget=slice_budget,
            )
        )

    sell_level_count = min(
        grid_config.initial_slices,
        grid_config.grid_levels_each_side,
    )
    init_lots = init_shares // lot_size
    if init_lots > 0 and sell_level_count > 0:
        active_sell_levels = min(sell_level_count, init_lots)
        base_lots = init_lots // active_sell_levels
        extra_lots = init_lots % active_sell_levels
        for k in range(1, active_sell_levels + 1):
            lots_for_level = base_lots + (1 if k > active_sell_levels - extra_lots else 0)
            pending.append(
                GridOrder(
                    side="SELL",
                    level=k,
                    price=grid_level_price(anchor_price, step, k),
                    shares=lots_for_level * lot_size,
                    budget=None,
                )
            )

    pending.sort(key=lambda o: o.level)
    state = GridState(
        anchor_price=anchor_price,
        step=step,
        slice_budget=slice_budget,
        total_slices=grid_config.total_slices,
        initial_slices=grid_config.initial_slices,
        initialized_date=trade_date,
        pending_orders=tuple(pending),
    )

    init_fill: GridFill | None = None
    if init_shares >= lot_size:
        init_fill = GridFill(
            side="BUY",
            action="GRID_INIT_BUY",
            level=0,
            trigger_price=anchor_price,
            exec_price=anchor_price,
            shares=init_shares,
            counter_order=None,
        )
    return state, init_fill


def apply_split_to_grid(grid_state: GridState, factor: float) -> GridState:
    """Adjust an active GridState proportionally on a share split effective date."""
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"Split factor must be finite and > 0, got {factor!r}")
    new_orders: list[GridOrder] = []
    for order in grid_state.pending_orders:
        new_shares = (
            int(round(order.shares / factor))
            if order.side == "SELL" and order.shares is not None
            else order.shares
        )
        new_orders.append(
            GridOrder(
                side=order.side,
                level=order.level,
                price=float(order.price * factor),
                shares=new_shares,
                budget=order.budget,
            )
        )
    new_orders.sort(key=lambda o: o.level)
    return GridState(
        anchor_price=float(grid_state.anchor_price * factor),
        step=grid_state.step,
        slice_budget=grid_state.slice_budget,
        total_slices=grid_state.total_slices,
        initial_slices=grid_state.initial_slices,
        initialized_date=grid_state.initialized_date,
        pending_orders=tuple(new_orders),
    )


def match_grid_bar(
    *,
    grid_state: GridState,
    open_price: float,
    high_price: float,
    low_price: float,
    sellable_shares: int,
    available_cash: float,
    lot_size: int = 100,
    commission_rate: float = 0.0002,
    min_commission: float = 0.1,
    stamp_duty_rate: float = 0.0,
) -> tuple[GridState, list[GridFill]]:
    """Match resting Grid limit orders against a single raw OHLC bar.

    Rules:
    - Buy trigger: `low_price <= order.price` -> `exec_price = min(open_price, order.price)`.
      After fill at `level`, places counter-SELL at `level + 1` (active next trading day).
    - Sell trigger: `high_price >= order.price` -> `exec_price = max(open_price, order.price)`.
      After fill at `level`, places counter-BUY at `level - 1` (active next trading day).
    - On rare wide-range bars where both resting sides are touched, resolves
      deterministically without level collisions or fake intraday round-trips:
      gap side takes priority; otherwise existing T-1 sell side executes first.
    - Counter-orders created today take effect starting on the next trading day.
    """
    buys = sorted(
        (o for o in grid_state.pending_orders if o.side == "BUY"),
        key=lambda o: o.level,
        reverse=True,
    )
    sells = sorted(
        (o for o in grid_state.pending_orders if o.side == "SELL"),
        key=lambda o: o.level,
    )

    buy_triggered = bool(buys) and (low_price <= buys[0].price)
    sell_triggered = bool(sells) and (sellable_shares > 0) and (high_price >= sells[0].price)

    if not buy_triggered and not sell_triggered:
        return grid_state, []

    if buy_triggered and not sell_triggered:
        side_to_run = "BUY"
    elif sell_triggered and not buy_triggered:
        side_to_run = "SELL"
    else:
        # Both sides touched on the same daily bar: gap side wins, else conservative SELL first.
        if open_price <= buys[0].price:
            side_to_run = "BUY"
        elif open_price >= sells[0].price:
            side_to_run = "SELL"
        else:
            side_to_run = "SELL"

    fills: list[GridFill] = []

    if side_to_run == "BUY":
        remaining_cash = float(available_cash)
        remaining_buys: list[GridOrder] = []
        new_sells: list[GridOrder] = list(sells)
        halted_by_cash = False

        for order in buys:
            if not halted_by_cash and low_price <= order.price:
                exec_price = float(min(open_price, order.price))
                order_budget = (
                    order.budget
                    if order.budget is not None
                    else grid_state.slice_budget
                )
                target_budget = min(order_budget, remaining_cash)
                buy_shares = max_affordable_shares(
                    budget=target_budget,
                    price=exec_price,
                    lot_size=lot_size,
                    commission_rate=commission_rate,
                    min_commission=min_commission,
                    stamp_duty_rate=stamp_duty_rate,
                )
                if buy_shares >= lot_size:
                    notional = calc_notional(buy_shares, exec_price)
                    fee = calc_commission(notional, commission_rate, min_commission)
                    stamp = calc_stamp_duty(notional, stamp_duty_rate)
                    remaining_cash = max(0.0, remaining_cash - (notional + fee + stamp))

                    counter_level = order.level + 1
                    counter_price = grid_level_price(
                        grid_state.anchor_price,
                        grid_state.step,
                        counter_level,
                    )
                    counter_order = GridOrder(
                        side="SELL",
                        level=counter_level,
                        price=counter_price,
                        shares=buy_shares,
                        budget=None,
                    )
                    new_sells.append(counter_order)
                    fills.append(
                        GridFill(
                            side="BUY",
                            action="GRID_BUY",
                            level=order.level,
                            trigger_price=order.price,
                            exec_price=exec_price,
                            shares=buy_shares,
                            counter_order=counter_order,
                        )
                    )
                else:
                    # Do not skip an unfilled level to avoid colliding counter-orders.
                    halted_by_cash = True
                    remaining_buys.append(order)
            else:
                remaining_buys.append(order)

        next_orders = sorted(remaining_buys + new_sells, key=lambda o: o.level)
    else:
        rem_sellable = int(sellable_shares)
        remaining_sells: list[GridOrder] = []
        new_buys: list[GridOrder] = list(buys)
        halted_by_shares = False

        for idx, order in enumerate(sells):
            if not halted_by_shares and high_price >= order.price and rem_sellable > 0:
                exec_price = float(max(open_price, order.price))
                is_last_sell = idx == len(sells) - 1
                if is_last_sell:
                    sell_qty = rem_sellable
                else:
                    target_qty = (
                        int(order.shares)
                        if order.shares is not None
                        else rem_sellable
                    )
                    sell_qty = min(target_qty, rem_sellable)

                if sell_qty > 0:
                    rem_sellable -= sell_qty
                    counter_level = order.level - 1
                    counter_price = grid_level_price(
                        grid_state.anchor_price,
                        grid_state.step,
                        counter_level,
                    )
                    counter_order = GridOrder(
                        side="BUY",
                        level=counter_level,
                        price=counter_price,
                        shares=None,
                        budget=grid_state.slice_budget,
                    )
                    new_buys.append(counter_order)
                    fills.append(
                        GridFill(
                            side="SELL",
                            action="GRID_SELL",
                            level=order.level,
                            trigger_price=order.price,
                            exec_price=exec_price,
                            shares=sell_qty,
                            counter_order=counter_order,
                        )
                    )
                else:
                    halted_by_shares = True
                    remaining_sells.append(order)
            else:
                remaining_sells.append(order)

        next_orders = sorted(new_buys + remaining_sells, key=lambda o: o.level)

    next_state = GridState(
        anchor_price=grid_state.anchor_price,
        step=grid_state.step,
        slice_budget=grid_state.slice_budget,
        total_slices=grid_state.total_slices,
        initial_slices=grid_state.initial_slices,
        initialized_date=grid_state.initialized_date,
        pending_orders=tuple(next_orders),
    )
    return next_state, fills
