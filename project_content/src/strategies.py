"""Fee math, prior-close Turtle signals, and deterministic grid matching."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
import math
from typing import Any, Mapping

import pandas as pd


def calc_notional(shares: int, price: float) -> float:
    """Use decimal arithmetic at share and price boundaries."""
    return float(shares * Decimal(str(price)))


def calc_commission(
    notional: float, commission_rate: float = 0.0002, min_commission: float = 0.1
) -> float:
    return max(notional * commission_rate, min_commission) if notional > 0.0 else 0.0


def trade_amounts(
    shares: int,
    price: float,
    commission_rate: float,
    min_commission: float,
    stamp_duty_rate: float,
) -> tuple[float, float, float]:
    notional = calc_notional(shares, price)
    commission = calc_commission(notional, commission_rate, min_commission)
    return notional, commission, notional * stamp_duty_rate


def max_affordable_shares(
    budget: float,
    price: float,
    lot_size: int = 100,
    commission_rate: float = 0.0002,
    min_commission: float = 0.1,
    stamp_duty_rate: float = 0.0,
) -> int:
    """Find the largest whole-lot order whose total cost fits both fee bounds."""
    if not math.isfinite(budget) or not math.isfinite(price) or min(budget, price) <= 0 or lot_size <= 0:
        return 0
    cash = Decimal(str(budget))
    stamp_multiplier = 1 + Decimal(str(stamp_duty_rate))
    max_notional = min(
        (cash - Decimal(str(min_commission))) / stamp_multiplier,
        cash / (stamp_multiplier + Decimal(str(commission_rate))),
    )
    return max(0, int(max_notional // (Decimal(str(price)) * lot_size))) * lot_size


@dataclass(frozen=True)
class TurtleConfig:
    entry_lookback: int = 55
    exit_lookback: int = 20

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TurtleConfig:
        if data["entry_lookback"] != 55 or data["exit_lookback"] != 20:
            raise ValueError("Turtle windows must be 55 and 20 trading bars")
        return cls()


@dataclass(frozen=True)
class SymbolIndicators:
    previous_close: float
    entry_channel: float | None
    exit_channel: float | None
    atr: float | None


def compute_symbol_indicators(
    history: pd.DataFrame, turtle: TurtleConfig, atr_window: int
) -> SymbolIndicators:
    """Use adjusted history ending at the last valid close before the open."""
    closes = history["close"].tolist()
    previous = closes[:-1]
    atr = None
    if len(closes) >= atr_window:
        prior_closes = [closes[0], *previous]
        # With exactly one window of history, the first bar's range is high minus low.
        atr = math.fsum(
            max(high - low, abs(high - prior), abs(low - prior))
            for high, low, prior in zip(
                history["high"].iloc[-atr_window:],
                history["low"].iloc[-atr_window:],
                prior_closes[-atr_window:],
            )
        ) / atr_window
    return SymbolIndicators(
        previous_close=closes[-1],
        entry_channel=(
            max(previous[-turtle.entry_lookback:]) if len(previous) >= turtle.entry_lookback else None
        ),
        exit_channel=(
            min(previous[-turtle.exit_lookback:]) if len(previous) >= turtle.exit_lookback else None
        ),
        atr=atr,
    )


@dataclass(frozen=True)
class TurtleSignal:
    side: str
    trigger_price: float


def evaluate_turtle_signal(
    indicators: SymbolIndicators, current_shares: int, sellable_shares: int
) -> TurtleSignal | None:
    if current_shares == 0:
        channel = indicators.entry_channel
        if channel is not None and indicators.previous_close >= channel:
            return TurtleSignal("BUY", channel)
    elif sellable_shares > 0:
        channel = indicators.exit_channel
        if channel is not None and indicators.previous_close <= channel:
            return TurtleSignal("SELL", channel)
    return None


@dataclass(frozen=True)
class GridConfig:
    atr_window: int = 20
    min_step_pct: float = 0.015
    grid_levels_each_side: int = 4
    total_slices: int = 8
    initial_position_ratio: float = 0.5

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> GridConfig:
        levels = data["grid_levels_each_side"]
        if type(levels) is not int or levels <= 0:
            raise ValueError("grid_levels_each_side must be a positive integer")
        if data["total_slices"] != 2 * levels or data["initial_slices"] != levels:
            raise ValueError("Grid slices must be split equally across both sides")
        if (
            data["step_mode"] != "atr20_locked"
            or data["atr_window"] != 20
            or data["min_step_pct"] != 0.015
            or data["initial_position_ratio"] != 0.5
        ):
            raise ValueError("Grid requires locked ATR20, a 0.015 step floor and 50% initial allocation")
        return cls(grid_levels_each_side=levels, total_slices=2 * levels)


@dataclass(frozen=True)
class GridOrder:
    side: str
    level: int
    price: float
    shares: int = 0


@dataclass(frozen=True)
class GridState:
    anchor_price: float
    step: float
    slice_budget: float
    initialized_date: str
    pending_orders: tuple[GridOrder, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GridState:
        return cls(
            anchor_price=data["anchor_price"],
            step=data["step"],
            slice_budget=data["slice_budget"],
            initialized_date=data["initialized_date"],
            pending_orders=tuple(GridOrder(**order) for order in data["pending_orders"]),
        )


@dataclass(frozen=True)
class GridFill:
    side: str
    level: int
    trigger_price: float
    exec_price: float
    shares: int
    counter_order: GridOrder


def compute_grid_step(atr20: float, previous_close: float, min_step_pct: float = 0.015) -> float:
    if not math.isfinite(atr20) or atr20 < 0 or not math.isfinite(previous_close) or previous_close <= 0:
        raise ValueError("Grid step requires a nonnegative ATR and a positive previous close")
    return max(atr20 / previous_close, min_step_pct)


def grid_level_price(anchor_price: float, step: float, level: int) -> float:
    return float(Decimal(str(anchor_price)) * (1 + level * Decimal(str(step))))


def initialize_grid(
    *,
    trade_date: str,
    open_price: float,
    atr20: float,
    previous_close: float,
    sector_budget: float,
    available_cash: float,
    grid_config: GridConfig,
    lot_size: int = 100,
    commission_rate: float = 0.0002,
    min_commission: float = 0.1,
    stamp_duty_rate: float = 0.0,
) -> tuple[GridState, int]:
    """Size the base position at the open and distribute its lots over sell levels."""
    step = compute_grid_step(atr20, previous_close, grid_config.min_step_pct)
    shares = max_affordable_shares(
        min(sector_budget * grid_config.initial_position_ratio, available_cash),
        open_price, lot_size, commission_rate, min_commission, stamp_duty_rate,
    )
    slice_budget = sector_budget / grid_config.total_slices
    orders: list[GridOrder] = []
    for level in range(-grid_config.grid_levels_each_side, 0):
        price = grid_level_price(open_price, step, level)
        if price > 0.0:
            orders.append(GridOrder("BUY", level, price))

    sell_levels = min(shares // lot_size, grid_config.grid_levels_each_side)
    if sell_levels:
        lots_per_level, extra_lots = divmod(shares // lot_size, sell_levels)
        for level in range(1, sell_levels + 1):
            lots = lots_per_level + (level > sell_levels - extra_lots)
            orders.append(
                GridOrder("SELL", level, grid_level_price(open_price, step, level), lots * lot_size)
            )
    return GridState(open_price, step, slice_budget, trade_date, tuple(orders)), shares


def apply_split_to_grid(grid_state: GridState, factor: float) -> GridState:
    multiplier = Decimal(str(factor))
    anchor_price = float(Decimal(str(grid_state.anchor_price)) * multiplier)
    orders = []
    for order in grid_state.pending_orders:
        updated = replace(
            order,
            price=float(Decimal(str(order.price)) * multiplier),
            shares=round(order.shares / factor),
        )
        if updated.side == "BUY" or updated.shares > 0:
            orders.append(updated)
    return replace(grid_state, anchor_price=anchor_price, pending_orders=tuple(orders))


def match_grid_bar(
    *,
    grid_state: GridState,
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    sellable_shares: int,
    available_cash: float,
    lot_size: int = 100,
    commission_rate: float = 0.0002,
    min_commission: float = 0.1,
    stamp_duty_rate: float = 0.0,
) -> tuple[GridState, list[GridFill]]:
    """Match the opening order book in OHLC path order, deferring all counter-orders."""
    buys = sorted(
        (order for order in grid_state.pending_orders if order.side == "BUY"),
        key=lambda order: order.level, reverse=True,
    )
    sells = sorted(
        (order for order in grid_state.pending_orders if order.side == "SELL"),
        key=lambda order: order.level,
    )
    legs = (("BUY", buys), ("SELL", sells))
    if close_price < open_price:
        legs = legs[::-1]

    remaining = set(grid_state.pending_orders)
    fills: list[GridFill] = []
    for side, orders in legs:
        for order in orders:
            if side == "BUY":
                if low_price > order.price:
                    break
                price = min(open_price, order.price)
                shares = max_affordable_shares(
                    min(grid_state.slice_budget, available_cash),
                    price, lot_size, commission_rate, min_commission, stamp_duty_rate,
                )
                if shares == 0:
                    break
                available_cash -= sum(
                    trade_amounts(shares, price, commission_rate, min_commission, stamp_duty_rate)
                )
                counter_level = order.level + 1
                counter_side = "SELL"
                counter_shares = shares
            else:
                if high_price < order.price or sellable_shares == 0:
                    break
                price = max(open_price, order.price)
                shares = min(order.shares, sellable_shares)
                notional, fee, duty = trade_amounts(
                    shares, price, commission_rate, min_commission, stamp_duty_rate
                )
                available_cash += notional - fee - duty
                sellable_shares -= shares
                counter_level = order.level - 1
                counter_side = "BUY"
                counter_shares = 0

            counter = GridOrder(
                counter_side, counter_level,
                grid_level_price(grid_state.anchor_price, grid_state.step, counter_level),
                counter_shares,
            )
            remaining.remove(order)
            fills.append(GridFill(side, order.level, order.price, price, shares, counter))

    orders = sorted(
        [*remaining, *(fill.counter_order for fill in fills)],
        key=lambda order: (order.level, order.side),
    )
    return replace(grid_state, pending_orders=tuple(orders)), fills
