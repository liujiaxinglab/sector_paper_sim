"""Deterministic unit tests for src.strategies and src.engine (Task 1t03).

Covers:
1. Lot-size (100) flooring and 0.02% commission with 0.1 CNY minimum fee floor
2. Ex-dividend automatic reinvestment (commission-free, 100-share floor, residual to cash)
3. Turtle strategy gap-up breakout buy (`max(Open, prev_55d_high)`), gap-down exit
   (`min(Open, prev_20d_low)`), risk/cap/cash sizing, and zero-lookahead indicators
4. Grid strategy ATR20-locked step clamping `[2%, 5%]`, geometric order book
   (2 initial slices, 3 buys below, 2 sells above), intraday & gap fills, counter-orders
5. Account A routing: `TRAP_OFF` unconditional next-open liquidation & order cancel,
   `UNRATED` treated as `TRANSITION` (cancel orders, hold existing, no new entries),
   and `TREND_UP` <-> `RANGE_GRID` open liquidation & clean re-initialization
6. Accounts C & D buy-and-hold start-day allocation, zero rebalancing, & excess returns
7. Incremental resume equivalence: 10-day full run vs 5-day + JSON `state_dict` + 5-day run
8. Split adjustments + real upstream 7-ETF end-to-end deterministic smoke & resume test
"""

from __future__ import annotations

from datetime import date, timedelta
import json
from typing import Any

import pandas as pd
import pytest

from src.data_loader import (
    DEFAULT_SIM_CONFIG_PATH,
    MarketBundle,
    build_forward_adjusted_daily,
    load_market_bundle,
)
from src.engine import SimulationEngine, run_simulation
from src.strategies import (
    GridConfig,
    calc_commission,
    compute_grid_step,
    compute_symbol_indicators,
    compute_turtle_risk_shares,
    grid_level_price,
    max_affordable_shares,
)

BENCHMARK = "159330.SZ"
SECTORS = [
    "159622.SZ",
    "510720.SH",
    "512400.SH",
    "512480.SH",
    "512880.SH",
    "515790.SH",
]
ALL_SYMBOLS = [BENCHMARK, *SECTORS]


def _make_weekdays(count: int, start: date = date(2026, 1, 5)) -> list[str]:
    dates: list[str] = []
    cur = start
    while len(dates) < count:
        if cur.weekday() < 5:
            dates.append(cur.isoformat())
        cur += timedelta(days=1)
    return dates


def _make_constant_bars(
    dates: list[str],
    price: float = 1.0,
    spread: float = 0.015,
    overrides: dict[int, tuple[float, float, float, float]] | None = None,
) -> pd.DataFrame:
    """Create deterministic flat OHLC bars with constant TR = 2 * spread unless overridden."""
    rows: list[dict[str, Any]] = []
    ov = overrides or {}
    for idx, dt in enumerate(dates):
        if idx in ov:
            o, h, l, c = ov[idx]
        else:
            o = price
            h = round(price + spread, 6)
            l = round(price - spread, 6)
            c = price
        rows.append(
            {
                "date": dt,
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "amount": 1_000_000.0,
                "volume": 1_000_000.0,
            }
        )
    return pd.DataFrame(rows)


def _make_synthetic_bundle(
    dates: list[str],
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
    regimes: dict[str, str] | None = None,
    dividends: dict[str, dict[str, float]] | None = None,
    splits: dict[str, list[dict[str, Any]]] | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> MarketBundle:
    config: dict[str, Any] = {
        "project_name": "sector_paper_sim_unit_test",
        "upstream_paths": {
            "tdx_daily_dir": "synthetic",
            "dividend_events_file": "synthetic",
            "watchlist_file": "synthetic",
        },
        "account_defaults": {
            "initial_capital": 100000.0,
            "sector_max_weight": 0.1666667,
            "lot_size": 100,
            "commission_rate": 0.0002,
            "min_commission": 0.1,
            "stamp_duty_rate": 0.0,
            "dividend_reinvest": True,
        },
        "strategies": {
            "turtle": {
                "entry_lookback": 55,
                "exit_lookback": 20,
                "atr_window": 20,
            },
            "grid": {
                "step_mode": "atr20_locked",
                "atr_window": 20,
                "step_min": 0.02,
                "step_max": 0.05,
                "grid_levels_each_side": 4,
                "total_slices": 5,
                "initial_slices": 2,
            },
        },
        "universe": {
            "benchmark": BENCHMARK,
            "sectors": list(SECTORS),
        },
    }
    if config_overrides:
        for k, v in config_overrides.items():
            if isinstance(v, dict) and isinstance(config.get(k), dict):
                config[k].update(v)
            else:
                config[k] = v

    bars: dict[str, pd.DataFrame] = {}
    for sym in ALL_SYMBOLS:
        if bars_by_symbol and sym in bars_by_symbol:
            bars[sym] = bars_by_symbol[sym].copy()
        else:
            bars[sym] = _make_constant_bars(dates, price=1.0, spread=0.015)

    reg_map = regimes or {}
    split_map = splits or {}
    div_map = dividends or {}

    watchlist: dict[str, dict[str, Any]] = {}
    full_dividends: dict[str, dict[str, float]] = {}
    adjusted_bars: dict[str, pd.DataFrame] = {}

    for sym in ALL_SYMBOLS:
        default_reg = "BENCHMARK" if sym == BENCHMARK else "TRANSITION"
        watchlist[sym] = {
            "symbol": sym,
            "name": sym,
            "role": "benchmark" if sym == BENCHMARK else "sector",
            "current_regime": reg_map.get(sym, default_reg),
            "split_events": list(split_map.get(sym, [])),
        }
        full_dividends[sym] = dict(div_map.get(sym, {}))
        adjusted_bars[sym] = build_forward_adjusted_daily(
            bars[sym],
            split_events=watchlist[sym]["split_events"],
            dividend_events=full_dividends[sym],
            cutoff_date=dates[-1],
        )

    return MarketBundle(
        config=config,
        benchmark=BENCHMARK,
        sectors=list(SECTORS),
        symbols=list(ALL_SYMBOLS),
        bars=bars,
        adjusted_bars=adjusted_bars,
        watchlist=watchlist,
        dividends=full_dividends,
        common_end_date=dates[-1],
        common_trading_dates=list(dates),
        metadata={
            "common_start_date": dates[0],
            "common_end_date": dates[-1],
            "latest_end_date": dates[-1],
            "common_trading_day_count": len(dates),
            "lag_warnings": [],
            "missing_benchmark_dates": {},
            "source_files": {},
            "symbol_stats": {},
        },
    )


# =============================================================================
# 1. Lot-size 100 flooring & 0.02% commission (including 0.1 CNY minimum floor)
# =============================================================================
def test_lot_size_flooring_and_min_commission_rules() -> None:
    """Verify 100-share flooring, 0.02% commission, 0.1 CNY minimum, and no cash overdraft."""
    # Minimum commission trigger: 100 * 1.0 * 0.0002 = 0.02 < 0.1 -> 0.1
    assert calc_commission(100.0, 0.0002, 0.1) == pytest.approx(0.1)
    assert calc_commission(499.0, 0.0002, 0.1) == pytest.approx(0.1)
    # Threshold at notional = 500.0: 500 * 0.0002 = 0.10
    assert calc_commission(500.0, 0.0002, 0.1) == pytest.approx(0.1)
    # Proportional commission above threshold: 10000 * 0.0002 = 2.0
    assert calc_commission(10000.0, 0.0002, 0.1) == pytest.approx(2.0)
    assert calc_commission(0.0, 0.0002, 0.1) == 0.0

    # Budget exactly covers 100 shares + 0.10 min commission
    assert max_affordable_shares(100.10, price=1.0, lot_size=100, commission_rate=0.0002, min_commission=0.1) == 100
    # 1 cent short of min commission -> floors down to 0 shares (never negative cash)
    assert max_affordable_shares(100.09, price=1.0, lot_size=100, commission_rate=0.0002, min_commission=0.1) == 0
    # Odd amount floors to 100-share lot boundary
    assert max_affordable_shares(299.99, price=1.0, lot_size=100, commission_rate=0.0002, min_commission=0.1) == 200
    # Large notional where 0.02% > 0.1: 10000 shares * 1.0 + 2.0 = 10002.0 > 10000.0 -> 9900
    assert max_affordable_shares(10000.0, price=1.0, lot_size=100, commission_rate=0.0002, min_commission=0.1) == 9900
    assert max_affordable_shares(10002.0, price=1.0, lot_size=100, commission_rate=0.0002, min_commission=0.1) == 10000

    # End-to-end check inside SimulationEngine triggering min_commission=0.1
    dates = _make_weekdays(5)
    bars = {
        sym: _make_constant_bars(dates, price=1.0, spread=0.01)
        for sym in ALL_SYMBOLS
    }
    bundle = _make_synthetic_bundle(
        dates=dates,
        bars_by_symbol=bars,
        config_overrides={"account_defaults": {"initial_capital": 350.0}},
    )
    engine = run_simulation(bundle, start_date=dates[0], end_date=dates[1])
    d_trades = engine.accounts["D"].trades
    assert len(d_trades) == 1
    t = d_trades[0]
    assert t["action"] == "BH_INIT_BUY"
    assert t["shares"] == 300  # 300 * 1.0 + 0.1 = 300.1 <= 350.0
    assert t["notional"] == pytest.approx(300.0)
    assert t["commission"] == pytest.approx(0.1)  # min_commission triggered!
    assert t["stamp_duty"] == 0.0
    assert engine.accounts["D"].cash == pytest.approx(350.0 - 300.1)


# =============================================================================
# 2. Ex-dividend automatic reinvestment (commission-free, 100-share floor, residual)
# =============================================================================
def test_dividend_reinvestment_commission_free_and_residual_cash() -> None:
    """Verify ex_date dividend reinvests at Open, 0 commission, 100-lot floor, residual to cash."""
    dates = _make_weekdays(5)
    # Day 0: buy at open=1.00; Day 2: ex_date with cash_per_unit=0.025 and open=1.00
    # Account D initial capital = 100,000 -> buys 99,900 shares at 1.00 (comm = 19.98, cash = 80.02)
    # On Day 2: div_cash = 99,900 * 0.025 = 2497.50
    # Reinvest at Open = 1.00 (commission-free, lot=100) -> 2400 shares (notional=2400.0, residual=97.50)
    # Final cash = 80.02 + 97.50 = 177.52, final shares = 99,900 + 2,400 = 102,300
    bars = {
        sym: _make_constant_bars(dates, price=1.0, spread=0.01)
        for sym in ALL_SYMBOLS
    }
    bars[BENCHMARK] = _make_constant_bars(
        dates,
        price=1.0,
        spread=0.01,
        overrides={
            2: (1.00, 1.01, 0.99, 1.00),  # ex-date open=1.00
            3: (1.00, 1.01, 0.99, 1.00),  # second small dividend < 1 lot
        },
    )
    bundle = _make_synthetic_bundle(
        dates=dates,
        bars_by_symbol=bars,
        dividends={
            BENCHMARK: {
                dates[2]: 0.025,   # div_cash = 2497.50 -> 2400 shares + 97.50 residual
                dates[3]: 0.0005,  # div_cash = 102300 * 0.0005 = 51.15 < 100 -> 0 shares, all cash
            }
        },
    )
    engine = run_simulation(bundle, start_date=dates[0], end_date=dates[-1])
    acct_d = engine.accounts["D"]

    reinvest_trades = [t for t in acct_d.trades if t["action"] == "DIVIDEND_REINVEST"]
    assert len(reinvest_trades) == 1
    rt = reinvest_trades[0]
    assert rt["date"] == dates[2]
    assert rt["side"] == "BUY"
    assert rt["shares"] == 2400
    assert rt["price"] == pytest.approx(1.00)
    assert rt["notional"] == pytest.approx(2400.0)
    assert rt["commission"] == 0.0
    assert rt["stamp_duty"] == 0.0

    # Day 3 dividend (51.15) is < 100 shares so no reinvest trade, full 51.15 goes to cash
    expected_cash = (100000.0 - 99900.0 - 19.98) + (2497.50 - 2400.0) + (102300 * 0.0005)
    assert acct_d.get_shares(BENCHMARK) == 102300
    assert acct_d.cash == pytest.approx(expected_cash)
    assert acct_d.equity == pytest.approx(expected_cash + 102300 * 1.00)


def test_dividend_reinvest_ieee754_exact_boundary() -> None:
    """Ensure 110.0 / 1.10 exact dividend boundary buys 100 shares with 0 residual (not 0 shares)."""
    assert max_affordable_shares(110.0, price=1.10, lot_size=100, commission_free=True) == 100


# =============================================================================
# 3. Turtle gap-up breakout buy, gap-down exit, risk/cap sizing & zero lookahead
# =============================================================================
def test_turtle_gap_breakout_buy_and_gap_breakdown_exit() -> None:
    """Verify Turtle max(Open, prev_55d_high) buy, min(Open, prev_20d_low) sell, and risk sizing."""
    # Build 65 trading days:
    # Days 0..54 (55 bars): flat price=1.00, high=1.05, low=0.95, close=1.00 -> TR = 0.10, ATR20 = 0.10
    #   prev_55d_high on day 55 = 1.05, prev_20d_low on day 55 = 0.95, atr20 on day 55 = 0.10
    # Day 55 (sim start): GAP UP open=1.12, high=1.15, low=1.10, close=1.14
    #   High(1.15) >= prev_55d_high(1.05) -> exec_price = max(1.12, 1.05) = 1.12!
    #   risk_shares = floor((100000 * 0.01) / 0.10 / 100) * 100 = 10000
    #   cap_budget = 100000 * 0.1666667 = 16666.67 -> cap_shares at 1.12 = 14800
    #   actual shares = min(10000, 14800, cash_shares) = 10000!
    # Day 56..57: normal hold (low > 0.95)
    # Day 58: GAP DOWN open=0.88, high=0.92, low=0.85, close=0.90
    #   prev_20d_low is still 0.95 (days 38..57 have min low = 0.95)
    #   Low(0.85) <= prev_20d_low(0.95) -> exec_price = min(0.88, 0.95) = 0.88!
    # Day 59: INTRADAY (non-gap) breakout test on another sector:
    #   510720.SH has open=1.02, high=1.08, low=1.01, close=1.07
    #   High(1.08) >= prev_55d_high(1.05) -> exec_price = max(1.02, 1.05) = 1.05!
    dates = _make_weekdays(62)
    sym_gap = "512400.SH"
    sym_intraday = "510720.SH"

    bars: dict[str, pd.DataFrame] = {
        sym: _make_constant_bars(dates, price=1.00, spread=0.05)
        for sym in ALL_SYMBOLS
    }
    # Other sectors stay strictly inside [0.96, 1.04] during sim days 55..61 so they don't trigger
    for sym in SECTORS:
        if sym not in (sym_gap, sym_intraday):
            ov = {i: (1.00, 1.02, 0.98, 1.00) for i in range(55, 62)}
            bars[sym] = _make_constant_bars(dates, price=1.00, spread=0.05, overrides=ov)

    bars[sym_gap] = _make_constant_bars(
        dates,
        price=1.00,
        spread=0.05,
        overrides={
            55: (1.12, 1.15, 1.10, 1.14),  # gap-up breakout > 1.05 -> fill at 1.12
            56: (1.13, 1.14, 1.11, 1.12),  # hold
            57: (1.11, 1.12, 1.09, 1.10),  # hold
            58: (0.88, 0.92, 0.85, 0.90),  # gap-down breakdown < 0.95 -> fill at 0.88
            59: (0.90, 0.92, 0.89, 0.91),
            60: (0.91, 0.92, 0.90, 0.91),
            61: (0.91, 0.92, 0.90, 0.91),
        },
    )
    # sym_intraday has small ATR = 0.02 (spread=0.01) so risk_shares=50000 > cap_shares!
    # Tests cap_shares limiting actual shares when risk_shares > cap_shares, and non-gap trigger fill.
    bars[sym_intraday] = _make_constant_bars(
        dates,
        price=1.00,
        spread=0.01,
        overrides={
            55: (1.00, 1.005, 0.995, 1.00),
            56: (1.00, 1.005, 0.995, 1.00),
            57: (1.00, 1.005, 0.995, 1.00),
            58: (1.00, 1.005, 0.995, 1.00),
            59: (1.00, 1.02, 0.998, 1.015),   # non-gap breakout: Open=1.00 < prev_55d_high=1.01 -> fill=1.01
            60: (1.01, 1.015, 0.985, 0.988),  # non-gap breakdown: Open=1.01 > prev_20d_low=0.99 -> fill=0.99
            61: (0.99, 0.995, 0.988, 0.992),
        },
    )

    bundle = _make_synthetic_bundle(
        dates=dates,
        bars_by_symbol=bars,
        regimes={sym_gap: "TREND_UP", sym_intraday: "TREND_UP"},
    )
    engine = run_simulation(bundle, start_date=dates[55], end_date=dates[61])

    # Check B1 (pure turtle) and A (routed TREND_UP) both got identical fills on these two symbols
    for acct_id in ("A", "B1"):
        acct = engine.accounts[acct_id]
        gap_trades = [t for t in acct.trades if t["symbol"] == sym_gap]
        assert len(gap_trades) == 2

        buy_t, sell_t = gap_trades
        assert buy_t["date"] == dates[55]
        assert buy_t["action"] == "TURTLE_ENTRY_BUY"
        assert buy_t["trigger_price"] == pytest.approx(1.05)
        assert buy_t["price"] == pytest.approx(1.12)  # max(Open=1.12, trigger=1.05)
        assert buy_t["shares"] == 10000  # risk_shares = floor(1000/0.10/100)*100 = 10000 < cap_shares
        assert buy_t["commission"] == pytest.approx(10000 * 1.12 * 0.0002)

        assert sell_t["date"] == dates[58]
        assert sell_t["action"] == "TURTLE_EXIT_SELL"
        assert sell_t["trigger_price"] == pytest.approx(0.95)
        assert sell_t["price"] == pytest.approx(0.88)  # min(Open=0.88, trigger=0.95)
        assert sell_t["shares"] == 10000
        assert sell_t["commission"] == pytest.approx(10000 * 0.88 * 0.0002)

        # Verify intraday (non-gap) trigger fills & cap_budget limiting on sym_intraday
        intra_trades = [t for t in acct.trades if t["symbol"] == sym_intraday]
        assert len(intra_trades) == 2
        ibuy, isell = intra_trades
        assert ibuy["date"] == dates[59]
        assert ibuy["trigger_price"] == pytest.approx(1.01)
        assert ibuy["price"] == pytest.approx(1.01)  # max(Open=1.00, trigger=1.01)
        # Risk shares >> cap shares, so shares are capped by account_equity * sector_max_weight
        prev_eq_day58 = acct.daily_records[3]["equity"]  # day 58 close equity
        expected_cap_shares = max_affordable_shares(
            prev_eq_day58 * 0.1666667,
            price=1.01,
            lot_size=100,
            commission_rate=0.0002,
            min_commission=0.1,
        )
        assert ibuy["shares"] == expected_cap_shares

        assert isell["date"] == dates[60]
        assert isell["trigger_price"] == pytest.approx(0.99)
        assert isell["price"] == pytest.approx(0.99)  # min(Open=1.01, trigger=0.99)
        assert isell["shares"] == ibuy["shares"]


def test_indicators_strict_no_lookahead_and_future_action_invariance() -> None:
    """Modifying future bars or adding a future split/dividend must NOT alter day-T indicators."""
    dates = _make_weekdays(70)
    raw_df = _make_constant_bars(dates, price=2.00, spread=0.04)
    # Truncated series up to day 58
    raw_trunc = raw_df.iloc[:59].copy()
    adj_trunc = build_forward_adjusted_daily(raw_trunc, cutoff_date=dates[58])
    ind_trunc = compute_symbol_indicators(adj_trunc)

    # Full series up to day 69 with massive future price jump, future 1:2 split, and future dividend
    raw_future = raw_df.copy()
    raw_future.loc[60:, ["open", "high", "low", "close"]] = [1.0, 1.5, 0.8, 1.2]
    adj_future = build_forward_adjusted_daily(
        raw_future,
        split_events=[{"effective_date": dates[60], "factor": 0.5}],
        dividend_events={dates[65]: 0.10},
        cutoff_date=dates[-1],
    )
    ind_future = compute_symbol_indicators(adj_future)

    # Every day <= 58 must have IDENTICAL spot-space indicators despite future split/dividend/prices
    for idx in range(55, 59):
        assert ind_trunc.loc[idx, "prev_55d_high"] == pytest.approx(
            ind_future.loc[idx, "prev_55d_high"], abs=1e-12
        )
        assert ind_trunc.loc[idx, "prev_20d_low"] == pytest.approx(
            ind_future.loc[idx, "prev_20d_low"], abs=1e-12
        )
        assert ind_trunc.loc[idx, "atr20"] == pytest.approx(
            ind_future.loc[idx, "atr20"], abs=1e-12
        )

    # Verify compute_turtle_risk_shares exact Decimal boundary
    assert compute_turtle_risk_shares(100000.0, atr20=0.05, risk_fraction=0.01, lot_size=100) == 20000


# =============================================================================
# 4. Grid step guardrails [2%, 5%], intraday & gap fills, and counter-orders
# =============================================================================
def test_grid_step_guardrails_and_intraday_gap_counter_orders() -> None:
    """Verify Grid step clipping [0.02, 0.05], geometric levels, gap/intraday fills, & counter-orders."""
    # Direct unit test on step clipping guardrails
    assert compute_grid_step(atr20=0.005, anchor_price=1.00) == pytest.approx(0.02)  # floored at 2%
    assert compute_grid_step(atr20=0.035, anchor_price=1.00) == pytest.approx(0.035)  # inside [2%, 5%]
    assert compute_grid_step(atr20=0.120, anchor_price=1.00) == pytest.approx(0.05)  # capped at 5%

    # End-to-end Grid test across 3 symbols with low, mid, and high ATRs
    dates = _make_weekdays(28)
    sym_low_atr = "159622.SZ"   # spread=0.005 -> TR=0.01 -> ATR20/1.0=1% -> clipped to 2%
    sym_mid_atr = "510720.SH"   # spread=0.015 -> TR=0.03 -> ATR20/1.0=3% -> locked at 3%
    sym_high_atr = "512400.SH"  # spread=0.040 -> TR=0.08 -> ATR20/1.0=8% -> clipped to 5%

    bars: dict[str, pd.DataFrame] = {
        sym: _make_constant_bars(dates, price=1.00, spread=0.015)
        for sym in ALL_SYMBOLS
    }
    bars[sym_low_atr] = _make_constant_bars(dates, price=1.00, spread=0.005)
    bars[sym_high_atr] = _make_constant_bars(dates, price=1.00, spread=0.040)

    # For sym_mid_atr (P0=1.00, step=0.03):
    #   Buy levels:  -1: 0.97, -2: 0.9409, -3: 0.912673
    #   Sell levels: +1: 1.03, +2: 1.0609
    # Day 20 (sim start): init at Close=1.00 -> buys 2 slices, hangs 3 buys & 2 sells
    # Day 21: Intraday touch buy -1 (open=0.99, low=0.965, high=0.995, close=0.975)
    #   -> fill BUY -1 at min(0.99, 0.97) = 0.97, places counter-SELL at level 0 (1.00)!
    # Day 22: Gap-up past counter-SELL at level 0 (open=1.015, high=1.02, low=1.01, close=1.015)
    #   -> fill SELL 0 at max(1.015, 1.00) = 1.015, places counter-BUY back at level -1 (0.97)!
    # Day 23: Gap-down past BUY -1 (open=0.955, high=0.96, low=0.95, close=0.955)
    #   -> fill BUY -1 at min(0.955, 0.97) = 0.955, places counter-SELL at level 0 (1.00)!
    # Day 24: Intraday touch SELL 0 and SELL +1 (open=0.99, high=1.04, low=0.985, close=1.035)
    #   -> fill SELL 0 at max(0.99, 1.00)=1.00 & SELL +1 at max(0.99, 1.03)=1.03,
    #      places counter-BUYs at -1 and 0!
    bars[sym_mid_atr] = _make_constant_bars(
        dates,
        price=1.00,
        spread=0.015,
        overrides={
            20: (1.00, 1.01, 0.99, 1.00),     # Init at Close=1.00
            21: (0.99, 0.995, 0.965, 0.975),  # Intraday BUY -1 at 0.97
            22: (1.015, 1.02, 1.01, 1.015),   # Gap-up SELL 0 at 1.015
            23: (0.955, 0.96, 0.95, 0.955),   # Gap-down BUY -1 at 0.955
            24: (0.99, 1.04, 0.985, 1.035),   # Multi-level intraday SELL 0 (1.00) & SELL +1 (1.03)
        },
    )

    bundle = _make_synthetic_bundle(
        dates=dates,
        bars_by_symbol=bars,
        regimes={
            sym_low_atr: "RANGE_GRID",
            sym_mid_atr: "RANGE_GRID",
            sym_high_atr: "RANGE_GRID",
        },
    )

    engine = SimulationEngine(bundle)
    # Run Day 20 only first to inspect initial grid construction
    engine.run(start_date=dates[20], end_date=dates[20])
    acct_a = engine.accounts["A"]

    assert acct_a.symbol_states[sym_low_atr].grid is not None
    assert acct_a.symbol_states[sym_low_atr].grid.step == pytest.approx(0.02)
    assert acct_a.symbol_states[sym_mid_atr].grid is not None
    assert acct_a.symbol_states[sym_mid_atr].grid.step == pytest.approx(0.03)
    assert acct_a.symbol_states[sym_high_atr].grid is not None
    assert acct_a.symbol_states[sym_high_atr].grid.step == pytest.approx(0.05)

    mid_grid = acct_a.symbol_states[sym_mid_atr].grid
    assert mid_grid.anchor_price == pytest.approx(1.00)
    buys = [o for o in mid_grid.pending_orders if o.side == "BUY"]
    sells = [o for o in mid_grid.pending_orders if o.side == "SELL"]
    assert [o.level for o in buys] == [-3, -2, -1]
    assert [o.level for o in sells] == [1, 2]
    assert [o.price for o in buys] == pytest.approx(
        [grid_level_price(1.0, 0.03, -3), grid_level_price(1.0, 0.03, -2), grid_level_price(1.0, 0.03, -1)]
    )
    assert [o.price for o in sells] == pytest.approx(
        [grid_level_price(1.0, 0.03, 1), grid_level_price(1.0, 0.03, 2)]
    )
    # 2 initial slices of (100000 * 0.1666667 / 5 = 3333.334) -> 2 * 3333.334 = 6666.668 -> 6600 shares
    assert acct_a.get_shares(sym_mid_atr) == 6600
    assert sum(o.shares for o in sells if o.shares is not None) == 6600

    # Advance Day 21: Intraday BUY -1 at 0.97 -> counter-SELL at level 0
    engine.step_day(dates[21])
    mid_trades = [t for t in acct_a.trades if t["symbol"] == sym_mid_atr]
    assert len(mid_trades) == 2  # INIT + BUY -1
    t_buy1 = mid_trades[1]
    assert t_buy1["action"] == "GRID_BUY"
    assert t_buy1["grid_level"] == -1
    assert t_buy1["price"] == pytest.approx(0.97)
    g21 = acct_a.symbol_states[sym_mid_atr].grid
    assert g21 is not None
    assert sorted(o.level for o in g21.pending_orders if o.side == "BUY") == [-3, -2]
    assert sorted(o.level for o in g21.pending_orders if o.side == "SELL") == [0, 1, 2]

    # Advance Day 22: Gap-up SELL 0 at max(Open=1.015, 1.00) = 1.015 -> counter-BUY at level -1
    engine.step_day(dates[22])
    mid_trades = [t for t in acct_a.trades if t["symbol"] == sym_mid_atr]
    assert len(mid_trades) == 3
    t_sell0 = mid_trades[2]
    assert t_sell0["action"] == "GRID_SELL"
    assert t_sell0["grid_level"] == 0
    assert t_sell0["price"] == pytest.approx(1.015)
    assert t_sell0["shares"] == t_buy1["shares"]
    g22 = acct_a.symbol_states[sym_mid_atr].grid
    assert g22 is not None
    assert sorted(o.level for o in g22.pending_orders if o.side == "BUY") == [-3, -2, -1]
    assert sorted(o.level for o in g22.pending_orders if o.side == "SELL") == [1, 2]

    # Advance Day 23: Gap-down BUY -1 at min(Open=0.955, 0.97) = 0.955 -> counter-SELL at level 0
    engine.step_day(dates[23])
    mid_trades = [t for t in acct_a.trades if t["symbol"] == sym_mid_atr]
    assert len(mid_trades) == 4
    t_buy_gap = mid_trades[3]
    assert t_buy_gap["action"] == "GRID_BUY"
    assert t_buy_gap["grid_level"] == -1
    assert t_buy_gap["price"] == pytest.approx(0.955)

    # Advance Day 24: Multi-level SELL 0 (1.00) & SELL +1 (1.03) -> counter-BUYs at -1 and 0
    engine.step_day(dates[24])
    mid_trades = [t for t in acct_a.trades if t["symbol"] == sym_mid_atr]
    assert len(mid_trades) == 6
    assert mid_trades[4]["action"] == "GRID_SELL"
    assert mid_trades[4]["grid_level"] == 0
    assert mid_trades[4]["price"] == pytest.approx(1.00)
    assert mid_trades[5]["action"] == "GRID_SELL"
    assert mid_trades[5]["grid_level"] == 1
    assert mid_trades[5]["price"] == pytest.approx(1.03)
    g24 = acct_a.symbol_states[sym_mid_atr].grid
    assert g24 is not None
    assert sorted(o.level for o in g24.pending_orders if o.side == "BUY") == [-3, -2, -1, 0]
    assert sorted(o.level for o in g24.pending_orders if o.side == "SELL") == [2]


# =============================================================================
# 5. Account A routing: TRAP_OFF liquidation, UNRATED==TRANSITION, TREND<->GRID switch
# =============================================================================
def test_account_a_trap_off_unrated_transition_and_mode_switch() -> None:
    """Verify TRAP_OFF open liquidation & order cancel, UNRATED==TRANSITION, and TREND<->GRID switch."""
    dates = _make_weekdays(65)
    sym_trap = "515790.SH"
    sym_unrated = "512880.SH"
    sym_switch = "512400.SH"
    sym_trans_hold = "510720.SH"

    bars: dict[str, pd.DataFrame] = {
        sym: _make_constant_bars(dates, price=1.00, spread=0.015)
        for sym in ALL_SYMBOLS
    }
    # sym_unrated has a huge breakout on day 55..59, yet UNRATED must NEVER open a position!
    bars[sym_unrated] = _make_constant_bars(
        dates,
        price=1.00,
        spread=0.015,
        overrides={
            55: (1.10, 1.20, 1.08, 1.18),
            56: (1.18, 1.25, 1.15, 1.22),
        },
    )
    # sym_trap opens at 0.92 on day 56 (TRAP_OFF day)
    bars[sym_trap] = _make_constant_bars(
        dates,
        price=1.00,
        spread=0.015,
        overrides={
            55: (1.00, 1.01, 0.99, 1.00),   # RANGE_GRID init at Close=1.00
            56: (0.92, 0.95, 0.90, 0.94),   # TRAP_OFF -> liquidate at Open=0.92!
        },
    )
    # sym_switch: day 55 TREND_UP breakout -> day 56 switches to RANGE_GRID (open=1.08, close=1.10)
    #             -> day 57 switches back to TREND_UP (open=1.12)
    bars[sym_switch] = _make_constant_bars(
        dates,
        price=1.00,
        spread=0.015,
        overrides={
            55: (1.03, 1.05, 1.02, 1.04),   # TREND_UP breakout > 1.015 -> buy at 1.03
            56: (1.08, 1.11, 1.07, 1.10),   # Switch TREND_UP -> RANGE_GRID: exit@1.08, grid init@1.10
            57: (1.12, 1.13, 1.11, 1.12),   # Switch RANGE_GRID -> TREND_UP: exit@1.12, cancel grid
        },
    )
    # sym_trans_hold: day 55 RANGE_GRID init -> day 56 switches to UNRATED (==TRANSITION):
    #   must cancel grid pending orders AND keep existing shares without trading!
    bars[sym_trans_hold] = _make_constant_bars(
        dates,
        price=1.00,
        spread=0.015,
        overrides={
            55: (1.00, 1.01, 0.99, 1.00),
            56: (0.90, 1.10, 0.85, 0.95),  # Wild swing would trigger grid if orders weren't cancelled!
        },
    )

    bundle = _make_synthetic_bundle(
        dates=dates,
        bars_by_symbol=bars,
        regimes={
            sym_trap: "RANGE_GRID",
            sym_unrated: "UNRATED",
            sym_switch: "TREND_UP",
            sym_trans_hold: "RANGE_GRID",
        },
    )

    regime_schedule = {
        dates[55]: {
            sym_trap: "RANGE_GRID",
            sym_unrated: "UNRATED",
            sym_switch: "TREND_UP",
            sym_trans_hold: "RANGE_GRID",
        },
        dates[56]: {
            sym_trap: "TRAP_OFF",
            sym_unrated: "UNRATED",
            sym_switch: "RANGE_GRID",
            sym_trans_hold: "UNRATED",
        },
        dates[57]: {
            sym_switch: "TREND_UP",
        },
    }

    engine = SimulationEngine(bundle)
    engine.step_day(dates[55], regime_schedule=regime_schedule)
    acct_a = engine.accounts["A"]

    # Day 55 checks
    assert acct_a.get_shares(sym_unrated) == 0
    assert acct_a.symbol_states[sym_unrated].raw_regime == "UNRATED"
    assert acct_a.symbol_states[sym_unrated].effective_regime == "TRANSITION"
    assert acct_a.get_shares(sym_trap) > 0
    assert acct_a.symbol_states[sym_trap].grid is not None
    assert acct_a.get_shares(sym_switch) > 0
    assert acct_a.get_shares(sym_trans_hold) > 0
    held_trans_day55 = acct_a.get_shares(sym_trans_hold)

    # Day 56 checks
    engine.step_day(dates[56], regime_schedule=regime_schedule)
    # 1) TRAP_OFF: unconditionally liquidated at Open=0.92 and grid orders cancelled
    assert acct_a.get_shares(sym_trap) == 0
    assert acct_a.symbol_states[sym_trap].grid is None
    trap_trades = [t for t in acct_a.trades if t["symbol"] == sym_trap]
    assert trap_trades[-1]["action"] == "TRAP_OFF_LIQUIDATION"
    assert trap_trades[-1]["date"] == dates[56]
    assert trap_trades[-1]["price"] == pytest.approx(0.92)

    # 2) UNRATED from flat: still 0 shares, 0 trades
    assert acct_a.get_shares(sym_unrated) == 0
    assert [t for t in acct_a.trades if t["symbol"] == sym_unrated] == []

    # 3) RANGE_GRID -> UNRATED (== TRANSITION): pending grid cancelled, existing shares held, 0 trades on day 56
    assert acct_a.symbol_states[sym_trans_hold].raw_regime == "UNRATED"
    assert acct_a.symbol_states[sym_trans_hold].effective_regime == "TRANSITION"
    assert acct_a.symbol_states[sym_trans_hold].grid is None
    assert acct_a.get_shares(sym_trans_hold) == held_trans_day55
    assert [t for t in acct_a.trades if t["symbol"] == sym_trans_hold and t["date"] == dates[56]] == []

    # 4) TREND_UP -> RANGE_GRID: exited TREND_UP at Open=1.08, then initialized Grid at Close=1.10
    sw_d56 = [t for t in acct_a.trades if t["symbol"] == sym_switch and t["date"] == dates[56]]
    assert [t["action"] for t in sw_d56] == ["REGIME_SWITCH_EXIT", "GRID_INIT_BUY"]
    assert sw_d56[0]["price"] == pytest.approx(1.08)
    assert sw_d56[1]["price"] == pytest.approx(1.10)
    assert acct_a.symbol_states[sym_switch].grid is not None
    assert acct_a.symbol_states[sym_switch].grid.anchor_price == pytest.approx(1.10)

    # Day 57 checks: RANGE_GRID -> TREND_UP exits grid position at Open=1.12 & clears grid
    engine.step_day(dates[57], regime_schedule=regime_schedule)
    sw_d57 = [t for t in acct_a.trades if t["symbol"] == sym_switch and t["date"] == dates[57]]
    assert sw_d57[0]["action"] == "REGIME_SWITCH_EXIT"
    assert sw_d57[0]["price"] == pytest.approx(1.12)
    assert acct_a.symbol_states[sym_switch].grid is None


# =============================================================================
# 6. Accounts C & D buy-and-hold start-day init, zero rebalancing & excess returns
# =============================================================================
def test_accounts_c_and_d_buy_and_hold_zero_rebalance_and_excess_returns() -> None:
    """Verify C buys 6 sectors at 1/6 on start Open, D buys benchmark full, 0 rebalances after."""
    dates = _make_weekdays(10)
    bars: dict[str, pd.DataFrame] = {}
    # Different price paths for sectors & benchmark to verify no rebalancing & accurate excess returns
    for i, sym in enumerate(SECTORS):
        ov = {
            d: (1.0 + 0.01 * (i + 1) * d, 1.02 + 0.01 * (i + 1) * d, 0.99 + 0.01 * (i + 1) * d, 1.0 + 0.01 * (i + 1) * d)
            for d in range(len(dates))
        }
        bars[sym] = _make_constant_bars(dates, price=1.0, spread=0.01, overrides=ov)

    ov_bench = {
        d: (2.0 + 0.02 * d, 2.02 + 0.02 * d, 1.98 + 0.02 * d, 2.0 + 0.02 * d)
        for d in range(len(dates))
    }
    bars[BENCHMARK] = _make_constant_bars(dates, price=2.0, spread=0.02, overrides=ov_bench)

    # Account A stays all-cash (all UNRATED/TRANSITION) so A cum_return == 0.0
    bundle = _make_synthetic_bundle(
        dates=dates,
        bars_by_symbol=bars,
        regimes={sym: "TRANSITION" for sym in SECTORS},
    )
    engine = run_simulation(bundle, start_date=dates[0], end_date=dates[-1])

    acct_c = engine.accounts["C"]
    acct_d = engine.accounts["D"]

    # Account C: exactly 6 BH_INIT_BUY trades on Day 0, zero trades thereafter
    assert len(acct_c.trades) == 6
    assert all(t["date"] == dates[0] and t["action"] == "BH_INIT_BUY" for t in acct_c.trades)
    assert [t["symbol"] for t in acct_c.trades] == sorted(SECTORS)
    for sym in SECTORS:
        # Budget per sector = 100000 / 6 = 16666.6667 at open=1.0 -> 16600 shares
        assert acct_c.get_shares(sym) == 16600

    # Account D: exactly 1 BH_INIT_BUY trade on Day 0 for 159330.SZ, zero trades thereafter
    assert len(acct_d.trades) == 1
    assert acct_d.trades[0]["date"] == dates[0]
    assert acct_d.trades[0]["symbol"] == BENCHMARK
    assert acct_d.trades[0]["action"] == "BH_INIT_BUY"
    # Budget = 100000 at open=2.0: 49900 * 2.0 = 99800, comm = 19.96, total = 99819.96 <= 100000
    assert acct_d.get_shares(BENCHMARK) == 49900
    assert acct_d.cash == pytest.approx(100000.0 - 99800.0 - 19.96)

    # Verify excess returns A_vs_C and A_vs_D match A_cum - other_cum
    summary = engine.get_summary()
    a_cum = summary["accounts"]["A"]["cum_return"]
    assert a_cum == pytest.approx(0.0)
    assert summary["excess_returns"]["A_vs_B1"] == pytest.approx(a_cum - summary["accounts"]["B1"]["cum_return"])
    assert summary["excess_returns"]["A_vs_B2"] == pytest.approx(a_cum - summary["accounts"]["B2"]["cum_return"])
    assert summary["excess_returns"]["A_vs_C"] == pytest.approx(a_cum - summary["accounts"]["C"]["cum_return"])
    assert summary["excess_returns"]["A_vs_D"] == pytest.approx(a_cum - summary["accounts"]["D"]["cum_return"])


# =============================================================================
# 7. Incremental resume equivalence: 10-day full run vs 5-day + JSON state + 5-day
# =============================================================================
def test_incremental_resume_10_days_vs_5_plus_5_exact_match() -> None:
    """Running 10 days continuously must match 5 days -> JSON state_dict -> 5 days identically."""
    # 56 warmup days + 10 active simulation days (indices 56..65)
    dates = _make_weekdays(66)
    bars: dict[str, pd.DataFrame] = {}
    for sym in ALL_SYMBOLS:
        bars[sym] = _make_constant_bars(dates, price=1.00, spread=0.015)

    # Inject diverse events across BOTH halves (days 56..60 and days 61..65):
    # - 512400.SH: Turtle breakout day 57, exit day 63, dividend day 60
    bars["512400.SH"] = _make_constant_bars(
        dates,
        price=1.00,
        spread=0.015,
        overrides={
            56: (1.00, 1.01, 0.99, 1.00),
            57: (1.03, 1.06, 1.02, 1.05),   # Turtle breakout
            58: (1.05, 1.07, 1.04, 1.06),
            59: (1.06, 1.08, 1.05, 1.07),
            60: (1.04, 1.06, 1.03, 1.05),   # Dividend ex_date on day 60!
            61: (1.04, 1.05, 1.01, 1.02),
            62: (1.01, 1.02, 0.99, 1.00),
            63: (0.94, 0.96, 0.93, 0.95),   # Turtle breakdown exit
            64: (0.95, 0.98, 0.94, 0.97),
            65: (0.98, 1.02, 0.97, 1.01),
        },
    )
    # - 510720.SH: Oscillating price triggering Grid buys & sells in both 5-day halves
    bars["510720.SH"] = _make_constant_bars(
        dates,
        price=1.00,
        spread=0.015,
        overrides={
            56: (1.00, 1.01, 0.99, 1.00),   # Grid init P0=1.00, step=0.03
            57: (0.98, 0.99, 0.96, 0.97),   # BUY -1
            58: (0.96, 0.97, 0.935, 0.94),  # BUY -2
            59: (0.96, 0.98, 0.955, 0.975), # SELL -1
            60: (0.99, 1.01, 0.985, 1.005), # SELL 0
            61: (1.02, 1.04, 1.015, 1.035), # SELL +1
            62: (1.00, 1.01, 0.995, 1.00),  # BUY 0
            63: (0.98, 0.99, 0.965, 0.97),  # BUY -1
            64: (1.00, 1.02, 0.995, 1.015), # SELL 0
            65: (1.03, 1.04, 1.02, 1.035),  # SELL +1
        },
    )
    # - 515790.SH: switches to TRAP_OFF on day 62 (second half!)
    bars["515790.SH"] = _make_constant_bars(
        dates,
        price=1.00,
        spread=0.015,
        overrides={
            56: (1.00, 1.01, 0.99, 1.00),
            62: (0.91, 0.93, 0.90, 0.92),
        },
    )
    # - 159330.SZ: benchmark moves up and down across both halves
    bars[BENCHMARK] = _make_constant_bars(
        dates,
        price=1.50,
        spread=0.02,
        overrides={
            56: (1.50, 1.52, 1.49, 1.51),
            59: (1.54, 1.56, 1.52, 1.55),
            62: (1.45, 1.47, 1.43, 1.44),
            65: (1.52, 1.55, 1.50, 1.53),
        },
    )

    bundle = _make_synthetic_bundle(
        dates=dates,
        bars_by_symbol=bars,
        dividends={"512400.SH": {dates[60]: 0.03}},
    )

    regime_schedule = {
        dates[56]: {
            "512400.SH": "TREND_UP",
            "510720.SH": "RANGE_GRID",
            "515790.SH": "RANGE_GRID",
            "512880.SH": "UNRATED",
            "159622.SZ": "TRANSITION",
            "512480.SH": "TREND_UP",
        },
        dates[62]: {
            "515790.SH": "TRAP_OFF",
            "512480.SH": "RANGE_GRID",
        },
    }

    # Path 1: Continuous 10-day run (days 56..65)
    full_engine = run_simulation(
        bundle,
        start_date=dates[56],
        end_date=dates[65],
        regime_schedule=regime_schedule,
    )
    full_state = full_engine.export_state()

    # Path 2: First 5 days (56..60) -> lossless JSON roundtrip -> next 5 days (61..65)
    part1_engine = run_simulation(
        bundle,
        start_date=dates[56],
        end_date=dates[60],
        regime_schedule=regime_schedule,
    )
    serialized_mid = json.dumps(part1_engine.export_state(), allow_nan=False, sort_keys=True)
    restored_mid_state = json.loads(serialized_mid)

    part2_engine = SimulationEngine.from_state(bundle, restored_mid_state)
    part2_engine.run(
        start_date=dates[61],
        end_date=dates[65],
        regime_schedule=regime_schedule,
    )
    split_state = part2_engine.export_state()

    # Verify JSON serialization succeeds strictly without NaN/Inf
    full_json = json.dumps(full_state, allow_nan=False, sort_keys=True)
    split_json = json.dumps(split_state, allow_nan=False, sort_keys=True)
    assert full_json == split_json

    # Explicit assertions across all 5 accounts
    for acct_id in ("A", "B1", "B2", "C", "D"):
        a_full = full_engine.accounts[acct_id]
        a_split = part2_engine.accounts[acct_id]
        assert a_full.nav == pytest.approx(a_split.nav, abs=1e-12)
        assert a_full.equity == pytest.approx(a_split.equity, abs=1e-12)
        assert a_full.cash == pytest.approx(a_split.cash, abs=1e-12)
        assert a_full.max_drawdown == pytest.approx(a_split.max_drawdown, abs=1e-12)
        assert a_full.trades == a_split.trades
        assert a_full.daily_records == a_split.daily_records
        assert len(a_full.daily_records) == 10
        for sym in a_full.positions:
            assert a_full.get_shares(sym) == a_split.get_shares(sym)

    # Confirm non-trivial activity actually occurred during the 10 days
    assert len(full_engine.accounts["A"].trades) >= 8
    assert len(full_engine.accounts["B1"].trades) >= 2
    assert len(full_engine.accounts["B2"].trades) >= 10


# =============================================================================
# 8. Split adjustment on active positions & Grid + Real upstream smoke test
# =============================================================================
def test_split_event_adjusts_positions_and_grid_without_false_signals() -> None:
    """1-for-2 split doubles held shares, halves grid anchor/orders, and avoids false Turtle exit."""
    dates = _make_weekdays(62)
    sym_split = "512480.SH"

    # Pre-split price = 2.00 (high=2.04, low=1.96) for days 0..56; day 55 breaks out to 2.08
    # Day 58: 1-for-2 split (factor=0.5) -> raw price halves to ~1.04 (not a breakdown!)
    overrides: dict[int, tuple[float, float, float, float]] = {
        55: (2.06, 2.10, 2.05, 2.08),  # Turtle breakout buy at 2.06
        56: (2.08, 2.10, 2.06, 2.08),  # Hold (Grid also active in B2 at anchor=2.08 on day 55)
        57: (2.08, 2.10, 2.06, 2.08),
        58: (1.04, 1.05, 1.03, 1.04),  # Split effective_date: raw price halves, no false exit!
        59: (1.04, 1.05, 1.03, 1.04),
    }
    bars = {
        sym: _make_constant_bars(dates, price=1.00, spread=0.015)
        for sym in ALL_SYMBOLS
    }
    bars[sym_split] = _make_constant_bars(dates, price=2.00, spread=0.04, overrides=overrides)

    bundle = _make_synthetic_bundle(
        dates=dates,
        bars_by_symbol=bars,
        splits={sym_split: [{"effective_date": dates[58], "factor": 0.5}]},
    )

    engine = run_simulation(bundle, start_date=dates[55], end_date=dates[57])
    b1_shares_pre = engine.accounts["B1"].get_shares(sym_split)
    b2_shares_pre = engine.accounts["B2"].get_shares(sym_split)
    b2_anchor_pre = engine.accounts["B2"].symbol_states[sym_split].grid.anchor_price
    assert b1_shares_pre > 0
    assert b2_shares_pre > 0
    assert b2_anchor_pre == pytest.approx(2.08)

    # Advance through split day 58 & day 59
    engine.step_day(dates[58])
    engine.step_day(dates[59])

    # B1 must NOT false-exit on split day, and held shares must double
    assert engine.accounts["B1"].get_shares(sym_split) == b1_shares_pre * 2
    b1_exits = [
        t for t in engine.accounts["B1"].trades
        if t["symbol"] == sym_split and t["action"] == "TURTLE_EXIT_SELL"
    ]
    assert b1_exits == []

    # B2 shares double and grid anchor halves from 2.08 -> 1.04
    assert engine.accounts["B2"].get_shares(sym_split) == b2_shares_pre * 2
    b2_grid_post = engine.accounts["B2"].symbol_states[sym_split].grid
    assert b2_grid_post is not None
    assert b2_grid_post.anchor_price == pytest.approx(1.04)


def test_real_upstream_30day_smoke_and_incremental_resume() -> None:
    """Read-only smoke test on real upstream 7-ETF bundle verifying conservation & resume."""
    bundle = load_market_bundle(DEFAULT_SIM_CONFIG_PATH)
    recent_dates = bundle.common_trading_dates[-30:]
    start_dt, mid_dt, next_dt, end_dt = (
        recent_dates[0],
        recent_dates[14],
        recent_dates[15],
        recent_dates[-1],
    )

    full_engine = run_simulation(bundle, start_date=start_dt, end_date=end_dt)

    part1 = run_simulation(bundle, start_date=start_dt, end_date=mid_dt)
    mid_json = json.dumps(part1.export_state(), allow_nan=False, sort_keys=True)
    part2 = SimulationEngine.from_state(bundle, json.loads(mid_json))
    part2.run(start_date=next_dt, end_date=end_dt)

    assert json.dumps(full_engine.export_state(), allow_nan=False, sort_keys=True) == json.dumps(
        part2.export_state(), allow_nan=False, sort_keys=True
    )

    # Real watchlist has 512880.SH=UNRATED & 515790.SH=TRAP_OFF -> Account A holds 0 of both
    acct_a = full_engine.accounts["A"]
    assert acct_a.get_shares("512880.SH") == 0
    assert acct_a.symbol_states["512880.SH"].effective_regime == "TRANSITION"
    assert acct_a.get_shares("515790.SH") == 0
    assert acct_a.symbol_states["515790.SH"].effective_regime == "TRAP_OFF"

    # Accounts C and D both initialized on start_dt and stayed fully invested
    assert full_engine.accounts["C"].initialized is True
    assert full_engine.accounts["D"].initialized is True
    assert full_engine.accounts["D"].get_shares("159330.SZ") > 0


def test_grid_wide_bar_double_touch_and_cash_exhaustion_invariants() -> None:
    """Wide bar touching both sides & low-cash buy halt must preserve unique ordered grid levels."""
    from src.strategies import GridOrder, GridState, match_grid_bar

    grid = GridState(
        anchor_price=1.00,
        step=0.03,
        slice_budget=3000.0,
        total_slices=5,
        initial_slices=2,
        initialized_date="2026-01-05",
        pending_orders=(
            GridOrder(side="BUY", level=-3, price=grid_level_price(1.00, 0.03, -3), budget=3000.0),
            GridOrder(side="BUY", level=-2, price=grid_level_price(1.00, 0.03, -2), budget=3000.0),
            GridOrder(side="BUY", level=-1, price=grid_level_price(1.00, 0.03, -1), budget=3000.0),
            GridOrder(side="SELL", level=1, price=grid_level_price(1.00, 0.03, 1), shares=3000),
            GridOrder(side="SELL", level=2, price=grid_level_price(1.00, 0.03, 2), shares=3000),
        ),
    )

    # Case 1: Wide bar touching both BUY -1 (0.97) and SELL +1 (1.03) with Open=1.00 ->
    # deterministic conservative SELL first, no level collision, no same-day round trip
    next_g1, fills1 = match_grid_bar(
        grid_state=grid,
        open_price=1.00,
        high_price=1.04,
        low_price=0.96,
        sellable_shares=6000,
        available_cash=10000.0,
    )
    assert len(fills1) == 1
    assert fills1[0].side == "SELL"
    assert fills1[0].level == 1
    levels1 = [o.level for o in next_g1.pending_orders]
    assert len(levels1) == len(set(levels1))
    assert sorted(o.level for o in next_g1.pending_orders if o.side == "BUY") == [-3, -2, -1, 0]
    assert sorted(o.level for o in next_g1.pending_orders if o.side == "SELL") == [2]

    # Case 2: Cash=95.0 cannot afford 100 shares at BUY -1 (0.97 -> 97.10), even though
    # 100 shares at BUY -2 (0.9409 -> 94.19) would be affordable -> must halt without skipping -1
    next_g2, fills2 = match_grid_bar(
        grid_state=grid,
        open_price=0.98,
        high_price=0.99,
        low_price=0.93,
        sellable_shares=6000,
        available_cash=95.0,
    )
    assert fills2 == []
    assert next_g2.pending_orders == grid.pending_orders

