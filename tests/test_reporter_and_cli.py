"""Unit and end-to-end CLI tests for src.reporter and src.run_weekly_sim (Task 1t04).

Covers:
1. Generation and field completeness of `state.json`, `trades.csv`, and `nav_weekly.md`
2. Deterministic `trades.csv` sorting by `(date, account, int(trade_id))` and atomic writes
3. Anti-lookahead regime locking: modifying `watchlist.json` after Week 1 without same-day
   refresh does NOT affect Week 2 execution; new regime locks at Week 2 close for Week 3
4. Same-day idempotent re-run: no duplicate trades or duplicate history rows in `nav_weekly.md`,
   and post-close watchlist updates cleanly refresh `next_effective_regime` & snapshot table
5. Lag warning guard (aborts by default with zero writes) and `--allow-lag` override
6. `--dry-run` prints summary without creating or mutating any files on disk
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any

import pytest

from src.reporter import (
    HISTORY_SECTION_HEADER,
    HISTORY_TABLE_COLUMNS,
    TRADES_CSV_COLUMNS,
    export_trades_csv,
    load_state_json,
    save_state_json,
)
from src.run_weekly_sim import main as cli_main

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

SYMBOL_NAMES = {
    "159330.SZ": "沪深300ETF",
    "159622.SZ": "沪港深创新药ETF",
    "510720.SH": "红利国企ETF",
    "512400.SH": "有色金属ETF",
    "512480.SH": "半导体ETF",
    "512880.SH": "证券ETF",
    "515790.SH": "光伏ETF",
}


def _make_weekdays(count: int, start: date = date(2026, 1, 5)) -> list[str]:
    dates: list[str] = []
    cur = start
    while len(dates) < count:
        if cur.weekday() < 5:
            dates.append(cur.isoformat())
        cur += timedelta(days=1)
    return dates


def _write_tdx_csv(
    tdx_dir: Path,
    symbol: str,
    dates: list[str],
    base_price: float = 1.0,
    spread: float = 0.015,
    overrides: dict[int, tuple[float, float, float, float]] | None = None,
) -> Path:
    sym_dir = tdx_dir / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sym_dir / "20261120_160000.csv"
    ov = overrides or {}
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["date", "open", "high", "low", "close", "amount", "volume"])
        for idx, dt in enumerate(dates):
            if idx in ov:
                o, h, l, c = ov[idx]
            else:
                o = base_price
                h = round(base_price + spread, 6)
                l = round(base_price - spread, 6)
                c = base_price
            writer.writerow([dt, f"{o:.6f}", f"{h:.6f}", f"{l:.6f}", f"{c:.6f}", "1000000.0", "1000000"])
    return csv_path


def _write_watchlist(watchlist_path: Path, regimes: dict[str, str]) -> None:
    universe: list[dict[str, Any]] = [
        {
            "symbol": BENCHMARK,
            "name": SYMBOL_NAMES[BENCHMARK],
            "role": "benchmark",
            "current_regime": "BENCHMARK",
            "split_events": [],
        }
    ]
    for sym in SECTORS:
        universe.append(
            {
                "symbol": sym,
                "name": SYMBOL_NAMES[sym],
                "role": "sector",
                "current_regime": regimes.get(sym, "TRANSITION"),
                "split_events": [],
            }
        )
    payload = {"universe": universe}
    watchlist_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _build_synthetic_env(
    tmp_path: Path,
    *,
    num_days: int = 80,
    regimes: dict[str, str] | None = None,
    lag_symbol: str | None = None,
    lag_days: int = 0,
    bar_overrides: dict[str, dict[int, tuple[float, float, float, float]]] | None = None,
) -> dict[str, Any]:
    dates = _make_weekdays(num_days)
    tdx_dir = tmp_path / "tdx"
    watchlist_path = tmp_path / "watchlist.json"
    dividends_path = tmp_path / "dividends.json"
    config_path = tmp_path / "sim_config.json"
    output_dir = tmp_path / "output"

    overrides_map = bar_overrides or {}
    for idx, sym in enumerate(ALL_SYMBOLS):
        sym_dates = dates[:-lag_days] if (lag_symbol == sym and lag_days > 0) else dates
        base_price = 1.0 + 0.1 * idx
        _write_tdx_csv(
            tdx_dir,
            sym,
            sym_dates,
            base_price=base_price,
            spread=0.015,
            overrides=overrides_map.get(sym),
        )

    _write_watchlist(watchlist_path, regimes or {})
    dividends_path.write_text(
        json.dumps({"events_by_symbol": {sym: [] for sym in ALL_SYMBOLS}}, indent=2) + "\n",
        encoding="utf-8",
    )

    sim_config = {
        "project_name": "sector_paper_sim_test",
        "upstream_paths": {
            "tdx_daily_dir": str(tdx_dir),
            "dividend_events_file": str(dividends_path),
            "watchlist_file": str(watchlist_path),
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
    config_path.write_text(
        json.dumps(sim_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "dates": dates,
        "tdx_dir": tdx_dir,
        "watchlist_path": watchlist_path,
        "dividends_path": dividends_path,
        "config_path": config_path,
        "output_dir": output_dir,
    }


def _extract_history_rows(md_text: str) -> list[list[str]]:
    lines = md_text.splitlines()
    in_history = False
    rows: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if stripped == HISTORY_SECTION_HEADER:
                in_history = True
                continue
            if in_history:
                break
        if not in_history:
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if cells and len(cells[0]) == 10 and cells[0][4] == "-" and cells[0][7] == "-":
                rows.append(cells)
    return rows


def test_outputs_generation_and_field_completeness(tmp_path: Path) -> None:
    """Verify state.json, trades.csv, and nav_weekly.md creation and full field integrity."""
    env = _build_synthetic_env(
        tmp_path,
        num_days=70,
        regimes={
            "512400.SH": "RANGE_GRID",
            "512480.SH": "TREND_UP",
            "515790.SH": "TRAP_OFF",
        },
    )
    dates = env["dates"]
    start_dt = dates[60]
    end_dt = dates[64]

    rc = cli_main(
        [
            "--config",
            str(env["config_path"]),
            "--output-dir",
            str(env["output_dir"]),
            "--start-date",
            start_dt,
            "--end-date",
            end_dt,
        ]
    )
    assert rc == 0

    state_path = env["output_dir"] / "state.json"
    trades_path = env["output_dir"] / "trades.csv"
    report_path = env["output_dir"] / "nav_weekly.md"

    assert state_path.is_file()
    assert trades_path.is_file()
    assert report_path.is_file()
    # Atomic write must not leave .tmp artifacts behind
    assert not list(env["output_dir"].glob("*.tmp"))

    # 1. state.json validation
    state = load_state_json(state_path)
    assert state["start_date"] == start_dt
    assert state["last_processed_date"] == end_dt
    assert state["trading_days_processed"] == 5
    assert set(state["accounts"].keys()) == {"A", "B1", "B2", "C", "D"}
    assert state["next_effective_regime"]["512400.SH"] == "RANGE_GRID"
    assert state["next_effective_regime"]["512480.SH"] == "TREND_UP"
    assert state["next_effective_regime"]["515790.SH"] == "TRAP_OFF"
    assert len(state["regime_audit_log"]) == 1
    assert state["regime_audit_log"][0]["recorded_at_market_date"] == end_dt

    # 2. trades.csv validation
    with trades_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames or ()) == TRADES_CSV_COLUMNS
        rows = list(reader)

    expected_total_trades = sum(len(state["accounts"][aid]["trades"]) for aid in ("A", "B1", "B2", "C", "D"))
    assert len(rows) == expected_total_trades
    assert len(rows) > 0  # Account A grid init + B2 grid inits + C & D B&H inits

    sort_keys = [(r["date"], r["account"], int(r["trade_id"])) for r in rows]
    assert sort_keys == sorted(sort_keys)
    for row in rows:
        assert int(row["shares"]) > 0
        assert float(row["price"]) > 0.0
        assert float(row["notional"]) > 0.0

    # 3. nav_weekly.md validation
    md_text = report_path.read_text(encoding="utf-8")
    for required_fragment in (
        f"**起始日**：`{start_dt}`",
        f"**最新净值日**：`{end_dt}`",
        "**已运行交易日数**：`5`",
        "**数据滞后检查**：`正常（无滞后）`",
        "A 路由实验盘",
        "B1 纯机械海龟",
        "B2 纯机械网格",
        "C 行业等权 B&H",
        "D 沪深300 B&H",
        "A vs B1",
        "A vs B2",
        "A vs C",
        "A vs D",
        "512400.SH",
        "有色金属ETF",
        "159330.SZ",
        HISTORY_SECTION_HEADER,
        "| " + " | ".join(HISTORY_TABLE_COLUMNS) + " |",
    ):
        assert required_fragment in md_text

    # Account A initialized grid for 512400.SH -> pending orders should be reported (买3/卖2)
    assert "5 (买3/卖2)" in md_text

    history_rows = _extract_history_rows(md_text)
    assert len(history_rows) == 1
    assert history_rows[0][0] == end_dt
    assert history_rows[0][1] == "5"
    assert len(history_rows[0]) == len(HISTORY_TABLE_COLUMNS)


def test_anti_lookahead_regime_locking_across_weeks(tmp_path: Path) -> None:
    """Modifying watchlist after Week 1 must NOT leak into Week 2; locks at Week 2 close for Week 3."""
    env = _build_synthetic_env(
        tmp_path,
        num_days=80,
        regimes={
            "512400.SH": "RANGE_GRID",
            "515790.SH": "TRAP_OFF",
        },
    )
    dates = env["dates"]
    w1_start, w1_end = dates[60], dates[64]  # Week 1 (5 trading days)
    w2_day2 = dates[66]                      # Week 2 first 2 trading days
    w3_day1 = dates[67]                      # Next trading day ("Week 3" start)

    # Week 1 initial run: 512400.SH enters RANGE_GRID and builds initial grid position
    rc1 = cli_main(
        [
            "--config",
            str(env["config_path"]),
            "--output-dir",
            str(env["output_dir"]),
            "--start-date",
            w1_start,
            "--end-date",
            w1_end,
        ]
    )
    assert rc1 == 0
    state_w1 = load_state_json(env["output_dir"] / "state.json")
    assert state_w1["accounts"]["A"]["positions"]["512400.SH"]["shares"] > 0
    assert state_w1["accounts"]["A"]["symbol_states"]["512400.SH"]["effective_regime"] == "RANGE_GRID"
    assert state_w1["next_effective_regime"]["512400.SH"] == "RANGE_GRID"

    # After Week 1 ends, change watchlist.json: 512400.SH -> TRAP_OFF (WITHOUT refreshing w1_end!)
    _write_watchlist(
        env["watchlist_path"],
        {
            "512400.SH": "TRAP_OFF",
            "515790.SH": "TRAP_OFF",
        },
    )

    # Advance Week 2 (2 days): MUST still run 512400.SH under Week 1 locked RANGE_GRID (no lookahead!)
    rc2 = cli_main(
        [
            "--config",
            str(env["config_path"]),
            "--output-dir",
            str(env["output_dir"]),
            "--end-date",
            w2_day2,
        ]
    )
    assert rc2 == 0
    state_w2 = load_state_json(env["output_dir"] / "state.json")
    assert state_w2["last_processed_date"] == w2_day2
    assert state_w2["trading_days_processed"] == 7
    # Still holding 512400.SH in RANGE_GRID throughout Week 2 (no premature TRAP_OFF liquidation!)
    assert state_w2["accounts"]["A"]["positions"]["512400.SH"]["shares"] > 0
    assert state_w2["accounts"]["A"]["symbol_states"]["512400.SH"]["effective_regime"] == "RANGE_GRID"
    assert state_w2["accounts"]["A"]["symbol_states"]["512400.SH"]["grid"] is not None
    w2_a_actions = [t["action"] for t in state_w2["accounts"]["A"]["trades"]]
    assert "TRAP_OFF_LIQUIDATION" not in w2_a_actions

    # After Week 2 finishes, new TRAP_OFF regime is locked for Week 3 and audited
    assert state_w2["next_effective_regime"]["512400.SH"] == "TRAP_OFF"
    assert len(state_w2["regime_audit_log"]) == 2
    latest_audit = state_w2["regime_audit_log"][-1]
    assert latest_audit["recorded_at_market_date"] == w2_day2
    assert latest_audit["changes"] == {"512400.SH": {"from": "RANGE_GRID", "to": "TRAP_OFF"}}

    # Advance Week 3: NOW locked TRAP_OFF takes effect at Open and liquidates 512400.SH
    rc3 = cli_main(
        [
            "--config",
            str(env["config_path"]),
            "--output-dir",
            str(env["output_dir"]),
            "--end-date",
            w3_day1,
        ]
    )
    assert rc3 == 0
    state_w3 = load_state_json(env["output_dir"] / "state.json")
    assert state_w3["last_processed_date"] == w3_day1
    assert state_w3["accounts"]["A"]["positions"]["512400.SH"]["shares"] == 0
    assert state_w3["accounts"]["A"]["symbol_states"]["512400.SH"]["effective_regime"] == "TRAP_OFF"
    assert state_w3["accounts"]["A"]["symbol_states"]["512400.SH"]["grid"] is None

    liq_trades = [
        t
        for t in state_w3["accounts"]["A"]["trades"]
        if t["symbol"] == "512400.SH" and t["action"] == "TRAP_OFF_LIQUIDATION"
    ]
    assert len(liq_trades) == 1
    assert liq_trades[0]["date"] == w3_day1

    # nav_weekly.md history table accumulated 3 distinct weekly dates
    history_rows = _extract_history_rows(
        (env["output_dir"] / "nav_weekly.md").read_text(encoding="utf-8")
    )
    assert [row[0] for row in history_rows] == [w1_end, w2_day2, w3_day1]


def test_same_day_idempotent_rerun_and_watchlist_refresh(tmp_path: Path) -> None:
    """Re-running CLI on the same end_date is idempotent and syncs watchlist changes cleanly."""
    env = _build_synthetic_env(
        tmp_path,
        num_days=70,
        regimes={
            "512400.SH": "TRANSITION",
            "512480.SH": "TRANSITION",
        },
    )
    dates = env["dates"]
    as_of = dates[64]

    # Run 1
    assert (
        cli_main(
            [
                "--config",
                str(env["config_path"]),
                "--output-dir",
                str(env["output_dir"]),
                "--start-date",
                dates[60],
                "--end-date",
                as_of,
            ]
        )
        == 0
    )
    trades_1 = (env["output_dir"] / "trades.csv").read_text(encoding="utf-8")
    state_1 = load_state_json(env["output_dir"] / "state.json")
    hist_1 = _extract_history_rows(
        (env["output_dir"] / "nav_weekly.md").read_text(encoding="utf-8")
    )
    assert len(hist_1) == 1

    # Run 2 (exact same end_date, unchanged watchlist)
    assert (
        cli_main(
            [
                "--config",
                str(env["config_path"]),
                "--output-dir",
                str(env["output_dir"]),
                "--end-date",
                as_of,
            ]
        )
        == 0
    )
    trades_2 = (env["output_dir"] / "trades.csv").read_text(encoding="utf-8")
    state_2 = load_state_json(env["output_dir"] / "state.json")
    hist_2 = _extract_history_rows(
        (env["output_dir"] / "nav_weekly.md").read_text(encoding="utf-8")
    )
    assert trades_2 == trades_1
    assert hist_2 == hist_1
    assert len(state_2["regime_audit_log"]) == len(state_1["regime_audit_log"])
    assert state_2["accounts"] == state_1["accounts"]

    # Run 3 (same end_date after editing watchlist.json: 512480.SH -> TREND_UP)
    _write_watchlist(
        env["watchlist_path"],
        {
            "512400.SH": "TRANSITION",
            "512480.SH": "TREND_UP",
        },
    )
    assert (
        cli_main(
            [
                "--config",
                str(env["config_path"]),
                "--output-dir",
                str(env["output_dir"]),
                "--end-date",
                as_of,
            ]
        )
        == 0
    )
    trades_3 = (env["output_dir"] / "trades.csv").read_text(encoding="utf-8")
    state_3 = load_state_json(env["output_dir"] / "state.json")
    md_3 = (env["output_dir"] / "nav_weekly.md").read_text(encoding="utf-8")
    hist_3 = _extract_history_rows(md_3)

    # Engine did not advance, trades unchanged, history row not duplicated
    assert trades_3 == trades_1
    assert state_3["trading_days_processed"] == state_1["trading_days_processed"]
    assert len(hist_3) == 1
    assert hist_3[0][0] == as_of
    # Next-week locked regime refreshed and audited
    assert state_3["next_effective_regime"]["512480.SH"] == "TREND_UP"
    assert len(state_3["regime_audit_log"]) == 2
    assert state_3["regime_audit_log"][-1]["recorded_at_market_date"] == as_of
    assert state_3["regime_audit_log"][-1]["changes"] == {
        "512480.SH": {"from": "TRANSITION", "to": "TREND_UP"}
    }
    assert "| 512480.SH | 半导体ETF | TRANSITION | TREND_UP |" in md_3


def test_lag_warnings_abort_by_default_and_allow_lag_override(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Non-empty lag_warnings aborts with non-zero exit & no writes unless --allow-lag is passed."""
    env = _build_synthetic_env(
        tmp_path,
        num_days=70,
        lag_symbol="512480.SH",
        lag_days=2,
    )

    rc_abort = cli_main(
        [
            "--config",
            str(env["config_path"]),
            "--output-dir",
            str(env["output_dir"]),
        ]
    )
    captured = capsys.readouterr()
    assert rc_abort != 0
    assert "512480.SH" in captured.err
    assert not (env["output_dir"] / "state.json").exists()
    assert not (env["output_dir"] / "trades.csv").exists()
    assert not (env["output_dir"] / "nav_weekly.md").exists()

    rc_allow = cli_main(
        [
            "--config",
            str(env["config_path"]),
            "--output-dir",
            str(env["output_dir"]),
            "--allow-lag",
        ]
    )
    assert rc_allow == 0
    assert (env["output_dir"] / "state.json").is_file()
    md_text = (env["output_dir"] / "nav_weekly.md").read_text(encoding="utf-8")
    assert "存在滞后警告" in md_text
    assert "512480.SH" in md_text


def test_dry_run_writes_nothing_on_init_or_incremental(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--dry-run prints summary without creating or modifying any files on disk."""
    env = _build_synthetic_env(tmp_path, num_days=70)
    dates = env["dates"]

    # 1. Dry-run on fresh output_dir
    rc_init_dry = cli_main(
        [
            "--config",
            str(env["config_path"]),
            "--output-dir",
            str(env["output_dir"]),
            "--start-date",
            dates[60],
            "--end-date",
            dates[62],
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert rc_init_dry == 0
    assert "[DRY-RUN]" in out
    assert not env["output_dir"].exists()

    # 2. Real init up to dates[62], then dry-run incremental up to dates[65]
    assert (
        cli_main(
            [
                "--config",
                str(env["config_path"]),
                "--output-dir",
                str(env["output_dir"]),
                "--start-date",
                dates[60],
                "--end-date",
                dates[62],
            ]
        )
        == 0
    )
    before_state = (env["output_dir"] / "state.json").read_text(encoding="utf-8")
    before_trades = (env["output_dir"] / "trades.csv").read_text(encoding="utf-8")
    before_md = (env["output_dir"] / "nav_weekly.md").read_text(encoding="utf-8")

    rc_inc_dry = cli_main(
        [
            "--config",
            str(env["config_path"]),
            "--output-dir",
            str(env["output_dir"]),
            "--end-date",
            dates[65],
            "--dry-run",
        ]
    )
    assert rc_inc_dry == 0
    assert (env["output_dir"] / "state.json").read_text(encoding="utf-8") == before_state
    assert (env["output_dir"] / "trades.csv").read_text(encoding="utf-8") == before_trades
    assert (env["output_dir"] / "nav_weekly.md").read_text(encoding="utf-8") == before_md


def test_reporter_state_roundtrip_and_numeric_trade_id_sorting(tmp_path: Path) -> None:
    """Verify UTF-8 JSON roundtrip and numeric trade_id ordering (2 before 10) in trades.csv."""
    sample_state = {
        "schema_version": "1.0.0",
        "symbol_names": {"512400.SH": "有色金属ETF"},
        "accounts": {
            "B2": {
                "trades": [
                    {
                        "trade_id": 10,
                        "date": "2026-11-20",
                        "account": "B2",
                        "symbol": "512400.SH",
                        "action": "GRID_BUY",
                        "side": "BUY",
                        "shares": 100,
                        "price": 1.25,
                        "notional": 125.0,
                        "commission": 0.1,
                        "stamp_duty": 0.0,
                        "net_cash_flow": -125.1,
                        "regime": "RANGE_GRID",
                        "trigger_price": 1.25,
                        "grid_level": -1,
                        "note": "level -1",
                    },
                    {
                        "trade_id": 2,
                        "date": "2026-11-20",
                        "account": "B2",
                        "symbol": "512400.SH",
                        "action": "GRID_INIT_BUY",
                        "side": "BUY",
                        "shares": 200,
                        "price": 1.30,
                        "notional": 260.0,
                        "commission": 0.1,
                        "stamp_duty": 0.0,
                        "net_cash_flow": -260.1,
                        "regime": "RANGE_GRID",
                        "trigger_price": 1.30,
                        "grid_level": 0,
                        "note": "init",
                    },
                ]
            },
            "A": {
                "trades": [
                    {
                        "trade_id": 1,
                        "date": "2026-11-19",
                        "account": "A",
                        "symbol": "512400.SH",
                        "action": "TURTLE_ENTRY_BUY",
                        "side": "BUY",
                        "shares": 500,
                        "price": 1.20,
                        "notional": 600.0,
                        "commission": 0.12,
                        "stamp_duty": 0.0,
                        "net_cash_flow": -600.12,
                        "regime": "TREND_UP",
                        "trigger_price": 1.20,
                        "grid_level": None,
                        "note": "",
                    }
                ]
            },
        },
    }

    state_file = tmp_path / "state.json"
    trades_file = tmp_path / "trades.csv"

    save_state_json(state_file_payload := sample_state, state_file)
    raw_json = state_file.read_text(encoding="utf-8")
    assert "有色金属ETF" in raw_json  # ensure_ascii=False
    assert load_state_json(state_file) == state_file_payload

    export_trades_csv(sample_state, trades_file)
    with trades_file.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert [(r["date"], r["account"], int(r["trade_id"])) for r in rows] == [
        ("2026-11-19", "A", 1),
        ("2026-11-20", "B2", 2),
        ("2026-11-20", "B2", 10),
    ]
    assert rows[0]["grid_level"] == ""
    assert rows[1]["grid_level"] == "0"
    assert rows[2]["grid_level"] == "-1"


def test_unrated_regime_handling_and_display(tmp_path: Path) -> None:
    """UNRATED symbol is preserved in next_effective_regime while routed as TRANSITION."""
    env = _build_synthetic_env(
        tmp_path,
        num_days=70,
        regimes={"512880.SH": "UNRATED"},
    )
    dates = env["dates"]
    rc = cli_main(
        [
            "--config",
            str(env["config_path"]),
            "--output-dir",
            str(env["output_dir"]),
            "--start-date",
            dates[62],
            "--end-date",
            dates[64],
        ]
    )
    assert rc == 0
    state = load_state_json(env["output_dir"] / "state.json")
    assert state["next_effective_regime"]["512880.SH"] == "UNRATED"
    assert state["accounts"]["A"]["symbol_states"]["512880.SH"]["effective_regime"] == "TRANSITION"
    md_text = (env["output_dir"] / "nav_weekly.md").read_text(encoding="utf-8")
    assert "TRANSITION (UNRATED)" in md_text


def test_cli_real_upstream_dry_run_smoke(tmp_path: Path) -> None:
    """Smoke-test real upstream sim_config.json in --dry-run mode with zero disk writes."""
    from src.data_loader import DEFAULT_SIM_CONFIG_PATH

    out_dir = tmp_path / "real_dry_out"
    rc = cli_main(
        [
            "--config",
            str(DEFAULT_SIM_CONFIG_PATH),
            "--output-dir",
            str(out_dir),
            "--start-date",
            "2026-11-20",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not out_dir.exists()

