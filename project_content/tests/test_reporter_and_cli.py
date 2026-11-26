"""Exercise the daily CLI with isolated on-disk BS-01, TDX and ledger inputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
import json
import os
from pathlib import Path
import subprocess

import pytest

from src.data_loader import DEFAULT_SIM_CONFIG_PATH, load_sim_config
from src.engine import ACCOUNT_IDS
from src.reporter import (
    NAV_CSV_COLUMNS,
    TRADES_CSV_COLUMNS,
    export_nav_daily_csv,
    export_trades_csv,
    load_state_json,
    read_csv_rows,
    save_daily_snapshot,
    save_state_json,
    snapshot_path,
)
from src.run_daily_sim import DEFAULT_OUTPUT_DIR, main as cli_main

START = "2026-11-23"
BENCHMARK = "159330.SZ"
SECTORS = ("159622.SZ", "510720.SH", "512400.SH", "512480.SH", "512880.SH", "515790.SH")
GRID = "510720.SH"
TREND = "512400.SH"
SPLIT = "512480.SH"
UNRATED = "512880.SH"
QUANT_PYTHON = "C:/Users/jxing/.conda/envs/quant/python.exe"


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


@dataclass
class Inputs:
    config: Path
    calendar: Path
    watchlist: Path
    ledger: Path
    regimes: Path
    dates: list[str]

    def args(self, output_dir, *flags, as_of=None):
        argv = ["--config", str(self.config), "--output-dir", str(output_dir), *flags]
        if as_of is not None:
            argv.extend(("--as-of", as_of))
        return argv


def make_inputs(tmp_path: Path, *, lag_symbol=None, all_lag=False) -> Inputs:
    begin = date(2026, 8, 17)
    end = date(2026, 12, 8)
    calendar_days = [begin + timedelta(days=offset) for offset in range((end - begin).days + 1)]
    trading_days = [day.isoformat() for day in calendar_days if day.weekday() < 5]
    dates = [day for day in trading_days if day >= START]
    calendar = tmp_path / "calendar" / "20261208_193000.csv"
    write_csv(calendar, ("calendar_date", "is_trading_day"), (
        (day.isoformat(), int(day.weekday() < 5)) for day in calendar_days
    ))

    bars_dir = tmp_path / "bars"
    symbols = [BENCHMARK, *SECTORS]
    for symbol in symbols:
        rows = []
        for day in trading_days:
            if day == dates[-1] and (all_lag or lag_symbol == symbol):
                continue
            if symbol == GRID and day == dates[2]:
                continue
            if symbol == BENCHMARK and day == START:
                continue
            price = 2.0 if symbol == SPLIT and day < dates[6] else 1.0
            ohlc = (price, price + 0.005, price - 0.005, price)
            if symbol == GRID and day in (dates[1], dates[3]):
                close = 1.02 if day == dates[1] else 0.98
                ohlc = (1.0, 1.07, 0.94, close)
            rows.append((day, *ohlc, 100000, 100000))
        write_csv(
            bars_dir / symbol / "20261208_180000.csv",
            ("date", "open", "high", "low", "close", "amount", "volume"),
            rows,
        )

    watchlist = tmp_path / "watchlist.json"
    write_json(watchlist, {
        "universe": [
            {"symbol": symbol, "name": f"ETF {symbol}", "current_regime": "BENCHMARK" if symbol == BENCHMARK else "TRAP_OFF"}
            for symbol in symbols
        ]
    })
    ledger = tmp_path / "corporate_actions_ledger.json"
    write_json(ledger, {
        "schema_version": "2.0.0",
        "as_of": dates[-1],
        "cash_dividends_by_symbol": {
            symbol: [{
                "ex_date": dates[2], "cash_per_unit": 0.06,
                "source_class": "OFFICIAL_TEST", "event_status": "VERIFIED",
                "source_pdf": "raw/test/dividend.pdf", "source_pdf_sha256": "a" * 64,
            }] if symbol == GRID else []
            for symbol in symbols
        },
        "splits_by_symbol": {
            symbol: [{
                "effective_date": dates[6], "factor": 0.5,
                "source_class": "OFFICIAL_TEST", "event_status": "VERIFIED",
                "source_pdf": "raw/test/split.pdf", "source_pdf_sha256": "b" * 64,
            }] if symbol == SPLIT else []
            for symbol in symbols
        },
    })
    regimes = tmp_path / "regime_history.json"
    write_json(regimes, {
        "schema_version": "1.0.0",
        "anchor_start_date": START,
        "initial_regimes": {
            **dict.fromkeys(SECTORS, "TRANSITION"),
            GRID: "RANGE_GRID",
            TREND: "TREND_UP",
            UNRATED: "UNRATED",
        },
        "records": [
            {"symbol": GRID, "effective_date": dates[2], "new_regime": "TRANSITION"},
            {"symbol": TREND, "effective_date": dates[5], "new_regime": "RANGE_GRID"},
            {"symbol": UNRATED, "effective_date": dates[7], "new_regime": "TREND_UP"},
        ],
    })
    config = tmp_path / "sim_config.json"
    contents = load_sim_config()
    assert contents["start_date"] == START
    contents["upstream_paths"] = {
        "tdx_daily_dir": str(bars_dir),
        "trade_calendar_dir": str(calendar.parent),
        "watchlist_file": str(watchlist),
        "corporate_actions_ledger_file": str(ledger),
        "regime_history_file": str(regimes),
    }
    write_json(config, contents)
    return Inputs(config, calendar, watchlist, ledger, regimes, dates)


def output_bytes(output_dir):
    return {path.relative_to(output_dir): path.read_bytes() for path in output_dir.rglob("*") if path.is_file()}


def test_daily_outputs_have_five_accounts_and_keep_only_the_checkpoint(tmp_path, capsys):
    inputs = make_inputs(tmp_path)
    output = tmp_path / "output"
    last_date = inputs.dates[-1]
    assert cli_main(inputs.args(output, "--from-scratch", as_of=last_date)) == 0
    state = load_state_json(output / "state.json")
    assert set(state) == {"schema_version", "start_date", "last_processed_date", "last_known_closes", "accounts"}
    assert state["start_date"] == START and state["last_processed_date"] == last_date
    assert set(state["accounts"]) == set(ACCOUNT_IDS)
    assert set(state["last_known_closes"]) == {BENCHMARK, *SECTORS}
    raw = (output / "state.json").read_text(encoding="utf-8")
    for historical_key in ("trades", "daily_records", "excess_history", "corporate_actions", "updated_at", "next_effective_regime"):
        assert f'"{historical_key}"' not in raw
    assert len(raw.splitlines()) < 800

    nav = read_csv_rows(output / "nav_daily.csv", NAV_CSV_COLUMNS)
    assert [(row["date"], row["account"]) for row in nav] == [
        (day, account) for day in inputs.dates for account in ACCOUNT_IDS
    ]
    assert all(row["A_vs_B1"] for row in nav if row["account"] == "A")
    assert all(row["A_vs_D"] == "" for row in nav if row["account"] != "A")
    suspended = next(row for row in nav if row["account"] == "A" and row["date"] == inputs.dates[2])
    assert suspended["suspended_symbols"] == GRID
    assert float(suspended["dividend_cash"]) > 0

    trades = read_csv_rows(output / "trades.csv", TRADES_CSV_COLUMNS)
    keys = [(t["date"], t["account"], int(t["trade_id"])) for t in trades]
    assert keys == sorted(set(keys))
    assert {row["action"] for row in trades} >= {
        "GRID_INIT_BUY", "GRID_BUY", "GRID_SELL", "TURTLE_ENTRY_BUY", "TURTLE_EXIT_SELL",
        "REGIME_SWITCH_EXIT", "DIVIDEND_REINVEST", "BH_INIT_BUY",
    }
    assert all(float(row["commission"]) >= 0.10 for row in trades)
    assert len(trades) == sum(account["next_trade_id"] - 1 for account in state["accounts"].values())
    snapshots = sorted((output / "history").glob("*.md"))
    assert snapshots == [snapshot_path(output, day) for day in inputs.dates]
    assert (output / "nav_latest.md").read_bytes() == snapshots[-1].read_bytes()
    marker = f"[停牌按前收估值 {inputs.dates[2]}]"
    suspended_report = snapshots[2].read_text(encoding="utf-8")
    assert marker in suspended_report and "DIVIDEND_CASH" in suspended_report
    assert marker in capsys.readouterr().out
    assert "SPLIT" in snapshots[6].read_text(encoding="utf-8")
    for data in output_bytes(output).values():
        assert b"\r" not in data and not data.startswith(b"\xef\xbb\xbf")
        data.decode("utf-8")


def test_from_scratch_matches_every_day_incremental_byte_for_byte(tmp_path):
    inputs = make_inputs(tmp_path)
    full = tmp_path / "full"
    incremental = tmp_path / "incremental"
    assert cli_main(inputs.args(full, "--from-scratch", as_of=inputs.dates[-1])) == 0
    for day in inputs.dates:
        assert cli_main(inputs.args(incremental, as_of=day)) == 0
    assert output_bytes(incremental) == output_bytes(full)

    # An invocation at the same cutoff is a no-op for the committed ledger and snapshots.
    before = output_bytes(incremental)
    assert cli_main(inputs.args(incremental, as_of=inputs.dates[-1])) == 0
    assert output_bytes(incremental) == before


def test_history_routes_activate_on_effective_day_and_ignore_current_watchlist(tmp_path):
    inputs = make_inputs(tmp_path)
    output = tmp_path / "output"
    assert cli_main(inputs.args(output, as_of=START)) == 0
    state = load_state_json(output / "state.json")["accounts"]["A"]
    assert state["symbol_states"][GRID]["active_regime"] == "RANGE_GRID"
    assert state["symbol_states"][UNRATED]["active_regime"] == "TRANSITION"
    assert state["positions"][UNRATED]["shares"] == 0

    assert cli_main(inputs.args(output, as_of=inputs.dates[2])) == 0
    state = load_state_json(output / "state.json")["accounts"]["A"]
    assert state["symbol_states"][GRID]["active_regime"] == "RANGE_GRID"
    assert state["symbol_states"][GRID]["pending_regime"] == "TRANSITION"
    before = state["positions"][GRID]["shares"]
    assert before > 0

    assert cli_main(inputs.args(output, as_of=inputs.dates[3])) == 0
    state = load_state_json(output / "state.json")["accounts"]["A"]
    assert state["positions"][GRID]["shares"] == 0
    sold, = [
        trade for trade in read_csv_rows(output / "trades.csv", TRADES_CSV_COLUMNS)
        if trade["account"] == "A" and trade["action"] == "REGIME_SWITCH_EXIT"
    ]
    assert (sold["date"], int(sold["shares"])) == (inputs.dates[3], before)


def test_dry_run_does_not_create_or_modify_output_files(tmp_path, capsys):
    inputs = make_inputs(tmp_path)
    output = tmp_path / "output"
    assert cli_main(inputs.args(output, "--dry-run", "--from-scratch", as_of=inputs.dates[3])) == 0
    assert "[DRY-RUN]" in capsys.readouterr().out
    assert not output.exists()
    assert cli_main(inputs.args(output, as_of=inputs.dates[1])) == 0
    before = output_bytes(output)
    assert cli_main(inputs.args(output, "--dry-run", as_of=inputs.dates[3])) == 0
    assert output_bytes(output) == before


@pytest.mark.parametrize("all_lag", [False, True])
def test_lag_aborts_unless_explicitly_allowed_by_calendar_date(tmp_path, capsys, all_lag):
    inputs = make_inputs(tmp_path, lag_symbol=GRID, all_lag=all_lag)
    output = tmp_path / "lag"
    requested = inputs.dates[-1]
    assert cli_main(inputs.args(output, as_of=requested)) == 2
    assert GRID in capsys.readouterr().err
    assert not output.exists()
    assert cli_main(inputs.args(output, "--allow-lag", as_of=requested)) == 0
    assert load_state_json(output / "state.json")["last_processed_date"] == requested
    marker = f"{GRID} [行情滞后按前收估值 {requested}]"
    assert marker in (output / "nav_latest.md").read_text(encoding="utf-8")
    assert marker in capsys.readouterr().out


def test_as_of_includes_only_calendar_open_days(tmp_path):
    inputs = make_inputs(tmp_path)
    output = tmp_path / "output"
    assert cli_main(inputs.args(output, as_of="2026-11-29")) == 0
    assert load_state_json(output / "state.json")["last_processed_date"] == "2026-11-27"
    assert len(list((output / "history").glob("*.md"))) == 5


@pytest.mark.parametrize("cutoff", ["2026-11-20", "20261123", "2026-11-31", "2026-12-31"])
def test_invalid_as_of_or_uncovered_calendar_fails_without_writing(tmp_path, capsys, cutoff):
    inputs = make_inputs(tmp_path)
    output = tmp_path / "output"
    assert cli_main(inputs.args(output, "--allow-lag", as_of=cutoff)) == 2
    assert "[运行失败]" in capsys.readouterr().err
    assert not output.exists()


def test_lag_override_cannot_bypass_missing_corporate_actions_coverage(tmp_path, capsys):
    inputs = make_inputs(tmp_path, all_lag=True)
    ledger = json.loads(inputs.ledger.read_text(encoding="utf-8"))
    ledger["as_of"] = inputs.dates[-2]
    write_json(inputs.ledger, ledger)
    output = tmp_path / "output"
    assert cli_main(inputs.args(output, "--allow-lag", as_of=inputs.dates[-1])) == 2
    assert "公司行为台账门禁阻断正式运行" in capsys.readouterr().err
    assert not output.exists()


def test_provisional_effective_event_blocks_formal_and_dry_run_without_output(tmp_path, capsys):
    inputs = make_inputs(tmp_path)
    ledger = json.loads(inputs.ledger.read_text(encoding="utf-8"))
    ledger["splits_by_symbol"][SPLIT] = [{
        "effective_date": inputs.dates[1], "factor": 0.5,
        "source_class": "PROVISIONAL_TEST", "event_status": "PROVISIONAL",
        "source_pdf": None, "source_pdf_sha256": None,
    }]
    write_json(inputs.ledger, ledger)
    output = tmp_path / "output"

    assert cli_main(inputs.args(output, "--dry-run", as_of=inputs.dates[2])) == 2
    diagnostic = capsys.readouterr().out
    assert "[DRY-RUN][BLOCKED]" in diagnostic
    assert "PROVISIONAL" in diagnostic and "缺少官方 PDF" in diagnostic
    assert SPLIT in diagnostic
    assert not output.exists()

    assert cli_main(inputs.args(output, as_of=inputs.dates[2])) == 2
    formal_error = capsys.readouterr().err
    assert "公司行为台账门禁阻断正式运行" in formal_error
    assert SPLIT in formal_error and "缺少官方 PDF" in formal_error
    assert not output.exists()


def test_from_scratch_replaces_legacy_checkpoint_and_rebuilds_current_ledgers(tmp_path, capsys):
    inputs = make_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    legacy_state = {"schema_version": "1.0.0", "last_processed_date": "2026-11-20"}
    save_state_json(legacy_state, output / "state.json")
    (output / "trades.csv").write_bytes(b"legacy")
    weekly = output / "nav_weekly.md"
    weekly.write_bytes(b"stage 1\n")
    assert cli_main(inputs.args(output, as_of=inputs.dates[1])) == 2
    assert "--from-scratch" in capsys.readouterr().err
    assert load_state_json(output / "state.json") == legacy_state

    assert cli_main(inputs.args(output, "--from-scratch", as_of=inputs.dates[1])) == 0
    assert load_state_json(output / "state.json")["last_processed_date"] == inputs.dates[1]
    assert read_csv_rows(output / "trades.csv", TRADES_CSV_COLUMNS)
    assert weekly.read_bytes() == b"stage 1\n"


def test_rebuild_keeps_already_published_daily_snapshots(tmp_path):
    inputs = make_inputs(tmp_path)
    output = tmp_path / "output"
    assert cli_main(inputs.args(output, as_of=inputs.dates[4])) == 0
    before = {day: snapshot_path(output, day).read_bytes() for day in inputs.dates[:5]}
    watchlist = json.loads(inputs.watchlist.read_text(encoding="utf-8"))
    watchlist["universe"][1]["name"] = "更新后的名称"
    write_json(inputs.watchlist, watchlist)

    assert cli_main(inputs.args(output, "--from-scratch", as_of=inputs.dates[1])) == 0
    assert load_state_json(output / "state.json")["last_processed_date"] == inputs.dates[1]
    assert {row["date"] for row in read_csv_rows(output / "nav_daily.csv", NAV_CSV_COLUMNS)} == set(inputs.dates[:2])
    assert "更新后的名称" in (output / "nav_latest.md").read_text(encoding="utf-8")
    assert {day: snapshot_path(output, day).read_bytes() for day in before} == before


def test_reporter_merge_replaces_duplicate_keys_and_sorts_numeric_trade_ids(tmp_path):
    trade_file = tmp_path / "trades.csv"
    rows = [
        {"date": START, "account": "B2", "trade_id": 10, "note": "old"},
        {"date": START, "account": "B2", "trade_id": 2},
        {"date": START, "account": "A", "trade_id": 3},
    ]
    export_trades_csv(rows, trade_file)
    previous = read_csv_rows(trade_file, TRADES_CSV_COLUMNS)
    export_trades_csv(
        [{"date": START, "account": "B2", "trade_id": "10", "note": "new"}],
        trade_file, previous_rows=previous,
    )
    rows = read_csv_rows(trade_file, TRADES_CSV_COLUMNS)
    assert [(row["account"], row["trade_id"], row["note"]) for row in rows] == [
        ("A", "3", ""), ("B2", "2", ""), ("B2", "10", "new"),
    ]

    nav_file = tmp_path / "nav_daily.csv"
    export_nav_daily_csv([{"date": START, "account": "A", "nav": 1}], nav_file)
    previous = read_csv_rows(nav_file, NAV_CSV_COLUMNS)
    export_nav_daily_csv([{"date": START, "account": "A", "nav": 1.01}], nav_file, previous_rows=previous)
    row, = read_csv_rows(nav_file, NAV_CSV_COLUMNS)
    assert row["nav"] == "1.01"


def test_atomic_writes_leave_prior_file_intact_and_clean_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    save_state_json({"field": "旧"}, path)

    def fail(temporary, destination):
        assert temporary.parent == destination.parent
        assert load_state_json(temporary) == {"field": "新"}
        raise OSError("simulated publish error")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="publish error"):
        save_state_json({"field": "新"}, path)
    assert load_state_json(path) == {"field": "旧"}
    assert list(tmp_path.iterdir()) == [path]
    save_daily_snapshot(tmp_path, START, "第一版\n")
    save_daily_snapshot(tmp_path, START, "第二版\n")
    assert snapshot_path(tmp_path, START).read_text(encoding="utf-8") == "第一版\n"


def test_retries_after_failed_checkpoint_publication(tmp_path, monkeypatch):
    inputs = make_inputs(tmp_path)
    output = tmp_path / "output"
    assert cli_main(inputs.args(output, as_of=START)) == 0
    before = output_bytes(output)
    replace = os.replace

    def fail_only_checkpoint(temporary, destination):
        if destination.name == "state.json" and destination.parent == output:
            raise OSError("simulated interruption before checkpoint")
        replace(temporary, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_only_checkpoint)
        assert cli_main(inputs.args(output, as_of=inputs.dates[2])) == 2
    assert (output / "state.json").read_bytes() == before[Path("state.json")]
    assert read_csv_rows(output / "nav_daily.csv", NAV_CSV_COLUMNS)[-1]["date"] == inputs.dates[2]
    assert cli_main(inputs.args(output, as_of=inputs.dates[2])) == 0

    clean = tmp_path / "clean"
    assert cli_main(inputs.args(clean, "--from-scratch", as_of=inputs.dates[2])) == 0
    assert output_bytes(output) == output_bytes(clean)


def test_absolute_script_entry_point_works_from_outside_repo(tmp_path):
    inputs = make_inputs(tmp_path)
    output = tmp_path / "output"
    script = DEFAULT_SIM_CONFIG_PATH.parents[1] / "src" / "run_daily_sim.py"
    assert DEFAULT_OUTPUT_DIR == DEFAULT_SIM_CONFIG_PATH.parents[1] / "output"
    proc = subprocess.run(
        [QUANT_PYTHON, str(script), *inputs.args(output, "--dry-run", as_of=START)],
        cwd=tmp_path, env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[DRY-RUN]" in proc.stdout and START in proc.stdout
    assert not output.exists()
