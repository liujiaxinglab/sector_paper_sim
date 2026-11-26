"""Exercise the daily CLI with isolated on-disk BS-01, TDX and ledger inputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from src.data_loader import (
    CHINA_TZ,
    DEFAULT_SIM_CONFIG_PATH,
    _extract_decision_entry_block,
    compute_regime_decision_sha256,
    load_sim_config,
)
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
QUANT_PYTHON = sys.executable


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


_UNSET = object()


def attach_live_proof(
    journal_dir: Path,
    record: dict,
    *,
    receipt_mtime: datetime | None = None,
    entry_sha256: str | None | object = _UNSET,
) -> Path:
    if entry_sha256 is _UNSET:
        decision_ref = str(record.get("decision_ref") or "").strip()
        if "#" not in decision_ref:
            decision_ref = f"2026_decisions.md#{record['decision_date']}-{record['symbol']}"
            record["decision_ref"] = decision_ref
        ref_file, anchor_id = decision_ref.split("#", 1)
        ref_path = journal_dir / ref_file
        existing = ref_path.read_text(encoding="utf-8") if ref_path.is_file() else ""
        marker = f'<a id="{anchor_id}"></a>'
        if marker not in existing:
            block_text = (
                f"{marker}\n\n### [{record['decision_date']}] {record['symbol']}\n"
                f"- {record.get('new_regime') or record.get('regime')}\n"
            )
            sep = "\n" if existing and not existing.endswith("\n") else ""
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            ref_path.write_text(existing + sep + block_text, encoding="utf-8", newline="\n")
        block = _extract_decision_entry_block(ref_path.read_text(encoding="utf-8"), anchor_id)
        assert block is not None
        resolved_entry_sha: str | None = hashlib.sha256(block.encode("utf-8")).hexdigest()
    else:
        resolved_entry_sha = entry_sha256  # type: ignore[assignment]

    recorded_dt = datetime.fromisoformat(record["recorded_at"].replace("Z", "+00:00")).astimezone(CHINA_TZ)
    decision_sha = compute_regime_decision_sha256(record, recorded_dt.isoformat())
    rel_path = f"pre_market_proofs/{record['record_id']}.json"
    receipt_path = journal_dir / rel_path
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(receipt_path, {
        "schema_version": "1.0.0",
        "record_id": record["record_id"],
        "decision_date": record["decision_date"],
        "effective_date": record["effective_date"],
        "symbol": record["symbol"],
        "prev_regime": "TRANSITION" if record["prev_regime"] == "UNRATED" else record["prev_regime"],
        "new_regime": "TRANSITION" if record["new_regime"] == "UNRATED" else record["new_regime"],
        "record_type": "LIVE",
        "recorded_at": record["recorded_at"],
        "recorded_at_cst": recorded_dt.isoformat(),
        "decision_sha256": decision_sha,
        "decision_ref": record.get("decision_ref"),
        "decision_entry_sha256": resolved_entry_sha,
    })
    ts_dt = receipt_mtime if receipt_mtime is not None else recorded_dt
    os.utime(receipt_path, (ts_dt.timestamp(), ts_dt.timestamp()))
    record["pre_market_proof"] = {
        "receipt_path": rel_path,
        "decision_sha256": decision_sha,
    }
    return receipt_path


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


def make_inputs(tmp_path: Path, *, lag_symbol=None, all_lag=False, include_live_records: bool = True) -> Inputs:
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
    regime_records = [
        {
            "record_id": "RH-TEST-001",
            "decision_date": dates[1],
            "effective_date": dates[2],
            "symbol": GRID,
            "prev_regime": "RANGE_GRID",
            "new_regime": "TRANSITION",
            "record_type": "LIVE",
            "recorded_at": f"{dates[1]}T16:00:00+08:00",
        },
        {
            "record_id": "RH-TEST-002",
            "decision_date": dates[4],
            "effective_date": dates[5],
            "symbol": TREND,
            "prev_regime": "TREND_UP",
            "new_regime": "RANGE_GRID",
            "record_type": "LIVE",
            "recorded_at": f"{dates[4]}T16:00:00+08:00",
        },
        {
            "record_id": "RH-TEST-003",
            "decision_date": dates[6],
            "effective_date": dates[7],
            "symbol": UNRATED,
            "prev_regime": "TRANSITION",
            "new_regime": "TREND_UP",
            "record_type": "LIVE",
            "recorded_at": f"{dates[6]}T16:00:00+08:00",
        },
    ] if include_live_records else []
    for rec in regime_records:
        attach_live_proof(tmp_path, rec)
    write_json(regimes, {
        "schema_version": "2.0.0",
        "anchor_start_date": START,
        "initial_regimes": {
            **dict.fromkeys(SECTORS, "TRANSITION"),
            GRID: "RANGE_GRID",
            TREND: "TREND_UP",
            UNRATED: "UNRATED",
        },
        "records": regime_records,
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
    assert set(state) == {
        "schema_version", "start_date", "last_processed_date", "last_known_closes", "routing_checkpoint", "accounts",
    }
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
    inputs = make_inputs(tmp_path, include_live_records=False)
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


def test_cli_formal_rejects_backfill_and_replay_reproduces_deterministically(tmp_path, capsys):
    inputs = make_inputs(tmp_path)
    payload = json.loads(inputs.regimes.read_text(encoding="utf-8"))
    payload["records"][0]["record_type"] = "BACKFILL"
    payload["records"][0]["recorded_at"] = None
    payload["records"][0]["time_evidence_boundary"] = "synthetic backfill window"
    payload["records"][0].pop("pre_market_proof", None)
    write_json(inputs.regimes, payload)

    formal_out = tmp_path / "formal_out"
    assert cli_main(inputs.args(formal_out, "--from-scratch", as_of=inputs.dates[-1])) == 2
    assert "BACKFILL" in capsys.readouterr().err
    assert not formal_out.exists()

    replay_full = tmp_path / "replay_full"
    replay_inc = tmp_path / "replay_inc"
    assert cli_main(inputs.args(replay_full, "--mode", "replay", "--from-scratch", as_of=inputs.dates[-1])) == 0
    for day in inputs.dates:
        assert cli_main(inputs.args(replay_inc, "--mode", "replay", as_of=day)) == 0
    assert output_bytes(replay_inc) == output_bytes(replay_full)
    report = (replay_full / "nav_latest.md").read_text(encoding="utf-8")
    assert "运行模式：replay" in report
    assert "已生效 LIVE 2 条" in report
    assert "已隔离 BACKFILL 1 条" in report


def test_forward_excess_returns_exclude_backfill_initialization_period(tmp_path):
    from src.run_daily_sim import run_daily_sim

    inputs = make_inputs(tmp_path)
    payload = json.loads(inputs.regimes.read_text(encoding="utf-8"))
    payload["records"][0]["record_type"] = "BACKFILL"
    payload["records"][0]["recorded_at"] = None
    payload["records"][0]["time_evidence_boundary"] = "init backfill before forward eval"
    payload["records"][0].pop("pre_market_proof", None)
    write_json(inputs.regimes, payload)

    cfg = json.loads(inputs.config.read_text(encoding="utf-8"))
    cfg["backfill_replay_window"] = {"start_date": START, "end_date": inputs.dates[3]}
    cfg["forward_eval_start_date"] = inputs.dates[4]
    write_json(inputs.config, cfg)

    output = tmp_path / "output"
    init_res = run_daily_sim(
        config_path=inputs.config, output_dir=output, as_of=inputs.dates[3], run_mode="replay"
    )
    assert init_res["forward_excess_returns"] == {}
    assert init_res["routing_stats"]["backfill_records_excluded"] == 1
    assert init_res["routing_stats"]["live_records_applied"] == 0

    fwd_res = run_daily_sim(
        config_path=inputs.config, output_dir=output, as_of=inputs.dates[-1], run_mode="formal"
    )
    assert fwd_res["routing_stats"]["backfill_records_excluded"] == 1
    assert fwd_res["routing_stats"]["live_records_applied"] == 2
    assert fwd_res["routing_stats"]["forward_live_records_applied"] == 2

    nav_rows_list = read_csv_rows(output / "nav_daily.csv", NAV_CSV_COLUMNS)
    base = {r["account"]: float(r["nav"]) for r in nav_rows_list if r["date"] == inputs.dates[3]}
    last = {r["account"]: float(r["nav"]) for r in nav_rows_list if r["date"] == inputs.dates[-1]}
    expected_a_vs_b2 = (last["A"] / base["A"] - 1.0) - (last["B2"] / base["B2"] - 1.0)
    assert fwd_res["forward_excess_returns"]["A_vs_B2"] == pytest.approx(expected_a_vs_b2)


def test_cross_repo_t_decision_t1_effective_synthesis_consistent_across_radar_and_sim(tmp_path):
    """合成跨仓闭环测试：T(周五)收盘决定 TRANSITION->RANGE_GRID，T 日看板与模拟盘均维持 TRANSITION，T+1(周一)才显示并执行 RANGE_GRID。"""
    import importlib.util

    radar_script = (
        DEFAULT_SIM_CONFIG_PATH.parents[3]
        / "sector_radar"
        / "project_content"
        / "scripts"
        / "run_daily_board.py"
    )
    if not radar_script.is_file():
        pytest.skip("Sibling sector_radar repository not present")
    spec = importlib.util.spec_from_file_location("sector_radar_board_probe", radar_script)
    assert spec is not None and spec.loader is not None
    board = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(board)

    # 2026-11-27 = Friday (T), 2026-11-30 = Monday (T+1)
    inputs = make_inputs(tmp_path)
    friday = "2026-11-27"
    monday = "2026-11-30"
    assert friday in inputs.dates and monday in inputs.dates

    qd_root = tmp_path / "qd"
    for sym_dir in (tmp_path / "bars").iterdir():
        dst = qd_root / "raw" / "tdx" / "vipdoc_parsed" / sym_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for csv_file in sym_dir.glob("*.csv"):
            (dst / csv_file.name).write_text(csv_file.read_text(encoding="utf-8"), encoding="utf-8")
    cal_dst = qd_root / "raw" / "baostock" / "query_trade_dates"
    cal_dst.mkdir(parents=True, exist_ok=True)
    (cal_dst / inputs.calendar.name).write_text(inputs.calendar.read_text(encoding="utf-8"), encoding="utf-8")

    # Add dummy CSI / CN10Y and corporate actions ledger for Radar
    csi_dir = qd_root / "raw" / "csi" / "index_perf" / "20261208_100000"
    csi_dir.mkdir(parents=True, exist_ok=True)
    for idx_code in ("000300", "000151"):
        write_json(csi_dir / f"{idx_code}.json", {"code": "200", "msg": "Success", "data": []})
    bond_dir = qd_root / "raw" / "akshare" / "bond_china_yield" / "20261208"
    bond_dir.mkdir(parents=True, exist_ok=True)
    write_csv(bond_dir / "20261208_100000.csv", ("曲线名称", "日期", "1年", "10年"), [("中债国债收益率曲线", monday, "1.3", "1.6")])

    all_symbols = [BENCHMARK, *SECTORS]
    radar_repo = tmp_path / "radar_repo"
    cards_dir = radar_repo / "project_content" / "radar_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    for sym in SECTORS:
        (cards_dir / f"{sym}_测试.md").write_text(f"# {sym}\n\n- **更新日期**：`{friday}`\n", encoding="utf-8")

    radar_ledger = qd_root / "reference" / "etf" / "corporate_actions" / "latest" / "corporate_actions_ledger.json"
    radar_ledger.parent.mkdir(parents=True, exist_ok=True)
    write_json(radar_ledger, {
        "schema_version": "2.0.0",
        "as_of": inputs.dates[-1],
        "generated_at": "2026-12-08T10:00:00+08:00",
        "authority_rule": "TEST",
        "raw_sources": {"sse_announcements_dir": "raw/sse", "cninfo_announcements_dir": "raw/cninfo", "eastmoney_discovery_years": ["2026"]},
        "symbols": {s: {"name": s, "exchange": "SSE" if s.endswith(".SH") else "SZSE", "official_source": "test"} for s in all_symbols},
        "cash_dividends_by_symbol": {s: [] for s in all_symbols},
        "splits_by_symbol": {s: [] for s in all_symbols},
    })

    shared_regime = radar_repo / "project_content" / "journal" / "regime_history.json"
    shared_regime.parent.mkdir(parents=True, exist_ok=True)
    decisions_md = radar_repo / "project_content" / "journal" / "2026_decisions.md"
    decisions_md.write_text(
        f'<a id="{friday}-{GRID}"></a>\n\n### [{friday}] {GRID}\n\n- 冻结：切换至 RANGE_GRID。\n',
        encoding="utf-8",
    )
    live_record = {
        "record_id": "RH-20261127-001",
        "decision_date": friday,
        "effective_date": monday,
        "symbol": GRID,
        "prev_regime": "TRANSITION",
        "new_regime": "RANGE_GRID",
        "record_type": "LIVE",
        "recorded_at": "2026-11-27T16:00:00+08:00",
        "trigger_alerts": ["[新触发] 20日跌破"],
        "card_path": f"project_content/radar_cards/{GRID}_测试.md",
        "decision_ref": f"2026_decisions.md#{friday}-{GRID}",
    }
    board.create_live_pre_market_proof(
        shared_regime,
        live_record,
        receipt_mtime=datetime(2026, 11, 27, 16, 0, tzinfo=CHINA_TZ),
    )
    write_json(shared_regime, {
        "schema_version": "2.0.0",
        "anchor_start_date": START,
        "initial_regimes": dict.fromkeys(SECTORS, "TRANSITION"),
        "records": [live_record],
    })

    shared_watchlist = radar_repo / "project_content" / "config" / "watchlist.json"
    shared_watchlist.parent.mkdir(parents=True, exist_ok=True)
    write_json(shared_watchlist, {
        "benchmark": BENCHMARK,
        "paths": {
            "qri_root": str(tmp_path / "qri"),
            "vipdoc_root": str(tmp_path / "vipdoc"),
            "quantdata_root": str(qd_root),
            "corporate_actions_ledger": str(radar_ledger),
            "tdx_parsed_dir": str(qd_root / "raw" / "tdx" / "vipdoc_parsed"),
            "trade_dates_dir": str(cal_dst),
            "csi_index_perf_dir": str(qd_root / "raw" / "csi" / "index_perf"),
            "danjuan_eva_dir": str(qd_root / "raw" / "danjuan" / "index_eva_detail"),
            "bond_yield_dir": str(qd_root / "raw" / "akshare" / "bond_china_yield"),
            "board_output": "project_content/output/board_latest.md",
            "board_history_dir": "project_content/output/history",
            "regime_history": "project_content/journal/regime_history.json",
            "radar_cards_dir": "project_content/radar_cards",
        },
        "alert_rules": {
            "price_pct_lookback": 750, "breakout_lookback": 55, "breakdown_lookback": 20,
            "excess_20d_abs_pct": 5.0, "atr_window": 20, "vol_ratio_window": 20,
            "vol_surge_ratio_20d": 1.8, "valuation_lookback_years": 5,
            "valuation_low_pct": 20.0, "valuation_high_pct": 80.0,
            "valuation_stale_days": 45, "overdue_review_trade_days": 20,
        },
        "universe": [
            {
                "symbol": s,
                "name": f"ETF {s}",
                "role": "benchmark" if s == BENCHMARK else "sector",
                "valuation_metric": "PE",
                "valuation_index_name": s,
                "valuation_index_code": "000300" if s == BENCHMARK else "000151",
                "danjuan_index_code": None,
                "current_regime": "BENCHMARK" if s == BENCHMARK else "TRANSITION",
                "manual_valuation": {"pe_pct": None, "pb_pct": None, "ps_pct": None, "as_of": None, "force": False, "note": ""},
            }
            for s in all_symbols
        ],
    })

    sim_cfg = json.loads(inputs.config.read_text(encoding="utf-8"))
    sim_cfg["upstream_paths"]["corporate_actions_ledger_file"] = str(radar_ledger)
    sim_cfg["upstream_paths"]["regime_history_file"] = str(shared_regime)
    sim_cfg["upstream_paths"]["watchlist_file"] = str(shared_watchlist)
    write_json(inputs.config, sim_cfg)
    sim_out = tmp_path / "sim_out"

    # On Friday (T): public Radar CLI sync & board, plus Paper Sim CLI all keep TRANSITION
    assert board.main(["--config", str(shared_watchlist), "--sync-watchlist-regime", "--as-of", friday]) == 0
    assert board.main(["--config", str(shared_watchlist), "--as-of", friday, "--dry-run"]) == 0
    radar_friday = board.run_daily_board(config_path=shared_watchlist, as_of=friday, dry_run=True, repo_root=radar_repo)
    assert next(r for r in radar_friday["results"] if r["symbol"] == GRID)["current_regime"] == "TRANSITION"
    assert cli_main(inputs.args(sim_out, as_of=friday)) == 0
    sim_state_friday = load_state_json(sim_out / "state.json")["accounts"]["A"]
    assert sim_state_friday["symbol_states"][GRID]["active_regime"] == "TRANSITION"
    assert sim_state_friday["positions"][GRID]["shares"] == 0
    assert [t for t in read_csv_rows(sim_out / "trades.csv", TRADES_CSV_COLUMNS) if t["account"] == "A"] == []

    # On Monday (T+1): first public Radar CLI --sync-watchlist-regime --as-of monday returns 0, board shows RANGE_GRID, Paper Sim opens RANGE_GRID
    assert board.main(["--config", str(shared_watchlist), "--sync-watchlist-regime", "--as-of", monday]) == 0
    assert board.main(["--config", str(shared_watchlist), "--as-of", monday]) == 0
    assert f"| `{GRID}` |" in (radar_repo / "project_content" / "output" / "board_latest.md").read_text(encoding="utf-8")
    assert "`RANGE_GRID`" in (radar_repo / "project_content" / "output" / "board_latest.md").read_text(encoding="utf-8")
    radar_monday = board.run_daily_board(config_path=shared_watchlist, as_of=monday, dry_run=True, repo_root=radar_repo)
    assert next(r for r in radar_monday["results"] if r["symbol"] == GRID)["current_regime"] == "RANGE_GRID"
    assert cli_main(inputs.args(sim_out, as_of=monday)) == 0
    sim_state_monday = load_state_json(sim_out / "state.json")["accounts"]["A"]
    assert sim_state_monday["symbol_states"][GRID]["active_regime"] == "RANGE_GRID"
    assert sim_state_monday["positions"][GRID]["shares"] > 0
    a_trades = [t for t in read_csv_rows(sim_out / "trades.csv", TRADES_CSV_COLUMNS) if t["account"] == "A"]
    assert [(t["date"], t["symbol"], t["action"]) for t in a_trades] == [(monday, GRID, "GRID_INIT_BUY")]


def test_cli_formal_blocks_unproven_late_and_illegal_offset_live_records(tmp_path, capsys):
    inputs = make_inputs(tmp_path, include_live_records=False)
    base_regimes = {
        "schema_version": "2.0.0",
        "anchor_start_date": START,
        "initial_regimes": dict.fromkeys(SECTORS, "TRANSITION"),
        "records": [
            {
                "record_id": "RH-LIVE-EDGE-001",
                "decision_date": START,
                "effective_date": inputs.dates[1],
                "symbol": GRID,
                "prev_regime": "TRANSITION",
                "new_regime": "RANGE_GRID",
                "record_type": "LIVE",
                "recorded_at": "2026-11-23T16:00:00+08:00",
            }
        ],
    }

    # 1. Missing pre_market_proof -> exit 2
    write_json(inputs.regimes, base_regimes)
    out = tmp_path / "out_missing"
    assert cli_main(inputs.args(out, "--from-scratch", as_of=inputs.dates[1])) == 2
    assert "pre_market_proof" in capsys.readouterr().err
    assert not out.exists()

    # 2. Post-market receipt file write time (>= 09:30+08:00) -> exit 2
    with_late_receipt = json.loads(json.dumps(base_regimes))
    attach_live_proof(tmp_path, with_late_receipt["records"][0], receipt_mtime=datetime(2026, 11, 26, 20, 48, 0, tzinfo=CHINA_TZ))
    write_json(inputs.regimes, with_late_receipt)
    assert cli_main(inputs.args(out, "--from-scratch", as_of=inputs.dates[1])) == 2
    assert "market open" in capsys.readouterr().err
    assert not out.exists()

    # 3. Post-market Git commit timestamp fails even if receipt file mtime is older -> exit 2
    git_dir = tmp_path / "git_late"
    git_dir.mkdir()
    git_rec = json.loads(json.dumps(base_regimes))
    git_receipt = attach_live_proof(git_dir, git_rec["records"][0], receipt_mtime=datetime(2026, 11, 23, 16, 0, tzinfo=CHINA_TZ))
    git_regimes = git_dir / "regime_history.json"
    write_json(git_regimes, git_rec)
    git_cfg = json.loads(inputs.config.read_text(encoding="utf-8"))
    git_cfg["upstream_paths"]["regime_history_file"] = str(git_regimes)
    git_cfg_path = tmp_path / "sim_config_git.json"
    write_json(git_cfg_path, git_cfg)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    for g_args, g_ts in [
        (["init"], None),
        (["config", "user.name", "Tester"], None),
        (["config", "user.email", "tester@example.com"], None),
        (["add", "."], None),
        (["commit", "-m", "post-market commit"], "2026-11-26T20:48:00+08:00"),
    ]:
        g_env = dict(env)
        if g_ts:
            g_env["GIT_AUTHOR_DATE"] = g_ts
            g_env["GIT_COMMITTER_DATE"] = g_ts
        assert subprocess.run(["git", "-C", str(git_dir), *g_args], env=g_env, capture_output=True, check=False).returncode == 0
    pre_ts = datetime(2026, 11, 23, 16, 0, 0, tzinfo=CHINA_TZ).timestamp()
    os.utime(git_receipt, (pre_ts, pre_ts))
    assert cli_main(["--config", str(git_cfg_path), "--output-dir", str(out), "--from-scratch", "--as-of", inputs.dates[1]]) == 2
    assert "Git commit timestamp" in capsys.readouterr().err
    assert not out.exists()

    # 4. Illegal ISO offset +07:60 -> exit 2
    bad_offset = json.loads(json.dumps(base_regimes))
    bad_offset["records"][0]["recorded_at"] = "2026-11-23T16:00:00+07:60"
    write_json(inputs.regimes, bad_offset)
    assert cli_main(inputs.args(out, "--from-scratch", as_of=inputs.dates[1])) == 2
    assert "ISO 8601" in capsys.readouterr().err

    # 5. Exact market open (09:30:00+08:00) -> exit 2
    at_open = json.loads(json.dumps(base_regimes))
    at_open["records"][0]["recorded_at"] = f"{inputs.dates[1]}T09:30:00+08:00"
    write_json(inputs.regimes, at_open)
    assert cli_main(inputs.args(out, "--from-scratch", as_of=inputs.dates[1])) == 2
    assert "market open" in capsys.readouterr().err

    # 6. Valid UTC equivalent time (01:29:59Z = 09:29:59+08:00) + pre-market local proof -> exit 0
    utc_ok = json.loads(json.dumps(base_regimes))
    utc_ok["records"][0]["recorded_at"] = f"{inputs.dates[1]}T01:29:59Z"
    attach_live_proof(tmp_path, utc_ok["records"][0], receipt_mtime=datetime(2026, 11, 24, 9, 29, 59, tzinfo=CHINA_TZ))
    write_json(inputs.regimes, utc_ok)
    assert cli_main(inputs.args(out, "--from-scratch", as_of=inputs.dates[1])) == 0


def test_cli_formal_resume_blocks_appended_prior_backfill_and_preserves_output_bytes(tmp_path, capsys):
    """复现并回归问题 2：forward_eval_start_date=null 时，正式续跑与增量 replay 均严禁认证检查点之前新增的 BACKFILL，且失败时输出保持原字节。"""
    inputs = make_inputs(tmp_path, include_live_records=False)
    wednesday = "2026-11-25"
    thursday = "2026-11-26"
    monday = "2026-11-30"
    tuesday = "2026-12-01"
    cfg = json.loads(inputs.config.read_text(encoding="utf-8"))
    cfg["backfill_replay_window"] = {"start_date": START, "end_date": "2026-11-26"}
    cfg["forward_eval_start_date"] = None
    write_json(inputs.config, cfg)

    write_json(inputs.regimes, {
        "schema_version": "2.0.0",
        "anchor_start_date": START,
        "initial_regimes": dict.fromkeys(SECTORS, "TRANSITION"),
        "records": [],
    })
    output = tmp_path / "formal_resume_out"
    assert cli_main(inputs.args(output, "--from-scratch", as_of=monday)) == 0
    before = output_bytes(output)

    # Append a BACKFILL taking effect at 2026-11-26 (inside window, <= checkpoint 2026-11-30)
    write_json(inputs.regimes, {
        "schema_version": "2.0.0",
        "anchor_start_date": START,
        "initial_regimes": dict.fromkeys(SECTORS, "TRANSITION"),
        "records": [
            {
                "record_id": "RH-20261125-BF",
                "decision_date": wednesday,
                "effective_date": thursday,
                "symbol": GRID,
                "prev_regime": "TRANSITION",
                "new_regime": "RANGE_GRID",
                "record_type": "BACKFILL",
                "recorded_at": None,
                "time_evidence_boundary": "post-hoc appended backfill",
            }
        ],
    })
    # Direct formal resume -> blocked & byte-identical
    assert cli_main(inputs.args(output, as_of=tuesday)) == 2
    assert "BACKFILL" in capsys.readouterr().err
    assert output_bytes(output) == before

    # Incremental replay (without --from-scratch) must ALSO reject without certifying un-replayed 11-26 history!
    assert cli_main(inputs.args(output, "--mode", "replay", as_of=tuesday)) == 2
    assert "BACKFILL" in capsys.readouterr().err
    assert output_bytes(output) == before

    # True --from-scratch replay recalculates from 11-23 and opens GRID_INIT_BUY on 11-26
    full_rebuilt = tmp_path / "rebuilt_full"
    assert cli_main(inputs.args(full_rebuilt, "--mode", "replay", "--from-scratch", as_of=tuesday)) == 0
    a_trades = [t for t in read_csv_rows(full_rebuilt / "trades.csv", TRADES_CSV_COLUMNS) if t["account"] == "A"]
    assert a_trades and (a_trades[0]["date"], a_trades[0]["symbol"], a_trades[0]["action"]) == (thursday, GRID, "GRID_INIT_BUY")


def test_cli_valid_replay_init_then_formal_resume_blocks_edit_delete_and_missing_window(tmp_path, capsys):
    inputs = make_inputs(tmp_path, include_live_records=False)
    cfg = json.loads(inputs.config.read_text(encoding="utf-8"))
    cfg["backfill_replay_window"] = {"start_date": START, "end_date": "2026-11-26"}
    cfg["forward_eval_start_date"] = None
    write_json(inputs.config, cfg)

    bf_rec = {
        "record_id": "RH-BF-001",
        "decision_date": inputs.dates[1],
        "effective_date": inputs.dates[2],  # 2026-11-25, inside 2026-11-23..2026-11-26
        "symbol": GRID,
        "prev_regime": "TRANSITION",
        "new_regime": "TRANSITION",
        "record_type": "BACKFILL",
        "recorded_at": None,
        "time_evidence_boundary": "verified replay window backfill",
        "trigger_alerts": ["[新触发] 20日跌破"],
        "card_path": "project_content/radar_cards/510720.SH_红利国企.md",
        "decision_ref": "2026_decisions.md#2026-11-24-510720.SH",
    }
    live_rec = {
        "record_id": "RH-LIVE-001",
        "decision_date": "2026-11-27",
        "effective_date": "2026-11-30",
        "symbol": GRID,
        "prev_regime": "TRANSITION",
        "new_regime": "RANGE_GRID",
        "record_type": "LIVE",
        "recorded_at": "2026-11-27T16:00:00+08:00",
    }
    attach_live_proof(tmp_path, live_rec, receipt_mtime=datetime(2026, 11, 27, 16, 0, tzinfo=CHINA_TZ))
    payload = {
        "schema_version": "2.0.0",
        "anchor_start_date": START,
        "initial_regimes": dict.fromkeys(SECTORS, "TRANSITION"),
        "records": [bf_rec, live_rec],
    }
    write_json(inputs.regimes, payload)

    output = tmp_path / "valid_resume_out"
    assert cli_main(inputs.args(output, "--mode", "replay", "--from-scratch", as_of="2026-11-26")) == 0
    init_before = output_bytes(output)

    # 1. Modifying ONLY decision_ref & trigger_alerts on replayed BACKFILL must fail formal resume & preserve bytes
    meta_tampered = json.loads(json.dumps(payload))
    meta_tampered["records"][0]["decision_ref"] = "2026_decisions.md#TAMPERED"
    meta_tampered["records"][0]["trigger_alerts"] = ["[新触发] 改写"]
    write_json(inputs.regimes, meta_tampered)
    assert cli_main(inputs.args(output, as_of="2026-11-27")) == 2
    assert "BACKFILL" in capsys.readouterr().err
    assert output_bytes(output) == init_before

    # 2. Deleting the already-replayed BACKFILL must fail formal resume & preserve bytes
    deleted = json.loads(json.dumps(payload))
    deleted["records"] = [live_rec]
    write_json(inputs.regimes, deleted)
    assert cli_main(inputs.args(output, as_of="2026-11-27")) == 2
    assert "BACKFILL" in capsys.readouterr().err
    assert output_bytes(output) == init_before

    # 3. Setting backfill_replay_window=null when BACKFILLs exist must fail formal resume & preserve bytes
    write_json(inputs.regimes, payload)
    cfg_null_win = json.loads(json.dumps(cfg))
    cfg_null_win["backfill_replay_window"] = None
    write_json(inputs.config, cfg_null_win)
    assert cli_main(inputs.args(output, as_of="2026-11-27")) == 2
    assert "backfill_replay_window" in capsys.readouterr().err
    assert output_bytes(output) == init_before

    # 4. Restore valid window -> formal resume succeeds cleanly
    write_json(inputs.config, cfg)
    assert cli_main(inputs.args(output, as_of=inputs.dates[-1])) == 0
    state_a = load_state_json(output / "state.json")["accounts"]["A"]
    assert state_a["symbol_states"][GRID]["active_regime"] == "RANGE_GRID"
    assert state_a["positions"][GRID]["shares"] > 0

    # 5. Legacy checkpoint without routing_checkpoint must fail incremental replay & preserve bytes
    legacy_out = tmp_path / "legacy_out"
    assert cli_main(inputs.args(legacy_out, "--mode", "replay", "--from-scratch", as_of="2026-11-26")) == 0
    legacy_state = load_state_json(legacy_out / "state.json")
    legacy_state.pop("routing_checkpoint", None)
    save_state_json(legacy_state, legacy_out / "state.json")
    legacy_before = output_bytes(legacy_out)
    assert cli_main(inputs.args(legacy_out, "--mode", "replay", as_of="2026-11-27")) == 2
    assert "旧检查点" in capsys.readouterr().err
    assert output_bytes(legacy_out) == legacy_before

    # 6. Empty BACKFILL set with backfill_replay_window=null is allowed in formal resume
    empty_inputs = make_inputs(tmp_path / "empty_case", include_live_records=False)
    empty_cfg = json.loads(empty_inputs.config.read_text(encoding="utf-8"))
    empty_cfg["backfill_replay_window"] = None
    empty_cfg["forward_eval_start_date"] = None
    write_json(empty_inputs.config, empty_cfg)
    empty_out = tmp_path / "empty_out"
    assert cli_main(empty_inputs.args(empty_out, "--from-scratch", as_of="2026-11-26")) == 0
    assert cli_main(empty_inputs.args(empty_out, as_of="2026-11-27")) == 0


def test_cli_formal_rejects_missing_ledger_anchor_empty_sha_or_tampered_text_without_writing(tmp_path, capsys):
    inputs = make_inputs(tmp_path, include_live_records=False)
    cfg = json.loads(inputs.config.read_text(encoding="utf-8"))
    cfg["backfill_replay_window"] = None
    cfg["forward_eval_start_date"] = None
    write_json(inputs.config, cfg)
    write_json(inputs.regimes, {
        "schema_version": "2.0.0",
        "anchor_start_date": START,
        "initial_regimes": dict.fromkeys(SECTORS, "TRANSITION"),
        "records": [],
    })

    out_resume = tmp_path / "resume_out"
    assert cli_main(inputs.args(out_resume, "--from-scratch", as_of=START)) == 0
    resume_before = output_bytes(out_resume)

    decisions_path = tmp_path / "2026_decisions.md"
    anchor_id = f"{START}-{GRID}"
    valid_decisions = f'<a id="{anchor_id}"></a>\n\n### [{START}] {GRID}\n- RANGE_GRID\n'

    def _write_live_case(decision_ref: str | None, *, entry_sha: str | None | object = _UNSET) -> Path:
        rec: dict = {
            "record_id": "RH-LIVE-ANCHOR-001",
            "decision_date": START,
            "effective_date": inputs.dates[1],
            "symbol": GRID,
            "prev_regime": "TRANSITION",
            "new_regime": "RANGE_GRID",
            "record_type": "LIVE",
            "recorded_at": f"{START}T16:00:00+08:00",
        }
        if decision_ref is not None:
            rec["decision_ref"] = decision_ref
        receipt_p = attach_live_proof(
            tmp_path,
            rec,
            receipt_mtime=datetime(2026, 11, 23, 16, 0, tzinfo=CHINA_TZ),
            entry_sha256=entry_sha,
        )
        write_json(inputs.regimes, {
            "schema_version": "2.0.0",
            "anchor_start_date": START,
            "initial_regimes": dict.fromkeys(SECTORS, "TRANSITION"),
            "records": [rec],
        })
        return receipt_p

    out_fresh = tmp_path / "fresh_out"

    # 1. Missing #anchor (file name only / trailing # / empty) with null decision_entry_sha256 -> exit 2, no writes
    decisions_path.write_text(valid_decisions, encoding="utf-8")
    for bad_ref in ("2026_decisions.md", "2026_decisions.md#", "#anchor", ""):
        _write_live_case(bad_ref, entry_sha=None)
        assert cli_main(inputs.args(out_fresh, "--from-scratch", as_of=inputs.dates[1])) == 2
        assert not out_fresh.exists()
        assert cli_main(inputs.args(out_resume, as_of=inputs.dates[1])) == 2
        assert output_bytes(out_resume) == resume_before
        capsys.readouterr()

    # 2. Valid #anchor syntax but decision ledger file missing (with null or non-null entry sha) -> exit 2, no writes
    decisions_path.unlink()
    for sha_val in (None, "a" * 64):
        _write_live_case(f"2026_decisions.md#{anchor_id}", entry_sha=sha_val)
        assert cli_main(inputs.args(out_fresh, "--from-scratch", as_of=inputs.dates[1])) == 2
        assert not out_fresh.exists()
        assert cli_main(inputs.args(out_resume, as_of=inputs.dates[1])) == 2
        assert output_bytes(out_resume) == resume_before
        capsys.readouterr()

    # 3. Ledger exists, #anchor missing -> exit 2, no writes
    decisions_path.write_text("# empty decisions\n", encoding="utf-8")
    _write_live_case(f"2026_decisions.md#{anchor_id}", entry_sha="a" * 64)
    assert cli_main(inputs.args(out_fresh, "--from-scratch", as_of=inputs.dates[1])) == 2
    assert not out_fresh.exists()
    assert cli_main(inputs.args(out_resume, as_of=inputs.dates[1])) == 2
    assert output_bytes(out_resume) == resume_before
    capsys.readouterr()

    # 4. Valid anchor + receipt, then set decision_entry_sha256=None/"" or tamper text -> exit 2, no writes
    decisions_path.write_text(valid_decisions, encoding="utf-8")
    receipt_p = _write_live_case(f"2026_decisions.md#{anchor_id}")
    saved_receipt = json.loads(receipt_p.read_text(encoding="utf-8"))
    for empty_val in (None, ""):
        bad_receipt = dict(saved_receipt)
        bad_receipt["decision_entry_sha256"] = empty_val
        write_json(receipt_p, bad_receipt)
        os.utime(receipt_p, (datetime(2026, 11, 23, 16, 0, tzinfo=CHINA_TZ).timestamp(),) * 2)
        assert cli_main(inputs.args(out_fresh, "--from-scratch", as_of=inputs.dates[1])) == 2
        assert "decision_entry_sha256" in capsys.readouterr().err
        assert not out_fresh.exists()
        assert cli_main(inputs.args(out_resume, as_of=inputs.dates[1])) == 2
        assert output_bytes(out_resume) == resume_before
        capsys.readouterr()
    write_json(receipt_p, saved_receipt)
    os.utime(receipt_p, (datetime(2026, 11, 23, 16, 0, tzinfo=CHINA_TZ).timestamp(),) * 2)

    # Also reject if pre_market_proof pointer itself embeds empty/null decision_entry_sha256
    reg_payload = json.loads(inputs.regimes.read_text(encoding="utf-8"))
    reg_payload["records"][0]["pre_market_proof"]["decision_entry_sha256"] = None
    write_json(inputs.regimes, reg_payload)
    assert cli_main(inputs.args(out_fresh, "--from-scratch", as_of=inputs.dates[1])) == 2
    assert not out_fresh.exists()
    assert cli_main(inputs.args(out_resume, as_of=inputs.dates[1])) == 2
    assert output_bytes(out_resume) == resume_before
    capsys.readouterr()
    reg_payload["records"][0]["pre_market_proof"].pop("decision_entry_sha256", None)
    write_json(inputs.regimes, reg_payload)

    decisions_path.write_text(valid_decisions.replace("RANGE_GRID", "TRAP_OFF"), encoding="utf-8")
    assert cli_main(inputs.args(out_fresh, "--from-scratch", as_of=inputs.dates[1])) == 2
    assert "decision_entry_sha256" in capsys.readouterr().err
    assert not out_fresh.exists()
    assert cli_main(inputs.args(out_resume, as_of=inputs.dates[1])) == 2
    assert output_bytes(out_resume) == resume_before
    capsys.readouterr()

    # 5. Restored valid text passes formal CLI cleanly
    decisions_path.write_text(valid_decisions, encoding="utf-8")
    assert cli_main(inputs.args(out_resume, as_of=inputs.dates[1])) == 0


def test_relocated_sibling_workspace_cli_works_from_repo_root_and_external_cwd(tmp_path):
    """Copy code and relative sim_config.json into a relocated sibling workspace and run CLI without --config/--output-dir."""
    inputs = make_inputs(tmp_path, include_live_records=False)
    ws_root = tmp_path / "relocated_ws" / "nested"
    sim_repo = ws_root / "sector_paper_sim"
    radar_repo = ws_root / "sector_radar"
    qd_root = ws_root / "QuantData"
    unrelated_cwd = tmp_path / "unrelated_cwd"
    unrelated_cwd.mkdir(parents=True, exist_ok=True)

    src_repo_root = DEFAULT_SIM_CONFIG_PATH.parents[2]
    dst_src = sim_repo / "project_content" / "src"
    shutil.copytree(src_repo_root / "project_content" / "src", dst_src)
    dst_cfg = sim_repo / "project_content" / "config" / "sim_config.json"
    dst_cfg.parent.mkdir(parents=True, exist_ok=True)
    sim_cfg = json.loads(DEFAULT_SIM_CONFIG_PATH.read_text(encoding="utf-8"))
    sim_cfg["backfill_replay_window"] = {"start_date": START, "end_date": inputs.dates[2]}
    sim_cfg["forward_eval_start_date"] = None
    write_json(dst_cfg, sim_cfg)

    for sym_dir in (tmp_path / "bars").iterdir():
        sym_dst = qd_root / "raw" / "tdx" / "vipdoc_parsed" / sym_dir.name
        sym_dst.mkdir(parents=True, exist_ok=True)
        for csv_file in sym_dir.glob("*.csv"):
            shutil.copy2(csv_file, sym_dst / csv_file.name)
    cal_dst = qd_root / "raw" / "baostock" / "query_trade_dates" / inputs.calendar.name
    cal_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs.calendar, cal_dst)
    ledger_dst = qd_root / "reference" / "etf" / "corporate_actions" / "latest" / "corporate_actions_ledger.json"
    ledger_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs.ledger, ledger_dst)

    regime_dst = radar_repo / "project_content" / "journal" / "regime_history.json"
    regime_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs.regimes, regime_dst)
    watchlist_dst = radar_repo / "project_content" / "config" / "watchlist.json"
    watchlist_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs.watchlist, watchlist_dst)

    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    help_proc = subprocess.run(
        [sys.executable, "project_content/src/run_daily_sim.py", "--help"],
        cwd=sim_repo,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert help_proc.returncode == 0, help_proc.stderr
    assert "--config" in help_proc.stdout and "--output-dir" in help_proc.stdout and "--mode" in help_proc.stdout

    root_run = subprocess.run(
        [
            sys.executable,
            "project_content/src/run_daily_sim.py",
            "--mode",
            "replay",
            "--from-scratch",
            "--as-of",
            inputs.dates[1],
        ],
        cwd=sim_repo,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert root_run.returncode == 0, root_run.stderr
    relocated_output = sim_repo / "project_content" / "output"
    assert (relocated_output / "state.json").is_file()
    assert (relocated_output / "nav_daily.csv").is_file()

    ext_dry = subprocess.run(
        [
            sys.executable,
            str(dst_src / "run_daily_sim.py"),
            "--mode",
            "replay",
            "--as-of",
            inputs.dates[2],
            "--dry-run",
        ],
        cwd=unrelated_cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert ext_dry.returncode == 0, ext_dry.stderr
    assert "[DRY-RUN]" in ext_dry.stdout and inputs.dates[2] in ext_dry.stdout
    assert list(unrelated_cwd.iterdir()) == []
