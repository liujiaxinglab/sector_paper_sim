"""Unit tests for BS-01 calendar, corporate-actions, and route-history loading."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from src.data_loader import (
    CHINA_TZ,
    DEFAULT_SIM_CONFIG_PATH,
    TDX_NUMERIC_COLUMNS,
    TDX_REQUIRED_COLUMNS,
    RegimeHistory,
    build_forward_adjusted_daily,
    compute_regime_decision_sha256,
    find_latest_tdx_csv,
    load_corporate_actions_ledger,
    load_market_bundle,
    load_regime_history,
    load_sim_config,
    load_tdx_daily,
    load_trade_calendar,
    assess_corporate_actions_gate,
)

EXPECTED_SYMBOLS = [
    "159330.SZ",
    "512400.SH",
    "510720.SH",
    "159622.SZ",
    "515790.SH",
    "512480.SH",
    "512880.SH",
]
ISO_DAYS = ["2026-11-23", "2026-11-24", "2026-11-25", "2026-11-26"]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_calendar(directory: Path, trading_dates: list[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rows = pd.date_range(trading_dates[0], trading_dates[-1], freq="D")
    path = directory / "20261126_100000.csv"
    path.write_text(
        "calendar_date,is_trading_day\n"
        + "".join(
            f"{day:%Y-%m-%d},{int(day.strftime('%Y-%m-%d') in trading_dates)}\n"
            for day in rows
        ),
        encoding="utf-8",
    )
    return path


def _write_tdx_batch(directory: Path, symbol: str, dates: list[str]) -> Path:
    symbol_dir = directory / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    path = symbol_dir / "20261126_100000.csv"
    path.write_text(
        "date,open,high,low,close,amount,volume,reserved\n"
        + "".join(
            f"{day},1.00,1.02,0.98,1.01,1000.0,1000,0\n" for day in dates
        ),
        encoding="utf-8",
    )
    return path


def _make_synthetic_config(tmp_path: Path, bar_dates: dict[str, list[str]]) -> dict:
    tdx_dir = tmp_path / "tdx"
    symbols = list(bar_dates)
    sectors = symbols[1:]
    for symbol, dates in bar_dates.items():
        _write_tdx_batch(tdx_dir, symbol, dates)
    calendar_dir = tmp_path / "calendar"
    _write_calendar(calendar_dir, ISO_DAYS)

    watchlist_file = tmp_path / "watchlist.json"
    _write_json(
        watchlist_file,
        {
            "universe": [
                {
                    "symbol": symbol,
                    "name": symbol,
                    "role": "benchmark" if index == 0 else "sector",
                    "current_regime": "BENCHMARK" if index == 0 else "UNRATED",
                }
                for index, symbol in enumerate(symbols)
            ]
        },
    )
    ledger_file = tmp_path / "corporate_actions_ledger.json"
    _write_json(
        ledger_file,
        {
            "schema_version": "2.0.0",
            "as_of": "2026-11-26",
            "cash_dividends_by_symbol": {symbol: [] for symbol in symbols},
            "splits_by_symbol": {symbol: [] for symbol in symbols},
        },
    )
    regime_file = tmp_path / "regime_history.json"
    _write_json(
        regime_file,
        {
            "schema_version": "2.0.0",
            "anchor_start_date": "2026-11-23",
            "initial_regimes": {symbol: "UNRATED" for symbol in sectors},
            "records": [],
        },
    )
    return {
        "project_name": "sector_paper_sim_test",
        "start_date": "2026-11-23",
        "upstream_paths": {
            "tdx_daily_dir": str(tdx_dir),
            "trade_calendar_dir": str(calendar_dir),
            "corporate_actions_ledger_file": str(ledger_file),
            "regime_history_file": str(regime_file),
            "watchlist_file": str(watchlist_file),
        },
        "account_defaults": {"initial_capital": 100000.0},
        "strategies": {},
        "universe": {"benchmark": symbols[0], "sectors": sectors},
    }


def test_config_is_relocated_and_uses_the_v2_contract() -> None:
    config = load_sim_config(DEFAULT_SIM_CONFIG_PATH)
    assert DEFAULT_SIM_CONFIG_PATH == (
        Path(__file__).resolve().parents[1] / "config" / "sim_config.json"
    )
    assert config["start_date"] == "2026-11-23"
    assert set(config["upstream_paths"]) == {
        "tdx_daily_dir",
        "trade_calendar_dir",
        "corporate_actions_ledger_file",
        "regime_history_file",
        "watchlist_file",
    }
    assert config["strategies"]["grid"] == {
        "step_mode": "atr20_locked",
        "atr_window": 20,
        "min_step_pct": 0.015,
        "grid_levels_each_side": 4,
        "total_slices": 8,
        "initial_slices": 4,
        "initial_position_ratio": 0.5,
    }
    assert "risk_fraction" not in config["strategies"]["turtle"]


def test_real_market_bundle_uses_bs01_calendar_and_ledger() -> None:
    bundle = load_market_bundle(DEFAULT_SIM_CONFIG_PATH)

    assert bundle.benchmark == EXPECTED_SYMBOLS[0]
    assert set(bundle.sectors) == set(EXPECTED_SYMBOLS[1:])
    assert set(bundle.symbols) == set(EXPECTED_SYMBOLS)
    assert set(bundle.bars) == set(EXPECTED_SYMBOLS)
    calendar_file = Path(bundle.metadata["calendar_file"])
    assert calendar_file.is_file()
    assert datetime.strptime(
        calendar_file.name, "%Y%m%d_%H%M%S.csv"
    ).strftime("%Y%m%d_%H%M%S.csv") == calendar_file.name
    corporate_actions_as_of = bundle.metadata["corporate_actions_as_of"]
    corporate_actions_as_of_date = date.fromisoformat(corporate_actions_as_of)
    assert corporate_actions_as_of_date.isoformat() == corporate_actions_as_of
    configured_start_date = date.fromisoformat(
        load_sim_config(DEFAULT_SIM_CONFIG_PATH)["start_date"]
    )
    assert corporate_actions_as_of_date >= date.fromisoformat(bundle.common_end_date)
    assert corporate_actions_as_of_date >= configured_start_date
    assert bundle.regime_history.anchor_start_date == "2026-11-23"
    assert bundle.authoritative_trading_dates == sorted(
        bundle.authoritative_trading_dates
    )
    assert "2026-11-23" in bundle.authoritative_trading_dates
    assert bundle.common_trading_dates[-1] == bundle.common_end_date

    for symbol in EXPECTED_SYMBOLS:
        bars = bundle.bars[symbol]
        assert list(bars.columns) == list(TDX_REQUIRED_COLUMNS)
        assert bars["date"].tolist() == sorted(set(bars["date"].tolist()))
        for column in TDX_NUMERIC_COLUMNS:
            assert pd.api.types.is_float_dtype(bars[column])
        assert symbol in bundle.splits
        assert symbol in bundle.dividends
        assert bundle.adjusted_bars[symbol]["adj_factor"].iloc[-1] == pytest.approx(1.0)

    assert bundle.regime_history.regime_for("510720.SH", "2026-11-23") == "TRANSITION"


def test_bs01_calendar_keeps_market_day_when_one_symbol_has_no_bar(tmp_path: Path) -> None:
    config = _make_synthetic_config(
        tmp_path,
        {
            "159330.SZ": ["2026-11-23", "2026-11-25", "2026-11-26"],
            "512400.SH": ISO_DAYS,
        },
    )
    bundle = load_market_bundle(config)

    assert bundle.common_trading_dates == ISO_DAYS
    assert bundle.authoritative_trading_dates == ISO_DAYS
    assert bundle.suspension_dates_by_symbol == {"159330.SZ": ["2026-11-24"]}
    assert bundle.metadata["missing_trading_dates"]["159330.SZ"] == ["2026-11-24"]
    assert "2026-11-24" not in bundle.bars["159330.SZ"]["date"].tolist()
    assert "2026-11-24" in bundle.bars["512400.SH"]["date"].tolist()


def test_latest_csv_lag_is_separate_from_in_range_suspension(tmp_path: Path) -> None:
    config = _make_synthetic_config(
        tmp_path,
        {
            "159330.SZ": ISO_DAYS,
            "512400.SH": ISO_DAYS[:-1],
        },
    )
    bundle = load_market_bundle(config)

    assert bundle.common_end_date == "2026-11-25"
    assert bundle.common_trading_dates == ISO_DAYS[:-1]
    assert bundle.lag_warnings[0]["symbol"] == "512400.SH"
    assert bundle.lag_warnings[0]["lag_trading_days"] == 1
    assert bundle.lagged_dates_by_symbol["512400.SH"] == ["2026-11-26"]
    assert "512400.SH" not in bundle.suspension_dates_by_symbol


def test_corporate_actions_ledger_sorts_events_and_validates_schema(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    _write_json(
        ledger_path,
        {
            "schema_version": "2.0.0",
            "as_of": "2026-11-26",
            "cash_dividends_by_symbol": {
                "510720.SH": [
                    {
                        "ex_date": "2026-11-25", "cash_per_unit": 0.02,
                        "source_class": "OFFICIAL_TEST", "event_status": "VERIFIED",
                        "source_pdf": "raw/test/dividend.pdf", "source_pdf_sha256": "a" * 64,
                    },
                    {
                        "ex_date": "2026-11-23", "cash_per_unit": 0.01,
                        "source_class": "OFFICIAL_TEST", "event_status": "VERIFIED",
                        "source_pdf": "raw/test/dividend.pdf", "source_pdf_sha256": "a" * 64,
                    },
                ]
            },
            "splits_by_symbol": {
                "512480.SH": [
                    {
                        "effective_date": "2026-11-25", "factor": 0.5,
                        "source_class": "OFFICIAL_TEST", "event_status": "VERIFIED",
                        "source_pdf": "raw/test/split.pdf", "source_pdf_sha256": "b" * 64,
                    },
                    {
                        "effective_date": "2026-11-23", "factor": 0.8,
                        "source_class": "OFFICIAL_TEST", "event_status": "VERIFIED",
                        "source_pdf": "raw/test/split.pdf", "source_pdf_sha256": "b" * 64,
                    },
                ]
            },
        },
    )
    ledger = load_corporate_actions_ledger(ledger_path)
    assert ledger.as_of == "2026-11-26"
    assert [event["ex_date"] for event in ledger.cash_dividends_by_symbol["510720.SH"]] == [
        "2026-11-23", "2026-11-25"
    ]
    assert [event["effective_date"] for event in ledger.splits_by_symbol["512480.SH"]] == [
        "2026-11-23", "2026-11-25"
    ]
    assert all(
        event["event_status"] == "VERIFIED"
        and event["source_class"] == "OFFICIAL_TEST"
        and event["source_pdf_sha256"] == "a" * 64
        for event in ledger.cash_dividends_by_symbol["510720.SH"]
    )

    invalid_path = tmp_path / "invalid_ledger.json"
    _write_json(invalid_path, {"schema_version": "1.0.0", "as_of": "2026-11-26"})
    with pytest.raises(ValueError, match="schema_version"):
        load_corporate_actions_ledger(invalid_path)


def test_provisional_actions_are_retained_but_never_applied(tmp_path: Path) -> None:
    config = _make_synthetic_config(
        tmp_path,
        {"159330.SZ": ISO_DAYS, "512400.SH": ISO_DAYS},
    )
    ledger_path = Path(config["upstream_paths"]["corporate_actions_ledger_file"])
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["splits_by_symbol"]["512400.SH"] = [
        {
            "effective_date": "2026-11-25", "factor": 0.5,
            "source_class": "PROVISIONAL_TEST", "event_status": "PROVISIONAL",
            "source_pdf": None, "source_pdf_sha256": None,
        }
    ]
    _write_json(ledger_path, payload)

    bundle = load_market_bundle(config)
    assert bundle.splits["512400.SH"] == []
    assert bundle.adjusted_bars["512400.SH"]["adj_factor"].tolist() == [1.0] * len(ISO_DAYS)
    pending = bundle.metadata["unverified_corporate_action_events"]
    assert len(pending) == 1
    assert pending[0]["event_status"] == "PROVISIONAL"
    assert pending[0]["source_class"] == "PROVISIONAL_TEST"
    assert pending[0]["date"] == "2026-11-25"
    assert "缺少官方 PDF" in pending[0]["evidence_gap"]


def test_provisional_pdf_and_hash_must_be_both_null_or_a_valid_pair(tmp_path: Path) -> None:
    config = _make_synthetic_config(
        tmp_path,
        {"159330.SZ": ISO_DAYS, "512400.SH": ISO_DAYS},
    )
    ledger_path = Path(config["upstream_paths"]["corporate_actions_ledger_file"])
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["splits_by_symbol"]["512400.SH"] = [{
        "effective_date": "2026-11-25", "factor": 0.5,
        "source_class": "PROVISIONAL_TEST", "event_status": "PROVISIONAL",
        "source_pdf": "raw/test/split.pdf", "source_pdf_sha256": None,
    }]
    _write_json(ledger_path, payload)

    with pytest.raises(ValueError, match="PROVISIONAL event source_pdf and source_pdf_sha256"):
        load_corporate_actions_ledger(ledger_path)


def test_shared_gate_blocks_lag_and_effective_provisional_events() -> None:
    future = [{"date": "2026-11-27", "symbol": "512400.SH", "event_status": "PROVISIONAL"}]
    assert assess_corporate_actions_gate("2026-11-26", "2026-11-26", future)["allowed"]
    lag = assess_corporate_actions_gate("2026-11-25", "2026-11-26", [])
    assert not lag["allowed"]
    effective = assess_corporate_actions_gate("2026-11-26", "2026-11-26", future[:0] + [
        {"date": "2026-11-26", "symbol": "512400.SH", "event_status": "PROVISIONAL"}
    ])
    assert not effective["allowed"]
    assert effective["unverified_events"][0]["symbol"] == "512400.SH"


def test_real_512480_split_adjustment_comes_from_the_ledger() -> None:
    bundle = load_market_bundle(DEFAULT_SIM_CONFIG_PATH)
    split_events = bundle.splits["512480.SH"]
    assert any(
        event["effective_date"] == "2026-07-03" and event["factor"] == 0.5
        for event in split_events
    )

    raw = bundle.bars["512480.SH"].set_index("date")
    adjusted = bundle.adjusted_bars["512480.SH"].set_index("date")
    assert raw.loc["2026-07-02", "close"] == pytest.approx(2.700)
    assert adjusted.loc["2026-07-02", "adj_factor"] == pytest.approx(0.5)
    assert adjusted.loc["2026-07-02", "close"] == pytest.approx(1.350)
    assert raw.loc["2026-07-03", "close"] == pytest.approx(1.331)
    assert adjusted.loc["2026-07-03", "adj_factor"] == pytest.approx(1.0)


_UNSET = object()


def _attach_live_proof(
    journal_dir: Path,
    record: dict,
    *,
    receipt_mtime: datetime | None = None,
    entry_sha256: str | None | object = _UNSET,
) -> Path:
    from src.data_loader import _extract_decision_entry_block

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
    recorded_cst = recorded_dt.isoformat()
    decision_sha = compute_regime_decision_sha256(record, recorded_cst)
    rel_path = f"pre_market_proofs/{record['record_id']}.json"
    receipt_path = journal_dir / rel_path
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": "1.0.0",
        "record_id": record["record_id"],
        "decision_date": record["decision_date"],
        "effective_date": record["effective_date"],
        "symbol": record["symbol"],
        "prev_regime": "TRANSITION" if record["prev_regime"] == "UNRATED" else record["prev_regime"],
        "new_regime": "TRANSITION" if record["new_regime"] == "UNRATED" else record["new_regime"],
        "record_type": "LIVE",
        "recorded_at": record["recorded_at"],
        "recorded_at_cst": recorded_cst,
        "decision_sha256": decision_sha,
        "decision_ref": record.get("decision_ref"),
        "decision_entry_sha256": resolved_entry_sha,
    }
    _write_json(receipt_path, receipt)
    ts_dt = receipt_mtime if receipt_mtime is not None else recorded_dt
    os.utime(receipt_path, (ts_dt.timestamp(), ts_dt.timestamp()))
    record["pre_market_proof"] = {
        "receipt_path": rel_path,
        "decision_sha256": decision_sha,
    }
    return receipt_path


def test_regime_history_uses_effective_date_and_maps_unrated(tmp_path: Path) -> None:
    history_path = tmp_path / "regime_history.json"
    rec_a = {
        "record_id": "RH-20261125-001",
        "decision_date": "2026-11-25",
        "effective_date": "2026-11-26",
        "symbol": "512400.SH",
        "prev_regime": "TREND_UP",
        "new_regime": "RANGE_GRID",
        "record_type": "LIVE",
        "recorded_at": "2026-11-25T16:00:00+08:00",
    }
    rec_b = {
        "record_id": "RH-20261123-001",
        "decision_date": "2026-11-23",
        "effective_date": "2026-11-24",
        "symbol": "512400.SH",
        "prev_regime": "UNRATED",
        "new_regime": "TREND_UP",
        "record_type": "LIVE",
        "recorded_at": "2026-11-23T16:00:00+08:00",
    }
    _attach_live_proof(tmp_path, rec_a, receipt_mtime=datetime(2026, 11, 25, 16, 0, tzinfo=CHINA_TZ))
    _attach_live_proof(tmp_path, rec_b, receipt_mtime=datetime(2026, 11, 23, 16, 0, tzinfo=CHINA_TZ))
    _write_json(
        history_path,
        {
            "schema_version": "2.0.0",
            "anchor_start_date": "2026-11-23",
            "initial_regimes": {"512400.SH": "UNRATED"},
            "records": [rec_a, rec_b],
        },
    )
    history = load_regime_history(history_path, trading_calendar=ISO_DAYS)
    assert history.regime_for("512400.SH", "2026-11-23") == "TRANSITION"
    assert history.regime_for("512400.SH", "2026-11-24") == "TREND_UP"
    assert history.regime_for("512400.SH", "2026-11-25") == "TREND_UP"
    assert history.regime_for("512400.SH", "2026-11-26") == "RANGE_GRID"
    history.validate_for_run_mode("formal", ["2026-11-23", "2026-11-24", "2026-11-25", "2026-11-26"])
    with pytest.raises(ValueError, match="precedes"):
        history.regime_for("512400.SH", "2026-11-22")


def test_regime_history_enforces_bs01_t1_and_live_backfill_timestamps(tmp_path: Path) -> None:
    calendar = ["2026-11-26", "2026-11-27", "2026-11-30", "2026-12-01"]
    base = {
        "schema_version": "2.0.0",
        "anchor_start_date": "2026-11-26",
        "initial_regimes": {"512400.SH": "TRANSITION"},
        "records": [
            {
                "record_id": "RH-20261127-001",
                "decision_date": "2026-11-27",
                "effective_date": "2026-11-30",
                "symbol": "512400.SH",
                "prev_regime": "TRANSITION",
                "new_regime": "RANGE_GRID",
                "record_type": "LIVE",
                "recorded_at": "2026-11-30T01:29:59Z",
            }
        ],
    }
    pre_open_ts = datetime(2026, 11, 30, 9, 29, 59, tzinfo=CHINA_TZ)
    receipt_path = _attach_live_proof(tmp_path, base["records"][0], receipt_mtime=pre_open_ts)
    history_path = tmp_path / "regime_history.json"
    _write_json(history_path, base)
    loaded = load_regime_history(history_path, trading_calendar=calendar)
    assert loaded.regime_for("512400.SH", "2026-11-27") == "TRANSITION"
    assert loaded.regime_for("512400.SH", "2026-11-30") == "RANGE_GRID"

    # Calendar +1 day (Saturday 2026-11-28) instead of BS-01 Monday 2026-11-30 must fail
    bad_weekend = json.loads(json.dumps(base))
    bad_weekend["records"][0]["effective_date"] = "2026-11-28"
    _write_json(history_path, bad_weekend)
    with pytest.raises(ValueError, match="BS-01 T\\+1"):
        load_regime_history(history_path, trading_calendar=calendar)

    for bad_mut, pattern in [
        (lambda d: d.__setitem__("schema_version", "1.0.0"), "schema_version"),
        (lambda d: d["records"][0].__setitem__("record_type", "OTHER"), "record_type"),
        (lambda d: d["records"][0].__setitem__("recorded_at", None), "recorded_at"),
        (lambda d: d["records"][0].__setitem__("recorded_at", "2026-11-27T16:00:00"), "ISO 8601"),
        (lambda d: d["records"][0].__setitem__("recorded_at", "2026-11-27T16:00:00+07:60"), "ISO 8601"),
        (lambda d: d["records"][0].__setitem__("recorded_at", "2026-11-27T16:00:00+24:00"), "ISO 8601"),
        (lambda d: d["records"][0].__setitem__("recorded_at", "2026-11-27T16:00:00-00:00"), "'-00:00'"),
        (lambda d: d["records"][0].__setitem__("recorded_at", "2026-11-26T23:59:59+08:00"), "precedes decision_date"),
        (lambda d: d["records"][0].__setitem__("recorded_at", "2026-11-30T09:30:00+08:00"), "market open"),
        (lambda d: d["records"][0].__setitem__("recorded_at", "2026-11-30T01:30:00Z"), "market open"),
        (lambda d: d["records"][0].pop("pre_market_proof"), "pre_market_proof"),
        (lambda d: d["records"][0]["pre_market_proof"].__setitem__("decision_sha256", "0" * 64), "does not match"),
        (lambda d: d["records"][0]["pre_market_proof"].__setitem__("receipt_path", "pre_market_proofs/MISSING.json"), "Missing LIVE pre_market_proof"),
        (lambda d: d["records"][0].__setitem__("new_regime", "TREND_UP"), "does not match"),
    ]:
        bad = json.loads(json.dumps(base))
        bad_mut(bad)
        _write_json(history_path, bad)
        with pytest.raises(ValueError, match=pattern):
            load_regime_history(history_path, trading_calendar=calendar)

    # Post-market receipt file mtime (>= 09:30:00+08:00) without pre-market Git commit must fail
    _write_json(history_path, base)
    for bad_ts in (datetime(2026, 11, 30, 9, 30, 0, tzinfo=CHINA_TZ), datetime(2026, 11, 30, 20, 48, 0, tzinfo=CHINA_TZ)):
        os.utime(receipt_path, (bad_ts.timestamp(), bad_ts.timestamp()))
        with pytest.raises(ValueError, match="market open"):
            load_regime_history(history_path, trading_calendar=calendar)
    os.utime(receipt_path, (pre_open_ts.timestamp(), pre_open_ts.timestamp()))

    # BACKFILL requires time_evidence_boundary when recorded_at is null
    backfill = json.loads(json.dumps(base))
    backfill["records"][0]["record_type"] = "BACKFILL"
    backfill["records"][0]["recorded_at"] = None
    backfill["records"][0].pop("pre_market_proof", None)
    _write_json(history_path, backfill)
    with pytest.raises(ValueError, match="time_evidence_boundary"):
        load_regime_history(history_path, trading_calendar=calendar)
    backfill["records"][0]["time_evidence_boundary"] = "backfill committed at 2026-11-30T10:30:00+08:00"
    _write_json(history_path, backfill)
    bf_loaded = load_regime_history(history_path, trading_calendar=calendar)
    with pytest.raises(ValueError, match="BACKFILL"):
        bf_loaded.validate_for_run_mode("formal", ["2026-11-27", "2026-11-30"])
    bf_loaded.validate_for_run_mode("replay", ["2026-11-27", "2026-11-30"])


def test_regime_history_git_proof_anchor_binding_and_in_memory_validation(tmp_path: Path) -> None:
    from src.data_loader import _extract_decision_entry_block

    repo = tmp_path / "radar_git"
    journal = repo / "project_content" / "journal"
    journal.mkdir(parents=True)
    record = {
        "record_id": "RH-20261123-001",
        "decision_date": "2026-11-23",
        "effective_date": "2026-11-24",
        "symbol": "510720.SH",
        "prev_regime": "TRANSITION",
        "new_regime": "RANGE_GRID",
        "record_type": "LIVE",
        "recorded_at": "2026-11-23T16:00:00+08:00",
        "decision_ref": "2026_decisions.md#2026-11-23-510720.SH",
    }
    decisions = journal / "2026_decisions.md"
    original_decisions = '<a id="2026-11-23-510720.SH"></a>\n\n### 510720.SH\n- RANGE_GRID\n'
    decisions.write_text(original_decisions, encoding="utf-8")
    block = _extract_decision_entry_block(original_decisions, "2026-11-23-510720.SH")
    assert block is not None
    entry_sha = hashlib.sha256(block.encode("utf-8")).hexdigest()

    receipt_path = _attach_live_proof(
        journal,
        record,
        receipt_mtime=datetime(2026, 11, 23, 16, 0, 0, tzinfo=CHINA_TZ),
        entry_sha256=entry_sha,
    )
    history_path = journal / "regime_history.json"
    _write_json(
        history_path,
        {
            "schema_version": "2.0.0",
            "anchor_start_date": "2026-11-23",
            "initial_regimes": {"510720.SH": "TRANSITION"},
            "records": [record],
        },
    )

    def _git(args: list[str], commit_ts: str | None = None) -> None:
        env = {**os.environ}
        if commit_ts:
            env["GIT_AUTHOR_DATE"] = commit_ts
            env["GIT_COMMITTER_DATE"] = commit_ts
        proc = subprocess.run(["git", "-C", str(repo), *args], env=env, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stderr

    _git(["init"])
    _git(["config", "user.name", "Tester"])
    _git(["config", "user.email", "tester@example.com"])
    _git(["add", "."])
    _git(["commit", "-m", "pre-market freeze"], commit_ts="2026-11-23T16:05:00+08:00")

    # 1. Pre-market clean Git commit passes even when file mtime is later (e.g. re-clone)
    post_ts = datetime(2026, 11, 26, 20, 48, 0, tzinfo=CHINA_TZ).timestamp()
    os.utime(receipt_path, (post_ts, post_ts))
    loaded = load_regime_history(history_path, trading_calendar=ISO_DAYS)
    assert loaded.regime_for("510720.SH", "2026-11-24") == "RANGE_GRID"

    # 2. Tampering with 2026_decisions.md anchor text fails entry SHA check
    decisions.write_text(original_decisions.replace("RANGE_GRID", "TRAP_OFF"), encoding="utf-8")
    with pytest.raises(ValueError, match="decision_entry_sha256"):
        load_regime_history(history_path, trading_calendar=ISO_DAYS)
    decisions.write_text(original_decisions, encoding="utf-8")

    # 3. Post-market Git commit fails even if file mtime is rolled back before open
    receipt_path.write_text(receipt_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _git(["add", "."])
    _git(["commit", "-m", "post-market rewrite"], commit_ts="2026-11-26T20:48:00+08:00")
    pre_ts = datetime(2026, 11, 23, 16, 0, 0, tzinfo=CHINA_TZ).timestamp()
    os.utime(receipt_path, (pre_ts, pre_ts))
    with pytest.raises(ValueError, match="Git commit timestamp"):
        load_regime_history(history_path, trading_calendar=ISO_DAYS)

    # 4. In-memory RegimeHistory (source_path=None) enforces record_type, recorded_at tz/open, regime consistency, and formal BACKFILL gate
    in_memory_late = RegimeHistory(
        schema_version="2.0.0",
        anchor_start_date="2026-11-23",
        initial_regimes={"510720.SH": "TRANSITION"},
        records_by_symbol={
            "510720.SH": [
                {
                    "record_id": "RH-MEM-LATE",
                    "decision_date": "2026-11-23",
                    "effective_date": "2026-11-24",
                    "prev_regime": "TRANSITION",
                    "regime": "RANGE_GRID",
                    "record_type": "LIVE",
                    "recorded_at": "2026-11-24T09:30:00+08:00",
                }
            ]
        },
        source_path=None,
    )
    with pytest.raises(ValueError, match="market open"):
        in_memory_late.validate_for_run_mode("formal", ["2026-11-23", "2026-11-24"])

    in_memory_valid = RegimeHistory(
        schema_version="2.0.0",
        anchor_start_date="2026-11-23",
        initial_regimes={"510720.SH": "TRANSITION"},
        records_by_symbol={
            "510720.SH": [
                {
                    "record_id": "RH-MEM-OK",
                    "decision_date": "2026-11-23",
                    "effective_date": "2026-11-24",
                    "prev_regime": "TRANSITION",
                    "regime": "RANGE_GRID",
                    "record_type": "LIVE",
                    "recorded_at": "2026-11-23T16:00:00+08:00",
                }
            ]
        },
        source_path=None,
    )
    in_memory_valid.validate_for_run_mode("formal", ["2026-11-23", "2026-11-24"])


def test_file_backed_live_proof_requires_decision_ref_anchor_ledger_and_entry_sha256(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    journal.mkdir(parents=True)
    decisions = journal / "2026_decisions.md"
    valid_decisions = '<a id="2026-11-23-510720.SH"></a>\n\n### 510720.SH\n- RANGE_GRID\n'
    decisions.write_text(valid_decisions, encoding="utf-8")

    def _make_record(decision_ref: str | None) -> dict:
        rec: dict = {
            "record_id": "RH-20261123-001",
            "decision_date": "2026-11-23",
            "effective_date": "2026-11-24",
            "symbol": "510720.SH",
            "prev_regime": "TRANSITION",
            "new_regime": "RANGE_GRID",
            "record_type": "LIVE",
            "recorded_at": "2026-11-23T16:00:00+08:00",
        }
        if decision_ref is not None:
            rec["decision_ref"] = decision_ref
        return rec

    history_path = journal / "regime_history.json"

    def _write_history(rec: dict) -> None:
        _write_json(
            history_path,
            {
                "schema_version": "2.0.0",
                "anchor_start_date": "2026-11-23",
                "initial_regimes": {"510720.SH": "TRANSITION"},
                "records": [rec],
            },
        )

    # 1. Missing decision_ref / missing #anchor / empty file or anchor (even with valid hex or null entry sha)
    valid_sha = hashlib.sha256(valid_decisions.strip().encode("utf-8")).hexdigest()
    for bad_ref in (None, "", "2026_decisions.md", "2026_decisions.md#", "#2026-11-23-510720.SH"):
        for sha_val in (None, "", valid_sha):
            rec = _make_record(bad_ref)
            _attach_live_proof(
                journal,
                rec,
                receipt_mtime=datetime(2026, 11, 23, 16, 0, 0, tzinfo=CHINA_TZ),
                entry_sha256=sha_val,
            )
            _write_history(rec)
            with pytest.raises(ValueError, match="decision_ref|decision_entry_sha256"):
                load_regime_history(history_path, trading_calendar=ISO_DAYS)

    # 2. Missing ledger file or missing anchor block in ledger file
    rec_missing_anchor = _make_record("2026_decisions.md#MISSING-ANCHOR")
    _attach_live_proof(
        journal,
        rec_missing_anchor,
        receipt_mtime=datetime(2026, 11, 23, 16, 0, 0, tzinfo=CHINA_TZ),
        entry_sha256=valid_sha,
    )
    _write_history(rec_missing_anchor)
    with pytest.raises(ValueError, match="anchor"):
        load_regime_history(history_path, trading_calendar=ISO_DAYS)

    # 3. Valid baseline passes, then missing ledger / null or empty entry sha / tampered text all fail
    rec_ok = _make_record("2026_decisions.md#2026-11-23-510720.SH")
    receipt_path = _attach_live_proof(
        journal,
        rec_ok,
        receipt_mtime=datetime(2026, 11, 23, 16, 0, 0, tzinfo=CHINA_TZ),
    )
    _write_history(rec_ok)
    loaded = load_regime_history(history_path, trading_calendar=ISO_DAYS)
    loaded.validate_for_run_mode("formal", ["2026-11-23", "2026-11-24"])

    saved_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for empty_val in (None, ""):
        bad_receipt = dict(saved_receipt)
        bad_receipt["decision_entry_sha256"] = empty_val
        _write_json(receipt_path, bad_receipt)
        os.utime(receipt_path, (datetime(2026, 11, 23, 16, 0, 0, tzinfo=CHINA_TZ).timestamp(),) * 2)
        with pytest.raises(ValueError, match="decision_entry_sha256"):
            load_regime_history(history_path, trading_calendar=ISO_DAYS)
        with pytest.raises(ValueError, match="decision_entry_sha256"):
            loaded.validate_for_run_mode("formal", ["2026-11-23", "2026-11-24"])
    _write_json(receipt_path, saved_receipt)
    os.utime(receipt_path, (datetime(2026, 11, 23, 16, 0, 0, tzinfo=CHINA_TZ).timestamp(),) * 2)

    # Tamper decision text after loading -> file-backed validate_for_run_mode must also reject
    decisions.write_text(valid_decisions.replace("RANGE_GRID", "TRAP_OFF"), encoding="utf-8")
    with pytest.raises(ValueError, match="decision_entry_sha256"):
        load_regime_history(history_path, trading_calendar=ISO_DAYS)
    with pytest.raises(ValueError, match="decision_entry_sha256"):
        loaded.validate_for_run_mode("formal", ["2026-11-23", "2026-11-24"])

    # Missing decision ledger file -> both load_regime_history and validate_for_run_mode reject
    decisions.unlink()
    with pytest.raises(ValueError, match="not found"):
        load_regime_history(history_path, trading_calendar=ISO_DAYS)
    with pytest.raises(ValueError, match="not found"):
        loaded.validate_for_run_mode("formal", ["2026-11-23", "2026-11-24"])


def test_tdx_latest_batch_deduplicates_and_validates_dates(tmp_path: Path) -> None:
    symbol_dir = tmp_path / "510720.SH"
    symbol_dir.mkdir()
    (symbol_dir / "20261120_093000.csv").write_text(
        "date,open,high,low,close,amount,volume\n"
        "2026-11-20,1,1,1,1,100,100\n",
        encoding="utf-8",
    )
    latest = symbol_dir / "20261123_151557.csv"
    latest.write_text(
        "date,open,high,low,close,amount,volume,reserved\n"
        "20261120,1.2,1.22,1.19,1.21,300000,250000,0\n"
        "2026/11/18,1.1,1.12,1.09,1.11,100000,90000,0\n"
        "2026-11-18,1.105,1.125,1.095,1.115,150000,135000,0\n",
        encoding="utf-8",
    )
    assert find_latest_tdx_csv(tmp_path, "510720.SH") == latest
    frame = load_tdx_daily(tmp_path, "510720.SH")
    assert frame["date"].tolist() == ["2026-11-18", "2026-11-20"]
    assert frame["close"].tolist() == pytest.approx([1.115, 1.21])
    assert list(frame.columns) == list(TDX_REQUIRED_COLUMNS)


def test_connectivity_accepts_rolling_batches_and_advancing_ledger_watermark(
    tmp_path: Path,
) -> None:
    newer_days = [*ISO_DAYS, "2026-11-27"]
    config = _make_synthetic_config(
        tmp_path,
        {"159330.SZ": newer_days, "512400.SH": newer_days},
    )
    upstream = config["upstream_paths"]
    tdx_dir = Path(upstream["tdx_daily_dir"])
    for symbol in [config["universe"]["benchmark"], *config["universe"]["sectors"]]:
        batch_dir = tdx_dir / symbol
        latest = batch_dir / "20261127_101530.csv"
        latest.write_text(
            "date,open,high,low,close,amount,volume,reserved\n"
            + "".join(
                f"{day},1.00,1.02,0.98,1.01,1000.0,1000,0\n"
                for day in newer_days
            ),
            encoding="utf-8",
        )
    calendar_dir = Path(upstream["trade_calendar_dir"])
    (calendar_dir / "20261127_101530.csv").write_text(
        "calendar_date,is_trading_day\n"
        + "2026-11-23,1\n2026-11-24,1\n2026-11-25,1\n"
        + "2026-11-26,1\n2026-11-27,1\n",
        encoding="utf-8",
    )
    ledger_path = Path(upstream["corporate_actions_ledger_file"])
    ledger_payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger_payload["as_of"] = "2026-11-27"
    ledger_payload["cash_dividends_by_symbol"]["512400.SH"] = [{
        "ex_date": "2026-11-27", "cash_per_unit": 0.01,
        "source_class": "OFFICIAL_TEST", "event_status": "VERIFIED",
        "source_pdf": "raw/test/new_dividend.pdf", "source_pdf_sha256": "c" * 64,
    }]
    _write_json(ledger_path, ledger_payload)

    bundle = load_market_bundle(config)
    calendar = load_trade_calendar(
        calendar_dir, start_date=config["start_date"], end_date="2026-11-27"
    )
    refreshed_ledger = load_corporate_actions_ledger(ledger_path)
    assert bundle.metadata["latest_end_date"] == "2026-11-27"
    assert all(
        bundle.metadata["symbol_stats"][symbol]["latest_csv"] == "20261127_101530.csv"
        for symbol in bundle.symbols
    )
    assert calendar == newer_days
    assert refreshed_ledger.as_of == "2026-11-27"
    assert refreshed_ledger.cash_dividends_by_symbol["512400.SH"][0]["ex_date"] == "2026-11-27"


@pytest.mark.parametrize(
    "bad_row",
    [
        "2026-11-20,,1.02,0.99,1.01,1000,1000",
        "not-a-date,1.00,1.02,0.99,1.01,1000,1000",
        "2026-11-20,1.00,1.02,0.99,-1.01,1000,1000",
    ],
)
def test_tdx_loader_rejects_corrupt_rows(tmp_path: Path, bad_row: str) -> None:
    symbol_dir = tmp_path / "159330.SZ"
    symbol_dir.mkdir()
    (symbol_dir / "20261123_120000.csv").write_text(
        "date,open,high,low,close,amount,volume\n" + bad_row + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_tdx_daily(tmp_path, "159330.SZ")


def test_forward_adjustment_applies_split_and_dividend_without_mutation() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-06"],
            "open": [2.00, 2.00, 1.00, 0.95],
            "high": [2.04, 2.02, 1.02, 0.98],
            "low": [1.98, 1.98, 0.98, 0.94],
            "close": [2.00, 2.00, 1.00, 0.96],
            "amount": [1000.0] * 4,
            "volume": [500.0, 500.0, 1000.0, 1050.0],
        }
    )
    before = raw.copy(deep=True)
    adjusted = build_forward_adjusted_daily(
        raw,
        split_events=[{"effective_date": "2026-07-03", "factor": 0.5}],
        dividend_events={"2026-07-06": 0.05},
    )
    pd.testing.assert_frame_equal(raw, before)
    assert adjusted["adj_factor"].tolist() == pytest.approx([0.475, 0.475, 0.95, 1.0])
    assert adjusted["close"].tolist() == pytest.approx([0.95, 0.95, 0.95, 0.96])


def test_bs01_calendar_loader_filters_trading_days(tmp_path: Path) -> None:
    calendar_dir = tmp_path / "calendar"
    _write_calendar(calendar_dir, ["2026-11-23", "2026-11-25"])
    calendar = load_trade_calendar(
        calendar_dir, start_date="2026-11-23", end_date="2026-11-25"
    )
    assert calendar == ["2026-11-23", "2026-11-25"]


def test_pytest_config_imports_package_from_an_external_working_directory(
    tmp_path: Path,
) -> None:
    repo_root = DEFAULT_SIM_CONFIG_PATH.parents[2]
    pyproject = repo_root / "pyproject.toml"
    probe_file = tmp_path / "test_external_import_probe.py"
    probe_file.write_text(
        "from importlib import import_module\n"
        "from src.data_loader import DEFAULT_SIM_CONFIG_PATH\n"
        "def test_package_import_resolves():\n"
        "    assert DEFAULT_SIM_CONFIG_PATH.is_file()\n",
        encoding="utf-8",
    )
    probe_file.write_text(
        probe_file.read_text(encoding="utf-8")
        + "    for module in ('src.engine', 'src.strategies', 'src.reporter', 'src.run_daily_sim'):\n"
        + "        assert import_module(module)\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(pyproject),
            "--rootdir",
            str(repo_root),
            str(probe_file),
            "-q",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "1 passed" in result.stdout


def test_committed_sim_config_is_portable_and_resolves_from_any_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = DEFAULT_SIM_CONFIG_PATH.parents[2]
    raw_cfg = json.loads(DEFAULT_SIM_CONFIG_PATH.read_text(encoding="utf-8"))
    raw_upstream = raw_cfg["upstream_paths"]
    assert raw_upstream["tdx_daily_dir"] == "../QuantData/raw/tdx/vipdoc_parsed"
    assert raw_upstream["trade_calendar_dir"] == "../QuantData/raw/baostock/query_trade_dates"
    assert raw_upstream["corporate_actions_ledger_file"] == "../QuantData/reference/etf/corporate_actions/latest/corporate_actions_ledger.json"
    assert raw_upstream["regime_history_file"] == "../sector_radar/project_content/journal/regime_history.json"
    assert raw_upstream["watchlist_file"] == "../sector_radar/project_content/config/watchlist.json"

    import re

    forbidden = (
        "Vibe" + "Coding",
        "new_" + "tdx64",
        ".con" + "da",
    )
    users_home_re = re.compile(r"[A-Za-z]:[/\\]Users[/\\][^/\\'\"]+")
    scan_targets = [
        DEFAULT_SIM_CONFIG_PATH,
        repo_root / "pyproject.toml",
        repo_root / "project_content" / "docs" / "01_撮合与切换确定性规则.md",
        repo_root / "project_content" / "docs" / "02_跨仓路径与CLI运行说明.md",
        *(repo_root / "project_content" / "src").glob("*.py"),
        *(repo_root / "project_content" / "tests").glob("*.py"),
    ]
    for target in scan_targets:
        text = target.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{target} must not contain machine path {token!r}"
        assert users_home_re.search(text) is None, f"{target} must not contain user home path"

    monkeypatch.chdir(tmp_path)
    resolved = load_sim_config(DEFAULT_SIM_CONFIG_PATH)
    workspace_parent = repo_root.parent
    assert Path(resolved["upstream_paths"]["tdx_daily_dir"]) == (workspace_parent / "QuantData" / "raw" / "tdx" / "vipdoc_parsed").resolve()
    assert Path(resolved["upstream_paths"]["trade_calendar_dir"]) == (workspace_parent / "QuantData" / "raw" / "baostock" / "query_trade_dates").resolve()
    assert Path(resolved["upstream_paths"]["corporate_actions_ledger_file"]) == (
        workspace_parent / "QuantData" / "reference" / "etf" / "corporate_actions" / "latest" / "corporate_actions_ledger.json"
    ).resolve()
    assert Path(resolved["upstream_paths"]["regime_history_file"]) == (
        workspace_parent / "sector_radar" / "project_content" / "journal" / "regime_history.json"
    ).resolve()
    assert Path(resolved["upstream_paths"]["watchlist_file"]) == (
        workspace_parent / "sector_radar" / "project_content" / "config" / "watchlist.json"
    ).resolve()
