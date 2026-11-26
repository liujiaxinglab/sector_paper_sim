"""Read-only connectivity checks for all configured upstream data sources."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import hashlib
from pathlib import Path
import re
import pytest

from src.data_loader import (
    DEFAULT_SIM_CONFIG_PATH,
    find_latest_tdx_csv,
    load_corporate_actions_ledger,
    load_market_bundle,
    load_regime_history,
    load_sim_config,
    load_tdx_daily,
    load_trade_calendar,
    load_watchlist,
)
from src.engine import run_simulation
from src.reporter import NAV_CSV_COLUMNS, read_csv_rows

EXPECTED_SYMBOLS = [
    "159330.SZ",
    "512400.SH",
    "510720.SH",
    "159622.SZ",
    "515790.SH",
    "512480.SH",
    "512880.SH",
]


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _assert_batch_filename(path: Path) -> None:
    match = re.fullmatch(r"(?P<day>\d{8})_(?P<time>\d{6})\.csv", path.name)
    assert match, f"Unexpected upstream batch filename: {path.name}"
    datetime.strptime(match["day"] + match["time"], "%Y%m%d%H%M%S")


def _assert_iso_dates(values: list[str]) -> None:
    parsed = [date.fromisoformat(value) for value in values]
    assert [item.isoformat() for item in parsed] == values
    assert values == sorted(set(values))


def _verify_event_evidence(event: dict, quantdata_root: Path) -> None:
    source = event.get("source_pdf")
    source_hash = event.get("source_pdf_sha256")
    assert source and re.fullmatch(r"[0-9a-f]{64}", str(source_hash))
    source_path = quantdata_root / source
    assert source_path.is_file(), f"Missing event evidence: {source_path}"
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
    discovery = event.get("discovery_raw_file")
    if discovery:
        discovery_path = quantdata_root / discovery
        assert discovery_path.is_file(), f"Missing event discovery record: {discovery_path}"
    result_source = event.get("result_announcement_pdf")
    if result_source:
        result_hash = event.get("result_announcement_pdf_sha256")
        assert re.fullmatch(r"[0-9a-f]{64}", str(result_hash))
        result_path = quantdata_root / result_source
        assert result_path.is_file(), f"Missing event evidence: {result_path}"
        assert hashlib.sha256(result_path.read_bytes()).hexdigest() == result_hash


def test_sim_config_points_to_existing_readonly_sources() -> None:
    config = load_sim_config(DEFAULT_SIM_CONFIG_PATH)
    upstream = config["upstream_paths"]
    assert config["start_date"] == "2026-11-23"
    assert Path(upstream["tdx_daily_dir"]).is_dir()
    assert Path(upstream["trade_calendar_dir"]).is_dir()
    assert Path(upstream["corporate_actions_ledger_file"]).is_file()
    assert Path(upstream["regime_history_file"]).is_file()
    assert Path(upstream["watchlist_file"]).is_file()
    assert "dividend_events_file" not in upstream


def test_upstream_artifacts_cover_all_symbols_without_being_modified() -> None:
    config = load_sim_config(DEFAULT_SIM_CONFIG_PATH)
    upstream = config["upstream_paths"]
    tdx_dir = Path(upstream["tdx_daily_dir"])
    calendar_dir = Path(upstream["trade_calendar_dir"])
    ledger_file = Path(upstream["corporate_actions_ledger_file"])
    regime_file = Path(upstream["regime_history_file"])
    watchlist_file = Path(upstream["watchlist_file"])

    latest_tdx = {symbol: find_latest_tdx_csv(tdx_dir, symbol) for symbol in EXPECTED_SYMBOLS}
    latest_calendar = sorted(calendar_dir.glob("????????_??????.csv"))[-1]
    source_paths = [
        ledger_file,
        regime_file,
        watchlist_file,
        latest_calendar,
        *latest_tdx.values(),
    ]
    before = {path: _file_signature(path) for path in source_paths}

    bundle = load_market_bundle(config)
    ledger = load_corporate_actions_ledger(ledger_file)
    regimes = load_regime_history(regime_file)
    watchlist = load_watchlist(watchlist_file)
    bs01_dates = load_trade_calendar(
        calendar_dir,
        start_date=config["start_date"],
        end_date=bundle.metadata["latest_end_date"],
    )

    assert set(ledger.cash_dividends_by_symbol) >= set(EXPECTED_SYMBOLS)
    assert set(ledger.splits_by_symbol) >= set(EXPECTED_SYMBOLS)
    assert ledger.schema_version == "2.0.0"
    assert date.fromisoformat(ledger.as_of).isoformat() == ledger.as_of
    assert ledger.as_of >= bundle.metadata["common_end_date"]
    assert set(watchlist) >= set(EXPECTED_SYMBOLS)
    assert set(regimes.initial_regimes) == set(EXPECTED_SYMBOLS[1:])
    assert all(regimes.regime_for(symbol, "2026-11-23") for symbol in EXPECTED_SYMBOLS[1:])
    assert config["start_date"] in bs01_dates
    assert bs01_dates[-1] >= bundle.metadata["latest_end_date"]
    assert Path(bundle.metadata["calendar_file"]) == latest_calendar
    _assert_batch_filename(latest_calendar)
    _assert_iso_dates(bs01_dates)

    quantdata_root = ledger_file.parents[4]
    for symbol_events in ledger.cash_dividends_by_symbol.values():
        event_dates = [event["ex_date"] for event in symbol_events]
        assert [date.fromisoformat(value).isoformat() for value in event_dates] == event_dates
        assert event_dates == sorted(event_dates)
        for event in symbol_events:
            if event.get("event_status") == "VERIFIED":
                _verify_event_evidence(event, quantdata_root)
    for symbol_events in ledger.splits_by_symbol.values():
        event_dates = [event["effective_date"] for event in symbol_events]
        assert [date.fromisoformat(value).isoformat() for value in event_dates] == event_dates
        assert event_dates == sorted(event_dates)
        for event in symbol_events:
            if event.get("event_status") == "VERIFIED":
                _verify_event_evidence(event, quantdata_root)

    for symbol, path in latest_tdx.items():
        _assert_batch_filename(path)
        frame = load_tdx_daily(tdx_dir, symbol)
        assert frame.attrs["source_csv"] == str(path)
        assert not frame.empty
        observed_dates = frame["date"].tolist()
        _assert_iso_dates(observed_dates)
        assert len(observed_dates) >= 250
        minimum_coverage = date.fromisoformat(config["start_date"]) - timedelta(days=365)
        assert date.fromisoformat(observed_dates[0]) <= minimum_coverage
        assert observed_dates[-1] >= config["start_date"]

    after = {path: _file_signature(path) for path in source_paths}
    assert after == before, "A loader changed an upstream source file"


def test_real_ledger_has_sorted_dividends_and_split_factor() -> None:
    config = load_sim_config(DEFAULT_SIM_CONFIG_PATH)
    ledger = load_corporate_actions_ledger(
        config["upstream_paths"]["corporate_actions_ledger_file"]
    )
    dividends = ledger.cash_dividends_by_symbol["510720.SH"]
    assert [event["ex_date"] for event in dividends] == sorted(
        event["ex_date"] for event in dividends
    )
    key_dividend = next(
        (event for event in dividends if event["ex_date"] == "2024-05-28"), None
    )
    assert key_dividend is not None
    assert key_dividend["cash_per_unit"] == 0.0031
    assert key_dividend["event_status"] == "VERIFIED"
    quantdata_root = Path(
        config["upstream_paths"]["corporate_actions_ledger_file"]
    ).parents[4]
    _verify_event_evidence(key_dividend, quantdata_root)

    splits = ledger.splits_by_symbol["512480.SH"]
    assert [event["effective_date"] for event in splits] == sorted(
        event["effective_date"] for event in splits
    )
    assert any(
        event["effective_date"] == "2026-07-03" and event["factor"] == 0.5
        for event in splits
    )


def test_real_backfill_regime_history_blocks_formal_and_replays_existing_nav() -> None:
    config = load_sim_config(DEFAULT_SIM_CONFIG_PATH)
    bundle = load_market_bundle(config)
    all_records = bundle.regime_history.all_records()
    assert bundle.regime_history.schema_version == "2.0.0"
    backfills = [rec for rec in all_records if rec["record_type"] == "BACKFILL"]
    assert backfills
    assert all(rec["recorded_at"] is None for rec in backfills)
    for rec in all_records:
        date.fromisoformat(rec["decision_date"])
        date.fromisoformat(rec["effective_date"])

    expected_nav_path = DEFAULT_SIM_CONFIG_PATH.parents[1] / "output" / "nav_daily.csv"
    stored_rows = read_csv_rows(expected_nav_path, NAV_CSV_COLUMNS)
    checkpoint = max(row["date"] for row in stored_rows)

    with pytest.raises(ValueError, match="BACKFILL"):
        run_simulation(bundle, end_date=checkpoint, run_mode="formal")

    replayed = run_simulation(bundle, end_date=checkpoint, run_mode="replay")
    expected_by_key = {(row["date"], row["account"]): float(row["nav"]) for row in stored_rows}
    for day in replayed.daily_results:
        for account_id, metrics in day["accounts"].items():
            assert metrics["nav"] == pytest.approx(expected_by_key[day["date"], account_id])
