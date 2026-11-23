"""Read-only connectivity checks for sim_config.json and upstream data sources."""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM_CONFIG_PATH = REPO_ROOT / "project_content" / "config" / "sim_config.json"
EXPECTED_ETFS = [
    "159330.SZ",
    "512400.SH",
    "510720.SH",
    "159622.SZ",
    "515790.SH",
    "512480.SH",
    "512880.SH",
]


def _load_json_readonly(path: Path) -> dict:
    assert path.is_file(), f"Missing JSON file: {path}"
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, dict), f"Expected top-level JSON object in {path}"
    return data


def _locate_etf_entry(tdx_daily_dir: Path, symbol: str) -> tuple[str, Path, list[Path]]:
    """Locate subdir or CSV file for an ETF without writing anything."""
    bare_code = symbol.split(".")[0]

    candidates = [
        ("subdir_with_suffix", tdx_daily_dir / symbol),
        ("subdir_without_suffix", tdx_daily_dir / bare_code),
        ("csv_with_suffix", tdx_daily_dir / f"{symbol}.csv"),
        ("csv_without_suffix", tdx_daily_dir / f"{bare_code}.csv"),
    ]
    for fmt, path in candidates:
        if fmt.startswith("subdir_") and path.is_dir():
            csv_files = sorted(path.glob("*.csv"))
            return fmt, path, csv_files
        if fmt.startswith("csv_") and path.is_file():
            return fmt, path, [path]
    raise AssertionError(
        f"No matching subdir or CSV found for {symbol} under {tdx_daily_dir}"
    )


def _inspect_latest_csv(csv_path: Path) -> tuple[list[str], int, str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        rows = [row for row in reader if row]
    assert header, f"Empty CSV header in {csv_path}"
    assert rows, f"No data rows in {csv_path}"
    return header, len(rows), rows[0][0], rows[-1][0]


def test_sim_config_json_valid() -> None:
    cfg = _load_json_readonly(SIM_CONFIG_PATH)
    assert cfg["project_name"] == "sector_paper_sim"

    upstream = cfg["upstream_paths"]
    for key in ("tdx_daily_dir", "dividend_events_file", "watchlist_file"):
        assert key in upstream and isinstance(upstream[key], str) and upstream[key]

    defaults = cfg["account_defaults"]
    assert defaults["initial_capital"] == 100000.0
    assert defaults["sector_max_weight"] == 0.1666667
    assert defaults["lot_size"] == 100
    assert defaults["commission_rate"] == 0.0002
    assert defaults["min_commission"] == 0.1
    assert defaults["stamp_duty_rate"] == 0.0
    assert defaults["dividend_reinvest"] is True

    strategies = cfg["strategies"]
    assert strategies["turtle"] == {
        "entry_lookback": 55,
        "exit_lookback": 20,
        "atr_window": 20,
    }
    assert strategies["grid"] == {
        "step_mode": "atr20_locked",
        "atr_window": 20,
        "step_min": 0.02,
        "step_max": 0.05,
        "grid_levels_each_side": 4,
        "total_slices": 5,
        "initial_slices": 2,
    }

    universe = cfg["universe"]
    assert universe["benchmark"] == "159330.SZ"
    assert universe["sectors"] == EXPECTED_ETFS[1:]


def test_upstream_json_files_readable() -> None:
    cfg = _load_json_readonly(SIM_CONFIG_PATH)

    watchlist_path = Path(cfg["upstream_paths"]["watchlist_file"])
    watchlist = _load_json_readonly(watchlist_path)
    watchlist_symbols = [item["symbol"] for item in watchlist.get("universe", [])]
    for symbol in EXPECTED_ETFS:
        assert symbol in watchlist_symbols, f"{symbol} missing from watchlist.json"

    dividend_path = Path(cfg["upstream_paths"]["dividend_events_file"])
    dividends = _load_json_readonly(dividend_path)
    events_by_symbol = dividends.get("events_by_symbol", {})
    for symbol in EXPECTED_ETFS:
        assert symbol in events_by_symbol, (
            f"{symbol} missing from watchlist_dividend_events.json"
        )


def test_tdx_daily_dir_etf_files_exist() -> None:
    cfg = _load_json_readonly(SIM_CONFIG_PATH)
    tdx_daily_dir = Path(cfg["upstream_paths"]["tdx_daily_dir"])
    assert tdx_daily_dir.is_dir(), f"Missing tdx_daily_dir: {tdx_daily_dir}"

    all_symbols = [cfg["universe"]["benchmark"], *cfg["universe"]["sectors"]]
    assert all_symbols == EXPECTED_ETFS

    for symbol in all_symbols:
        fmt, entry_path, csv_files = _locate_etf_entry(tdx_daily_dir, symbol)
        assert csv_files, f"No CSV files found for {symbol} at {entry_path} ({fmt})"
        header, row_count, start_date, end_date = _inspect_latest_csv(csv_files[-1])
        assert {"date", "open", "high", "low", "close"}.issubset(set(header))
        assert row_count > 0 and start_date <= end_date


def print_readonly_report() -> None:
    """Print an objective diagnostic summary for 1t01 receipt and 1t02 handoff."""
    cfg = _load_json_readonly(SIM_CONFIG_PATH)
    print(f"[sim_config] OK: {SIM_CONFIG_PATH}")
    print(f"  project_name = {cfg['project_name']}")

    watchlist_path = Path(cfg["upstream_paths"]["watchlist_file"])
    watchlist = _load_json_readonly(watchlist_path)
    regimes = {
        item["symbol"]: item.get("current_regime")
        for item in watchlist.get("universe", [])
    }
    print(f"[watchlist_file] OK: {watchlist_path} (universe_count={len(regimes)})")
    for sym in EXPECTED_ETFS:
        print(f"  {sym}: current_regime={regimes.get(sym)}")

    dividend_path = Path(cfg["upstream_paths"]["dividend_events_file"])
    dividends = _load_json_readonly(dividend_path)
    events_by_symbol = dividends.get("events_by_symbol", {})
    print(
        f"[dividend_events_file] OK: {dividend_path} "
        f"(schema_version={dividends.get('schema_version')}, as_of={dividends.get('as_of')})"
    )
    for sym in EXPECTED_ETFS:
        print(f"  {sym}: dividend_events={len(events_by_symbol.get(sym, []))}")

    tdx_daily_dir = Path(cfg["upstream_paths"]["tdx_daily_dir"])
    print(f"[tdx_daily_dir] OK: {tdx_daily_dir}")
    for sym in EXPECTED_ETFS:
        fmt, entry_path, csv_files = _locate_etf_entry(tdx_daily_dir, sym)
        latest_csv = csv_files[-1]
        header, row_count, start_date, end_date = _inspect_latest_csv(latest_csv)
        print(
            f"  {sym}: format={fmt}, entry={entry_path.name}, "
            f"csv_count={len(csv_files)}, latest={latest_csv.name}, "
            f"rows={row_count}, range={start_date}..{end_date}, "
            f"columns={','.join(header)}"
        )


if __name__ == "__main__":
    test_sim_config_json_valid()
    test_upstream_json_files_readable()
    test_tdx_daily_dir_etf_files_exist()
    print_readonly_report()
