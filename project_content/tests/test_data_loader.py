"""Unit tests for BS-01 calendar, corporate-actions, and route-history loading."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from src.data_loader import (
    DEFAULT_SIM_CONFIG_PATH,
    TDX_NUMERIC_COLUMNS,
    TDX_REQUIRED_COLUMNS,
    build_forward_adjusted_daily,
    find_latest_tdx_csv,
    load_corporate_actions_ledger,
    load_market_bundle,
    load_regime_history,
    load_sim_config,
    load_tdx_daily,
    load_trade_calendar,
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
            "schema_version": "1.0.0",
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
    assert bundle.metadata["calendar_file"].endswith("20261126_100336.csv")
    assert bundle.metadata["corporate_actions_as_of"] == "2026-11-25"
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
                    {"ex_date": "2026-11-25", "cash_per_unit": 0.02},
                    {"ex_date": "2026-11-23", "cash_per_unit": 0.01},
                ]
            },
            "splits_by_symbol": {
                "512480.SH": [
                    {"effective_date": "2026-11-25", "factor": 0.5},
                    {"effective_date": "2026-11-23", "factor": 0.8},
                ]
            },
        },
    )
    ledger = load_corporate_actions_ledger(ledger_path)
    assert ledger.as_of == "2026-11-26"
    assert ledger.cash_dividends_by_symbol["510720.SH"] == [
        {"ex_date": "2026-11-23", "cash_per_unit": 0.01},
        {"ex_date": "2026-11-25", "cash_per_unit": 0.02},
    ]
    assert ledger.splits_by_symbol["512480.SH"] == [
        {"effective_date": "2026-11-23", "factor": 0.8},
        {"effective_date": "2026-11-25", "factor": 0.5},
    ]

    invalid_path = tmp_path / "invalid_ledger.json"
    _write_json(invalid_path, {"schema_version": "1.0.0", "as_of": "2026-11-26"})
    with pytest.raises(ValueError, match="schema_version"):
        load_corporate_actions_ledger(invalid_path)


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


def test_regime_history_uses_effective_date_and_maps_unrated(tmp_path: Path) -> None:
    history_path = tmp_path / "regime_history.json"
    _write_json(
        history_path,
        {
            "schema_version": "1.0.0",
            "anchor_start_date": "2026-11-23",
            "initial_regimes": {"512400.SH": "UNRATED"},
            "records": [
                {
                    "effective_date": "2026-11-26",
                    "symbol": "512400.SH",
                    "new_regime": "RANGE_GRID",
                },
                {
                    "effective_date": "2026-11-24",
                    "symbol": "512400.SH",
                    "new_regime": "TREND_UP",
                },
            ],
        },
    )
    history = load_regime_history(history_path)
    assert history.regime_for("512400.SH", "2026-11-23") == "TRANSITION"
    assert history.regime_for("512400.SH", "2026-11-24") == "TREND_UP"
    assert history.regime_for("512400.SH", "2026-11-25") == "TREND_UP"
    assert history.regime_for("512400.SH", "2026-11-26") == "RANGE_GRID"
    with pytest.raises(ValueError, match="precedes"):
        history.regime_for("512400.SH", "2026-11-22")


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
            "C:/Users/jxing/.conda/envs/quant/python.exe",
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
