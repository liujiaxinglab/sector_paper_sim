"""Offline unit and real-upstream read-only tests for src.data_loader."""

from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd
import pytest

from src.data_loader import (
    DEFAULT_SIM_CONFIG_PATH,
    TDX_NUMERIC_COLUMNS,
    TDX_REQUIRED_COLUMNS,
    build_forward_adjusted_daily,
    find_latest_tdx_csv,
    load_dividend_events,
    load_market_bundle,
    load_sim_config,
    load_tdx_daily,
    load_watchlist,
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

MIN_BASELINE_ROWS = {
    "159330.SZ": 546,
    "512400.SH": 2236,
    "510720.SH": 613,
    "159622.SZ": 859,
    "515790.SH": 1435,
    "512480.SH": 1807,
    "512880.SH": 2497,
}

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def test_load_market_bundle_real_upstream() -> None:
    """Read-only load of all 7 real ETFs: row counts, consistent latest date, no lag warnings."""
    config = load_sim_config(DEFAULT_SIM_CONFIG_PATH)
    bundle = load_market_bundle(config)

    assert bundle.benchmark == "159330.SZ"
    assert bundle.sectors == EXPECTED_SYMBOLS[1:]
    assert bundle.symbols == EXPECTED_SYMBOLS
    assert set(bundle.bars.keys()) == set(EXPECTED_SYMBOLS)
    assert set(bundle.adjusted_bars.keys()) == set(EXPECTED_SYMBOLS)

    end_dates: dict[str, str] = {}
    for symbol in EXPECTED_SYMBOLS:
        df = bundle.bars[symbol]
        assert list(df.columns) == list(TDX_REQUIRED_COLUMNS)
        assert "reserved" not in df.columns
        assert len(df) >= MIN_BASELINE_ROWS[symbol]
        assert len(df) == bundle.metadata["symbol_stats"][symbol]["row_count"]

        dates = df["date"].tolist()
        assert all(isinstance(d, str) and ISO_DATE_RE.match(d) for d in dates)
        assert dates == sorted(set(dates))

        for col in TDX_NUMERIC_COLUMNS:
            assert pd.api.types.is_float_dtype(df[col])
        for col in ("open", "high", "low", "close"):
            assert not df[col].isna().any()
            assert (df[col] > 0.0).all()

        end_dates[symbol] = dates[-1]

        adj_df = bundle.adjusted_bars[symbol]
        assert len(adj_df) == len(df)
        assert adj_df["adj_factor"].iloc[-1] == pytest.approx(1.0)
        assert adj_df["close"].iloc[-1] == pytest.approx(df["close"].iloc[-1])

    # All 7 ETFs share the same latest trading date and zero lag warnings
    assert len(set(end_dates.values())) == 1
    expected_end_date = next(iter(end_dates.values()))
    assert bundle.common_end_date == expected_end_date
    assert bundle.metadata["common_end_date"] == expected_end_date
    assert bundle.metadata["latest_end_date"] == expected_end_date
    assert bundle.common_trading_dates[0] == "2024-08-19"
    assert bundle.common_trading_dates[-1] == expected_end_date
    assert len(bundle.common_trading_dates) >= 546
    assert bundle.lag_warnings == []
    assert bundle.metadata["lag_warnings"] == []


def test_load_dividend_events_real_upstream() -> None:
    """Verify 510720.SH (31 events) and 512400.SH (3 events) dividend parsing completeness."""
    config = load_sim_config(DEFAULT_SIM_CONFIG_PATH)
    dividends = load_dividend_events(config["upstream_paths"]["dividend_events_file"])

    for symbol in EXPECTED_SYMBOLS:
        assert symbol in dividends

    # 510720.SH: 31 monthly cash dividend events
    div_510720 = dividends["510720.SH"]
    assert len(div_510720) == 31
    dates_510720 = list(div_510720.keys())
    assert dates_510720 == sorted(dates_510720)
    assert all(ISO_DATE_RE.match(dt) for dt in dates_510720)
    assert all(isinstance(val, float) and val > 0.0 for val in div_510720.values())
    assert dates_510720[0] == "2024-05-28"
    assert div_510720["2024-05-28"] == pytest.approx(0.0031)
    assert dates_510720[-1] == "2026-11-13"
    assert div_510720["2026-11-13"] == pytest.approx(0.0031)

    # 512400.SH: 3 annual cash dividend events
    div_512400 = dividends["512400.SH"]
    assert len(div_512400) == 3
    assert list(div_512400.keys()) == ["2024-09-18", "2025-09-15", "2026-09-09"]
    assert div_512400["2024-09-18"] == pytest.approx(0.01)
    assert div_512400["2025-09-15"] == pytest.approx(0.015)
    assert div_512400["2026-09-09"] == pytest.approx(0.08)

    # Remaining 5 ETFs have 0 cash dividend events
    for symbol in ("159330.SZ", "159622.SZ", "515790.SH", "512480.SH", "512880.SH"):
        assert dividends[symbol] == {}


def test_load_watchlist_real_upstream_includes_unrated_and_splits() -> None:
    """512880.SH UNRATED parses cleanly; 512480.SH split_events and others default to []."""
    config = load_sim_config(DEFAULT_SIM_CONFIG_PATH)
    watchlist = load_watchlist(config["upstream_paths"]["watchlist_file"])

    assert set(EXPECTED_SYMBOLS).issubset(set(watchlist.keys()))

    assert watchlist["512880.SH"]["current_regime"] == "UNRATED"
    assert watchlist["512880.SH"]["split_events"] == []

    assert watchlist["159330.SZ"]["current_regime"] == "BENCHMARK"
    assert watchlist["515790.SH"]["current_regime"] == "TRAP_OFF"

    split_512480 = watchlist["512480.SH"]["split_events"]
    assert len(split_512480) == 1
    assert split_512480[0]["effective_date"] == "2026-07-03"
    assert split_512480[0]["factor"] == pytest.approx(0.5)

    for symbol in ("159330.SZ", "512400.SH", "510720.SH", "159622.SZ", "515790.SH", "512880.SH"):
        assert watchlist[symbol]["split_events"] == []


def test_load_tdx_daily_latest_batch_and_dedup_sort(tmp_path: Path) -> None:
    """Synthetic multi-batch directory selects latest YYYYMMDD_HHMMSS.csv and sorts/dedups dates."""
    symbol = "510720.SH"
    symbol_dir = tmp_path / symbol
    symbol_dir.mkdir(parents=True)

    # Older batch (should not be selected)
    (symbol_dir / "20261120_093000.csv").write_text(
        "date,open,high,low,close,amount,volume,reserved\n"
        "2026-11-18,0.900,0.910,0.890,0.905,100000.0,110000,65536\n",
        encoding="utf-8",
    )

    # Non-matching filenames (must be ignored even if lexicographically larger)
    (symbol_dir / "99999999_extra.csv").write_text(
        "date,open,high,low,close,amount,volume,reserved\n"
        "2026-11-25,9.999,9.999,9.999,9.999,10.0,10,0\n",
        encoding="utf-8",
    )
    (symbol_dir / "latest.csv").write_text(
        "date,open,high,low,close,amount,volume,reserved\n"
        "2026-11-25,8.888,8.888,8.888,8.888,10.0,10,0\n",
        encoding="utf-8",
    )
    (symbol_dir / "20261124_120000.txt").write_text("not a csv", encoding="utf-8")

    # Latest valid batch with unsorted dates, mixed date formats, and a duplicate date
    latest_path = symbol_dir / "20261123_151557.csv"
    latest_path.write_text(
        "date,open,high,low,close,amount,volume,reserved\n"
        "20261120,1.200,1.220,1.190,1.210,300000.0,250000,65536\n"
        "2026/11/18,1.100,1.120,1.090,1.110,100000.0,90000,65536\n"
        "2026-11-19,1.110,1.130,1.100,1.120,200000.0,180000,65536\n"
        "2026-11-18,1.105,1.125,1.095,1.115,150000.0,135000,65536\n",
        encoding="utf-8",
    )

    assert find_latest_tdx_csv(tmp_path, symbol) == latest_path

    df = load_tdx_daily(tmp_path, symbol)
    assert list(df.columns) == list(TDX_REQUIRED_COLUMNS)
    assert "reserved" not in df.columns
    assert df["date"].tolist() == ["2026-11-18", "2026-11-19", "2026-11-20"]

    # Duplicate 2026-11-18 keeps the later occurrence in file order (close=1.115)
    assert df["close"].tolist() == pytest.approx([1.115, 1.120, 1.210])
    assert df["open"].tolist() == pytest.approx([1.105, 1.110, 1.200])
    assert df["volume"].tolist() == pytest.approx([135000.0, 180000.0, 250000.0])
    for col in TDX_NUMERIC_COLUMNS:
        assert pd.api.types.is_float_dtype(df[col])


@pytest.mark.parametrize(
    "bad_row",
    [
        "2026-11-20,,1.02,0.99,1.01,1000.0,1000,0",  # NaN open
        "2026-11-20,1.00,1.02,0.0,1.01,1000.0,1000,0",  # zero low
        "2026-11-20,1.00,1.02,0.99,-1.01,1000.0,1000,0",  # negative close
        "not-a-date,1.00,1.02,0.99,1.01,1000.0,1000,0",  # invalid date
    ],
)
def test_load_tdx_daily_rejects_corrupted_rows(tmp_path: Path, bad_row: str) -> None:
    """NaN, <= 0 OHLC values, or unparseable dates must raise ValueError."""
    symbol_dir = tmp_path / "159330.SZ"
    symbol_dir.mkdir(parents=True)
    (symbol_dir / "20261123_120000.csv").write_text(
        "date,open,high,low,close,amount,volume,reserved\n"
        f"{bad_row}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_tdx_daily(tmp_path, "159330.SZ")


def test_load_watchlist_rejects_unknown_regime(tmp_path: Path) -> None:
    """Unsupported current_regime values must raise ValueError."""
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text(
        json.dumps(
            {
                "universe": [
                    {"symbol": "159330.SZ", "current_regime": "INVALID_STATE"}
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported current_regime"):
        load_watchlist(watchlist_path)


def test_load_market_bundle_reports_lag_warnings(tmp_path: Path) -> None:
    """When one symbol ends behind others, common_end_date shrinks and lag_warnings records it."""
    tdx_dir = tmp_path / "tdx"
    for sym, rows in {
        "159330.SZ": [
            "2026-11-18,1.00,1.01,0.99,1.00,1000,1000,0",
            "2026-11-19,1.01,1.02,1.00,1.01,1000,1000,0",
            "2026-11-20,1.02,1.03,1.01,1.02,1000,1000,0",
        ],
        "512400.SH": [
            "2026-11-18,2.00,2.02,1.98,2.00,1000,500,0",
            "2026-11-19,2.02,2.04,2.00,2.02,1000,500,0",
            "2026-11-20,2.04,2.06,2.02,2.04,1000,500,0",
        ],
        "512880.SH": [
            "2026-11-18,1.50,1.52,1.48,1.50,1000,700,0",
        ],
    }.items():
        sdir = tdx_dir / sym
        sdir.mkdir(parents=True)
        (sdir / "20261123_120000.csv").write_text(
            "date,open,high,low,close,amount,volume,reserved\n"
            + "\n".join(rows)
            + "\n",
            encoding="utf-8",
        )

    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text(
        json.dumps(
            {
                "universe": [
                    {"symbol": "159330.SZ", "current_regime": "BENCHMARK"},
                    {"symbol": "512400.SH", "current_regime": "TREND_UP"},
                    {"symbol": "512880.SH", "current_regime": "UNRATED"},
                ]
            }
        ),
        encoding="utf-8",
    )

    dividend_path = tmp_path / "dividends.json"
    dividend_path.write_text(
        json.dumps(
            {
                "events_by_symbol": {
                    "159330.SZ": [],
                    "512400.SH": [],
                    "512880.SH": [],
                }
            }
        ),
        encoding="utf-8",
    )

    config = {
        "project_name": "sector_paper_sim_test",
        "upstream_paths": {
            "tdx_daily_dir": str(tdx_dir),
            "dividend_events_file": str(dividend_path),
            "watchlist_file": str(watchlist_path),
        },
        "account_defaults": {"initial_capital": 100000.0},
        "strategies": {},
        "universe": {
            "benchmark": "159330.SZ",
            "sectors": ["512400.SH", "512880.SH"],
        },
    }

    bundle = load_market_bundle(config)
    assert bundle.common_end_date == "2026-11-18"
    assert bundle.common_trading_dates == ["2026-11-18"]
    assert bundle.metadata["latest_end_date"] == "2026-11-20"
    assert len(bundle.lag_warnings) == 1
    warning = bundle.lag_warnings[0]
    assert warning["symbol"] == "512880.SH"
    assert warning["symbol_end_date"] == "2026-11-18"
    assert warning["latest_end_date"] == "2026-11-20"
    assert warning["lag_trading_days"] == 2


def test_build_forward_adjusted_daily_split_and_dividend() -> None:
    """Forward adjustment combines split_events and dividend_events without mutating raw bars."""
    raw_df = pd.DataFrame(
        {
            "date": ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-06"],
            "open": [2.00, 2.00, 1.00, 0.95],
            "high": [2.04, 2.02, 1.02, 0.98],
            "low": [1.98, 1.98, 0.98, 0.94],
            "close": [2.00, 2.00, 1.00, 0.96],
            "amount": [1000.0, 1000.0, 1000.0, 1000.0],
            "volume": [500.0, 500.0, 1000.0, 1050.0],
        }
    )
    raw_copy = raw_df.copy()

    adj_df = build_forward_adjusted_daily(
        raw_df,
        split_events=[{"effective_date": "2026-07-03", "factor": 0.5}],
        dividend_events={"2026-07-06": 0.05},
    )

    # Raw input DataFrame must remain untouched
    pd.testing.assert_frame_equal(raw_df, raw_copy)

    # 2026-07-06 dividend step factor: (1.00 - 0.05) / 1.00 = 0.95 (applies to dates < 2026-07-06)
    # 2026-07-03 split factor: 0.5 (applies to dates < 2026-07-03)
    expected_factors = [0.5 * 0.95, 0.5 * 0.95, 0.95, 1.0]
    assert adj_df["adj_factor"].tolist() == pytest.approx(expected_factors)
    assert adj_df["close"].tolist() == pytest.approx([0.95, 0.95, 0.95, 0.96])


def test_load_tdx_daily_missing_dir_or_batch_raises(tmp_path: Path) -> None:
    """Missing symbol directory or absent YYYYMMDD_HHMMSS.csv must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_tdx_daily(tmp_path, "999999.SH")

    empty_dir = tmp_path / "512480.SH"
    empty_dir.mkdir(parents=True)
    (empty_dir / "not_a_batch.csv").write_text(
        "date,open,high,low,close,amount,volume\n"
        "2026-11-20,1.0,1.0,1.0,1.0,100.0,100\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        load_tdx_daily(tmp_path, "512480.SH")


def test_real_512480_split_forward_adjustment() -> None:
    """Verify real 512480.SH 2026-07-03 1-for-2 split forward adjustment matches spot post-split."""
    bundle = load_market_bundle(DEFAULT_SIM_CONFIG_PATH)
    raw = bundle.bars["512480.SH"].set_index("date")
    adj = bundle.adjusted_bars["512480.SH"].set_index("date")

    assert raw.loc["2026-07-02", "close"] == pytest.approx(2.700)
    assert adj.loc["2026-07-02", "adj_factor"] == pytest.approx(0.5)
    assert adj.loc["2026-07-02", "close"] == pytest.approx(1.350)

    assert raw.loc["2026-07-03", "close"] == pytest.approx(1.331)
    assert adj.loc["2026-07-03", "adj_factor"] == pytest.approx(1.0)
    assert adj.loc["2026-07-03", "close"] == pytest.approx(1.331)

