"""Read-only upstream data loading and trading calendar alignment for sector_paper_sim.

All functions in this module strictly perform read-only access against local
config and upstream QuantData / sector_radar artifacts without mutating any
upstream files or falling back to network sources.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import date
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SIM_CONFIG_PATH = REPO_ROOT / "project_content" / "config" / "sim_config.json"

TDX_BATCH_CSV_PATTERN = re.compile(r"^\d{8}_\d{6}\.csv$")
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

TDX_REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "amount",
    "volume",
)
TDX_PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
TDX_NUMERIC_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "amount",
    "volume",
)

ALLOWED_REGIMES: frozenset[str] = frozenset(
    {
        "BENCHMARK",
        "TREND_UP",
        "RANGE_GRID",
        "TRANSITION",
        "TRAP_OFF",
        "UNRATED",
    }
)


@dataclass(frozen=True)
class MarketBundle:
    """Unified read-only market dataset, metadata, and aligned trading calendar."""

    config: dict[str, Any]
    benchmark: str
    sectors: list[str]
    symbols: list[str]
    bars: dict[str, pd.DataFrame]
    adjusted_bars: dict[str, pd.DataFrame]
    watchlist: dict[str, dict[str, Any]]
    dividends: dict[str, dict[str, float]]
    common_end_date: str
    common_trading_dates: list[str]
    metadata: dict[str, Any]

    @property
    def lag_warnings(self) -> list[dict[str, Any]]:
        """Explicit warnings for symbols lagging behind the latest trading date."""
        return self.metadata["lag_warnings"]


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label} ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object in {label}: {path}")
    return payload


def _normalize_iso_date(raw_value: Any, label: str) -> str:
    if not isinstance(raw_value, str):
        raise ValueError(f"{label} must be a YYYY-MM-DD string, got {raw_value!r}")
    text = raw_value.strip()
    if not ISO_DATE_PATTERN.match(text):
        raise ValueError(f"{label} must match YYYY-MM-DD, got {raw_value!r}")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid calendar date: {raw_value!r}") from exc
    return parsed.isoformat()


def _parse_positive_float(raw_value: Any, label: str) -> float:
    if isinstance(raw_value, bool) or raw_value in (None, ""):
        raise ValueError(f"{label} must be a positive number, got {raw_value!r}")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not a valid number: {raw_value!r}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be finite and > 0, got {raw_value!r}")
    return value


def _validate_sim_config(config: Mapping[str, Any], source_label: str) -> dict[str, Any]:
    required_top_keys = (
        "project_name",
        "upstream_paths",
        "account_defaults",
        "strategies",
        "universe",
    )
    for key in required_top_keys:
        if key not in config:
            raise ValueError(f"Missing required config key '{key}' in {source_label}")

    upstream = config["upstream_paths"]
    if not isinstance(upstream, Mapping):
        raise ValueError(f"'upstream_paths' must be a JSON object in {source_label}")
    for key in ("tdx_daily_dir", "dividend_events_file", "watchlist_file"):
        val = upstream.get(key)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(
                f"'upstream_paths.{key}' must be a non-empty string in {source_label}"
            )

    universe = config["universe"]
    if not isinstance(universe, Mapping):
        raise ValueError(f"'universe' must be a JSON object in {source_label}")
    benchmark = universe.get("benchmark")
    sectors = universe.get("sectors")
    if not isinstance(benchmark, str) or not benchmark.strip():
        raise ValueError(f"'universe.benchmark' must be a non-empty string in {source_label}")
    if (
        not isinstance(sectors, list)
        or not sectors
        or any(not isinstance(sym, str) or not sym.strip() for sym in sectors)
    ):
        raise ValueError(
            f"'universe.sectors' must be a non-empty list of symbol strings in {source_label}"
        )

    return dict(config)


def load_sim_config(
    config_path: str | Path = DEFAULT_SIM_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate sim_config.json."""
    path = Path(config_path)
    raw = _read_json_object(path, "sim_config")
    return _validate_sim_config(raw, str(path))


def find_latest_tdx_csv(tdx_daily_dir: str | Path, symbol: str) -> Path:
    """Locate the latest YYYYMMDD_HHMMSS.csv batch file under {tdx_daily_dir}/{symbol}/."""
    symbol_dir = Path(tdx_daily_dir) / symbol
    if not symbol_dir.is_dir():
        raise FileNotFoundError(
            f"TDX daily directory not found for {symbol}: {symbol_dir}"
        )

    matched_files = sorted(
        (
            entry
            for entry in symbol_dir.iterdir()
            if entry.is_file() and TDX_BATCH_CSV_PATTERN.match(entry.name)
        ),
        key=lambda p: p.name,
    )
    if not matched_files:
        raise FileNotFoundError(
            f"No YYYYMMDD_HHMMSS.csv files found for {symbol} under {symbol_dir}"
        )
    return matched_files[-1]


def load_tdx_daily(tdx_daily_dir: str | Path, symbol: str) -> pd.DataFrame:
    """Load, clean, sort, deduplicate, and validate the latest TDX daily CSV for a symbol."""
    csv_path = find_latest_tdx_csv(tdx_daily_dir, symbol)
    raw_df = pd.read_csv(csv_path, dtype={"date": str})

    missing_cols = [col for col in TDX_REQUIRED_COLUMNS if col not in raw_df.columns]
    if missing_cols:
        raise ValueError(
            f"{symbol} TDX CSV missing required columns {missing_cols}: {csv_path}"
        )
    if raw_df.empty:
        raise ValueError(f"{symbol} TDX CSV has no data rows: {csv_path}")

    df = raw_df.loc[:, list(TDX_REQUIRED_COLUMNS)].copy()

    raw_dates = df["date"].fillna("").astype(str).str.strip()
    parsed_dates = pd.to_datetime(raw_dates, format="mixed", errors="coerce")
    if parsed_dates.isna().any():
        bad_count = int(parsed_dates.isna().sum())
        raise ValueError(
            f"{symbol} TDX CSV contains {bad_count} invalid date value(s): {csv_path}"
        )
    df["date"] = parsed_dates.dt.strftime("%Y-%m-%d").astype(str)

    for col in TDX_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    for col in TDX_PRICE_COLUMNS:
        values = df[col].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            bad_count = int((~np.isfinite(values)).sum())
            raise ValueError(
                f"{symbol} TDX CSV contains {bad_count} NaN/non-finite value(s) in '{col}': {csv_path}"
            )
        if (values <= 0.0).any():
            bad_count = int((values <= 0.0).sum())
            raise ValueError(
                f"{symbol} TDX CSV contains {bad_count} non-positive (<= 0) value(s) in '{col}': {csv_path}"
            )

    for col in ("amount", "volume"):
        values = df[col].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            bad_count = int((~np.isfinite(values)).sum())
            raise ValueError(
                f"{symbol} TDX CSV contains {bad_count} NaN/non-finite value(s) in '{col}': {csv_path}"
            )
        if (values < 0.0).any():
            bad_count = int((values < 0.0).sum())
            raise ValueError(
                f"{symbol} TDX CSV contains {bad_count} negative value(s) in '{col}': {csv_path}"
            )

    df = (
        df.sort_values("date", kind="mergesort")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    df.attrs["symbol"] = symbol
    df.attrs["source_csv"] = str(csv_path)
    return df


def _parse_split_events(raw_events: Any, symbol: str) -> list[dict[str, Any]]:
    if raw_events is None:
        return []
    if not isinstance(raw_events, list):
        raise ValueError(f"{symbol} split_events must be a list, got {type(raw_events).__name__}")

    parsed: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{symbol} split_events[{idx}] must be an object")
        effective_date = _normalize_iso_date(
            raw.get("effective_date"), f"{symbol} split_events[{idx}].effective_date"
        )
        raw_record = raw.get("record_date")
        record_date = (
            _normalize_iso_date(raw_record, f"{symbol} split_events[{idx}].record_date")
            if raw_record not in (None, "")
            else None
        )
        factor = _parse_positive_float(
            raw.get("factor"), f"{symbol} split_events[{idx}].factor"
        )
        parsed.append(
            {
                "effective_date": effective_date,
                "record_date": record_date,
                "factor": factor,
                "ratio_desc": str(raw.get("ratio_desc") or ""),
                "source": str(raw.get("source") or ""),
                "note": str(raw.get("note") or ""),
            }
        )
    parsed.sort(key=lambda ev: ev["effective_date"])
    return parsed


def load_watchlist(watchlist_file: str | Path) -> dict[str, dict[str, Any]]:
    """Load watchlist.json indexed by symbol with validated regime and split_events."""
    path = Path(watchlist_file)
    payload = _read_json_object(path, "watchlist_file")
    universe = payload.get("universe")
    if not isinstance(universe, list) or not universe:
        raise ValueError(f"watchlist 'universe' must be a non-empty list: {path}")

    result: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(universe):
        if not isinstance(item, Mapping):
            raise ValueError(f"watchlist universe[{idx}] must be a JSON object: {path}")
        raw_symbol = item.get("symbol")
        if not isinstance(raw_symbol, str) or not raw_symbol.strip():
            raise ValueError(f"watchlist universe[{idx}] missing valid 'symbol': {path}")
        symbol = raw_symbol.strip()
        if symbol in result:
            raise ValueError(f"Duplicate symbol '{symbol}' in watchlist: {path}")

        raw_regime = item.get("current_regime")
        if not isinstance(raw_regime, str) or raw_regime.strip() not in ALLOWED_REGIMES:
            raise ValueError(
                f"{symbol} has unsupported current_regime {raw_regime!r}; "
                f"allowed={sorted(ALLOWED_REGIMES)}"
            )
        regime = raw_regime.strip()

        split_events = _parse_split_events(item.get("split_events"), symbol)
        result[symbol] = {
            "symbol": symbol,
            "name": str(item.get("name") or ""),
            "role": str(item.get("role") or ""),
            "current_regime": regime,
            "split_events": split_events,
        }
    return result


def load_dividend_events(
    dividend_events_file: str | Path,
) -> dict[str, dict[str, float]]:
    """Load watchlist_dividend_events.json as {symbol: {ex_date: cash_per_unit}}."""
    path = Path(dividend_events_file)
    payload = _read_json_object(path, "dividend_events_file")
    events_by_symbol = payload.get("events_by_symbol")
    if not isinstance(events_by_symbol, Mapping):
        raise ValueError(f"Missing or invalid 'events_by_symbol' object in {path}")

    result: dict[str, dict[str, float]] = {}
    for raw_symbol, raw_events in events_by_symbol.items():
        if not isinstance(raw_symbol, str) or not raw_symbol.strip():
            raise ValueError(f"Invalid symbol key in events_by_symbol: {raw_symbol!r}")
        symbol = raw_symbol.strip()
        if not isinstance(raw_events, list):
            raise ValueError(
                f"{symbol} entry in events_by_symbol must be a list: {path}"
            )

        by_date: dict[str, float] = {}
        for idx, ev in enumerate(raw_events):
            if not isinstance(ev, Mapping):
                raise ValueError(
                    f"{symbol} dividend event [{idx}] must be an object: {path}"
                )
            ex_date = _normalize_iso_date(
                ev.get("ex_date"), f"{symbol} dividend[{idx}].ex_date"
            )
            raw_per_unit = ev.get("cash_per_unit")
            raw_per_10 = ev.get("cash_per_10_units")
            if raw_per_unit is None and raw_per_10 is None:
                raise ValueError(
                    f"{symbol} dividend[{idx}] missing 'cash_per_unit' and 'cash_per_10_units': {path}"
                )

            cash_per_unit = (
                _parse_positive_float(
                    raw_per_unit, f"{symbol} dividend[{idx}].cash_per_unit"
                )
                if raw_per_unit is not None
                else None
            )
            cash_per_10 = (
                _parse_positive_float(
                    raw_per_10, f"{symbol} dividend[{idx}].cash_per_10_units"
                )
                if raw_per_10 is not None
                else None
            )
            if cash_per_unit is not None and cash_per_10 is not None:
                if abs(cash_per_unit - cash_per_10 / 10.0) > 1e-6:
                    raise ValueError(
                        f"{symbol} dividend[{idx}] inconsistent cash_per_unit={cash_per_unit} "
                        f"vs cash_per_10_units={cash_per_10}: {path}"
                    )
            resolved_cash = (
                cash_per_unit
                if cash_per_unit is not None
                else float(cash_per_10) / 10.0
            )
            if ex_date in by_date:
                raise ValueError(
                    f"{symbol} has duplicate dividend ex_date '{ex_date}': {path}"
                )
            by_date[ex_date] = resolved_cash

        result[symbol] = dict(sorted(by_date.items()))
    return result


def build_forward_adjusted_daily(
    raw_df: pd.DataFrame,
    split_events: Sequence[Mapping[str, Any]] | None = None,
    dividend_events: Mapping[str, float] | None = None,
    cutoff_date: str | None = None,
) -> pd.DataFrame:
    """Compute forward-adjusted OHLC bars matching sector_radar's split + dividend formula.

    The raw input DataFrame is never mutated. Prices on and after the last active
    corporate action carry adj_factor == 1.0 so latest comparable prices match spot prices.
    """
    if raw_df.empty:
        raise ValueError("Cannot build forward-adjusted bars from an empty DataFrame")

    dates = raw_df["date"].astype(str).tolist()
    n = len(dates)
    resolved_cutoff = (
        _normalize_iso_date(cutoff_date, "cutoff_date")
        if cutoff_date is not None
        else dates[-1]
    )

    split_multipliers = np.ones(n, dtype=float)
    if split_events:
        active_splits = [
            ev
            for ev in split_events
            if _normalize_iso_date(ev["effective_date"], "split.effective_date")
            <= resolved_cutoff
        ]
        for ev in active_splits:
            eff_date = _normalize_iso_date(ev["effective_date"], "split.effective_date")
            factor = _parse_positive_float(ev["factor"], "split.factor")
            idx = bisect_left(dates, eff_date)
            if idx > 0:
                split_multipliers[:idx] *= factor

    div_multipliers = np.ones(n, dtype=float)
    if dividend_events:
        raw_closes = raw_df["close"].to_numpy(dtype=float)
        step_factors = np.ones(n, dtype=float)
        has_active_div = False
        for raw_ex_date, raw_cash in sorted(dividend_events.items()):
            ex_date = _normalize_iso_date(raw_ex_date, "dividend.ex_date")
            if ex_date > resolved_cutoff:
                continue
            cash = _parse_positive_float(raw_cash, f"dividend[{ex_date}].cash_per_unit")
            idx = bisect_left(dates, ex_date)
            if idx <= 0:
                continue
            prev_close = float(raw_closes[idx - 1])
            if prev_close <= 0.0 or cash >= prev_close:
                raise ValueError(
                    f"Dividend on {ex_date} ({cash}) cannot be applied against prev_close={prev_close}"
                )
            step_factors[idx - 1] *= (prev_close - cash) / prev_close
            has_active_div = True

        if has_active_div:
            running = 1.0
            for i in range(n - 1, -1, -1):
                running *= float(step_factors[i])
                div_multipliers[i] = running

    adj_factors = split_multipliers * div_multipliers
    adj_df = raw_df.copy()
    for col in TDX_PRICE_COLUMNS:
        adj_df[col] = adj_df[col].to_numpy(dtype=float) * adj_factors
    adj_df["adj_factor"] = adj_factors
    return adj_df


def load_market_bundle(config: Mapping[str, Any] | str | Path) -> MarketBundle:
    """Load all configured ETFs, watchlist, dividends, and aligned common calendar."""
    resolved_config = (
        load_sim_config(config)
        if isinstance(config, (str, Path))
        else _validate_sim_config(config, "config")
    )

    benchmark = resolved_config["universe"]["benchmark"].strip()
    sectors = [sym.strip() for sym in resolved_config["universe"]["sectors"]]
    symbols = [benchmark, *sectors]
    if len(set(symbols)) != len(symbols):
        raise ValueError(f"Duplicate symbols detected in universe: {symbols}")

    upstream = resolved_config["upstream_paths"]
    tdx_daily_dir = Path(upstream["tdx_daily_dir"])
    watchlist_file = Path(upstream["watchlist_file"])
    dividend_events_file = Path(upstream["dividend_events_file"])

    watchlist = load_watchlist(watchlist_file)
    missing_wl = [sym for sym in symbols if sym not in watchlist]
    if missing_wl:
        raise ValueError(
            f"Universe symbol(s) missing from watchlist ({watchlist_file}): {missing_wl}"
        )

    all_dividends = load_dividend_events(dividend_events_file)
    dividends: dict[str, dict[str, float]] = {
        sym: dict(all_dividends.get(sym, {})) for sym in symbols
    }

    bars: dict[str, pd.DataFrame] = {
        sym: load_tdx_daily(tdx_daily_dir, sym) for sym in symbols
    }

    end_dates: dict[str, str] = {
        sym: str(df["date"].iloc[-1]) for sym, df in bars.items()
    }
    common_end_date = min(end_dates.values())
    latest_end_date = max(end_dates.values())

    date_sets: dict[str, set[str]] = {
        sym: set(df["date"].tolist()) for sym, df in bars.items()
    }
    common_trading_dates = sorted(set.intersection(*date_sets.values()))
    if not common_trading_dates:
        raise ValueError(
            "No common trading dates found across configured universe symbols"
        )

    union_dates = sorted(set().union(*date_sets.values()))
    lag_warnings: list[dict[str, Any]] = []
    for sym in symbols:
        sym_end = end_dates[sym]
        if sym_end < latest_end_date:
            lag_trading_days = sum(
                1 for dt in union_dates if sym_end < dt <= latest_end_date
            )
            lag_warnings.append(
                {
                    "symbol": sym,
                    "symbol_end_date": sym_end,
                    "latest_end_date": latest_end_date,
                    "lag_trading_days": lag_trading_days,
                    "message": (
                        f"{sym} latest trading date {sym_end} lags behind "
                        f"{latest_end_date} by {lag_trading_days} trading day(s)"
                    ),
                }
            )

    common_start_date = common_trading_dates[0]
    benchmark_window_dates = {
        dt
        for dt in date_sets[benchmark]
        if common_start_date <= dt <= common_end_date
    }
    missing_benchmark_dates: dict[str, list[str]] = {}
    for sym in symbols:
        missing = sorted(benchmark_window_dates - date_sets[sym])
        if missing:
            missing_benchmark_dates[sym] = missing

    adjusted_bars: dict[str, pd.DataFrame] = {
        sym: build_forward_adjusted_daily(
            bars[sym],
            split_events=watchlist[sym]["split_events"],
            dividend_events=dividends[sym],
            cutoff_date=common_end_date,
        )
        for sym in symbols
    }

    source_files: dict[str, str] = {
        sym: str(bars[sym].attrs.get("source_csv", "")) for sym in symbols
    }
    symbol_stats: dict[str, dict[str, Any]] = {
        sym: {
            "row_count": len(bars[sym]),
            "start_date": str(bars[sym]["date"].iloc[0]),
            "end_date": end_dates[sym],
            "latest_csv": Path(source_files[sym]).name if source_files[sym] else "",
            "current_regime": watchlist[sym]["current_regime"],
            "split_event_count": len(watchlist[sym]["split_events"]),
            "dividend_event_count": len(dividends[sym]),
        }
        for sym in symbols
    }

    metadata: dict[str, Any] = {
        "common_start_date": common_start_date,
        "common_end_date": common_end_date,
        "latest_end_date": latest_end_date,
        "common_trading_day_count": len(common_trading_dates),
        "lag_warnings": lag_warnings,
        "missing_benchmark_dates": missing_benchmark_dates,
        "source_files": source_files,
        "symbol_stats": symbol_stats,
    }

    return MarketBundle(
        config=resolved_config,
        benchmark=benchmark,
        sectors=sectors,
        symbols=symbols,
        bars=bars,
        adjusted_bars=adjusted_bars,
        watchlist=watchlist,
        dividends=dividends,
        common_end_date=common_end_date,
        common_trading_dates=common_trading_dates,
        metadata=metadata,
    )
