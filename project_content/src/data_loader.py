"""Read-only upstream data loading and trading calendar alignment for sector_paper_sim.

All functions in this module strictly perform read-only access against local
config and upstream QuantData / sector_radar artifacts without mutating any
upstream files or falling back to network sources.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIM_CONFIG_PATH = REPO_ROOT / "project_content" / "config" / "sim_config.json"

TDX_BATCH_CSV_PATTERN = re.compile(r"^\d{8}_\d{6}\.csv$")
TRADE_CALENDAR_CSV_PATTERN = re.compile(r"^\d{8}_\d{6}\.csv$")
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_DATETIME_TZ_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CORPORATE_ACTIONS_SCHEMA_VERSION = "2.0.0"
REGIME_HISTORY_SCHEMA_VERSION = "2.0.0"
REGIME_RECORD_TYPES: frozenset[str] = frozenset({"LIVE", "BACKFILL"})
CHINA_TZ = timezone(timedelta(hours=8))
MARKET_OPEN_TIME = time(9, 30, 0)
ROUTABLE_REGIMES: frozenset[str] = frozenset(
    {"TREND_UP", "RANGE_GRID", "TRANSITION", "TRAP_OFF", "UNRATED"}
)

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


def _parse_tz_aware_iso_datetime(raw_value: Any, label: str) -> datetime:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"{label} must be a timezone-aware ISO 8601 timestamp, got {raw_value!r}")
    text = raw_value.strip()
    if not ISO_DATETIME_TZ_PATTERN.fullmatch(text):
        raise ValueError(
            f"{label} must be a full timezone-aware ISO 8601 timestamp, got {raw_value!r}"
        )
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid ISO 8601 timestamp: {raw_value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include timezone offset: {raw_value!r}")
    return parsed.astimezone(CHINA_TZ)


def _validate_live_recorded_at(
    raw_recorded_at: Any,
    decision_date: str,
    effective_date: str,
    label: str,
) -> str:
    recorded_cst = _parse_tz_aware_iso_datetime(raw_recorded_at, f"{label}.recorded_at")
    dec_dt = date.fromisoformat(decision_date)
    eff_dt = date.fromisoformat(effective_date)
    dec_start = datetime(dec_dt.year, dec_dt.month, dec_dt.day, 0, 0, 0, tzinfo=CHINA_TZ)
    eff_open = datetime(
        eff_dt.year, eff_dt.month, eff_dt.day, MARKET_OPEN_TIME.hour, MARKET_OPEN_TIME.minute, 0, tzinfo=CHINA_TZ
    )
    if recorded_cst < dec_start:
        raise ValueError(
            f"{label}.recorded_at ({recorded_cst.isoformat()}) precedes decision_date ({decision_date})"
        )
    if recorded_cst >= eff_open:
        raise ValueError(
            f"{label}.recorded_at ({recorded_cst.isoformat()}) must precede effective_date market open "
            f"({eff_open.isoformat()})"
        )
    return recorded_cst.isoformat()


@dataclass(frozen=True)
class RegimeHistory:
    """Validated regime history with effective-date lookup and LIVE/BACKFILL provenance."""

    schema_version: str
    anchor_start_date: str
    initial_regimes: dict[str, str]
    records_by_symbol: dict[str, list[dict[str, Any]]]

    def regime_for(self, symbol: str, trade_date: str) -> str:
        """Return the latest regime effective on ``trade_date`` for ``symbol``."""
        resolved_date = _normalize_iso_date(trade_date, "trade_date")
        if resolved_date < self.anchor_start_date:
            raise ValueError(
                f"trade_date {resolved_date} precedes regime-history anchor "
                f"{self.anchor_start_date}"
            )
        if symbol not in self.initial_regimes:
            raise KeyError(f"No initial regime for symbol {symbol!r}")

        result = self.initial_regimes[symbol]
        for record in self.records_by_symbol.get(symbol, []):
            if record["effective_date"] > resolved_date:
                break
            result = record["regime"]
        return result

    def all_records(self) -> list[dict[str, Any]]:
        """Return all regime records sorted deterministically by (effective_date, symbol)."""
        flattened = [
            {"symbol": symbol, **record}
            for symbol, records in self.records_by_symbol.items()
            for record in records
        ]
        flattened.sort(
            key=lambda rec: (
                rec["effective_date"],
                rec.get("decision_date") or "",
                rec.get("record_id") or "",
                rec["symbol"],
            )
        )
        return flattened

    def records_effective_on(self, trade_date: str) -> list[dict[str, Any]]:
        """Return records taking effect at market open on ``trade_date``."""
        resolved_date = _normalize_iso_date(trade_date, "trade_date")
        return [
            record
            for record in self.all_records()
            if record["effective_date"] == resolved_date
        ]

    def validate_for_run_mode(
        self,
        run_mode: str,
        processed_dates: Sequence[str],
        *,
        forward_eval_start_date: str | None = None,
    ) -> None:
        """Enforce formal forward-run vs explicit replay boundary on LIVE/BACKFILL records."""
        if run_mode not in {"formal", "replay"}:
            raise ValueError(f"Unsupported run_mode {run_mode!r}; expected 'formal' or 'replay'")
        stepped_set = {
            _normalize_iso_date(day, "processed_date") for day in processed_dates
        }
        eval_start = (
            _normalize_iso_date(forward_eval_start_date, "forward_eval_start_date")
            if forward_eval_start_date is not None
            else None
        )
        for symbol, records in sorted(self.records_by_symbol.items()):
            for idx, record in enumerate(records):
                label = f"regime_history[{symbol}][{idx}]"
                record_type = record.get("record_type")
                if record_type not in REGIME_RECORD_TYPES:
                    raise ValueError(
                        f"{label} has invalid record_type {record_type!r}; "
                        f"expected one of {sorted(REGIME_RECORD_TYPES)}"
                    )
                effective_date = record["effective_date"]
                decision_date = record.get("decision_date")
                if record_type == "LIVE":
                    if not decision_date:
                        raise ValueError(f"{label} LIVE record lacks decision_date")
                    _validate_live_recorded_at(
                        record.get("recorded_at"),
                        decision_date,
                        effective_date,
                        label,
                    )
                elif run_mode == "formal":
                    if effective_date in stepped_set or (
                        eval_start is not None and effective_date >= eval_start
                    ):
                        raise ValueError(
                            f"正式前瞻运行拒绝消费 BACKFILL 路由记录: {symbol} "
                            f"{record.get('record_id') or '(no-id)'} effective_date={effective_date}; "
                            "历史回放或状态初始化请显式使用 --mode replay"
                        )

    def routing_stats_through(
        self,
        trade_date: str,
        *,
        forward_eval_start_date: str | None = None,
    ) -> dict[str, Any]:
        """Summarize LIVE vs BACKFILL counts through ``trade_date``, excluding BACKFILL from LIVE stats."""
        resolved_date = _normalize_iso_date(trade_date, "trade_date")
        eval_start = (
            _normalize_iso_date(forward_eval_start_date, "forward_eval_start_date")
            if forward_eval_start_date is not None
            else None
        )
        through_records = [
            rec for rec in self.all_records() if rec["effective_date"] <= resolved_date
        ]
        live_records = [rec for rec in through_records if rec.get("record_type") == "LIVE"]
        backfill_records = [rec for rec in through_records if rec.get("record_type") == "BACKFILL"]
        forward_live_records = (
            [rec for rec in live_records if rec["effective_date"] >= eval_start]
            if eval_start is not None
            else []
        )
        return {
            "through_date": resolved_date,
            "forward_eval_start_date": eval_start,
            "live_records_applied": len(live_records),
            "forward_live_records_applied": len(forward_live_records),
            "backfill_records_excluded": len(backfill_records),
            "live_record_ids": [rec.get("record_id") or "" for rec in live_records],
            "backfill_record_ids": [rec.get("record_id") or "" for rec in backfill_records],
        }


@dataclass(frozen=True)
class CorporateActionsLedger:
    """Normalized and validated QuantData corporate-actions ledger."""

    schema_version: str
    as_of: str
    cash_dividends_by_symbol: dict[str, list[dict[str, Any]]]
    splits_by_symbol: dict[str, list[dict[str, Any]]]
    unverified_events: list[dict[str, Any]]


@dataclass(frozen=True)
class MarketBundle:
    """Unified read-only market data, metadata, and QRI calendar alignment."""

    config: dict[str, Any]
    benchmark: str
    sectors: list[str]
    symbols: list[str]
    bars: dict[str, pd.DataFrame]
    adjusted_bars: dict[str, pd.DataFrame]
    watchlist: dict[str, dict[str, Any]]
    dividends: dict[str, dict[str, float]]
    splits: dict[str, list[dict[str, Any]]]
    regime_history: RegimeHistory
    common_end_date: str
    common_trading_dates: list[str]
    authoritative_trading_dates: list[str]
    suspension_dates_by_symbol: dict[str, list[str]]
    lagged_dates_by_symbol: dict[str, list[str]]
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
    for key in (
        "tdx_daily_dir",
        "trade_calendar_dir",
        "corporate_actions_ledger_file",
        "regime_history_file",
        "watchlist_file",
    ):
        val = upstream.get(key)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(
                f"'upstream_paths.{key}' must be a non-empty string in {source_label}"
            )

    if "start_date" not in config:
        raise ValueError(f"Missing required config key 'start_date' in {source_label}")
    start_date = _normalize_iso_date(config["start_date"], "start_date")

    if config.get("forward_eval_start_date") is not None:
        fwd_start = _normalize_iso_date(
            config["forward_eval_start_date"], "forward_eval_start_date"
        )
        if fwd_start < start_date:
            raise ValueError(
                f"forward_eval_start_date {fwd_start} precedes start_date {start_date}"
            )
    if config.get("backfill_replay_window") is not None:
        window = config["backfill_replay_window"]
        if not isinstance(window, Mapping):
            raise ValueError(f"'backfill_replay_window' must be a JSON object in {source_label}")
        win_start = _normalize_iso_date(window.get("start_date"), "backfill_replay_window.start_date")
        win_end = _normalize_iso_date(window.get("end_date"), "backfill_replay_window.end_date")
        if win_start > win_end:
            raise ValueError(
                f"backfill_replay_window start_date {win_start} exceeds end_date {win_end}"
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


def _normalized_regime(raw_value: Any, label: str) -> str:
    if not isinstance(raw_value, str) or raw_value.strip() not in ROUTABLE_REGIMES:
        raise ValueError(
            f"{label} must be one of {sorted(ROUTABLE_REGIMES)}, got {raw_value!r}"
        )
    value = raw_value.strip()
    return "TRANSITION" if value == "UNRATED" else value


def load_corporate_actions_ledger(
    ledger_file: str | Path,
) -> CorporateActionsLedger:
    """Load and validate the versioned QuantData corporate-actions ledger."""
    path = Path(ledger_file)
    payload = _read_json_object(path, "corporate_actions_ledger_file")
    schema_version = payload.get("schema_version")
    if schema_version != CORPORATE_ACTIONS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported corporate-actions schema_version {schema_version!r} "
            f"in {path}; expected {CORPORATE_ACTIONS_SCHEMA_VERSION!r}"
        )
    as_of = _normalize_iso_date(payload.get("as_of"), "corporate_actions.as_of")

    dividends_raw = payload.get("cash_dividends_by_symbol")
    splits_raw = payload.get("splits_by_symbol")
    if not isinstance(dividends_raw, Mapping):
        raise ValueError(
            f"Missing or invalid 'cash_dividends_by_symbol' object in {path}"
        )
    if not isinstance(splits_raw, Mapping):
        raise ValueError(f"Missing or invalid 'splits_by_symbol' object in {path}")

    unverified_events: list[dict[str, Any]] = []

    def parse_symbol_map(
        raw_map: Mapping[str, Any], *, event_kind: str
    ) -> dict[str, list[dict[str, Any]]]:
        parsed_map: dict[str, list[dict[str, Any]]] = {}
        date_key = "ex_date" if event_kind == "dividend" else "effective_date"
        value_key = "cash_per_unit" if event_kind == "dividend" else "factor"
        map_name = (
            "cash_dividends_by_symbol" if event_kind == "dividend" else "splits_by_symbol"
        )

        for raw_symbol, raw_events in raw_map.items():
            if not isinstance(raw_symbol, str) or not raw_symbol.strip():
                raise ValueError(f"Invalid symbol key in {map_name}: {raw_symbol!r}")
            symbol = raw_symbol.strip()
            if not isinstance(raw_events, list):
                raise ValueError(f"{symbol} entry in {map_name} must be a list: {path}")

            parsed_events: list[dict[str, Any]] = []
            for idx, raw_event in enumerate(raw_events):
                if not isinstance(raw_event, Mapping):
                    raise ValueError(
                        f"{symbol} {event_kind}[{idx}] must be an object in {path}"
                    )
                event_date = _normalize_iso_date(
                    raw_event.get(date_key),
                    f"{symbol} {event_kind}[{idx}].{date_key}",
                )
                value = _parse_positive_float(
                    raw_event.get(value_key),
                    f"{symbol} {event_kind}[{idx}].{value_key}",
                )
                status = raw_event.get("event_status")
                if not isinstance(status, str) or status not in {"VERIFIED", "PROVISIONAL"}:
                    raise ValueError(
                        f"{symbol} {event_kind}[{idx}].event_status must be VERIFIED or PROVISIONAL"
                    )
                source_class = raw_event.get("source_class")
                if not isinstance(source_class, str) or not source_class.strip():
                    raise ValueError(
                        f"{symbol} {event_kind}[{idx}].source_class must be non-empty"
                    )
                source_pdf = raw_event.get("source_pdf")
                source_hash = raw_event.get("source_pdf_sha256")
                if status == "VERIFIED":
                    if not isinstance(source_pdf, str) or not source_pdf.strip():
                        raise ValueError(
                            f"{symbol} {event_kind}[{idx}] VERIFIED event lacks source_pdf"
                        )
                    if not isinstance(source_hash, str) or not SHA256_PATTERN.fullmatch(source_hash):
                        raise ValueError(
                            f"{symbol} {event_kind}[{idx}] VERIFIED event has invalid source_pdf_sha256"
                        )
                elif source_pdf is not None or source_hash is not None:
                    if (
                        not isinstance(source_pdf, str) or not source_pdf.strip()
                        or not isinstance(source_hash, str)
                        or not SHA256_PATTERN.fullmatch(source_hash)
                    ):
                        raise ValueError(
                            f"{symbol} {event_kind}[{idx}] PROVISIONAL event source_pdf and "
                            "source_pdf_sha256 must both be null or both contain valid values"
                        )
                event = {
                    **dict(raw_event),
                    "symbol": symbol,
                    "kind": "cash_dividend" if event_kind == "dividend" else "split",
                    date_key: event_date,
                    value_key: value,
                    "event_status": status,
                    "source_class": source_class.strip(),
                    "source_pdf": source_pdf,
                    "source_pdf_sha256": source_hash,
                }
                if status == "VERIFIED":
                    parsed_events.append(event)
                else:
                    unverified_events.append(
                        {
                            **event,
                            "date": event_date,
                            "evidence_gap": (
                                "缺少官方 PDF / SHA-256"
                                if not source_pdf or not source_hash
                                else "事件尚未达到 VERIFIED 状态"
                            ),
                        }
                    )

            parsed_events.sort(key=lambda event: event[date_key])
            parsed_map[symbol] = parsed_events
        return parsed_map

    return CorporateActionsLedger(
        schema_version=schema_version,
        as_of=as_of,
        cash_dividends_by_symbol=parse_symbol_map(dividends_raw, event_kind="dividend"),
        splits_by_symbol=parse_symbol_map(splits_raw, event_kind="split"),
        unverified_events=sorted(
            unverified_events, key=lambda event: (event["date"], event["symbol"], event["kind"])
        ),
    )


def assess_corporate_actions_gate(
    as_of: str,
    target_date: str,
    unverified_events: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply the shared hard gate for ledger coverage and effective unverified events."""
    resolved_as_of = _normalize_iso_date(as_of, "corporate_actions.as_of")
    resolved_target = _normalize_iso_date(target_date, "corporate_actions.target_date")
    reasons: list[str] = []
    if resolved_target > resolved_as_of:
        reasons.append(
            f"公司行为台账覆盖截止 {resolved_as_of} 早于目标交易日 {resolved_target}"
        )
    active_unverified: list[dict[str, Any]] = []
    for raw_event in unverified_events or []:
        effective_date = _normalize_iso_date(
            raw_event.get("date"), "corporate_actions.unverified_event.date"
        )
        if effective_date <= resolved_target:
            active_unverified.append(dict(raw_event))
    if active_unverified:
        reasons.append(f"{len(active_unverified)} 个未 VERIFIED 公司行为已在目标日生效")
        for event in active_unverified:
            reasons.append(
                f"{event.get('event_status')} {event.get('kind')} {event.get('symbol')} "
                f"effective={event.get('date')}: {event.get('evidence_gap') or '证据状态待核验'}"
            )
    return {
        "allowed": not reasons,
        "target_date": resolved_target,
        "as_of": resolved_as_of,
        "reasons": reasons,
        "unverified_events": active_unverified,
    }


def load_regime_history(
    regime_history_file: str | Path,
    *,
    trading_calendar: Sequence[str] | None = None,
) -> RegimeHistory:
    """Load route history, enforce Schema/T+1/LIVE-BACKFILL rules, and expose effective-date lookup."""
    path = Path(regime_history_file)
    payload = _read_json_object(path, "regime_history_file")
    schema_version = payload.get("schema_version")
    if schema_version != REGIME_HISTORY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported regime-history schema_version {schema_version!r} in {path}; "
            f"expected {REGIME_HISTORY_SCHEMA_VERSION!r}"
        )
    anchor = _normalize_iso_date(
        payload.get("anchor_start_date"), "regime_history.anchor_start_date"
    )
    calendar_list = list(trading_calendar) if trading_calendar is not None else None
    calendar_set = set(calendar_list) if calendar_list is not None else None
    if calendar_set is not None and anchor not in calendar_set:
        raise ValueError(
            f"regime_history.anchor_start_date {anchor} is not a BS-01 trading day"
        )

    raw_initial = payload.get("initial_regimes")
    if not isinstance(raw_initial, Mapping) or not raw_initial:
        raise ValueError(f"Missing or invalid 'initial_regimes' object in {path}")

    initial: dict[str, str] = {}
    for raw_symbol, raw_regime in raw_initial.items():
        if not isinstance(raw_symbol, str) or not raw_symbol.strip():
            raise ValueError(f"Invalid symbol key in regime_history.initial_regimes: {raw_symbol!r}")
        symbol = raw_symbol.strip()
        if symbol in initial:
            raise ValueError(f"Duplicate initial regime symbol {symbol!r} in {path}")
        initial[symbol] = _normalized_regime(
            raw_regime, f"regime_history.initial_regimes[{symbol}]"
        )

    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError(f"Missing or invalid 'records' list in {path}")
    records_by_symbol: dict[str, list[dict[str, Any]]] = {
        symbol: [] for symbol in initial
    }
    seen: set[tuple[str, str]] = set()
    seen_ids: set[str] = set()
    for idx, raw_record in enumerate(raw_records):
        label = f"regime_history.records[{idx}]"
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"{label} must be an object in {path}")
        raw_symbol = raw_record.get("symbol")
        if not isinstance(raw_symbol, str) or not raw_symbol.strip():
            raise ValueError(f"{label}.symbol is invalid in {path}")
        symbol = raw_symbol.strip()
        if symbol not in initial:
            raise ValueError(
                f"{label} symbol {symbol!r} has no initial regime"
            )

        record_id = raw_record.get("record_id")
        if record_id is not None:
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"{label}.record_id must be a non-empty string")
            record_id = record_id.strip()
            if record_id in seen_ids:
                raise ValueError(f"Duplicate regime record_id {record_id!r} in {path}")
            seen_ids.add(record_id)

        decision_date = _normalize_iso_date(
            raw_record.get("decision_date"),
            f"{label}.decision_date",
        )
        effective_date = _normalize_iso_date(
            raw_record.get("effective_date"),
            f"{label}.effective_date",
        )
        if decision_date < anchor:
            raise ValueError(
                f"{label} decision_date {decision_date} precedes anchor {anchor}"
            )
        if effective_date <= decision_date:
            raise ValueError(
                f"{label} effective_date {effective_date} must strictly follow decision_date {decision_date}"
            )
        if calendar_list is not None and calendar_set is not None:
            if decision_date not in calendar_set:
                raise ValueError(
                    f"{label} decision_date {decision_date} is not a BS-01 trading day"
                )
            next_idx = bisect_right(calendar_list, decision_date)
            if next_idx >= len(calendar_list):
                raise ValueError(
                    f"BS-01 calendar lacks T+1 trading day after {decision_date} for {label}"
                )
            expected_effective = calendar_list[next_idx]
            if effective_date != expected_effective:
                raise ValueError(
                    f"{label} effective_date {effective_date} must strictly equal BS-01 T+1 "
                    f"{expected_effective} (not calendar +1 day)"
                )

        if (symbol, effective_date) in seen:
            raise ValueError(
                f"Duplicate regime effective_date {effective_date} for {symbol} in {path}"
            )
        seen.add((symbol, effective_date))

        prev_regime = _normalized_regime(
            raw_record.get("prev_regime"),
            f"{label}.prev_regime",
        )
        new_regime = _normalized_regime(
            raw_record.get("new_regime"),
            f"{label}.new_regime",
        )

        record_type = raw_record.get("record_type")
        if not isinstance(record_type, str) or record_type not in REGIME_RECORD_TYPES:
            raise ValueError(
                f"{label}.record_type must be one of {sorted(REGIME_RECORD_TYPES)}, got {record_type!r}"
            )
        if "recorded_at" not in raw_record:
            raise ValueError(f"{label} missing required field 'recorded_at'")
        raw_recorded_at = raw_record.get("recorded_at")
        raw_boundary = raw_record.get("time_evidence_boundary")
        if record_type == "LIVE":
            normalized_recorded_at: str | None = _validate_live_recorded_at(
                raw_recorded_at, decision_date, effective_date, label
            )
            if raw_boundary is not None and (not isinstance(raw_boundary, str) or not raw_boundary.strip()):
                raise ValueError(f"{label}.time_evidence_boundary must be non-empty if provided")
        else:
            if raw_recorded_at is None:
                if not isinstance(raw_boundary, str) or not raw_boundary.strip():
                    raise ValueError(
                        f"{label} BACKFILL record with recorded_at=null requires non-empty time_evidence_boundary"
                    )
                normalized_recorded_at = None
            else:
                normalized_recorded_at = _parse_tz_aware_iso_datetime(
                    raw_recorded_at, f"{label}.recorded_at"
                ).isoformat()
                if raw_boundary is not None and (not isinstance(raw_boundary, str) or not raw_boundary.strip()):
                    raise ValueError(f"{label}.time_evidence_boundary must be non-empty if provided")

        records_by_symbol[symbol].append(
            {
                "record_id": record_id or "",
                "decision_date": decision_date,
                "effective_date": effective_date,
                "prev_regime": prev_regime,
                "regime": new_regime,
                "record_type": record_type,
                "recorded_at": normalized_recorded_at,
                "time_evidence_boundary": raw_boundary.strip() if isinstance(raw_boundary, str) else None,
            }
        )

    for symbol, records in records_by_symbol.items():
        records.sort(key=lambda record: (record["effective_date"], record["decision_date"], record["record_id"]))
        expected_prev = initial[symbol]
        for rec in records:
            if rec["prev_regime"] != expected_prev:
                raise ValueError(
                    f"regime_history {symbol} effective_date={rec['effective_date']} "
                    f"prev_regime {rec['prev_regime']!r} breaks chain from {expected_prev!r}"
                )
            expected_prev = rec["regime"]
    return RegimeHistory(
        schema_version=schema_version,
        anchor_start_date=anchor,
        initial_regimes=initial,
        records_by_symbol=records_by_symbol,
    )


def _read_trade_calendar(
    trade_calendar_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[list[str], Path, list[str]]:
    """Read the newest valid BS-01 CSV covering the requested full date range."""
    directory = Path(trade_calendar_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"trade_calendar_dir not found: {directory}")
    candidates = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and TRADE_CALENDAR_CSV_PATTERN.match(path.name)
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No YYYYMMDD_HHMMSS.csv BS-01 calendar files found under {directory}"
        )

    resolved_start = (
        _normalize_iso_date(start_date, "calendar.start_date")
        if start_date is not None
        else None
    )
    resolved_end = (
        _normalize_iso_date(end_date, "calendar.end_date")
        if end_date is not None
        else None
    )
    failures: list[str] = []
    for path in candidates:
        try:
            frame = pd.read_csv(
                path, dtype={"calendar_date": str, "is_trading_day": str}
            )
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            failures.append(f"{path.name}: {exc}")
            continue
        required = {"calendar_date", "is_trading_day"}
        if not required.issubset(frame.columns):
            failures.append(
                f"{path.name}: missing columns {sorted(required - set(frame.columns))}"
            )
            continue
        if frame.empty:
            failures.append(f"{path.name}: no calendar rows")
            continue

        raw_dates = frame["calendar_date"].fillna("").astype(str).str.strip()
        parsed_dates = pd.to_datetime(raw_dates, format="%Y-%m-%d", errors="coerce")
        if parsed_dates.isna().any():
            failures.append(f"{path.name}: invalid calendar_date values")
            continue
        dates = parsed_dates.dt.strftime("%Y-%m-%d").tolist()
        if len(dates) != len(set(dates)):
            failures.append(f"{path.name}: duplicate calendar_date values")
            continue
        if dates != sorted(dates):
            failures.append(f"{path.name}: calendar_date values are not sorted")
            continue
        expected_days = pd.date_range(dates[0], dates[-1], freq="D").strftime(
            "%Y-%m-%d"
        ).tolist()
        if dates != expected_days:
            failures.append(f"{path.name}: calendar has missing calendar_date rows")
            continue

        flags = frame["is_trading_day"].fillna("").astype(str).str.strip()
        if not flags.isin({"0", "1"}).all():
            failures.append(f"{path.name}: is_trading_day must contain only 0/1")
            continue
        trading_dates = [dt for dt, flag in zip(dates, flags) if flag == "1"]
        if not trading_dates:
            failures.append(f"{path.name}: no is_trading_day == 1 rows")
            continue
        if resolved_start is not None and dates[0] > resolved_start:
            failures.append(
                f"{path.name}: begins at {dates[0]}, after required {resolved_start}"
            )
            continue
        if resolved_end is not None and dates[-1] < resolved_end:
            failures.append(
                f"{path.name}: ends at {dates[-1]}, before required {resolved_end}"
            )
            continue
        if resolved_start is not None and resolved_end is not None:
            if not any(resolved_start <= dt <= resolved_end for dt in trading_dates):
                failures.append(
                    f"{path.name}: no trading days in required interval "
                    f"{resolved_start}..{resolved_end}"
                )
                continue
        return trading_dates, path, dates

    detail = "; ".join(failures)
    raise ValueError(
        f"No valid BS-01 calendar covers the requested interval in {directory}. "
        f"Checked {len(candidates)} file(s): {detail}"
    )


def load_trade_calendar(
    trade_calendar_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[str]:
    """Load authoritative BS-01 dates with ``is_trading_day == 1``."""
    trading_dates, _, _ = _read_trade_calendar(
        trade_calendar_dir, start_date=start_date, end_date=end_date
    )
    if start_date is not None:
        resolved_start = _normalize_iso_date(start_date, "calendar.start_date")
        trading_dates = [dt for dt in trading_dates if dt >= resolved_start]
    if end_date is not None:
        resolved_end = _normalize_iso_date(end_date, "calendar.end_date")
        trading_dates = [dt for dt in trading_dates if dt <= resolved_end]
    return trading_dates


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
    """Load configured data and align it to the authoritative BS-01 calendar."""
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
    ledger_file = Path(upstream["corporate_actions_ledger_file"])
    calendar_dir = Path(upstream["trade_calendar_dir"])
    regime_history_file = Path(upstream["regime_history_file"])

    watchlist = load_watchlist(watchlist_file)
    missing_watchlist = [sym for sym in symbols if sym not in watchlist]
    if missing_watchlist:
        raise ValueError(
            f"Universe symbol(s) missing from watchlist ({watchlist_file}): "
            f"{missing_watchlist}"
        )

    ledger = load_corporate_actions_ledger(ledger_file)
    missing_dividends = [
        sym for sym in symbols if sym not in ledger.cash_dividends_by_symbol
    ]
    missing_splits = [sym for sym in symbols if sym not in ledger.splits_by_symbol]
    if missing_dividends or missing_splits:
        raise ValueError(
            "Corporate-actions ledger does not cover the configured universe: "
            f"missing dividends={missing_dividends}, missing splits={missing_splits}"
        )

    dividends: dict[str, dict[str, float]] = {}
    splits: dict[str, list[dict[str, Any]]] = {}
    for symbol in symbols:
        cash_by_date: dict[str, float] = {}
        for event in ledger.cash_dividends_by_symbol[symbol]:
            ex_date = event["ex_date"]
            cash_by_date[ex_date] = cash_by_date.get(ex_date, 0.0) + float(
                event["cash_per_unit"]
            )
        dividends[symbol] = dict(sorted(cash_by_date.items()))
        splits[symbol] = [dict(event) for event in ledger.splits_by_symbol[symbol]]
        # Keep the pre-2t03 engine's watchlist shape while making the ledger the
        # sole authoritative source of split events.
        watchlist[symbol] = dict(watchlist[symbol])
        watchlist[symbol]["split_events"] = [dict(event) for event in splits[symbol]]

    bars: dict[str, pd.DataFrame] = {
        sym: load_tdx_daily(tdx_daily_dir, sym) for sym in symbols
    }
    end_dates = {sym: str(df["date"].iloc[-1]) for sym, df in bars.items()}
    start_dates = {sym: str(df["date"].iloc[0]) for sym, df in bars.items()}
    common_start_date = max(start_dates.values())
    common_end_date = min(end_dates.values())
    latest_end_date = max(end_dates.values())
    if common_start_date > common_end_date:
        raise ValueError(
            "Configured symbols have no synchronized date interval: "
            f"{common_start_date}..{common_end_date}"
        )

    all_calendar_dates, calendar_file, all_calendar_rows = _read_trade_calendar(
        calendar_dir, start_date=common_start_date, end_date=latest_end_date
    )
    regime_history = load_regime_history(
        regime_history_file, trading_calendar=all_calendar_dates
    )
    missing_regimes = [sym for sym in sectors if sym not in regime_history.initial_regimes]
    if missing_regimes:
        raise ValueError(
            f"Regime history is missing initial routes for configured sectors: "
            f"{missing_regimes} ({regime_history_file})"
        )
    trading_calendar = [
        dt for dt in all_calendar_dates if common_start_date <= dt <= latest_end_date
    ]
    if not trading_calendar:
        raise ValueError(
            f"No BS-01 trading dates cover {common_start_date}..{latest_end_date}"
        )
    calendar_set = set(all_calendar_dates)
    for symbol, frame in bars.items():
        off_calendar = sorted(set(frame["date"].tolist()) - calendar_set)
        if off_calendar:
            raise ValueError(
                f"{symbol} TDX rows contain dates absent from the BS-01 calendar: "
                f"{off_calendar[:5]}"
            )

    common_trading_dates = [
        dt for dt in trading_calendar if dt <= common_end_date
    ]
    authoritative_trading_dates = trading_calendar
    date_sets = {sym: set(df["date"].tolist()) for sym, df in bars.items()}

    lag_warnings: list[dict[str, Any]] = []
    lagged_dates_by_symbol: dict[str, list[str]] = {}
    for symbol in symbols:
        sym_end = end_dates[symbol]
        lagged_dates = [
            dt for dt in authoritative_trading_dates if sym_end < dt <= latest_end_date
        ]
        lagged_dates_by_symbol[symbol] = lagged_dates
        if sym_end < latest_end_date:
            lag_warnings.append(
                {
                    "symbol": symbol,
                    "symbol_end_date": sym_end,
                    "latest_end_date": latest_end_date,
                    "lag_trading_days": len(lagged_dates),
                    "message": (
                        f"{symbol} latest CSV date {sym_end} lags behind "
                        f"{latest_end_date} by {len(lagged_dates)} BS-01 trading day(s)"
                    ),
                }
            )

    suspension_dates_by_symbol = {
        symbol: [
            dt for dt in common_trading_dates if dt not in date_sets[symbol]
        ]
        for symbol in symbols
    }
    suspension_dates_by_symbol = {
        symbol: dates
        for symbol, dates in suspension_dates_by_symbol.items()
        if dates
    }
    missing_trading_dates = {
        symbol: [dt for dt in common_trading_dates if dt not in date_sets[symbol]]
        for symbol in symbols
    }
    missing_trading_dates = {
        symbol: dates for symbol, dates in missing_trading_dates.items() if dates
    }

    adjusted_bars: dict[str, pd.DataFrame] = {
        symbol: build_forward_adjusted_daily(
            bars[symbol],
            split_events=splits[symbol],
            dividend_events=dividends[symbol],
            cutoff_date=common_end_date,
        )
        for symbol in symbols
    }

    source_files = {
        sym: str(bars[sym].attrs.get("source_csv", "")) for sym in symbols
    }
    symbol_stats: dict[str, dict[str, Any]] = {
        sym: {
            "row_count": len(bars[sym]),
            "start_date": start_dates[sym],
            "end_date": end_dates[sym],
            "latest_csv": Path(source_files[sym]).name if source_files[sym] else "",
            "current_regime": watchlist[sym]["current_regime"],
            "split_event_count": len(splits[sym]),
            "dividend_event_count": len(ledger.cash_dividends_by_symbol[sym]),
            "suspension_dates": suspension_dates_by_symbol.get(sym, []),
            "lagged_dates": lagged_dates_by_symbol[sym],
        }
        for sym in symbols
    }
    metadata: dict[str, Any] = {
        "common_start_date": common_start_date,
        "common_end_date": common_end_date,
        "latest_end_date": latest_end_date,
        "common_trading_day_count": len(common_trading_dates),
        "authoritative_trading_day_count": len(authoritative_trading_dates),
        "lag_warnings": lag_warnings,
        "suspension_dates_by_symbol": suspension_dates_by_symbol,
        "lagged_dates_by_symbol": lagged_dates_by_symbol,
        "missing_trading_dates": missing_trading_dates,
        "missing_benchmark_dates": missing_trading_dates.get(benchmark, []),
        "calendar_file": str(calendar_file),
        "calendar_start_date": all_calendar_rows[0],
        "calendar_end_date": all_calendar_rows[-1],
        "corporate_actions_as_of": ledger.as_of,
        "unverified_corporate_action_events": [dict(event) for event in ledger.unverified_events],
        "regime_history_anchor_start_date": regime_history.anchor_start_date,
        "backfill_replay_window": resolved_config.get("backfill_replay_window"),
        "forward_eval_start_date": resolved_config.get("forward_eval_start_date"),
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
        splits=splits,
        regime_history=regime_history,
        common_end_date=common_end_date,
        common_trading_dates=common_trading_dates,
        authoritative_trading_dates=authoritative_trading_dates,
        suspension_dates_by_symbol=suspension_dates_by_symbol,
        lagged_dates_by_symbol=lagged_dates_by_symbol,
        metadata=metadata,
    )
