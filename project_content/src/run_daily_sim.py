"""Run or resume the daily simulation against local, read-only market data."""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
import sys
from typing import Any, Sequence

if not __package__:
    # Support invocation by absolute script path from any working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import DEFAULT_SIM_CONFIG_PATH, MarketBundle, load_market_bundle, load_trade_calendar
from src.engine import ACCOUNT_IDS, ENGINE_SCHEMA_VERSION, SimulationEngine
from src.reporter import (
    NAV_CSV_COLUMNS,
    TRADES_CSV_COLUMNS,
    build_daily_report,
    export_nav_daily_csv,
    export_trades_csv,
    load_state_json,
    nav_rows,
    read_csv_rows,
    save_daily_snapshot,
    save_latest_report,
    save_state_json,
    snapshot_path,
    valuation_markers,
)

DEFAULT_OUTPUT_DIR = DEFAULT_SIM_CONFIG_PATH.parents[1] / "output"


def _iso_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() == value:
            return value
    except ValueError:
        pass
    raise ValueError(f"--as-of requires a valid YYYY-MM-DD date, got {value!r}")


def _lag_warnings(bundle: MarketBundle, last_session: str) -> list[str]:
    if last_session <= bundle.metadata["latest_end_date"]:
        return [warning["message"] for warning in bundle.lag_warnings]
    return [
        f"{symbol} latest CSV date {stats['end_date']} lags behind requested BS-01 date {last_session}"
        for symbol, stats in sorted(bundle.metadata["symbol_stats"].items())
        if stats["end_date"] < last_session
    ]


def _committed_ledgers(
    output_dir: Path, state: dict[str, Any], dates: Sequence[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]], bool]:
    """Read the published prefix, discarding rows added before an interrupted checkpoint write."""
    through = state["last_processed_date"]
    stored_nav = read_csv_rows(output_dir / "nav_daily.csv", NAV_CSV_COLUMNS)
    stored_trades = read_csv_rows(output_dir / "trades.csv", TRADES_CSV_COLUMNS)
    nav = {
        (row["date"], row["account"]): row
        for row in stored_nav if row["date"] <= through
    }
    trades = {
        (row["date"], row["account"], int(row["trade_id"])): row
        for row in stored_trades if row["date"] <= through
    }
    committed_dates = {day for day in dates if day <= through}
    expected_nav = {(day, account) for day in committed_dates for account in ACCOUNT_IDS}
    error = ValueError("CSV ledger does not match state.json. Rebuild with --from-scratch.")
    if set(nav) != expected_nav:
        raise error

    ids = {account: [] for account in ACCOUNT_IDS}
    for trade_date, account, trade_id in trades:
        if account not in ids or trade_date not in committed_dates:
            raise error
        ids[account].append(trade_id)
    for account, trade_ids in ids.items():
        saved = state["accounts"][account]
        if sorted(trade_ids) != list(range(1, saved["next_trade_id"])):
            raise error
        latest = nav[through, account]
        for key in ("cash", "equity", "nav", "peak_equity", "max_drawdown"):
            if float(latest[key]) != saved[key]:
                raise error

    needs_refresh = len(nav) != len(stored_nav) or len(trades) != len(stored_trades)
    return list(nav.values()), list(trades.values()), needs_refresh


def run_daily_sim(
    *,
    config_path: str | Path = DEFAULT_SIM_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    as_of: str | None = None,
    from_scratch: bool = False,
    allow_lag: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Load, replay and publish one daily run, or return the dry-run calculation."""
    bundle = load_market_bundle(config_path)
    start = bundle.config["start_date"]
    cutoff = _iso_date(as_of) if as_of is not None else bundle.metadata["latest_end_date"]
    if cutoff < start:
        raise ValueError(f"--as-of {cutoff} precedes configured start_date {start}")
    dates = load_trade_calendar(
        bundle.config["upstream_paths"]["trade_calendar_dir"], start_date=start, end_date=cutoff
    )

    lag_warnings = _lag_warnings(bundle, dates[-1])
    for warning in lag_warnings:
        print(f"[行情滞后] {warning}", file=sys.stderr)
    if lag_warnings and not allow_lag:
        raise ValueError("行情滞后，推进已中止。--allow-lag 可显式放行。")

    output_dir = Path(output_dir)
    state_path = output_dir / "state.json"
    previous_state = None
    previous_nav = []
    previous_trades = []
    refresh_ledgers = False
    if not from_scratch and state_path.exists():
        previous_state = load_state_json(state_path)
        if previous_state.get("schema_version") != ENGINE_SCHEMA_VERSION:
            raise ValueError("Incompatible state.json schema_version. Rebuild with --from-scratch.")
        if previous_state["last_processed_date"] > dates[-1]:
            raise ValueError("--as-of precedes the checkpoint. Rebuild with --from-scratch to rewind.")

    engine = SimulationEngine(bundle, previous_state, trading_dates=dates)
    if previous_state is not None:
        previous_nav, previous_trades, refresh_ledgers = _committed_ledgers(output_dir, previous_state, dates)
    summary = engine.run()
    state = engine.export_state()
    days = engine.daily_results

    if not dry_run:
        report_path = output_dir / "nav_latest.md"
        if days or refresh_ledgers or not report_path.is_file():
            if days:
                symbol_names = {symbol: bundle.watchlist[symbol]["name"] for symbol in engine.sectors}
                first_day_number = engine.trading_days_processed - len(days) + 1
                for number, day in enumerate(days, first_day_number):
                    report = build_daily_report(day, start_date=start, trading_days=number, symbol_names=symbol_names)
                    save_daily_snapshot(output_dir, day["date"], report)
            else:
                report = snapshot_path(output_dir, engine.last_processed_date).read_text(encoding="utf-8")
            export_trades_csv(
                (trade for day in days for trade in day["trades"]),
                output_dir / "trades.csv", previous_rows=previous_trades,
            )
            export_nav_daily_csv(
                (row for day in days for row in nav_rows(day)),
                output_dir / "nav_daily.csv", previous_rows=previous_nav,
            )
            save_latest_report(output_dir, report)
            # Publish the checkpoint after the ledgers and reports are available.
            save_state_json(state, state_path)

    prefix = "[DRY-RUN] " if dry_run else ""
    print(
        f"{prefix}起算日 {start} | 净值日 {engine.last_processed_date} | "
        f"本次推进 {len(days)} 个交易日 | 累计 {engine.trading_days_processed} 个交易日"
    )
    print(" | ".join(f"{account} NAV {ledger.nav:.6f}" for account, ledger in engine.accounts.items()))
    for day in days:
        for marker in valuation_markers(day):
            print(marker)
    return {**summary, "state": state, "lag_warnings": lag_warnings}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="sector_paper_sim 日频模拟撮合与账本输出")
    parser.add_argument("--config", type=Path, default=DEFAULT_SIM_CONFIG_PATH, help="模拟配置路径")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="日频账本输出目录")
    parser.add_argument("--as-of", help="截止日 YYYY-MM-DD，默认取全池最新行情日")
    parser.add_argument("--from-scratch", action="store_true", help="从配置起算日重放")
    parser.add_argument("--allow-lag", action="store_true", help="允许行情滞后时按 BS-01 日历推进")
    parser.add_argument("--dry-run", action="store_true", help="计算净值并仅输出终端摘要")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        run_daily_sim(
            config_path=args.config, output_dir=args.output_dir, as_of=args.as_of,
            from_scratch=args.from_scratch, allow_lag=args.allow_lag, dry_run=args.dry_run,
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, csv.Error) as exc:
        print(f"[运行失败] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
