"""Run or resume the daily simulation against local, read-only market data."""

from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if not __package__:
    # Support invocation by absolute script path from any working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import (
    DEFAULT_SIM_CONFIG_PATH,
    MarketBundle,
    assess_corporate_actions_gate,
    load_market_bundle,
    load_trade_calendar,
)
from src.engine import ACCOUNT_IDS, ENGINE_SCHEMA_VERSION, SimulationEngine
from src.experiment_manifest import (
    build_daily_input_manifest,
    build_experiment_manifest,
    canonical_json_bytes,
    daily_input_manifest_path,
    input_manifest_chain_sha256,
    manifest_sha256,
    sha256_bytes,
    validate_resume_manifest,
)
from src.reporter import (
    NAV_CSV_COLUMNS,
    TRADES_CSV_COLUMNS,
    build_daily_report,
    export_nav_daily_csv,
    export_trades_csv,
    load_json_file,
    load_state_json,
    nav_rows,
    read_csv_rows,
    render_csv,
    save_daily_snapshot,
    save_json_file,
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


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _has_simulation_artifacts(output_dir: Path) -> bool:
    return any(
        (output_dir / name).exists()
        for name in (
            "state.json", "nav_daily.csv", "trades.csv", "nav_latest.md",
            "experiment_manifest.json",
        )
    ) or any((output_dir / "history").glob("nav_????????.md")) or any(
        (output_dir / "daily_inputs").glob("inputs_????????.json")
    )


def _check_existing_text(path: Path, expected: str, *, description: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") != expected:
        raise ValueError(
            f"{description} conflicts with the existing immutable experiment at {path}; "
            "use an archived or new output directory"
        )


def _check_state_matches(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.is_file():
        return
    actual = load_state_json(path)
    expected_state = json.loads(json.dumps(expected, allow_nan=False))
    identity_fields = (
        "experiment_id", "experiment_manifest_sha256",
        "daily_input_manifest_chain_sha256",
    )
    if not any(field in actual for field in identity_fields):
        # A legacy checkpoint may be adopted only after its full replay matches.
        for field in identity_fields:
            expected_state.pop(field, None)
    if actual != expected_state:
        changed = [
            key for key in sorted(set(actual) | set(expected_state))
            if actual.get(key) != expected_state.get(key)
        ]
        raise ValueError(
            f"Full rebuild conflicts with existing state fields {changed}; "
            f"existing state was preserved at {path}; "
            "use an archived or new output directory"
        )


def _validate_committed_input_manifests(
    output_dir: Path,
    state: Mapping[str, Any],
    dates: Sequence[str],
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    through = str(state["last_processed_date"])
    committed_dates = [day for day in dates if day <= through]
    loaded: dict[str, dict[str, Any]] = {}
    for trade_date in committed_dates:
        path = daily_input_manifest_path(output_dir, trade_date)
        if not path.is_file():
            raise ValueError(
                f"Committed day {trade_date} has no daily input manifest at {path}; "
                "run a full comparison in a new output directory"
            )
        daily = load_json_file(path)
        if (
            daily.get("experiment_id") != manifest.get("experiment_id")
            or daily.get("trade_date") != trade_date
            or daily.get("run_date") != trade_date
        ):
            raise ValueError(f"Daily input manifest identity mismatch for {trade_date}: {path}")
        loaded[trade_date] = daily
        if trade_date == manifest.get("simulation_start_date"):
            input_hash = sha256_bytes(canonical_json_bytes(daily.get("inputs", {})))
            expected_hash = manifest.get("initialization_input_version", {}).get("sha256")
            if input_hash != expected_hash:
                raise ValueError(
                    "The first daily input manifest does not match the experiment initialization inputs"
                )
    expected_chain = input_manifest_chain_sha256(loaded)
    if state.get("daily_input_manifest_chain_sha256") != expected_chain:
        raise ValueError(
            "Committed daily input manifest chain does not match state.json; "
            "refusing to continue with a mixed or edited experiment history."
        )
    committed_snapshot = snapshot_path(output_dir, through)
    latest_report = output_dir / "nav_latest.md"
    if not committed_snapshot.is_file() or not latest_report.is_file():
        raise ValueError(f"Committed NAV report or history snapshot is missing for {through}")
    latest_bytes = latest_report.read_bytes()
    uncommitted_reports = [
        snapshot_path(output_dir, day)
        for day in dates
        if day > through and snapshot_path(output_dir, day).is_file()
    ]
    if (
        committed_snapshot.read_bytes() != latest_bytes
        and not any(path.read_bytes() == latest_bytes for path in uncommitted_reports)
    ):
        raise ValueError(
            "nav_latest.md does not match the committed snapshot or a recoverable later "
            f"snapshot for {through}"
        )
    return loaded


def _compute_forward_excess_returns(
    all_nav_rows: Sequence[Mapping[str, Any]],
    *,
    start_date: str,
    last_processed_date: str | None,
    forward_eval_start_date: str | None,
    regime_history: Any,
) -> dict[str, float]:
    """Compute forward-only A vs B1/B2/C/D excess returns excluding BACKFILL-contaminated periods."""
    if not forward_eval_start_date or not last_processed_date or last_processed_date < forward_eval_start_date:
        return {}
    if any(
        rec.get("record_type") == "BACKFILL" and rec["effective_date"] >= forward_eval_start_date
        for rec in regime_history.all_records()
        if rec["effective_date"] <= last_processed_date
    ):
        return {}

    nav_by_date: dict[str, dict[str, float]] = {}
    for row in all_nav_rows:
        nav_by_date.setdefault(str(row["date"]), {})[str(row["account"])] = float(row["nav"])
    if last_processed_date not in nav_by_date:
        return {}

    if forward_eval_start_date == start_date:
        baseline_nav = {account: 1.0 for account in ACCOUNT_IDS}
    else:
        prior_dates = [day for day in sorted(nav_by_date) if day < forward_eval_start_date]
        if not prior_dates:
            return {}
        baseline_nav = nav_by_date[prior_dates[-1]]

    latest_nav = nav_by_date[last_processed_date]
    forward_returns = {
        account: latest_nav[account] / baseline_nav[account] - 1.0
        for account in ACCOUNT_IDS
    }
    return {
        f"A_vs_{account}": forward_returns["A"] - forward_returns[account]
        for account in ACCOUNT_IDS[1:]
    }


def run_daily_sim(
    *,
    config_path: str | Path = DEFAULT_SIM_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    as_of: str | None = None,
    from_scratch: bool = False,
    allow_lag: bool = False,
    dry_run: bool = False,
    run_mode: str = "formal",
    forward_eval_start_date: str | None = None,
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

    corporate_actions_gate = assess_corporate_actions_gate(
        bundle.metadata["corporate_actions_as_of"],
        dates[-1],
        bundle.metadata.get("unverified_corporate_action_events", []),
    )
    if not corporate_actions_gate["allowed"]:
        if dry_run:
            print(
                "[DRY-RUN][BLOCKED] "
                + json.dumps(corporate_actions_gate, ensure_ascii=False, sort_keys=True)
            )
            return {
                "status": "BLOCKED",
                "corporate_actions_gate": corporate_actions_gate,
                "dry_run": True,
            }
        details = "; ".join(corporate_actions_gate["reasons"])
        raise ValueError(f"公司行为台账门禁阻断正式运行: {details}")

    lag_warnings = _lag_warnings(bundle, dates[-1])
    for warning in lag_warnings:
        print(f"[行情滞后] {warning}", file=sys.stderr)
    if lag_warnings and not allow_lag:
        raise ValueError("行情滞后，推进已中止。--allow-lag 可显式放行。")

    output_dir = Path(output_dir)
    state_path = output_dir / "state.json"
    manifest_path = output_dir / "experiment_manifest.json"
    previous_state = None
    previous_nav: list[dict[str, str]] = []
    previous_trades: list[dict[str, str]] = []
    committed_daily_inputs: dict[str, dict[str, Any]] = {}
    refresh_ledgers = False
    saved_manifest = load_json_file(manifest_path) if manifest_path.is_file() else None
    candidate_manifest = build_experiment_manifest(
        bundle, config_path=config_path, run_mode=run_mode
    )

    if not from_scratch:
        if state_path.is_file() and saved_manifest is None:
            raise ValueError(
                "Existing state.json has no experiment_manifest.json. Resume is blocked; "
                "use --from-scratch for a full result comparison or choose a new output directory."
            )
        if saved_manifest is not None and not state_path.is_file():
            raise ValueError(
                "experiment_manifest.json exists without state.json. Use --from-scratch "
                "to verify and complete the interrupted full replay."
            )
        if saved_manifest is None and _has_simulation_artifacts(output_dir):
            raise ValueError(
                "Output contains legacy simulation files without an experiment manifest. "
                "Use --from-scratch for a full result comparison or choose a new output directory."
            )
        if saved_manifest is not None:
            validate_resume_manifest(saved_manifest, candidate_manifest)
        if state_path.is_file():
            previous_state = load_state_json(state_path)
            if previous_state.get("schema_version") != ENGINE_SCHEMA_VERSION:
                raise ValueError("Incompatible state.json schema_version. Rebuild with --from-scratch.")
            if previous_state["last_processed_date"] > dates[-1]:
                raise ValueError("--as-of precedes the checkpoint. Rebuild with --from-scratch to rewind.")
            expected_manifest_hash = manifest_sha256(saved_manifest)
            if (
                previous_state.get("experiment_id") != saved_manifest.get("experiment_id")
                or previous_state.get("experiment_manifest_sha256") != expected_manifest_hash
            ):
                raise ValueError(
                    "state.json does not reference the saved experiment manifest; "
                    "resume is blocked to avoid mixing experiment results."
                )
            committed_daily_inputs = _validate_committed_input_manifests(
                output_dir, previous_state, dates, saved_manifest
            )
            previous_nav, previous_trades, refresh_ledgers = _committed_ledgers(
                output_dir, previous_state, dates
            )

    if from_scratch and saved_manifest is not None and saved_manifest != candidate_manifest:
        changed = [
            key for key in sorted(set(saved_manifest) | set(candidate_manifest))
            if saved_manifest.get(key) != candidate_manifest.get(key)
        ]
        raise ValueError(
            "Full rebuild would change immutable experiment manifest fields "
            f"{changed}; use an archived or new output directory."
        )

    manifest = saved_manifest if saved_manifest is not None else candidate_manifest
    engine = SimulationEngine(
        bundle,
        previous_state,
        trading_dates=dates,
        run_mode=run_mode,
        forward_eval_start_date=forward_eval_start_date,
    )
    summary = engine.run()
    state = engine.export_state()
    days = engine.daily_results
    all_nav = [*previous_nav, *(row for day in days for row in nav_rows(day))]
    forward_excess = _compute_forward_excess_returns(
        all_nav,
        start_date=start,
        last_processed_date=engine.last_processed_date,
        forward_eval_start_date=engine.forward_eval_start_date,
        regime_history=bundle.regime_history,
    )

    experiment_hash = manifest_sha256(manifest)
    state["experiment_id"] = manifest["experiment_id"]
    state["experiment_manifest_sha256"] = experiment_hash
    symbol_names = {symbol: bundle.watchlist[symbol]["name"] for symbol in engine.sectors}
    first_day_number = engine.trading_days_processed - len(days) + 1
    reports: dict[str, str] = {}
    for number, day in enumerate(days, first_day_number):
        reports[day["date"]] = build_daily_report(
            day, start_date=start, trading_days=number, symbol_names=symbol_names
        )

    input_dates = dates if from_scratch else [day["date"] for day in days]
    daily_inputs = {
        day: build_daily_input_manifest(
            bundle, trade_date=day, experiment_id=str(manifest["experiment_id"]),
            run_mode=run_mode, allow_lag=allow_lag,
            forward_eval_start_date=engine.forward_eval_start_date,
        )
        for day in input_dates
    }
    all_daily_inputs = {**committed_daily_inputs, **daily_inputs}
    state["daily_input_manifest_chain_sha256"] = input_manifest_chain_sha256(
        all_daily_inputs
    )
    nav_content = render_csv(
        NAV_CSV_COLUMNS,
        (row for day in days for row in nav_rows(day)),
        previous_rows=previous_nav,
        key=lambda row: (row["date"], row["account"]),
    )
    trades_content = render_csv(
        TRADES_CSV_COLUMNS,
        (trade for day in days for trade in day["trades"]),
        previous_rows=previous_trades,
        key=lambda trade: (trade["date"], trade["account"], int(trade["trade_id"])),
    )
    report_path = output_dir / "nav_latest.md"
    latest_report: str | None = None
    if days:
        latest_report = reports[days[-1]["date"]]
    elif refresh_ledgers or not report_path.is_file():
        latest_report = snapshot_path(output_dir, engine.last_processed_date).read_text(
            encoding="utf-8"
        )

    if not dry_run:
        _check_existing_text(manifest_path, _json_text(manifest), description="Experiment manifest")
        if from_scratch:
            expected_snapshots = {
                snapshot_path(output_dir, day): reports[day] for day in reports
            }
            existing_snapshots = set((output_dir / "history").glob("nav_????????.md"))
            unexpected_snapshots = existing_snapshots - set(expected_snapshots)
            if unexpected_snapshots:
                names = sorted(path.name for path in unexpected_snapshots)
                raise ValueError(
                    f"Full rebuild conflicts with existing history snapshots {names}; "
                    "use an archived or new output directory"
                )
            for path, content in expected_snapshots.items():
                _check_existing_text(path, content, description="History snapshot")

            expected_input_paths = {
                daily_input_manifest_path(output_dir, day): _json_text(payload)
                for day, payload in daily_inputs.items()
            }
            existing_inputs = set((output_dir / "daily_inputs").glob("inputs_????????.json"))
            unexpected_inputs = existing_inputs - set(expected_input_paths)
            if unexpected_inputs:
                names = sorted(path.name for path in unexpected_inputs)
                raise ValueError(
                    f"Full rebuild conflicts with existing daily input manifests {names}; "
                    "use an archived or new output directory"
                )
            for path, content in expected_input_paths.items():
                _check_existing_text(path, content, description="Daily input manifest")

            _check_existing_text(output_dir / "nav_daily.csv", nav_content, description="NAV ledger")
            _check_existing_text(output_dir / "trades.csv", trades_content, description="Trade ledger")
            if latest_report is not None:
                _check_existing_text(report_path, latest_report, description="Latest NAV report")
            _check_state_matches(state_path, state)
        else:
            for day, payload in daily_inputs.items():
                _check_existing_text(
                    daily_input_manifest_path(output_dir, day), _json_text(payload),
                    description="Daily input manifest",
                )
            for day, report in reports.items():
                _check_existing_text(
                    snapshot_path(output_dir, day), report, description="History snapshot"
                )

        save_json_file(manifest, manifest_path, immutable=True)
        for day, payload in daily_inputs.items():
            save_json_file(
                payload, daily_input_manifest_path(output_dir, day), immutable=True
            )
        for day, report in reports.items():
            save_daily_snapshot(output_dir, day, report)
        export_trades_csv(
            (trade for day in days for trade in day["trades"]),
            output_dir / "trades.csv", previous_rows=previous_trades,
        )
        export_nav_daily_csv(
            (row for day in days for row in nav_rows(day)),
            output_dir / "nav_daily.csv", previous_rows=previous_nav,
        )
        if latest_report is not None:
            save_latest_report(output_dir, latest_report)
        # The checkpoint is published last and binds the state to its immutable manifest.
        save_state_json(state, state_path)

    prefix = "[DRY-RUN] " if dry_run else ""
    print(
        f"{prefix}模式 {run_mode} | 起算日 {start} | 净值日 {engine.last_processed_date} | "
        f"本次推进 {len(days)} 个交易日 | 累计 {engine.trading_days_processed} 个交易日"
    )
    print(" | ".join(f"{account} NAV {ledger.nav:.6f}" for account, ledger in engine.accounts.items()))
    for day in days:
        for marker in valuation_markers(day):
            print(marker)
    return {
        **summary,
        "state": state,
        "lag_warnings": lag_warnings,
        "forward_excess_returns": forward_excess,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="sector_paper_sim 日频模拟撮合与账本输出")
    parser.add_argument("--config", type=Path, default=DEFAULT_SIM_CONFIG_PATH, help="模拟配置路径")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="日频账本输出目录")
    parser.add_argument("--as-of", help="截止日 YYYY-MM-DD，默认取全池最新行情日")
    parser.add_argument("--from-scratch", action="store_true", help="从配置起算日重放")
    parser.add_argument("--allow-lag", action="store_true", help="允许行情滞后时按 BS-01 日历推进")
    parser.add_argument("--dry-run", action="store_true", help="计算净值并仅输出终端摘要")
    parser.add_argument(
        "--mode",
        choices=("formal", "replay"),
        default="formal",
        help="运行模式：formal（正式前瞻，仅接受时序合格 LIVE 决策）或 replay（显式历史重放/状态初始化，允许 BACKFILL）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        outcome = run_daily_sim(
            config_path=args.config, output_dir=args.output_dir, as_of=args.as_of,
            from_scratch=args.from_scratch, allow_lag=args.allow_lag, dry_run=args.dry_run,
            run_mode=args.mode,
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, csv.Error) as exc:
        print(f"[运行失败] {exc}", file=sys.stderr)
        return 2
    if outcome.get("status") == "BLOCKED":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
