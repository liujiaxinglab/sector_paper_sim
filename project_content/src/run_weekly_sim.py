"""Weekly post-close CLI entry point for sector_paper_sim.

Supports:
- First-time baseline initialization (no existing `state.json`)
- Incremental advancement of new common trading days using previously locked
  `next_effective_regime` (strict anti-lookahead discipline)
- Same-day idempotent refresh when no new trading days exist (including syncing
  updated `watchlist.json` regimes into `next_effective_regime` for next week)
- Lag warning guard (`--allow-lag` to override) and `--dry-run` preview mode
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from src.data_loader import DEFAULT_SIM_CONFIG_PATH, MarketBundle, load_market_bundle
from src.engine import ACCOUNT_IDS, SimulationEngine, normalize_regime
from src.reporter import (
    ACCOUNT_LABELS,
    append_or_update_nav_weekly_md,
    export_trades_csv,
    load_state_json,
    save_state_json,
)

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_iso_date(raw: str, flag_name: str) -> str:
    cleaned = raw.strip()
    if not _ISO_DATE_RE.match(cleaned):
        raise ValueError(f"{flag_name} must be YYYY-MM-DD, got {raw!r}")
    return cleaned


def _extract_watchlist_regimes(bundle: MarketBundle) -> dict[str, str]:
    regimes: dict[str, str] = {}
    for sym in sorted(bundle.sectors):
        raw_regime = str(bundle.watchlist[sym]["current_regime"]).strip()
        # Validate regime via engine's canonical validator (keeps raw string e.g. UNRATED)
        eff = normalize_regime(raw_regime)
        if eff == "BENCHMARK":
            raise ValueError(f"Sector symbol {sym!r} cannot have BENCHMARK regime")
        regimes[sym] = raw_regime
    return regimes


def _extract_symbol_names(bundle: MarketBundle) -> dict[str, str]:
    return {
        sym: str(bundle.watchlist.get(sym, {}).get("name") or "")
        for sym in sorted(bundle.symbols)
    }


def _diff_regimes(
    previous: Mapping[str, str] | None,
    current: Mapping[str, str],
) -> dict[str, dict[str, str | None]]:
    changes: dict[str, dict[str, str | None]] = {}
    prev_map = previous or {}
    for sym in sorted(set(prev_map.keys()) | set(current.keys())):
        old_val = prev_map.get(sym)
        new_val = current.get(sym)
        if old_val != new_val:
            changes[sym] = {"from": old_val, "to": new_val}
    return changes


def _resolve_locked_regimes_from_state(
    state_dict: Mapping[str, Any],
    bundle: MarketBundle,
) -> dict[str, str]:
    locked = state_dict.get("next_effective_regime")
    if isinstance(locked, Mapping) and locked:
        result: dict[str, str] = {}
        for sym in sorted(bundle.sectors):
            if sym in locked:
                val = str(locked[sym]).strip()
                normalize_regime(val)
                result[sym] = val
            else:
                result[sym] = str(bundle.watchlist[sym]["current_regime"]).strip()
        return result

    # Defensive fallback if loading a bare engine state without next_effective_regime
    acct_a_states = (
        state_dict.get("accounts", {}).get("A", {}).get("symbol_states", {})
    )
    fallback: dict[str, str] = {}
    for sym in sorted(bundle.sectors):
        sym_st = acct_a_states.get(sym, {}) if isinstance(acct_a_states, Mapping) else {}
        regime = sym_st.get("raw_regime") or sym_st.get("effective_regime")
        fallback[sym] = (
            str(regime).strip()
            if regime
            else str(bundle.watchlist[sym]["current_regime"]).strip()
        )
    return fallback


def _format_terminal_summary(
    *,
    state_dict: Mapping[str, Any],
    mode: str,
    processed_dates: Sequence[str],
    regime_changes: Mapping[str, Mapping[str, str | None]],
    dry_run: bool,
    output_dir: Path,
) -> str:
    accounts = state_dict["accounts"]
    excess_hist = state_dict.get("excess_history") or []
    latest_excess = (
        excess_hist[-1]
        if excess_hist
        else {"A_vs_B1": 0.0, "A_vs_B2": 0.0, "A_vs_C": 0.0, "A_vs_D": 0.0}
    )

    lines: list[str] = []
    prefix = "[DRY-RUN] " if dry_run else ""
    lines.append(f"{prefix}=== sector_paper_sim 周更运行汇总 ({mode}) ===")
    lines.append(
        f"起始日: {state_dict.get('start_date')} | "
        f"最新净值日: {state_dict.get('last_processed_date')} | "
        f"累计交易日: {state_dict.get('trading_days_processed')} | "
        f"本次推进交易日: {len(processed_dates)}"
    )
    if processed_dates:
        lines.append(f"推进区间: {processed_dates[0]} .. {processed_dates[-1]}")

    lines.append("")
    lines.append("五账户净值摘要:")
    for acct_id in ACCOUNT_IDS:
        acct = accounts[acct_id]
        nav = float(acct["nav"])
        cum_pct = (nav - 1.0) * 100.0
        dd_pct = float(acct["max_drawdown"]) * 100.0
        equity = float(acct["equity"])
        trades = len(acct.get("trades", []))
        lines.append(
            f"  - {ACCOUNT_LABELS[acct_id]:<14} | NAV={nav:.4f} | "
            f"累计={cum_pct:+.2f}% | 最大回撤={dd_pct:.2f}% | "
            f"总资产={equity:,.2f} | 成交={trades}笔"
        )

    lines.append("")
    lines.append(
        "A 组超额收益(百分点): "
        f"A-B1={float(latest_excess['A_vs_B1']) * 100.0:+.2f}pp | "
        f"A-B2={float(latest_excess['A_vs_B2']) * 100.0:+.2f}pp | "
        f"A-C={float(latest_excess['A_vs_C']) * 100.0:+.2f}pp | "
        f"A-D={float(latest_excess['A_vs_D']) * 100.0:+.2f}pp"
    )

    next_locked = state_dict.get("next_effective_regime", {})
    locked_desc = ", ".join(f"{sym}={next_locked[sym]}" for sym in sorted(next_locked.keys()))
    lines.append(f"下周锁定路由: {locked_desc}")
    if regime_changes:
        diff_desc = ", ".join(
            f"{sym}: {ch.get('from')} -> {ch.get('to')}"
            for sym, ch in sorted(regime_changes.items())
        )
        lines.append(f"路由审计变更: {diff_desc}")
    else:
        lines.append("路由审计变更: 无变化")

    if dry_run:
        lines.append(f"输出模式: --dry-run（未向 {output_dir} 写入任何文件）")
    else:
        lines.append(
            f"已落盘: {output_dir / 'state.json'}, "
            f"{output_dir / 'trades.csv'}, {output_dir / 'nav_weekly.md'}"
        )
    return "\n".join(lines)


def run_weekly_sim(
    *,
    config_path: str | Path = DEFAULT_SIM_CONFIG_PATH,
    output_dir: str | Path = "output",
    start_date: str | None = None,
    end_date: str | None = None,
    allow_lag: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute one weekly post-close simulation cycle with anti-lookahead regime locking."""
    resolved_config_path = Path(config_path)
    resolved_output_dir = Path(output_dir)
    state_path = resolved_output_dir / "state.json"
    trades_path = resolved_output_dir / "trades.csv"
    report_path = resolved_output_dir / "nav_weekly.md"

    bundle = load_market_bundle(resolved_config_path)

    # Guard against stale / lagging upstream ETF CSVs unless --allow-lag is explicitly given
    if bundle.lag_warnings:
        warning_lines = [
            "[WARNING] 检测到上游 ETF 行情日期滞后（可能个别标的 TDX 数据未更新）："
        ]
        for warn in bundle.lag_warnings:
            warning_lines.append(f"  - {warn.get('message', warn)}")
        warning_text = "\n".join(warning_lines)
        print(warning_text, file=sys.stderr)
        if not allow_lag:
            print(
                "[ABORT] 已中止推进以防止跨标的日期错位；若确认按共同截止日强制推进，请传入 --allow-lag。",
                file=sys.stderr,
            )
            return {
                "status": "ABORTED_LAG",
                "exit_code": 2,
                "lag_warnings": list(bundle.lag_warnings),
                "processed_dates": [],
            }

    resolved_end = (
        _validate_iso_date(end_date, "--end-date")
        if end_date is not None
        else bundle.common_end_date
    )
    latest_watchlist_regimes = _extract_watchlist_regimes(bundle)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    now_iso = datetime.now().astimezone().isoformat(timespec="seconds")

    if not state_path.is_file():
        # =====================================================================
        # Initial run: establish baseline from start_date..resolved_end
        # =====================================================================
        mode = "INIT"
        resolved_start = (
            _validate_iso_date(start_date, "--start-date")
            if start_date is not None
            else resolved_end
        )
        if resolved_start > resolved_end:
            raise ValueError(
                f"start_date ({resolved_start}) cannot be later than end_date ({resolved_end})"
            )

        candidate_dates = [
            dt
            for dt in bundle.common_trading_dates
            if resolved_start <= dt <= resolved_end
        ]
        if not candidate_dates:
            raise ValueError(
                f"No common trading dates available in [{resolved_start}, {resolved_end}] "
                f"(common_end_date={bundle.common_end_date})"
            )

        engine = SimulationEngine(bundle=bundle)
        run_summary = engine.run(
            start_date=resolved_start,
            end_date=resolved_end,
            regime_schedule=latest_watchlist_regimes,
        )
        processed_dates = list(run_summary["processed_dates"])
        state_dict = engine.export_state()

        regime_changes = _diff_regimes(None, latest_watchlist_regimes)
        audit_log: list[dict[str, Any]] = [
            {
                "recorded_at_market_date": state_dict["last_processed_date"],
                "recorded_at": now_iso,
                "event": "INIT",
                "changes": regime_changes,
            }
        ]
        state_dict["next_effective_regime"] = dict(latest_watchlist_regimes)
        state_dict["regime_audit_log"] = audit_log
        state_dict["symbol_names"] = _extract_symbol_names(bundle)
        state_dict["updated_at"] = now_str
    else:
        # =====================================================================
        # Incremental run / Same-day idempotent refresh
        # =====================================================================
        prev_state = load_state_json(state_path)
        engine = SimulationEngine.from_state(bundle=bundle, state_dict=prev_state)
        last_processed = engine.last_processed_date
        locked_regimes = _resolve_locked_regimes_from_state(prev_state, bundle)
        audit_log = list(prev_state.get("regime_audit_log", []))

        if last_processed is not None and resolved_end > last_processed:
            new_dates = [
                dt
                for dt in bundle.common_trading_dates
                if last_processed < dt <= resolved_end
            ]
        elif last_processed is None:
            new_dates = [
                dt for dt in bundle.common_trading_dates if dt <= resolved_end
            ]
        else:
            new_dates = []

        if new_dates:
            mode = "INCREMENTAL"
            # Strictly use previously locked regimes for newly arrived trading days
            run_summary = engine.run(
                end_date=resolved_end,
                regime_schedule=locked_regimes,
            )
            processed_dates = list(run_summary["processed_dates"])
        else:
            mode = "REFRESH"
            processed_dates = []

        state_dict = engine.export_state()

        # After advancing (or on same-day refresh), sync latest watchlist into next_effective_regime
        regime_changes = _diff_regimes(locked_regimes, latest_watchlist_regimes)
        if regime_changes:
            audit_log.append(
                {
                    "recorded_at_market_date": state_dict["last_processed_date"],
                    "recorded_at": now_iso,
                    "event": "UPDATE",
                    "changes": regime_changes,
                }
            )
        state_dict["next_effective_regime"] = dict(latest_watchlist_regimes)
        state_dict["regime_audit_log"] = audit_log
        state_dict["symbol_names"] = _extract_symbol_names(bundle)
        state_dict["updated_at"] = now_str

    if not dry_run:
        save_state_json(state_dict, state_path)
        export_trades_csv(state_dict, trades_path)
        append_or_update_nav_weekly_md(
            state_dict,
            report_path,
            lag_warnings=bundle.lag_warnings,
        )

    summary_text = _format_terminal_summary(
        state_dict=state_dict,
        mode=mode,
        processed_dates=processed_dates,
        regime_changes=regime_changes,
        dry_run=dry_run,
        output_dir=resolved_output_dir,
    )
    print(summary_text)

    return {
        "status": "OK",
        "exit_code": 0,
        "mode": mode,
        "dry_run": dry_run,
        "processed_dates": processed_dates,
        "regime_changes": regime_changes,
        "state": state_dict,
        "lag_warnings": list(bundle.lag_warnings),
        "summary_text": summary_text,
    }


def _default_config_arg() -> str:
    rel = Path("project_content/config/sim_config.json")
    if rel.is_file():
        return str(rel)
    return str(DEFAULT_SIM_CONFIG_PATH)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.run_weekly_sim",
        description="sector_paper_sim 每周五盘后增量模拟运行与周报输出入口",
    )
    parser.add_argument(
        "--config",
        default=_default_config_arg(),
        help="sim_config.json 路径（默认 project_content/config/sim_config.json）",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="输出目录路径（默认 output）",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="起始交易日 YYYY-MM-DD（仅首次无 state.json 时生效；默认取 bundle.common_end_date）",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="截止交易日 YYYY-MM-DD（默认取 bundle.common_end_date）",
    )
    parser.add_argument(
        "--allow-lag",
        action="store_true",
        help="当上游个别 ETF 行情存在日期滞后警告时仍允许按共同截止日继续推进",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅计算并打印终端汇总，不向磁盘写入 state.json / trades.csv / nav_weekly.md",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_weekly_sim(
        config_path=args.config,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        allow_lag=args.allow_lag,
        dry_run=args.dry_run,
    )
    return int(result.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
