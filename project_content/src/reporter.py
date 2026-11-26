"""Reporting and lossless state persistence layer for sector_paper_sim.

Responsibilities:
- Atomic UTF-8 JSON persistence for full engine + routing lock state (`output/state.json`)
- Deterministic merged trade ledger export across all 5 accounts (`output/trades.csv`)
- Human-readable weekly Markdown report with idempotent history table updates (`output/nav_weekly.md`)
"""

from __future__ import annotations

import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

ACCOUNT_ORDER: tuple[str, ...] = ("A", "B1", "B2", "C", "D")

ACCOUNT_LABELS: dict[str, str] = {
    "A": "A 路由实验盘",
    "B1": "B1 纯机械海龟",
    "B2": "B2 纯机械网格",
    "C": "C 行业等权 B&H",
    "D": "D 沪深300 B&H",
}

TRADES_CSV_COLUMNS: tuple[str, ...] = (
    "date",
    "account",
    "trade_id",
    "symbol",
    "action",
    "side",
    "shares",
    "price",
    "notional",
    "commission",
    "stamp_duty",
    "net_cash_flow",
    "regime",
    "trigger_price",
    "grid_level",
    "note",
)

HISTORY_SECTION_HEADER = "## 历史净值追踪表"
HISTORY_TABLE_COLUMNS: tuple[str, ...] = (
    "日期",
    "交易日数",
    "A净值",
    "B1净值",
    "B2净值",
    "C净值",
    "D净值",
    "A最大回撤",
    "A-B1",
    "A-B2",
    "A-C",
    "A-D",
)

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _atomic_write_text(path: Path, content: str) -> Path:
    """Atomically write UTF-8 text to `path` via a same-directory `.tmp` file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return path


def save_state_json(
    state_dict: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Atomically persist full simulation state as UTF-8 JSON (`indent=2`, `ensure_ascii=False`)."""
    if not isinstance(state_dict, Mapping):
        raise ValueError("state_dict must be a mapping")
    path = Path(output_path)
    serialized = json.dumps(dict(state_dict), indent=2, ensure_ascii=False) + "\n"
    return _atomic_write_text(path, serialized)


def load_state_json(input_path: str | Path) -> dict[str, Any]:
    """Load simulation state JSON from `input_path` and validate top-level object structure."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"State JSON file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in state file ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object in state file: {path}")
    return payload


def _format_csv_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"


def _format_csv_int(value: Any) -> str:
    if value is None or value == "":
        return ""
    return str(int(value))


def export_trades_csv(
    state_dict: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Merge all 5 accounts' trades, sort by `(date, account, trade_id)`, and export UTF-8 CSV."""
    accounts = state_dict.get("accounts")
    if not isinstance(accounts, Mapping):
        raise ValueError("state_dict missing 'accounts' mapping")

    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    ordered_ids = [aid for aid in ACCOUNT_ORDER if aid in accounts] + sorted(
        aid for aid in accounts.keys() if aid not in ACCOUNT_ORDER
    )

    for acct_id in ordered_ids:
        if acct_id in seen_ids:
            continue
        seen_ids.add(acct_id)
        acct_data = accounts[acct_id]
        if not isinstance(acct_data, Mapping):
            continue
        for trade in acct_data.get("trades", []):
            if not isinstance(trade, Mapping):
                continue
            row = dict(trade)
            row.setdefault("account", acct_id)
            merged.append(row)

    merged.sort(
        key=lambda t: (
            str(t.get("date", "")),
            str(t.get("account", "")),
            int(t.get("trade_id", 0)),
        )
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=list(TRADES_CSV_COLUMNS),
                lineterminator="\n",
            )
            writer.writeheader()
            for trade in merged:
                writer.writerow(
                    {
                        "date": str(trade.get("date", "")),
                        "account": str(trade.get("account", "")),
                        "trade_id": _format_csv_int(trade.get("trade_id")),
                        "symbol": str(trade.get("symbol", "")),
                        "action": str(trade.get("action", "")),
                        "side": str(trade.get("side", "")),
                        "shares": _format_csv_int(trade.get("shares")),
                        "price": _format_csv_float(trade.get("price")),
                        "notional": _format_csv_float(trade.get("notional")),
                        "commission": _format_csv_float(trade.get("commission")),
                        "stamp_duty": _format_csv_float(trade.get("stamp_duty")),
                        "net_cash_flow": _format_csv_float(trade.get("net_cash_flow")),
                        "regime": "" if trade.get("regime") is None else str(trade.get("regime")),
                        "trigger_price": _format_csv_float(trade.get("trigger_price")),
                        "grid_level": _format_csv_int(trade.get("grid_level")),
                        "note": "" if trade.get("note") is None else str(trade.get("note")),
                    }
                )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return path


def _fmt_signed_pct(decimal_return: float) -> str:
    pct = decimal_return * 100.0
    if abs(pct) < 1e-12:
        pct = 0.0
    return f"{pct:+.2f}%"


def _fmt_drawdown_pct(decimal_dd: float) -> str:
    pct = decimal_dd * 100.0
    if abs(pct) < 1e-12:
        pct = 0.0
    return f"{pct:.2f}%"


def _fmt_signed_pp(decimal_diff: float) -> str:
    pp = decimal_diff * 100.0
    if abs(pp) < 1e-12:
        pp = 0.0
    return f"{pp:+.2f}pp"


def _fmt_money(amount: float) -> str:
    return f"{amount:,.2f}"


def _fmt_price(price: float | None) -> str:
    if price is None:
        return "-"
    return f"{float(price):.4f}"


def _parse_existing_history_rows(md_path: Path) -> dict[str, list[str]]:
    """Parse existing `## 历史净值追踪表` data rows keyed by ISO date (`as_of_date`)."""
    if not md_path.is_file():
        return {}

    lines = md_path.read_text(encoding="utf-8").splitlines()
    in_history = False
    rows_by_date: dict[str, list[str]] = {}
    expected_cols = len(HISTORY_TABLE_COLUMNS)

    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("## "):
            if stripped == HISTORY_SECTION_HEADER:
                in_history = True
                continue
            if in_history:
                break
        if not in_history:
            continue
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != expected_cols:
            continue
        first_cell = cells[0]
        if _ISO_DATE_RE.match(first_cell):
            rows_by_date[first_cell] = cells

    return rows_by_date


def _compute_latest_excess(state_dict: Mapping[str, Any]) -> dict[str, float]:
    excess_history = state_dict.get("excess_history")
    if isinstance(excess_history, list) and excess_history:
        latest = excess_history[-1]
        if isinstance(latest, Mapping):
            return {
                "A_vs_B1": float(latest.get("A_vs_B1", 0.0)),
                "A_vs_B2": float(latest.get("A_vs_B2", 0.0)),
                "A_vs_C": float(latest.get("A_vs_C", 0.0)),
                "A_vs_D": float(latest.get("A_vs_D", 0.0)),
            }

    accounts = state_dict.get("accounts", {})
    a_cum = float(accounts["A"]["nav"]) - 1.0
    return {
        "A_vs_B1": a_cum - (float(accounts["B1"]["nav"]) - 1.0),
        "A_vs_B2": a_cum - (float(accounts["B2"]["nav"]) - 1.0),
        "A_vs_C": a_cum - (float(accounts["C"]["nav"]) - 1.0),
        "A_vs_D": a_cum - (float(accounts["D"]["nav"]) - 1.0),
    }


def _describe_grid_orders(sym_state: Mapping[str, Any]) -> str:
    grid = sym_state.get("grid")
    if not isinstance(grid, Mapping):
        return "0"
    pending = grid.get("pending_orders", [])
    if not isinstance(pending, list) or not pending:
        return "0"
    buys = sum(1 for item in pending if isinstance(item, Mapping) and item.get("side") == "BUY")
    sells = sum(1 for item in pending if isinstance(item, Mapping) and item.get("side") == "SELL")
    return f"{len(pending)} (买{buys}/卖{sells})"


def _format_lag_status(lag_warnings: Sequence[Mapping[str, Any]] | None) -> tuple[str, list[str]]:
    if not lag_warnings:
        return "正常（无滞后）", []
    details: list[str] = []
    for item in lag_warnings:
        msg = str(item.get("message") or "").strip()
        if not msg:
            sym = item.get("symbol", "UNKNOWN")
            end_dt = item.get("symbol_end_date", "?")
            latest_dt = item.get("latest_end_date", "?")
            days = item.get("lag_trading_days", "?")
            msg = f"{sym} 最新日期 {end_dt} 滞后于 {latest_dt}（{days} 个交易日）"
        details.append(msg)
    return f"存在滞后警告（{len(details)} 条，已通过 --allow-lag 放行）", details


def append_or_update_nav_weekly_md(
    state_dict: Mapping[str, Any],
    output_path: str | Path,
    lag_warnings: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Generate or idempotently update `output/nav_weekly.md` from `state_dict`.

    Maintains:
    1. Top metadata block (start date, latest NAV date, trading days, update time, lag status)
    2. Latest 5-account comparison table (A, B1, B2, C, D)
    3. Latest Account A excess return table (A vs B1/B2/C/D in percentage points)
    4. Current effective & next-week locked regime / holdings / grid orders snapshot table
    5. Weekly historical NAV tracking table keyed by `as_of_date` (idempotent row overwrite)
    """
    accounts = state_dict.get("accounts")
    if not isinstance(accounts, Mapping):
        raise ValueError("state_dict missing 'accounts' mapping")
    for acct_id in ACCOUNT_ORDER:
        if acct_id not in accounts:
            raise ValueError(f"state_dict['accounts'] missing required account '{acct_id}'")

    md_path = Path(output_path)
    start_date = state_dict.get("start_date") or "-"
    as_of_date = state_dict.get("last_processed_date") or "-"
    trading_days = int(state_dict.get("trading_days_processed", 0))
    updated_at = str(
        state_dict.get("updated_at")
        or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    lag_summary, lag_details = _format_lag_status(lag_warnings)

    excess = _compute_latest_excess(state_dict)

    lines: list[str] = [
        "# sector_paper_sim 周度净值报告",
        "",
        "## 元信息",
        "",
        f"- **起始日**：`{start_date}`",
        f"- **最新净值日**：`{as_of_date}`",
        f"- **已运行交易日数**：`{trading_days}`",
        f"- **最近更新时间**：`{updated_at}`",
        f"- **数据滞后检查**：`{lag_summary}`",
    ]
    for detail in lag_details:
        lines.append(f"  - ⚠️ {detail}")

    # Section 2: Latest 5-account comparison table
    lines.extend(
        [
            "",
            f"## 最新期五账户横向对比（截至 {as_of_date}）",
            "",
            "| 账户 | 净值(NAV) | 累计收益(%) | 最大回撤(%) | 现金 | 持仓市值 | 总资产 | 累计成交笔数 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for acct_id in ACCOUNT_ORDER:
        acct = accounts[acct_id]
        nav = float(acct["nav"])
        cum_ret = nav - 1.0
        max_dd = float(acct["max_drawdown"])
        cash = float(acct["cash"])
        equity = float(acct["equity"])
        market_value = equity - cash
        trade_count = len(acct.get("trades", []))
        lines.append(
            f"| {ACCOUNT_LABELS[acct_id]} | {nav:.4f} | {_fmt_signed_pct(cum_ret)} | "
            f"{_fmt_drawdown_pct(max_dd)} | {_fmt_money(cash)} | {_fmt_money(market_value)} | "
            f"{_fmt_money(equity)} | {trade_count} |"
        )

    # Section 3: Account A excess returns
    a_cum = float(accounts["A"]["nav"]) - 1.0
    excess_rows = [
        ("A vs B1", "B1", excess["A_vs_B1"]),
        ("A vs B2", "B2", excess["A_vs_B2"]),
        ("A vs C", "C", excess["A_vs_C"]),
        ("A vs D", "D", excess["A_vs_D"]),
    ]
    lines.extend(
        [
            "",
            "## 最新期 A 组超额收益（百分点）",
            "",
            "| 对比组 | 对照账户 | A组累计收益(%) | 对照累计收益(%) | 超额收益(百分点) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for label, opp_id, diff in excess_rows:
        opp_cum = float(accounts[opp_id]["nav"]) - 1.0
        lines.append(
            f"| {label} | {ACCOUNT_LABELS[opp_id]} | {_fmt_signed_pct(a_cum)} | "
            f"{_fmt_signed_pct(opp_cum)} | {_fmt_signed_pp(diff)} |"
        )

    # Section 4: Per-symbol effective/next-locked regime & holdings snapshot
    acct_a = accounts["A"]
    acct_d = accounts["D"]
    a_positions = acct_a.get("positions", {})
    a_sym_states = acct_a.get("symbol_states", {})
    d_positions = acct_d.get("positions", {})
    last_closes = state_dict.get("last_known_closes", {})
    symbol_names = state_dict.get("symbol_names", {})
    next_locked = state_dict.get("next_effective_regime", {})

    config_snap = state_dict.get("config_snapshot", {})
    universe_snap = config_snap.get("universe", {}) if isinstance(config_snap, Mapping) else {}
    sectors = sorted(
        universe_snap.get("sectors")
        or list(a_sym_states.keys())
        or list(a_positions.keys())
    )
    benchmark = universe_snap.get("benchmark")
    if not benchmark and isinstance(d_positions, Mapping) and d_positions:
        benchmark = sorted(d_positions.keys())[0]

    lines.extend(
        [
            "",
            "## 当前各标的生效/下周锁定路由与持仓快照",
            "",
            "| 标的 | 名称 | 本期生效路由 | 下周锁定路由 | A组策略模式 | A组持仓股数 | A组持仓市值 | A组网格挂单数 | 最新收盘价 |",
            "| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: |",
        ]
    )
    for sym in sectors:
        name = str(symbol_names.get(sym, "-")) or "-"
        sym_st = a_sym_states.get(sym, {}) if isinstance(a_sym_states, Mapping) else {}
        eff_regime = str(sym_st.get("effective_regime") or "-")
        raw_regime = sym_st.get("raw_regime")
        if raw_regime and raw_regime != eff_regime:
            eff_display = f"{eff_regime} ({raw_regime})"
        else:
            eff_display = eff_regime
        locked_regime = str(next_locked.get(sym) or eff_regime)
        pos_mode = str(sym_st.get("position_mode") or "-")
        pos = a_positions.get(sym, {}) if isinstance(a_positions, Mapping) else {}
        shares = int(pos.get("shares", 0))
        close_px = last_closes.get(sym)
        mv = float(shares) * float(close_px) if close_px is not None else 0.0
        grid_desc = _describe_grid_orders(sym_st)
        lines.append(
            f"| {sym} | {name} | {eff_display} | {locked_regime} | {pos_mode} | "
            f"{shares:,} | {_fmt_money(mv)} | {grid_desc} | {_fmt_price(close_px)} |"
        )

    if benchmark:
        b_name = str(symbol_names.get(benchmark, "基准ETF")) or "基准ETF"
        b_pos = d_positions.get(benchmark, {}) if isinstance(d_positions, Mapping) else {}
        b_shares = int(b_pos.get("shares", 0))
        b_close = last_closes.get(benchmark)
        b_mv = float(b_shares) * float(b_close) if b_close is not None else 0.0
        lines.append(
            f"| {benchmark} | {b_name}（基准） | BENCHMARK | BENCHMARK | D组BH | "
            f"D:{b_shares:,} | D:{_fmt_money(b_mv)} | - | {_fmt_price(b_close)} |"
        )

    # Optional compact audit log preview if present
    audit_log = state_dict.get("regime_audit_log")
    if isinstance(audit_log, list) and audit_log:
        recent_logs = audit_log[-10:]
        lines.extend(
            [
                "",
                "## 路由锁定审计记录（最近变更）",
                "",
                "| 行情日期 | 事件 | 路由变更明细 |",
                "| --- | --- | --- |",
            ]
        )
        for entry in recent_logs:
            if not isinstance(entry, Mapping):
                continue
            mkt_date = str(entry.get("recorded_at_market_date") or "-")
            event = str(entry.get("event") or "UPDATE")
            changes = entry.get("changes", {})
            if isinstance(changes, Mapping) and changes:
                parts: list[str] = []
                for sym in sorted(changes.keys()):
                    ch = changes[sym]
                    if isinstance(ch, Mapping):
                        old_v = ch.get("from")
                        new_v = ch.get("to")
                        if old_v is None:
                            parts.append(f"{sym}: {new_v}")
                        else:
                            parts.append(f"{sym}: {old_v} -> {new_v}")
                    else:
                        parts.append(f"{sym}: {ch}")
                detail_str = "; ".join(parts)
            else:
                detail_str = "-"
            lines.append(f"| {mkt_date} | {event} | {detail_str} |")

    # Section 5: Weekly historical NAV tracking table (idempotent by as_of_date)
    history_rows = _parse_existing_history_rows(md_path)
    if _ISO_DATE_RE.match(str(as_of_date)):
        history_rows[str(as_of_date)] = [
            str(as_of_date),
            str(trading_days),
            f"{float(accounts['A']['nav']):.4f}",
            f"{float(accounts['B1']['nav']):.4f}",
            f"{float(accounts['B2']['nav']):.4f}",
            f"{float(accounts['C']['nav']):.4f}",
            f"{float(accounts['D']['nav']):.4f}",
            _fmt_drawdown_pct(float(accounts["A"]["max_drawdown"])),
            _fmt_signed_pp(excess["A_vs_B1"]),
            _fmt_signed_pp(excess["A_vs_B2"]),
            _fmt_signed_pp(excess["A_vs_C"]),
            _fmt_signed_pp(excess["A_vs_D"]),
        ]

    lines.extend(
        [
            "",
            HISTORY_SECTION_HEADER,
            "",
            "| " + " | ".join(HISTORY_TABLE_COLUMNS) + " |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for dt in sorted(history_rows.keys()):
        lines.append("| " + " | ".join(history_rows[dt]) + " |")

    content = "\n".join(lines) + "\n"
    return _atomic_write_text(md_path, content)
