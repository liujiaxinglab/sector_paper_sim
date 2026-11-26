"""Atomic daily checkpoint, CSV ledgers and Markdown reports."""

from __future__ import annotations

import csv
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Iterable, Mapping

from src.engine import ACCOUNT_IDS

ACCOUNT_LABELS = {
    "A": "A 路由实验盘",
    "B1": "B1 纯机械海龟",
    "B2": "B2 纯机械网格",
    "C": "C 行业等权 B&H",
    "D": "D 沪深300 B&H",
}

TRADES_CSV_COLUMNS = (
    "date", "account", "trade_id", "symbol", "action", "side", "shares", "price",
    "notional", "commission", "stamp_duty", "net_cash_flow", "regime", "trigger_price",
    "grid_level", "note",
)
EXCESS_COLUMNS = tuple(f"A_vs_{account}" for account in ACCOUNT_IDS[1:])
NAV_CSV_COLUMNS = (
    "date", "account", "nav", "cash", "market_value", "equity", "daily_return", "cum_return",
    "peak_equity", "drawdown", "max_drawdown", "dividend_cash", "trades_today", "suspended_symbols",
    "lagged_symbols",
    *EXCESS_COLUMNS,
)


def _atomic_write_text(path: Path, content: str, *, immutable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(
                f"Immutable output conflict at {path}; archive it or use a new experiment directory"
            )
        return path
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            # Publish only if the date does not already have a snapshot.
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_text(encoding="utf-8") != content:
                    raise ValueError(
                        f"Immutable output conflict at {path}; archive it or use a new experiment directory"
                    )
        else:
            os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def save_state_json(state: Mapping[str, Any], output_path: str | Path) -> Path:
    content = json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    return _atomic_write_text(Path(output_path), content)


def load_state_json(input_path: str | Path) -> dict[str, Any]:
    path = Path(input_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def save_json_file(
    payload: Mapping[str, Any], output_path: str | Path, *, immutable: bool = False,
) -> Path:
    content = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    return _atomic_write_text(Path(output_path), content, immutable=immutable)


def load_json_file(input_path: str | Path) -> dict[str, Any]:
    path = Path(input_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def read_csv_rows(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"Unexpected CSV header in {path}")
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"Invalid CSV row width in {path}")
    return rows


def _export_csv(
    path: Path, columns: tuple[str, ...], rows: Iterable[Mapping[str, Any]],
    previous_rows: Iterable[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], tuple],
) -> Path:
    return _atomic_write_text(
        path, render_csv(columns, rows, previous_rows=previous_rows, key=key)
    )


def render_csv(
    columns: tuple[str, ...], rows: Iterable[Mapping[str, Any]],
    *, previous_rows: Iterable[Mapping[str, Any]] = (),
    key: Callable[[Mapping[str, Any]], tuple],
) -> str:
    merged = {key(row): row for row in previous_rows}
    merged.update((key(row), row) for row in rows)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(merged[k] for k in sorted(merged))
    return buffer.getvalue()


def export_trades_csv(
    trades: Iterable[Mapping[str, Any]], output_path: str | Path,
    *, previous_rows: Iterable[Mapping[str, Any]] = (),
) -> Path:
    return _export_csv(
        Path(output_path), TRADES_CSV_COLUMNS, trades, previous_rows,
        key=lambda trade: (trade["date"], trade["account"], int(trade["trade_id"])),
    )


def export_nav_daily_csv(
    rows: Iterable[Mapping[str, Any]], output_path: str | Path,
    *, previous_rows: Iterable[Mapping[str, Any]] = (),
) -> Path:
    return _export_csv(
        Path(output_path), NAV_CSV_COLUMNS, rows, previous_rows,
        key=lambda row: (row["date"], row["account"]),
    )


def nav_rows(day: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for account_id, metrics in day["accounts"].items():
        yield {
            "date": day["date"], "account": account_id, **metrics,
            "suspended_symbols": " ".join(day["suspended"]),
            "lagged_symbols": " ".join(day["lagged"]),
            **(day["excess_returns"] if account_id == "A" else {}),
        }


def valuation_markers(day: Mapping[str, Any]) -> Iterable[str]:
    for symbol in day["suspended"]:
        yield f"{symbol} [停牌按前收估值 {day['date']}]"
    for symbol in day["lagged"]:
        yield f"{symbol} [行情滞后按前收估值 {day['date']}]"


def build_daily_report(
    day: Mapping[str, Any], *, start_date: str, trading_days: int,
    symbol_names: Mapping[str, str],
) -> str:
    lines = [
        "# sector_paper_sim 日频净值",
        "",
        f"- 起算日 {start_date}",
        f"- 净值日 {day['date']}",
        f"- 累计交易日 {trading_days}",
        "",
        "| 账户 | 净值 | 日收益 | 累计收益 | 回撤 | 最大回撤 | 现金 | 持仓市值 | 权益 | 当日成交 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for account_id, metrics in day["accounts"].items():
        lines.append(
            f"| {ACCOUNT_LABELS[account_id]} | {metrics['nav']:.6f} | {metrics['daily_return']:+.2%} | "
            f"{metrics['cum_return']:+.2%} | {metrics['drawdown']:.2%} | {metrics['max_drawdown']:.2%} | "
            f"{metrics['cash']:,.2f} | {metrics['market_value']:,.2f} | {metrics['equity']:,.2f} | "
            f"{metrics['trades_today']} |"
        )
    lines.extend(["", "## A 组超额收益（百分点）", ""])
    for account in ACCOUNT_IDS[1:]:
        lines.append(f"- A 对 {account} {day['excess_returns'][f'A_vs_{account}'] * 100:+.2f}")

    lines.extend([
        "", "## A 组收盘持仓与路由", "",
        "| 标的 | 名称 | 目标路由 | 已执行路由 | 待清仓 | 持仓股数 | 网格挂单 | 估值价 |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ])
    for symbol, position in day["routed_positions"].items():
        close = day["closes"].get(symbol)
        valuation = f"{close:.4f}" if close is not None else "—"
        lines.append(
            f"| {symbol} | {symbol_names[symbol]} | {position['target_regime']} | "
            f"{position['active_regime'] or '—'} | {position['pending_liquidation'] or '—'} | "
            f"{position['shares']:,} | {position['grid_orders']} | {valuation} |"
        )

    if day["suspended"] or day["lagged"]:
        lines.extend(["", "## 停牌与缺失行情", ""])
        lines.extend(f"- {marker}" for marker in valuation_markers(day))
    if day["events"]:
        lines.extend(["", "## 公司行为", ""])
        for event in day["events"]:
            label = f"{event['account']} {event['symbol']} {event['event_type']}"
            if event["event_type"] == "DIVIDEND_CASH":
                lines.append(f"- {label} {event['div_cash']:,.6f} 元")
            else:
                lines.append(f"- {label} {event['old_shares']:,} 股 → {event['new_shares']:,} 股")

    run_mode = day.get("run_mode", "formal")
    fwd_start = day.get("forward_eval_start_date")
    backfill_window = day.get("backfill_replay_window")
    routing_stats = day.get("routing_stats") or {}
    window_desc = (
        f"{backfill_window['start_date']}..{backfill_window['end_date']}"
        if isinstance(backfill_window, Mapping)
        else "未配置"
    )
    fwd_desc = fwd_start if fwd_start else "待用户确认（不混入 BACKFILL 信息增量）"
    lines.extend([
        "",
        "## 路由来源与前瞻边界",
        "",
        f"- 运行模式：{run_mode}",
        f"- 模拟资金起算日：{start_date} ｜ 历史初始化回放区间：{window_desc} ｜ 正式前瞻评价起点：{fwd_desc}",
        (
            f"- 截至当日路由记录统计：已生效 LIVE {routing_stats.get('live_records_applied', 0)} 条"
            f"（前瞻期内 LIVE {routing_stats.get('forward_live_records_applied', 0)} 条），"
            f"已隔离 BACKFILL {routing_stats.get('backfill_records_excluded', 0)} 条"
            "（BACKFILL 仅用于状态初始化或历史重放，不计入信息路由增量统计）"
        ),
    ])
    effective_records = day.get("effective_route_records") or []
    if effective_records:
        for rec in effective_records:
            rec_ts = rec.get("recorded_at") or "null（历史补录时间边界见台账）"
            lines.append(
                f"- 当日生效路由：{rec['symbol']} {rec.get('prev_regime', '—')} -> {rec['regime']} "
                f"[{rec['record_type']}]（decision_date={rec.get('decision_date') or '—'}, "
                f"effective_date={rec['effective_date']}, recorded_at={rec_ts}）"
            )
    return "\n".join(lines) + "\n"


def snapshot_path(output_dir: Path, trade_date: str) -> Path:
    return output_dir / "history" / f"nav_{trade_date.replace('-', '')}.md"


def save_daily_snapshot(output_dir: Path, trade_date: str, content: str) -> Path:
    return _atomic_write_text(snapshot_path(output_dir, trade_date), content, immutable=True)


def save_latest_report(output_dir: Path, content: str) -> Path:
    return _atomic_write_text(output_dir / "nav_latest.md", content)
