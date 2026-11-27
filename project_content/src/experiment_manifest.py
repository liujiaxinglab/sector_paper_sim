"""Immutable experiment identity and per-day upstream input records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tomllib
from typing import Any, Mapping

from src.data_loader import MarketBundle

MANIFEST_SCHEMA_VERSION = "1.0.0"
RULES_VERSION = "2.0.0"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = _REPO_ROOT.parent


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve(strict=True)
    try:
        display_path = resolved.relative_to(_WORKSPACE_ROOT.resolve()).as_posix()
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "file_name": resolved.name,
        "sha256": sha256_file(resolved),
    }


def current_program_identity() -> dict[str, str]:
    pyproject = _REPO_ROOT / "pyproject.toml"
    if pyproject.is_file():
        with pyproject.open("rb") as stream:
            project = tomllib.load(stream).get("project", {})
        version = str(project.get("version", "0.2.0"))
    else:
        version = "0.2.0"

    digest = hashlib.sha256()
    source_root = _REPO_ROOT / "project_content" / "src"
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(_REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    try:
        result = subprocess.run(
            [
                "git", "log", "-1", "--format=%H", "--",
                "project_content/src", "pyproject.toml",
            ],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"

    return {
        "version": version,
        "commit": commit,
        "source_sha256": digest.hexdigest(),
        "rules_sha256": sha256_bytes(RULES_VERSION.encode("utf-8")),
    }


def daily_input_sources(bundle: MarketBundle, trade_date: str) -> dict[str, Any]:
    upstream = bundle.config["upstream_paths"]
    tdx_batches: list[dict[str, Any]] = []
    for symbol in bundle.symbols:
        path = bundle.metadata["source_files"].get(symbol)
        if not path:
            raise ValueError(f"No TDX source file recorded for {symbol}")
        record = _source_path(path)
        record.update({
            "symbol": symbol,
            "data_through": min(bundle.metadata["symbol_stats"][symbol]["end_date"], trade_date),
        })
        tdx_batches.append(record)

    calendar = _source_path(bundle.metadata["calendar_file"])
    calendar.update({
        "schema": "BS-01 calendar_date,is_trading_day",
        "coverage_through": min(bundle.metadata["calendar_end_date"], trade_date),
    })

    ledger_path = upstream["corporate_actions_ledger_file"]
    ledger_payload = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    ledger = _source_path(ledger_path)
    ledger.update({
        "as_of": ledger_payload["as_of"],
        "schema_version": ledger_payload["schema_version"],
    })

    regime_path = upstream["regime_history_file"]
    regime_payload = json.loads(Path(regime_path).read_text(encoding="utf-8"))
    regime = _source_path(regime_path)
    regime.update({"schema_version": regime_payload["schema_version"]})

    return {
        "tdx_batches": tdx_batches,
        "trade_calendar": calendar,
        "corporate_actions_ledger": ledger,
        "regime_history": regime,
        "watchlist": _source_path(upstream["watchlist_file"]),
    }


def build_daily_input_manifest(
    bundle: MarketBundle,
    *,
    trade_date: str,
    experiment_id: str,
    run_mode: str,
    allow_lag: bool,
    forward_eval_start_date: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "run_date": trade_date,
        "trade_date": trade_date,
        "run_mode": run_mode,
        "allow_lag": allow_lag,
        "forward_eval_start_date": forward_eval_start_date,
        "inputs": daily_input_sources(bundle, trade_date),
    }


def build_experiment_manifest(
    bundle: MarketBundle,
    *,
    config_path: str | Path,
    run_mode: str,
) -> dict[str, Any]:
    config_bytes = Path(config_path).read_bytes()
    config_sha = sha256_bytes(config_bytes)
    program = current_program_identity()
    first_date = bundle.config["start_date"]
    initial_sources = daily_input_sources(bundle, first_date)
    initial_inputs_sha = sha256_bytes(canonical_json_bytes(initial_sources))
    account_strategy = {
        "account_defaults": bundle.config["account_defaults"],
        "strategies": bundle.config["strategies"],
        "universe": bundle.config["universe"],
    }
    identity = {
        "config_sha256": config_sha,
        "program_version": program["version"],
        "program_commit": program["commit"],
        "program_source_sha256": program["source_sha256"],
        "rules_version": RULES_VERSION,
        "rules_sha256": program["rules_sha256"],
        "initialization_run_mode": run_mode,
        "initialization_input_sha256": initial_inputs_sha,
    }
    experiment_id = "exp-" + sha256_bytes(canonical_json_bytes(identity))[:20]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "simulation_start_date": first_date,
        "initial_capital": bundle.config["account_defaults"]["initial_capital"],
        "config_snapshot": json.loads(config_bytes),
        "account_strategy_config_summary": account_strategy,
        "config_file": _source_path(config_path),
        "config_sha256": config_sha,
        "program_version": program["version"],
        "program_commit": program["commit"],
        "program_source_sha256": program["source_sha256"],
        "rules_version": RULES_VERSION,
        "rules_sha256": program["rules_sha256"],
        "initialization_run_mode": run_mode,
        "initialization_input_version": {
            "trade_date": first_date,
            "sha256": initial_inputs_sha,
            "inputs": initial_sources,
        },
    }


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(manifest))


def input_manifest_chain_sha256(
    manifests: Mapping[str, Mapping[str, Any]],
) -> str:
    chain = hashlib.sha256(b"sector_paper_sim_daily_inputs_v1").digest()
    for trade_date in sorted(manifests):
        item_hash = manifest_sha256(manifests[trade_date]).encode("ascii")
        chain = hashlib.sha256(
            chain + b"\0" + trade_date.encode("ascii") + b"\0" + item_hash
        ).digest()
    return chain.hex()


def validate_resume_manifest(
    saved: Mapping[str, Any], current: Mapping[str, Any],
) -> None:
    immutable_fields = (
        "schema_version",
        "simulation_start_date",
        "initial_capital",
        "account_strategy_config_summary",
        "program_version",
        "program_commit",
        "program_source_sha256",
        "rules_version",
        "rules_sha256",
    )
    differences = [
        f"{field}: saved={saved.get(field)!r}, current={current.get(field)!r}"
        for field in immutable_fields
        if saved.get(field) != current.get(field)
    ]
    if saved.get("account_strategy_config_summary") != current.get("account_strategy_config_summary"):
        changed_config = _changed_paths(
            saved.get("account_strategy_config_summary", {}),
            current.get("account_strategy_config_summary", {}),
        )
        if changed_config:
            differences.append("core config fields changed: " + ", ".join(changed_config))
    if differences:
        raise ValueError(
            "Experiment manifest mismatch; refusing to resume. " + "; ".join(differences)
        )


def _changed_paths(saved: Any, current: Any, prefix: str = "") -> list[str]:
    if isinstance(saved, Mapping) and isinstance(current, Mapping):
        paths: list[str] = []
        for key in sorted(set(saved) | set(current)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in saved or key not in current:
                paths.append(path)
            else:
                paths.extend(_changed_paths(saved[key], current[key], path))
        return paths
    if isinstance(saved, list) and isinstance(current, list):
        if len(saved) != len(current):
            return [prefix]
        paths = []
        for index, (old, new) in enumerate(zip(saved, current)):
            paths.extend(_changed_paths(old, new, f"{prefix}[{index}]"))
        return paths
    return [] if saved == current else [prefix]


def daily_input_manifest_path(output_dir: str | Path, trade_date: str) -> Path:
    return Path(output_dir) / "daily_inputs" / f"inputs_{trade_date.replace('-', '')}.json"
