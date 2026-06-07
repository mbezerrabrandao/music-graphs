from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import hash_value, resolve_repo_path


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def fingerprint_stage(
    *,
    config: dict[str, Any],
    stage_id: str,
    stage_config: dict[str, Any],
) -> dict[str, Any]:
    input_records: list[dict[str, Any]] = []

    for raw_path in stage_config.get("inputs", []):
        path = resolve_repo_path(config, raw_path)
        record: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
        }

        if path.exists() and path.is_file():
            record["sha256"] = hash_file(path)
            record["size_bytes"] = path.stat().st_size

        input_records.append(record)

    payload = {
        "stage_id": stage_id,
        "command": stage_config.get("command", []),
        "parameters": stage_config.get("parameters", {}),
        "inputs": input_records,
    }

    return {
        "payload": payload,
        "fingerprint": hash_value(payload),
    }


def state_path(config: dict[str, Any], stage_id: str) -> Path:
    root = resolve_repo_path(
        config,
        config.get("project", {}).get(
            "state_dir",
            "results/.pipeline_state",
        ),
    )

    root.mkdir(parents=True, exist_ok=True)
    return root / f"{stage_id}.json"


def load_state(config: dict[str, Any], stage_id: str) -> dict[str, Any] | None:
    path = state_path(config, stage_id)

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ValueError(f"Invalid state file: {path}")

    return value


def save_state(
    config: dict[str, Any],
    stage_id: str,
    value: dict[str, Any],
) -> None:
    with state_path(config, stage_id).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def outputs_exist(
    *,
    config: dict[str, Any],
    stage_config: dict[str, Any],
) -> bool:
    outputs = stage_config.get("outputs", [])

    if not outputs:
        return False

    return all(resolve_repo_path(config, raw_path).exists() for raw_path in outputs)
