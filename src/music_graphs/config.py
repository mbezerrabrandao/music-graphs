from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config and attach an absolute repository root."""
    config_path = path.resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping.")

    repo_root = config_path.parent.parent.resolve()
    config["_meta"] = {
        "config_path": str(config_path),
        "repo_root": str(repo_root),
    }

    return config


def stable_json(value: Any) -> str:
    """Serialize deterministically for hashing and state files."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def hash_value(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def resolve_repo_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return (Path(config["_meta"]["repo_root"]) / path).resolve()
