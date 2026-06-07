from __future__ import annotations

import datetime as dt
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .state import fingerprint_stage, load_state, outputs_exist, save_state
from .stages import ALL_STAGES, StageSpec


@dataclass(frozen=True)
class PlannedStage:
    spec: StageSpec
    enabled: bool
    status: str
    reason: str


def _enabled_optional_groups(config: dict[str, Any]) -> set[str]:
    return {
        key
        for key, enabled in config.get("optional", {}).items()
        if bool(enabled)
    }


def _stage_config(config: dict[str, Any], stage_id: str) -> dict[str, Any]:
    value = config.get("stages", {}).get(stage_id, {})

    if not isinstance(value, dict):
        raise ValueError(f"Stage config must be a mapping: {stage_id}")

    return value


def _slice_stages(
    stages: list[StageSpec],
    *,
    from_stage: str | None,
    to_stage: str | None,
) -> list[StageSpec]:
    ids = [stage.stage_id for stage in stages]
    start = 0 if from_stage is None else ids.index(from_stage)
    stop = len(stages) if to_stage is None else ids.index(to_stage) + 1
    return stages[start:stop]


def build_plan(
    *,
    config: dict[str, Any],
    from_stage: str | None = None,
    to_stage: str | None = None,
    forced_stages: set[str] | None = None,
) -> list[PlannedStage]:
    forced_stages = forced_stages or set()
    optional_groups = _enabled_optional_groups(config)
    stages = _slice_stages(
        ALL_STAGES,
        from_stage=from_stage,
        to_stage=to_stage,
    )

    plan: list[PlannedStage] = []

    for spec in stages:
        stage_config = _stage_config(config, spec.stage_id)
        enabled = bool(stage_config.get("enabled", True)) and (
            spec.optional_group is None
            or spec.optional_group in optional_groups
        )

        if not enabled:
            plan.append(
                PlannedStage(spec, False, "disabled", "Disabled by configuration.")
            )
            continue

        command = stage_config.get("command", [])

        if not command:
            plan.append(
                PlannedStage(spec, True, "unwired", "No command has been mapped yet.")
            )
            continue

        fingerprint = fingerprint_stage(
            config=config,
            stage_id=spec.stage_id,
            stage_config=stage_config,
        )

        previous = load_state(config, spec.stage_id)
        cached = (
            spec.stage_id not in forced_stages
            and previous is not None
            and previous.get("fingerprint") == fingerprint["fingerprint"]
            and outputs_exist(config=config, stage_config=stage_config)
        )

        plan.append(
            PlannedStage(
                spec,
                True,
                "cached" if cached else "run",
                "Fingerprint and declared outputs match."
                if cached
                else "Stage must be executed.",
            )
        )

    return plan


def _format_command(config: dict[str, Any], command: list[str]) -> list[str]:
    return [
        str(token).format(
            repo_root=config["_meta"]["repo_root"],
            python=sys.executable,
        )
        for token in command
    ]


def execute_plan(
    *,
    config: dict[str, Any],
    plan: list[PlannedStage],
) -> None:
    for item in plan:
        stage_id = item.spec.stage_id
        stage_config = _stage_config(config, stage_id)

        print()
        print(f"== {stage_id} ==")
        print(item.spec.description)
        print(f"status: {item.status}")

        if item.status in {"disabled", "cached"}:
            print(item.reason)
            continue

        if item.status == "unwired":
            raise RuntimeError(
                f"Stage '{stage_id}' has not been wired yet. "
                "Map its command in the YAML config before executing."
            )

        command = _format_command(config, stage_config["command"])

        print("command:")
        print(" ".join(command))

        started_at = dt.datetime.now(tz=dt.timezone.utc)

        completed = subprocess.run(
            command,
            cwd=Path(config["_meta"]["repo_root"]),
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"Stage '{stage_id}' failed with exit code {completed.returncode}."
            )

        fingerprint = fingerprint_stage(
            config=config,
            stage_id=stage_id,
            stage_config=stage_config,
        )

        save_state(
            config,
            stage_id,
            {
                **fingerprint,
                "status": "completed",
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": dt.datetime.now(
                    tz=dt.timezone.utc
                ).isoformat(),
                "declared_outputs": stage_config.get("outputs", []),
            },
        )


def render_plan(plan: list[PlannedStage]) -> str:
    lines = [
        "stage_id | status | description",
        "--- | --- | ---",
    ]

    for item in plan:
        lines.append(
            f"{item.spec.stage_id} | {item.status} | {item.spec.description}"
        )

    return "\n".join(lines)
