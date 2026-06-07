from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import resolve_repo_path


def _render(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return "```json\n" + json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n```"

    return f"`{value}`"


def write_selection_report(config: dict[str, Any]) -> Path:
    output = resolve_repo_path(
        config,
        config.get("project", {}).get(
            "selection_report",
            "results/selection_report.md",
        ),
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Automatic methodological decision report",
        "",
        "Generated from the reproducible pipeline configuration and available "
        "selection-evidence files.",
        "",
        "## General separation of signals",
        "",
        "Behavioral graph construction and unsupervised representation learning "
        "do not use MusicBrainz genre labels. External genres are used for "
        "evaluation and for the explicitly supervised downstream inference task.",
        "",
        "## Declared manuscript decisions",
        "",
    ]

    for index, decision in enumerate(config.get("decisions", []), start=1):
        lines.extend(
            [
                f"### {index}. {decision['decision']}",
                "",
                f"- **Status:** `{decision.get('status', 'unspecified')}`",
                f"- **Selected value:** {_render(decision.get('selected'))}",
                f"- **Selection rule:** {decision.get('selection_rule', 'Not declared')}",
                f"- **Uses external genres:** `{decision.get('uses_external_genres', False)}`",
            ]
        )

        if "candidates" in decision:
            lines.append(f"- **Candidates:** {_render(decision['candidates'])}")

        if "evidence" in decision:
            lines.append(f"- **Evidence:** {_render(decision['evidence'])}")

        if "notes" in decision:
            lines.append(f"- **Notes:** {decision['notes']}")

        lines.append("")

    evidence_dir = resolve_repo_path(config, "results/selection_evidence")

    lines.extend(
        [
            "## Generated evidence",
            "",
        ]
    )

    evidence_files = sorted(evidence_dir.glob("*.md")) if evidence_dir.exists() else []

    if not evidence_files:
        lines.extend(
            [
                "_No generated evidence files are available yet._",
                "",
            ]
        )

    for evidence_file in evidence_files:
        lines.extend(
            [
                evidence_file.read_text(encoding="utf-8").strip(),
                "",
            ]
        )

    output.write_text("\n".join(lines), encoding="utf-8")
    return output
