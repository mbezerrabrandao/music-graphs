from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the tested session threshold closest to the p95 "
            "inter-scrobble gap and validate the frozen manuscript choice."
        )
    )

    parser.add_argument("gap_quantiles_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=[15, 30, 45, 60, 90, 120],
    )
    parser.add_argument("--expected-selected", type=int, default=60)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    table = pd.read_csv(args.gap_quantiles_csv)

    required = {"quantile", "gap_minutes"}

    if not required.issubset(table.columns):
        raise ValueError(
            f"Expected columns {sorted(required)} in {args.gap_quantiles_csv}"
        )

    table["quantile"] = pd.to_numeric(table["quantile"], errors="raise")
    table["gap_minutes"] = pd.to_numeric(table["gap_minutes"], errors="raise")

    row_index = (table["quantile"] - 0.95).abs().idxmin()
    p95 = float(table.loc[row_index, "gap_minutes"])

    selected = min(
        args.candidates,
        key=lambda candidate: (
            abs(float(candidate) - p95),
            candidate,
        ),
    )

    if selected != args.expected_selected:
        raise RuntimeError(
            "The automatically selected threshold differs from the frozen "
            f"manuscript choice: selected={selected}, "
            f"expected={args.expected_selected}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "decision": "session_threshold_minutes",
        "candidates": args.candidates,
        "selected": selected,
        "selection_rule": (
            "closest tested threshold to the p95 inter-scrobble gap"
        ),
        "evidence": {
            "p95_gap_minutes": p95,
        },
        "uses_external_genres": False,
    }

    json_path = args.output_dir / "session_threshold_selection.json"
    md_path = args.output_dir / "session_threshold_selection.md"

    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md_path.write_text(
        "\n".join(
            [
                "### Generated evidence: session threshold",
                "",
                f"- p95 inter-scrobble gap: `{p95:.6f}` minutes",
                f"- Tested thresholds: `{args.candidates}`",
                f"- Selected threshold: `{selected}` minutes",
                "- Rule: choose the tested threshold closest to the p95 gap.",
                "- Uses external genres: `False`",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2))
