from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the graph-support sensitivity table and validate "
            "the frozen repeated-session threshold."
        )
    )

    parser.add_argument("sensitivity_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-support", type=int, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    table = pd.read_csv(args.sensitivity_csv)

    required = {
        "min_shared_sessions",
        "node_count_with_edges",
        "edge_count",
        "largest_component_node_count",
        "largest_component_percentage_of_active_nodes",
    }

    if not required.issubset(table.columns):
        raise ValueError(
            f"Expected columns {sorted(required)} in {args.sensitivity_csv}"
        )

    for column in required:
        table[column] = pd.to_numeric(table[column], errors="raise")

    matches = table.loc[
        table["min_shared_sessions"] == args.selected_support
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one row for selected support "
            f"{args.selected_support}."
        )

    selected_row = matches.iloc[0].to_dict()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "decision": "minimum_shared_sessions",
        "selected": args.selected_support,
        "selection_rule": (
            "require repeated session support while retaining a broad "
            "active graph and a dominant connected component"
        ),
        "selected_row": selected_row,
        "uses_external_genres": False,
    }

    json_path = args.output_dir / "graph_support_selection.json"
    md_path = args.output_dir / "graph_support_selection.md"

    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "### Generated evidence: repeated-session edge support",
        "",
        f"- Selected minimum shared-session support: `{args.selected_support}`",
        "- Rule: require repeated co-occurrence while retaining a broad graph.",
        "- Uses external genres: `False`",
        "",
        "| support | active nodes | edges | largest component nodes | largest component % |",
        "|---:|---:|---:|---:|---:|",
    ]

    for row in table.sort_values("min_shared_sessions").itertuples(index=False):
        lines.append(
            "| "
            f"{int(row.min_shared_sessions)} | "
            f"{int(row.node_count_with_edges)} | "
            f"{int(row.edge_count)} | "
            f"{int(row.largest_component_node_count)} | "
            f"{float(row.largest_component_percentage_of_active_nodes):.4f} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
