from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the Louvain/Leiden resolution grid and document "
            "the frozen balanced manuscript scale."
        )
    )

    parser.add_argument("runs_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--balanced-resolution", type=float, default=2.25)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runs = pd.read_csv(args.runs_csv)

    required = {
        "algorithm",
        "resolution",
        "seed",
        "modularity",
        "nmi",
        "ami",
        "ari",
        "purity_labeled_nodes",
        "community_count",
        "singleton_community_count",
        "median_community_size",
        "maximum_community_size",
    }

    if not required.issubset(runs.columns):
        raise ValueError(
            f"Expected columns {sorted(required)} in {args.runs_csv}"
        )

    numeric_columns = sorted(required - {"algorithm"})

    for column in numeric_columns:
        runs[column] = pd.to_numeric(runs[column], errors="raise")

    summary = (
        runs.groupby(["algorithm", "resolution"], as_index=False)
        .agg(
            run_count=("seed", "size"),
            modularity_mean=("modularity", "mean"),
            nmi_mean=("nmi", "mean"),
            ami_mean=("ami", "mean"),
            ari_mean=("ari", "mean"),
            purity_mean=("purity_labeled_nodes", "mean"),
            community_count_mean=("community_count", "mean"),
            singleton_community_count_mean=("singleton_community_count", "mean"),
            median_community_size_mean=("median_community_size", "mean"),
            maximum_community_size_mean=("maximum_community_size", "mean"),
        )
        .sort_values(["algorithm", "resolution"])
        .reset_index(drop=True)
    )

    selected = summary.loc[
        (summary["resolution"] - args.balanced_resolution).abs() < 1e-12
    ]

    if selected.empty:
        raise RuntimeError(
            "Frozen balanced resolution is absent from the generated grid: "
            f"{args.balanced_resolution}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        args.output_dir / "modularity_resolution_grid_summary.csv",
        index=False,
        encoding="utf-8",
    )

    payload = {
        "decision": "modularity_balanced_resolution",
        "selected": args.balanced_resolution,
        "selection_rule": (
            "frozen manuscript scale after exploratory trade-off analysis; "
            "reported transparently rather than automatically optimized "
            "against external genres"
        ),
        "uses_external_genres": True,
    }

    (args.output_dir / "modularity_resolution_selection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "### Generated evidence: Louvain and Leiden resolution grid",
        "",
        f"- Frozen balanced manuscript resolution: `{args.balanced_resolution}`",
        "- Rule: report the exploratory trade-off transparently; do not "
        "automatically optimize resolution against genres for a new user.",
        "- Uses external genres in exploratory analysis: `True`",
        "",
        "| algorithm | resolution | modularity | NMI | AMI | ARI | purity | communities | singletons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary.itertuples(index=False):
        lines.append(
            "| "
            f"{row.algorithm} | "
            f"{float(row.resolution):.2f} | "
            f"{float(row.modularity_mean):.4f} | "
            f"{float(row.nmi_mean):.4f} | "
            f"{float(row.ami_mean):.4f} | "
            f"{float(row.ari_mean):.4f} | "
            f"{float(row.purity_mean):.4f} | "
            f"{float(row.community_count_mean):.1f} | "
            f"{float(row.singleton_community_count_mean):.1f} |"
        )

    (args.output_dir / "modularity_resolution_selection.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2))
