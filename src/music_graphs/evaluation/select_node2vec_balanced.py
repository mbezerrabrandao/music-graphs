from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select Node2Vec p/q by maximum mean silhouette across seeds "
            "at the frozen balanced cluster scale."
        )
    )

    parser.add_argument("runs_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--balanced-k", type=int, default=54)
    parser.add_argument("--expected-p", type=float, default=2.0)
    parser.add_argument("--expected-q", type=float, default=2.0)
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runs = pd.read_csv(args.runs_csv)

    required = {
        "p",
        "q",
        "seed",
        "k",
        "silhouette_score",
    }

    if not required.issubset(runs.columns):
        raise ValueError(
            f"Expected columns {sorted(required)} in {args.runs_csv}"
        )

    for column in required:
        runs[column] = pd.to_numeric(runs[column], errors="raise")

    candidates = runs.loc[runs["k"] == args.balanced_k].copy()

    if candidates.empty:
        raise RuntimeError(
            f"No Node2Vec runs found for balanced k={args.balanced_k}."
        )

    summary = (
        candidates.groupby(["p", "q", "k"], as_index=False)
        .agg(
            run_count=("seed", "size"),
            seed_count=("seed", "nunique"),
            silhouette_mean=("silhouette_score", "mean"),
            silhouette_std=("silhouette_score", "std"),
        )
        .fillna(0.0)
        .sort_values(
            ["silhouette_mean", "p", "q"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )

    selected = summary.iloc[0]
    selected_p = float(selected["p"])
    selected_q = float(selected["q"])

    expected_seed_count = len(set(args.expected_seeds))

    if int(selected["seed_count"]) != expected_seed_count:
        raise RuntimeError(
            "Selected Node2Vec candidate does not contain the expected "
            f"number of seeds: {int(selected['seed_count'])} != "
            f"{expected_seed_count}."
        )

    if (
        abs(selected_p - args.expected_p) > 1e-12
        or abs(selected_q - args.expected_q) > 1e-12
    ):
        raise RuntimeError(
            "Automatically selected Node2Vec walk bias differs from the "
            "frozen manuscript choice: "
            f"selected=(p={selected_p}, q={selected_q}), "
            f"expected=(p={args.expected_p}, q={args.expected_q})."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        args.output_dir / "node2vec_balanced_pq_summary.csv",
        index=False,
        encoding="utf-8",
    )

    payload = {
        "decision": "node2vec_walk_bias",
        "balanced_k": args.balanced_k,
        "selected": {
            "p": selected_p,
            "q": selected_q,
        },
        "selection_rule": (
            "maximum mean silhouette across the expected seeds within "
            "the frozen balanced cluster scale"
        ),
        "evidence": {
            "run_count": int(selected["run_count"]),
            "silhouette_mean": float(selected["silhouette_mean"]),
            "silhouette_std": float(selected["silhouette_std"]),
        },
        "uses_external_genres": False,
    }

    (args.output_dir / "node2vec_walk_bias_selection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "### Generated evidence: Node2Vec walk-bias selection",
        "",
        f"- Balanced cluster scale: `k={args.balanced_k}`",
        f"- Selected walk bias: `p={selected_p}`, `q={selected_q}`",
        "- Rule: maximize mean silhouette across seeds at the balanced scale.",
        "- Uses external genres: `False`",
        "",
        "| p | q | k | runs | silhouette mean | silhouette std |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary.itertuples(index=False):
        lines.append(
            "| "
            f"{float(row.p):.1f} | "
            f"{float(row.q):.1f} | "
            f"{int(row.k)} | "
            f"{int(row.run_count)} | "
            f"{float(row.silhouette_mean):.6f} | "
            f"{float(row.silhouette_std):.6f} |"
        )

    (args.output_dir / "node2vec_walk_bias_selection.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2))
