from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def variant_name(
    *,
    mode: str,
    window_size: int,
    window_minutes: float,
) -> str:
    if mode == "sequential":
        return f"sequential_k{window_size}"

    formatted = (
        str(int(window_minutes))
        if float(window_minutes).is_integer()
        else str(window_minutes).replace(".", "_")
    )

    return f"temporal_{formatted}m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run edge construction and graph materialization as one "
            "resumable behavioral-graph stage."
        )
    )

    parser.add_argument("sessions_csv", type=Path)
    parser.add_argument("--edge-output-root", type=Path, required=True)
    parser.add_argument("--processed-output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["sequential", "temporal"],
        default="sequential",
    )
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--window-minutes", type=float, default=30.0)
    parser.add_argument(
        "--min-shared-session-thresholds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 10],
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--min-scrobbles", type=int, default=3)
    parser.add_argument("--min-shared-sessions", type=int, default=2)
    parser.add_argument(
        "--weight-column",
        default="multi_user_shared_session_cosine",
    )

    return parser.parse_args()


def run(command: list[str]) -> None:
    print()
    print("Running:")
    print(" ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    build_edges = script_dir / "build_listening_graph_edges.py"
    materialize = script_dir / "materialize_listening_graph.py"

    run(
        [
            sys.executable,
            str(build_edges),
            str(args.sessions_csv),
            "--output-root",
            str(args.edge_output_root),
            "--mode",
            args.mode,
            "--window-size",
            str(args.window_size),
            "--window-minutes",
            str(args.window_minutes),
            "--min-shared-session-thresholds",
            *[
                str(value)
                for value in args.min_shared_session_thresholds
            ],
            "--progress-every",
            str(args.progress_every),
        ]
    )

    variant = variant_name(
        mode=args.mode,
        window_size=args.window_size,
        window_minutes=args.window_minutes,
    )

    variant_dir = args.edge_output_root / variant

    run(
        [
            sys.executable,
            str(materialize),
            str(variant_dir / "nodes_activity.csv"),
            str(variant_dir / "edges_raw.csv"),
            "--output-dir",
            str(args.processed_output_dir),
            "--min-scrobbles",
            str(args.min_scrobbles),
            "--min-shared-sessions",
            str(args.min_shared_sessions),
            "--weight-column",
            args.weight_column,
        ]
    )
