from __future__ import annotations

import argparse
from pathlib import Path

from music_graphs.config import load_config
from music_graphs.report import write_selection_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = write_selection_report(load_config(args.config))
    print(f"Selection report written to: {output}")
