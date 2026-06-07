from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the validated MusicBrainz fetcher using a contact value "
            "provided through the MUSICBRAINZ_CONTACT environment variable."
        )
    )

    parser.add_argument("canonical_artists_csv", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delay-seconds", type=float, default=1.1)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--save-progress-every", type=int, default=25)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    contact = os.environ.get("MUSICBRAINZ_CONTACT", "").strip()

    if not contact:
        raise RuntimeError(
            "Set MUSICBRAINZ_CONTACT before running this stage. "
            "Example in PowerShell: "
            '$env:MUSICBRAINZ_CONTACT = "your-email@example.com"'
        )

    fetcher = (
        Path(__file__).resolve().parent
        / "fetch_musicbrainz_artist_genres.py"
    )

    command = [
        sys.executable,
        str(fetcher),
        str(args.canonical_artists_csv),
        "--cache-dir",
        str(args.cache_dir),
        "--output-dir",
        str(args.output_dir),
        "--contact",
        contact,
        "--delay-seconds",
        str(args.delay_seconds),
        "--max-retries",
        str(args.max_retries),
        "--save-progress-every",
        str(args.save_progress_every),
    ]

    print("Running:")
    print(" ".join(command[:-8] + ["--contact", "<redacted>"] + command[-6:]))
    subprocess.run(command, check=True)
