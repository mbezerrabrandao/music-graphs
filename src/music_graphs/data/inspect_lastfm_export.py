from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "uts",
    "utc_time",
    "artist",
    "artist_mbid",
    "album",
    "album_mbid",
    "track",
    "track_mbid",
]


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text_column(series: pd.Series) -> pd.Series:
    """Convert whitespace-only strings to missing values."""
    return series.astype("string").str.strip().replace("", pd.NA)


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def infer_user_id(input_csv: Path) -> str:
    match = re.match(
        r"^recenttracks-(?P<user>.+)-\d+$",
        input_csv.stem,
    )

    if match:
        return normalize_text(match.group("user"))

    return normalize_text(input_csv.stem)


def resolve_input_csvs(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        return [input_path]

    csv_paths = sorted(
        path
        for path in input_path.glob("*.csv")
        if path.is_file()
    )

    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files were found in input directory: {input_path}"
        )

    return csv_paths


def inspect_single_export(input_csv: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, low_memory=False)
    df.columns = [column.strip() for column in df.columns]

    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = clean_text_column(df[column])

    missing_expected_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    unexpected_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))

    # Parse timestamps without altering the raw file.
    time_source = None
    parsed_time = None

    if "uts" in df.columns:
        parsed_time = pd.to_datetime(
            pd.to_numeric(df["uts"], errors="coerce"),
            unit="s",
            utc=True,
            errors="coerce",
        )
        time_source = "uts"
    elif "utc_time" in df.columns:
        parsed_time = pd.to_datetime(df["utc_time"], utc=True, errors="coerce")
        time_source = "utc_time"

    valid_times = parsed_time.dropna() if parsed_time is not None else pd.Series(dtype="datetime64[ns, UTC]")

    missing_values = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": [int(df[column].isna().sum()) for column in df.columns],
            "missing_percentage": [
                round(float(df[column].isna().mean() * 100), 6)
                for column in df.columns
            ],
        }
    )
    missing_values.to_csv(output_dir / "missing_values_by_column.csv", index=False)

    exact_duplicates = df[df.duplicated(keep=False)].copy()
    exact_duplicates.to_csv(output_dir / "exact_duplicate_rows.csv", index=False)

    artist_summary = None
    if "artist" in df.columns:
        artist_columns = ["artist"]
        if "artist_mbid" in df.columns:
            artist_columns.append("artist_mbid")

        artist_data = df[artist_columns].copy()

        if "artist_mbid" in artist_data.columns:
            artist_summary = (
                artist_data.groupby("artist", dropna=False)
                .agg(
                    scrobble_count=("artist", "size"),
                    scrobbles_with_mbid=("artist_mbid", lambda values: int(values.notna().sum())),
                    scrobbles_without_mbid=("artist_mbid", lambda values: int(values.isna().sum())),
                    distinct_mbids=(
                        "artist_mbid",
                        lambda values: int(values.dropna().nunique()),
                    ),
                    mbids=(
                        "artist_mbid",
                        lambda values: "|".join(sorted(set(values.dropna().astype(str)))),
                    ),
                )
                .reset_index()
                .sort_values(["scrobble_count", "artist"], ascending=[False, True])
            )
        else:
            artist_summary = (
                artist_data.groupby("artist", dropna=False)
                .size()
                .rename("scrobble_count")
                .reset_index()
                .sort_values(["scrobble_count", "artist"], ascending=[False, True])
            )

        artist_summary.to_csv(output_dir / "artist_mbid_summary.csv", index=False)

    summary = {
        "input_file": str(input_csv),
        "file_name": input_csv.name,
        "user_id": infer_user_id(input_csv),
        "file_size_bytes": input_csv.stat().st_size,
        "sha256": sha256sum(input_csv),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "missing_expected_columns": missing_expected_columns,
        "unexpected_columns": unexpected_columns,
        "exact_duplicate_row_count": int(df.duplicated().sum()),
        "timestamp_source": time_source,
        "first_scrobble_utc": valid_times.min().isoformat() if not valid_times.empty else None,
        "last_scrobble_utc": valid_times.max().isoformat() if not valid_times.empty else None,
    }

    if "artist" in df.columns:
        summary["unique_artist_names"] = int(df["artist"].nunique(dropna=True))

    if "artist_mbid" in df.columns:
        summary["scrobbles_without_artist_mbid"] = int(df["artist_mbid"].isna().sum())
        summary["scrobbles_with_artist_mbid"] = int(df["artist_mbid"].notna().sum())
        summary["unique_artist_mbids"] = int(df["artist_mbid"].nunique(dropna=True))

        if artist_summary is not None:
            summary["artist_names_without_any_mbid"] = int(
                (artist_summary["scrobbles_with_mbid"] == 0).sum()
            )
            summary["artist_names_with_multiple_mbids"] = int(
                (artist_summary["distinct_mbids"] > 1).sum()
            )

    with (output_dir / "lastfm_export_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return summary


def inspect_export(input_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for csv_path in resolve_input_csvs(input_csv):
        summaries.append(
            inspect_single_export(
                input_csv=csv_path,
                output_dir=output_dir / csv_path.stem,
            )
        )

    summary = {
        "input_path": str(input_csv),
        "input_type": "directory" if input_csv.is_dir() else "file",
        "file_count": int(len(summaries)),
        "user_count": int(
            len({item["user_id"] for item in summaries})
        ),
        "row_count": int(
            sum(item["row_count"] for item in summaries)
        ),
        "unique_artist_names_sum": int(
            sum(item.get("unique_artist_names", 0) for item in summaries)
        ),
        "files": summaries,
    }

    with (output_dir / "lastfm_exports_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    print(f"Audit files written to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a raw CSV exported from LastFM Data Export."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/data_audit"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inspect_export(args.input_csv, args.output_dir)
