from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

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


def normalize_text(value: Any) -> str | None:
    """
    Normalize text for matching while preserving the original text
    in separate columns.

    Examples:
        "  Artist   Name " -> "artist name"
        Unicode variants are normalized with NFKC.
    """
    if pd.isna(value):
        return None

    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return None

    return text.casefold()


def normalize_mbid(value: Any) -> str | None:
    """Normalize MusicBrainz IDs without assuming that every value is valid."""
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    return text or None


def make_fallback_artist_id(normalized_artist_name: str) -> str:
    """
    Create a stable identifier for artists without a MusicBrainz ID.

    SHA-1 is used only as a deterministic identifier, not for security.
    """
    digest = hashlib.sha1(
        normalized_artist_name.encode("utf-8")
    ).hexdigest()

    return f"name:{digest}"


def most_common_non_null(series: pd.Series) -> str | None:
    """Return the most frequent non-null display value."""
    clean = series.dropna()

    if clean.empty:
        return None

    return str(clean.value_counts().index[0])


def validate_columns(df: pd.DataFrame) -> None:
    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))

    if missing:
        raise ValueError(
            "The input CSV is missing required columns: "
            + ", ".join(missing)
        )


def infer_user_id(input_csv: Path) -> str:
    """
    Infer a stable user identifier from Last.fm export names.

    Expected multi-user files are named like:
    recenttracks-username-exportid.csv
    """
    match = re.match(
        r"^recenttracks-(?P<user>.+)-\d+$",
        input_csv.stem,
    )

    if match:
        return normalize_text(match.group("user")) or match.group("user")

    return normalize_text(input_csv.stem) or input_csv.stem


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


def read_raw_exports(input_path: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    manifests = []

    for csv_path in resolve_input_csvs(input_path):
        user_id = infer_user_id(csv_path)

        print(f"Reading raw export: {csv_path} [user_id={user_id}]")
        frame = pd.read_csv(
            csv_path,
            dtype="string",
            keep_default_na=True,
            low_memory=False,
        )

        frame.columns = [column.strip() for column in frame.columns]
        validate_columns(frame)

        frame.insert(0, "source_file", csv_path.name)
        frame.insert(1, "user_id", user_id)
        frame.insert(2, "raw_csv_line", range(2, len(frame) + 2))

        frames.append(frame)
        manifests.append(
            {
                "path": str(csv_path),
                "file_name": csv_path.name,
                "user_id": user_id,
                "raw_row_count": int(len(frame)),
                "file_size_bytes": int(csv_path.stat().st_size),
            }
        )

    return pd.concat(frames, ignore_index=True), manifests


def build_canonical_tables(input_csv: Path, output_dir: Path) -> None:
    input_path = input_csv

    output_dir.mkdir(parents=True, exist_ok=True)

    df, input_manifest = read_raw_exports(input_path)

    raw_row_count = len(df)

    original_columns = EXPECTED_COLUMNS.copy()

    # Clean whitespace-only values without changing the raw CSV.
    for column in original_columns:
        df[column] = df[column].str.strip().replace("", pd.NA)

    # Remove duplicate records, keeping the first occurrence.
    duplicate_mask = df.duplicated(
        subset=["user_id", *original_columns],
        keep="first",
    )

    duplicate_rows = df.loc[duplicate_mask].copy()
    duplicate_rows.to_csv(
        output_dir / "removed_exact_duplicates.csv",
        index=False,
        encoding="utf-8",
    )

    df = df.loc[~duplicate_mask].copy()

    # Parse timestamps required for session construction.
    numeric_uts = pd.to_numeric(df["uts"], errors="coerce")
    df["scrobble_time_utc"] = pd.to_datetime(
        numeric_uts,
        unit="s",
        utc=True,
        errors="coerce",
    )

    # Normalize identity fields.
    df["artist_name_normalized"] = df["artist"].map(normalize_text)
    df["artist_mbid_original"] = df["artist_mbid"].map(normalize_mbid)

    # Rows without timestamp or artist name cannot be used to build sessions.
    rejected_mask = (
        df["scrobble_time_utc"].isna()
        | df["artist_name_normalized"].isna()
    )

    rejected_rows = df.loc[rejected_mask].copy()
    rejected_rows.to_csv(
        output_dir / "rejected_scrobbles.csv",
        index=False,
        encoding="utf-8",
    )

    df = df.loc[~rejected_mask].copy()

    # Build a normalized-name -> MBID mapping only for unambiguous names.
    name_mbid_pairs = (
        df.loc[
            df["artist_mbid_original"].notna(),
            ["artist_name_normalized", "artist_mbid_original"],
        ]
        .drop_duplicates()
        .copy()
    )

    mbid_counts_by_name = (
        name_mbid_pairs.groupby("artist_name_normalized")[
            "artist_mbid_original"
        ]
        .nunique()
        .rename("distinct_mbids")
        .reset_index()
    )

    unambiguous_names = set(
        mbid_counts_by_name.loc[
            mbid_counts_by_name["distinct_mbids"] == 1,
            "artist_name_normalized",
        ]
    )

    ambiguous_names = mbid_counts_by_name.loc[
        mbid_counts_by_name["distinct_mbids"] > 1
    ].copy()

    ambiguous_names.to_csv(
        output_dir / "ambiguous_normalized_artist_names.csv",
        index=False,
        encoding="utf-8",
    )

    name_to_mbid = (
        name_mbid_pairs.loc[
            name_mbid_pairs["artist_name_normalized"].isin(
                unambiguous_names
            )
        ]
        .drop_duplicates(subset=["artist_name_normalized"])
        .set_index("artist_name_normalized")["artist_mbid_original"]
        .to_dict()
    )

    mapped_mbid = df["artist_name_normalized"].map(name_to_mbid)

    df["artist_mbid_resolved"] = (
        df["artist_mbid_original"]
        .fillna(mapped_mbid)
        .astype("string")
    )

    df["artist_mbid_resolution"] = "missing"
    df.loc[
        df["artist_mbid_original"].notna(),
        "artist_mbid_resolution",
    ] = "original"

    df.loc[
        df["artist_mbid_original"].isna()
        & df["artist_mbid_resolved"].notna(),
        "artist_mbid_resolution",
    ] = "backfilled_from_artist_name"

    def build_artist_id(row: pd.Series) -> str:
        resolved_mbid = row["artist_mbid_resolved"]

        if pd.notna(resolved_mbid):
            return f"mbid:{resolved_mbid}"

        return make_fallback_artist_id(
            row["artist_name_normalized"]
        )

    df["artist_id"] = df.apply(build_artist_id, axis=1)

    # Prepare values useful for aggregation.
    df["scrobble_date_utc"] = df["scrobble_time_utc"].dt.date

    # Sort chronologically because later steps will derive listening sessions.
    df = df.sort_values(
        ["user_id", "scrobble_time_utc", "raw_csv_line"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    df.insert(0, "scrobble_id", range(1, len(df) + 1))

    canonical_scrobbles_path = output_dir / "canonical_scrobbles.csv"
    df.to_csv(
        canonical_scrobbles_path,
        index=False,
        encoding="utf-8",
    )

    # Aggregate one record per artist.
    artists = (
        df.groupby("artist_id", dropna=False)
        .agg(
            artist_name=("artist", most_common_non_null),
            artist_name_normalized=(
                "artist_name_normalized",
                most_common_non_null,
            ),
            artist_mbid=("artist_mbid_resolved", most_common_non_null),
            scrobble_count=("scrobble_id", "size"),
            user_count=("user_id", "nunique"),
            first_scrobble_utc=("scrobble_time_utc", "min"),
            last_scrobble_utc=("scrobble_time_utc", "max"),
            active_days=("scrobble_date_utc", "nunique"),
            unique_tracks=("track", "nunique"),
            unique_albums=("album", "nunique"),
            scrobbles_with_original_mbid=(
                "artist_mbid_resolution",
                lambda values: int((values == "original").sum()),
            ),
            scrobbles_with_backfilled_mbid=(
                "artist_mbid_resolution",
                lambda values: int(
                    (values == "backfilled_from_artist_name").sum()
                ),
            ),
            scrobbles_without_mbid=(
                "artist_mbid_resolution",
                lambda values: int((values == "missing").sum()),
            ),
        )
        .reset_index()
    )

    artists["artist_id_source"] = artists["artist_mbid"].map(
        lambda value: "musicbrainz_mbid"
        if pd.notna(value)
        else "normalized_name_hash"
    )

    artists = artists.sort_values(
        ["scrobble_count", "artist_name"],
        ascending=[False, True],
    ).reset_index(drop=True)

    artists.to_csv(
        output_dir / "canonical_artists.csv",
        index=False,
        encoding="utf-8",
    )

    summary = {
        "input_file": str(input_csv),
        "input_type": "directory" if input_path.is_dir() else "file",
        "input_files": input_manifest,
        "user_count": int(df["user_id"].nunique()) if not df.empty else 0,
        "raw_row_count": int(raw_row_count),
        "removed_exact_duplicate_count": int(len(duplicate_rows)),
        "rejected_scrobble_count": int(len(rejected_rows)),
        "canonical_scrobble_count": int(len(df)),
        "canonical_artist_count": int(len(artists)),
        "scrobbles_with_original_artist_mbid": int(
            (df["artist_mbid_resolution"] == "original").sum()
        ),
        "scrobbles_with_backfilled_artist_mbid": int(
            (
                df["artist_mbid_resolution"]
                == "backfilled_from_artist_name"
            ).sum()
        ),
        "scrobbles_still_without_artist_mbid": int(
            (df["artist_mbid_resolution"] == "missing").sum()
        ),
        "artists_with_resolved_mbid": int(
            artists["artist_mbid"].notna().sum()
        ),
        "artists_using_name_fallback": int(
            artists["artist_mbid"].isna().sum()
        ),
        "ambiguous_normalized_artist_name_count": int(
            len(ambiguous_names)
        ),
        "first_scrobble_utc": (
            df["scrobble_time_utc"].min().isoformat()
            if not df.empty
            else None
        ),
        "last_scrobble_utc": (
            df["scrobble_time_utc"].max().isoformat()
            if not df.empty
            else None
        ),
        "outputs": {
            "canonical_scrobbles": str(canonical_scrobbles_path),
            "canonical_artists": str(
                output_dir / "canonical_artists.csv"
            ),
        },
    }

    with (
        output_dir / "canonicalization_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    print(f"Canonical tables written to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build canonical scrobble and artist tables from a raw "
            "Last.fm Data Export CSV."
        )
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the raw Last.fm export CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim"),
        help="Directory for generated canonical tables.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    build_canonical_tables(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
    )
