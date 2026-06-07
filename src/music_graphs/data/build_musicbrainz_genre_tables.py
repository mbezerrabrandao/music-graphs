from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_text(value: Any) -> str | None:
    """
    Normalize text for deterministic matching while preserving the
    original display value in a separate column.
    """
    if value is None or pd.isna(value):
        return None

    text = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:
        return None

    return text.casefold()


def normalize_mbid(value: Any) -> str | None:
    """
    Normalize a MusicBrainz identifier.
    """
    if value is None or pd.isna(value):
        return None

    text = str(value).strip().lower()

    return text or None


def safe_non_negative_int(
    value: Any,
    field_name: str,
    default: int = 0,
) -> int:
    """
    Convert a scalar value to a non-negative integer.
    """
    if value is None or pd.isna(value):
        return default

    try:
        result = int(value)

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise ValueError(
            f"Invalid integer value for {field_name}: {value!r}"
        ) from error

    if result < 0:
        raise ValueError(
            f"Negative integer value for {field_name}: {value!r}"
        )

    return result


def load_canonical_artists(
    input_csv: Path,
) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Canonical artist table does not exist: {input_csv}"
        )

    artists = pd.read_csv(
        input_csv,
        dtype={
            "artist_id": "string",
            "artist_name": "string",
            "artist_mbid": "string",
        },
        low_memory=False,
    )

    required_columns = {
        "artist_id",
        "artist_name",
        "artist_mbid",
        "scrobble_count",
    }

    missing_columns = sorted(
        required_columns
        - set(artists.columns)
    )

    if missing_columns:
        raise ValueError(
            "Canonical artist table is missing required columns: "
            + ", ".join(missing_columns)
        )

    artists["artist_id"] = (
        artists["artist_id"]
        .astype("string")
        .str.strip()
    )

    if artists["artist_id"].isna().any():
        raise ValueError(
            "Found missing artist_id values in canonical artist table."
        )

    if artists["artist_id"].duplicated().any():
        raise ValueError(
            "Found duplicated artist_id values in canonical artist table."
        )

    artists["artist_mbid"] = (
        artists["artist_mbid"]
        .map(normalize_mbid)
        .astype("string")
    )

    artists["scrobble_count"] = pd.to_numeric(
        artists["scrobble_count"],
        errors="raise",
    )

    return artists


def cache_file_for_mbid(
    cache_dir: Path,
    mbid: str,
) -> Path:
    return cache_dir / f"{mbid}.json"


def load_cached_payload(
    cache_file: Path,
) -> dict[str, Any]:
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Missing cached MusicBrainz response: {cache_file}"
        )

    try:
        with cache_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON cache file: {cache_file}"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object in cache file: {cache_file}"
        )

    return payload


def extract_genre_records(
    artist_mbid: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    genres = payload.get(
        "genres",
        [],
    )

    if genres is None:
        return []

    if not isinstance(genres, list):
        raise ValueError(
            f"Expected a genre list for artist MBID: {artist_mbid}"
        )

    records: list[dict[str, Any]] = []

    for genre in genres:
        if not isinstance(genre, dict):
            continue

        genre_name = genre.get(
            "name"
        )

        normalized_name = normalize_text(
            genre_name
        )

        if normalized_name is None:
            continue

        records.append(
            {
                "artist_mbid": artist_mbid,
                "genre_id": str(
                    genre.get(
                        "id",
                        "",
                    )
                    or ""
                ).strip(),
                "genre_name": str(
                    genre_name
                ).strip(),
                "genre_name_normalized": normalized_name,
                "genre_disambiguation": str(
                    genre.get(
                        "disambiguation",
                        "",
                    )
                    or ""
                ).strip(),
                "annotation_count": safe_non_negative_int(
                    genre.get(
                        "count",
                        0,
                    ),
                    "genre.count",
                    default=0,
                ),
                "label_source": "musicbrainz_api",
            }
        )

    return records


def build_genres_long(
    artists: pd.DataFrame,
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mbids = sorted(
        {
            str(mbid)
            for mbid in (
                artists["artist_mbid"]
                .dropna()
                .astype(str)
            )
            if mbid
        }
    )

    genre_records: list[
        dict[str, Any]
    ] = []

    cache_audit_records: list[
        dict[str, Any]
    ] = []

    for index, mbid in enumerate(
        mbids,
        start=1,
    ):
        cache_file = cache_file_for_mbid(
            cache_dir,
            mbid,
        )

        payload = load_cached_payload(
            cache_file
        )

        records = extract_genre_records(
            artist_mbid=mbid,
            payload=payload,
        )

        genre_records.extend(
            records
        )

        cache_audit_records.append(
            {
                "artist_mbid": mbid,
                "cache_file": str(
                    cache_file
                ),
                "genre_annotation_count": len(
                    records
                ),
                "has_genres": bool(
                    records
                ),
            }
        )

        if index % 250 == 0:
            print(
                f"Processed {index:,}/"
                f"{len(mbids):,} cached artists."
            )

    genres_long = pd.DataFrame.from_records(
        genre_records
    )

    if genres_long.empty:
        genres_long = pd.DataFrame(
            columns=[
                "artist_mbid",
                "genre_id",
                "genre_name",
                "genre_name_normalized",
                "genre_disambiguation",
                "annotation_count",
                "label_source",
            ]
        )

    else:
        # Defensive consolidation in case an API response contains
        # duplicate normalized genre names.
        genres_long = (
            genres_long.sort_values(
                [
                    "artist_mbid",
                    "genre_name_normalized",
                    "annotation_count",
                    "genre_name",
                ],
                ascending=[
                    True,
                    True,
                    False,
                    True,
                ],
            )
            .drop_duplicates(
                subset=[
                    "artist_mbid",
                    "genre_name_normalized",
                ],
                keep="first",
            )
            .reset_index(drop=True)
        )

    cache_audit = pd.DataFrame.from_records(
        cache_audit_records
    )

    return genres_long, cache_audit


def build_dominant_genre_table(
    genres_long: pd.DataFrame,
) -> pd.DataFrame:
    if genres_long.empty:
        return pd.DataFrame(
            columns=[
                "artist_mbid",
                "dominant_genre",
                "dominant_genre_normalized",
                "dominant_genre_annotation_count",
                "genre_annotation_count",
                "label_source",
            ]
        )

    genre_counts = (
        genres_long.groupby(
            "artist_mbid"
        )
        .size()
        .rename(
            "genre_annotation_count"
        )
        .reset_index()
    )

    dominant = (
        genres_long.sort_values(
            [
                "artist_mbid",
                "annotation_count",
                "genre_name_normalized",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .drop_duplicates(
            subset=[
                "artist_mbid",
            ],
            keep="first",
        )
        .rename(
            columns={
                "genre_name": "dominant_genre",
                "genre_name_normalized": (
                    "dominant_genre_normalized"
                ),
                "annotation_count": (
                    "dominant_genre_annotation_count"
                ),
            }
        )[
            [
                "artist_mbid",
                "dominant_genre",
                "dominant_genre_normalized",
                "dominant_genre_annotation_count",
                "label_source",
            ]
        ]
    )

    return (
        dominant.merge(
            genre_counts,
            on="artist_mbid",
            how="left",
            validate="one_to_one",
        )
        .sort_values(
            "artist_mbid"
        )
        .reset_index(drop=True)
    )


def build_artist_coverage_table(
    artists: pd.DataFrame,
    dominant_genres: pd.DataFrame,
) -> pd.DataFrame:
    coverage = artists.merge(
        dominant_genres,
        on="artist_mbid",
        how="left",
        validate="many_to_one",
    )

    coverage[
        "metadata_group"
    ] = "name_only_without_mbid"

    coverage.loc[
        coverage["artist_mbid"].notna(),
        "metadata_group",
    ] = "mbid_without_genres"

    coverage.loc[
        coverage[
            "dominant_genre"
        ].notna(),
        "metadata_group",
    ] = "labeled_musicbrainz"

    coverage["label_available"] = (
        coverage[
            "metadata_group"
        ]
        .eq(
            "labeled_musicbrainz"
        )
    )

    coverage[
        "genre_annotation_count"
    ] = (
        coverage[
            "genre_annotation_count"
        ]
        .fillna(0)
        .astype(int)
    )

    coverage[
        "dominant_genre_annotation_count"
    ] = (
        coverage[
            "dominant_genre_annotation_count"
        ]
        .fillna(0)
        .astype(int)
    )

    return (
        coverage.sort_values(
            [
                "metadata_group",
                "scrobble_count",
                "artist_name",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def build_label_table(
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "artist_id",
        "artist_name",
        "artist_mbid",
        "scrobble_count",
        "metadata_group",
        "label_available",
        "dominant_genre",
        "dominant_genre_normalized",
        "dominant_genre_annotation_count",
        "genre_annotation_count",
        "label_source",
    ]

    return (
        coverage[
            columns
        ]
        .copy()
        .sort_values(
            [
                "label_available",
                "scrobble_count",
                "artist_name",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def group_counts(
    table: pd.DataFrame,
    column: str,
) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in (
            table[column]
            .value_counts(
                dropna=False
            )
            .sort_index()
            .items()
        )
    }


def group_percentages(
    table: pd.DataFrame,
    column: str,
) -> dict[str, float]:
    total = len(
        table
    )

    if total == 0:
        return {}

    return {
        str(key): round(
            float(
                value
                / total
                * 100
            ),
            6,
        )
        for key, value in (
            table[column]
            .value_counts(
                dropna=False
            )
            .sort_index()
            .items()
        )
    }


def build_graph_subset_coverage(
    graph_nodes_csv: Path,
    label_table: pd.DataFrame,
    output_csv: Path,
) -> pd.DataFrame:
    if not graph_nodes_csv.exists():
        raise FileNotFoundError(
            f"Graph node table does not exist: {graph_nodes_csv}"
        )

    graph_nodes = pd.read_csv(
        graph_nodes_csv,
        dtype={
            "artist_id": "string",
        },
        low_memory=False,
    )

    if "artist_id" not in graph_nodes.columns:
        raise ValueError(
            "Graph node table does not contain artist_id."
        )

    if graph_nodes[
        "artist_id"
    ].duplicated().any():
        raise ValueError(
            "Graph node table contains duplicated artist_id values."
        )

    label_columns = [
        "artist_id",
        "artist_mbid",
        "metadata_group",
        "label_available",
        "dominant_genre",
        "dominant_genre_normalized",
        "dominant_genre_annotation_count",
        "genre_annotation_count",
        "label_source",
    ]

    graph_coverage = graph_nodes.merge(
        label_table[
            label_columns
        ],
        on="artist_id",
        how="left",
        validate="one_to_one",
    )

    if graph_coverage[
        "metadata_group"
    ].isna().any():
        raise ValueError(
            "Some graph nodes were not found in the canonical "
            "artist label table."
        )

    graph_coverage.to_csv(
        output_csv,
        index=False,
        encoding="utf-8",
    )

    return graph_coverage


def build_tables(
    canonical_artists_csv: Path,
    cache_dir: Path,
    output_dir: Path,
    graph_nodes_csv: Path | None,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Reading canonical artists: "
        f"{canonical_artists_csv}"
    )

    print(
        f"Reading cached MusicBrainz JSON files: "
        f"{cache_dir}"
    )

    artists = load_canonical_artists(
        canonical_artists_csv
    )

    genres_long, cache_audit = build_genres_long(
        artists=artists,
        cache_dir=cache_dir,
    )

    dominant_genres = build_dominant_genre_table(
        genres_long
    )

    coverage = build_artist_coverage_table(
        artists=artists,
        dominant_genres=dominant_genres,
    )

    label_table = build_label_table(
        coverage
    )

    genres_long_output = (
        output_dir
        / "artist_genres_long.csv"
    )

    labels_output = (
        output_dir
        / "artist_genre_labels.csv"
    )

    coverage_output = (
        output_dir
        / "artist_genre_coverage.csv"
    )

    cache_audit_output = (
        output_dir
        / "artist_genre_cache_audit.csv"
    )

    genres_long.to_csv(
        genres_long_output,
        index=False,
        encoding="utf-8",
    )

    label_table.to_csv(
        labels_output,
        index=False,
        encoding="utf-8",
    )

    coverage.to_csv(
        coverage_output,
        index=False,
        encoding="utf-8",
    )

    cache_audit.to_csv(
        cache_audit_output,
        index=False,
        encoding="utf-8",
    )

    summary: dict[
        str,
        Any,
    ] = {
        "inputs": {
            "canonical_artists_csv": str(
                canonical_artists_csv
            ),
            "musicbrainz_cache_dir": str(
                cache_dir
            ),
            "graph_nodes_csv": (
                str(
                    graph_nodes_csv
                )
                if graph_nodes_csv
                is not None
                else None
            ),
        },
        "canonical_artist_universe": {
            "artist_count": int(
                len(
                    coverage
                )
            ),
            "metadata_group_counts": group_counts(
                coverage,
                "metadata_group",
            ),
            "metadata_group_percentages": group_percentages(
                coverage,
                "metadata_group",
            ),
            "unique_genre_count": int(
                genres_long[
                    "genre_name_normalized"
                ].nunique()
            ),
            "artist_genre_relation_count": int(
                len(
                    genres_long
                )
            ),
        },
        "outputs": {
            "artist_genres_long": str(
                genres_long_output
            ),
            "artist_genre_labels": str(
                labels_output
            ),
            "artist_genre_coverage": str(
                coverage_output
            ),
            "artist_genre_cache_audit": str(
                cache_audit_output
            ),
        },
    }

    if graph_nodes_csv is not None:
        graph_subset_output = (
            output_dir
            / "largest_component_genre_coverage.csv"
        )

        graph_coverage = (
            build_graph_subset_coverage(
                graph_nodes_csv=graph_nodes_csv,
                label_table=label_table,
                output_csv=graph_subset_output,
            )
        )

        summary[
            "behavioral_graph_largest_component"
        ] = {
            "artist_count": int(
                len(
                    graph_coverage
                )
            ),
            "metadata_group_counts": group_counts(
                graph_coverage,
                "metadata_group",
            ),
            "metadata_group_percentages": group_percentages(
                graph_coverage,
                "metadata_group",
            ),
            "labeled_artist_count": int(
                graph_coverage[
                    "label_available"
                ].sum()
            ),
            "labeled_artist_percentage": round(
                float(
                    graph_coverage[
                        "label_available"
                    ].mean()
                    * 100
                ),
                6,
            ),
        }

        summary[
            "outputs"
        ][
            "largest_component_genre_coverage"
        ] = str(
            graph_subset_output
        )

    summary_output = (
        output_dir
        / "artist_genre_tables_summary.json"
    )

    with summary_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        f"Genre tables written to: "
        f"{output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transform cached MusicBrainz artist genre responses "
            "into normalized label and coverage tables."
        )
    )

    parser.add_argument(
        "canonical_artists_csv",
        type=Path,
        help=(
            "Path to canonical_artists.csv."
        ),
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "data/raw/musicbrainz/artist_genres"
        ),
        help=(
            "Directory containing cached MusicBrainz "
            "artist JSON files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/interim/musicbrainz"
        ),
        help=(
            "Directory for normalized MusicBrainz "
            "tables."
        ),
    )

    parser.add_argument(
        "--graph-nodes-csv",
        type=Path,
        default=None,
        help=(
            "Optional graph node table used to measure "
            "label coverage inside the behavioral graph."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    build_tables(
        canonical_artists_csv=(
            args.canonical_artists_csv
        ),
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        graph_nodes_csv=args.graph_nodes_csv,
    )