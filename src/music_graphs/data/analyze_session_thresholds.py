from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_THRESHOLDS_MINUTES = [15, 30, 45, 60, 90, 120]


def load_scrobbles(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_csv}")

    df = pd.read_csv(
        input_csv,
        low_memory=False,
    )

    required_columns = {
        "scrobble_id",
        "scrobble_time_utc",
        "artist_id",
    }

    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    df["scrobble_time_utc"] = pd.to_datetime(
        df["scrobble_time_utc"],
        utc=True,
        errors="coerce",
    )

    invalid_time_count = int(df["scrobble_time_utc"].isna().sum())

    if invalid_time_count:
        raise ValueError(
            f"Found {invalid_time_count} rows with invalid timestamps."
        )

    df = (
        df.sort_values(
            ["scrobble_time_utc", "scrobble_id"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )

    df["gap_minutes"] = (
        df["scrobble_time_utc"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    return df


def calculate_gap_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    gaps = df["gap_minutes"].dropna()

    quantiles = [
        0.00,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        0.995,
        0.999,
        1.00,
    ]

    return pd.DataFrame(
        {
            "quantile": quantiles,
            "gap_minutes": [
                round(float(gaps.quantile(q)), 6)
                for q in quantiles
            ],
        }
    )


def calculate_gap_bins(df: pd.DataFrame) -> pd.DataFrame:
    gaps = df["gap_minutes"].dropna()

    bin_edges = [
        0,
        5,
        10,
        15,
        30,
        45,
        60,
        90,
        120,
        240,
        480,
        1440,
        float("inf"),
    ]

    labels = [
        "0-5 min",
        "5-10 min",
        "10-15 min",
        "15-30 min",
        "30-45 min",
        "45-60 min",
        "60-90 min",
        "90-120 min",
        "2-4 h",
        "4-8 h",
        "8-24 h",
        ">24 h",
    ]

    categories = pd.cut(
        gaps,
        bins=bin_edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    counts = (
        categories.value_counts(sort=False)
        .rename_axis("gap_interval")
        .reset_index(name="gap_count")
    )

    counts["percentage"] = (
        counts["gap_count"]
        .div(len(gaps))
        .mul(100)
        .round(6)
    )

    return counts


def summarize_threshold(
    df: pd.DataFrame,
    threshold_minutes: int,
) -> dict:
    work = df.copy()

    work["starts_new_session"] = (
        work["gap_minutes"].isna()
        | work["gap_minutes"].gt(threshold_minutes)
    )

    work["session_id"] = (
        work["starts_new_session"]
        .cumsum()
        .astype(int)
    )

    sessions = (
        work.groupby("session_id")
        .agg(
            first_scrobble_utc=("scrobble_time_utc", "min"),
            last_scrobble_utc=("scrobble_time_utc", "max"),
            scrobble_count=("scrobble_id", "size"),
            unique_artist_count=("artist_id", "nunique"),
        )
        .reset_index()
    )

    sessions["duration_minutes"] = (
        sessions["last_scrobble_utc"]
        .sub(sessions["first_scrobble_utc"])
        .dt.total_seconds()
        .div(60)
    )

    return {
        "threshold_minutes": int(threshold_minutes),
        "session_count": int(len(sessions)),
        "singleton_session_count": int(
            (sessions["scrobble_count"] == 1).sum()
        ),
        "singleton_session_percentage": round(
            float(
                (sessions["scrobble_count"] == 1).mean()
                * 100
            ),
            6,
        ),
        "mean_scrobbles_per_session": round(
            float(sessions["scrobble_count"].mean()),
            6,
        ),
        "median_scrobbles_per_session": round(
            float(sessions["scrobble_count"].median()),
            6,
        ),
        "p90_scrobbles_per_session": round(
            float(sessions["scrobble_count"].quantile(0.90)),
            6,
        ),
        "max_scrobbles_in_session": int(
            sessions["scrobble_count"].max()
        ),
        "mean_unique_artists_per_session": round(
            float(sessions["unique_artist_count"].mean()),
            6,
        ),
        "median_unique_artists_per_session": round(
            float(sessions["unique_artist_count"].median()),
            6,
        ),
        "p90_unique_artists_per_session": round(
            float(
                sessions["unique_artist_count"]
                .quantile(0.90)
            ),
            6,
        ),
        "median_session_duration_minutes": round(
            float(sessions["duration_minutes"].median()),
            6,
        ),
        "p90_session_duration_minutes": round(
            float(
                sessions["duration_minutes"]
                .quantile(0.90)
            ),
            6,
        ),
        "max_session_duration_minutes": round(
            float(sessions["duration_minutes"].max()),
            6,
        ),
    }


def analyze_thresholds(
    input_csv: Path,
    output_dir: Path,
    thresholds: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading canonical scrobbles: {input_csv}")

    df = load_scrobbles(input_csv)

    negative_gap_count = int(
        (df["gap_minutes"].dropna() < 0).sum()
    )

    zero_gap_count = int(
        (df["gap_minutes"].dropna() == 0).sum()
    )

    gap_quantiles = calculate_gap_quantiles(df)
    gap_quantiles.to_csv(
        output_dir / "inter_scrobble_gap_quantiles.csv",
        index=False,
        encoding="utf-8",
    )

    gap_bins = calculate_gap_bins(df)
    gap_bins.to_csv(
        output_dir / "inter_scrobble_gap_bins.csv",
        index=False,
        encoding="utf-8",
    )

    threshold_records = [
        summarize_threshold(df, threshold)
        for threshold in thresholds
    ]

    threshold_sensitivity = pd.DataFrame(
        threshold_records
    )

    threshold_sensitivity.to_csv(
        output_dir / "session_threshold_sensitivity.csv",
        index=False,
        encoding="utf-8",
    )

    summary = {
        "input_file": str(input_csv),
        "scrobble_count": int(len(df)),
        "first_scrobble_utc": (
            df["scrobble_time_utc"].min().isoformat()
        ),
        "last_scrobble_utc": (
            df["scrobble_time_utc"].max().isoformat()
        ),
        "negative_gap_count": negative_gap_count,
        "zero_gap_count": zero_gap_count,
        "tested_thresholds_minutes": thresholds,
        "outputs": {
            "gap_quantiles": str(
                output_dir
                / "inter_scrobble_gap_quantiles.csv"
            ),
            "gap_bins": str(
                output_dir
                / "inter_scrobble_gap_bins.csv"
            ),
            "threshold_sensitivity": str(
                output_dir
                / "session_threshold_sensitivity.csv"
            ),
        },
    }

    with (
        output_dir / "session_gap_audit_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Inter-scrobble gap quantiles")
    print("----------------------------")
    print(gap_quantiles.to_string(index=False))

    print()
    print("Session threshold sensitivity")
    print("-----------------------------")
    print(threshold_sensitivity.to_string(index=False))

    print()
    print(
        f"Analysis written to: {output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze inter-scrobble gaps and compare candidate "
            "listening-session thresholds."
        )
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to canonical_scrobbles.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/session_audit"),
        help="Directory for generated audit files.",
    )

    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=DEFAULT_THRESHOLDS_MINUTES,
        help=(
            "Candidate inactivity thresholds in minutes. "
            "Default: 15 30 45 60 90 120"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    analyze_thresholds(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        thresholds=args.thresholds,
    )