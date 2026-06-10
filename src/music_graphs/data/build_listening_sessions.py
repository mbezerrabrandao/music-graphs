from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_scrobbles(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)

    required_columns = {
        "scrobble_id",
        "user_id",
        "scrobble_time_utc",
        "artist_id",
        "artist",
        "track",
        "album",
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

    invalid_timestamp_count = int(
        df["scrobble_time_utc"].isna().sum()
    )

    if invalid_timestamp_count:
        raise ValueError(
            f"Found {invalid_timestamp_count} invalid timestamps."
        )

    return (
        df.sort_values(
            ["user_id", "scrobble_time_utc", "scrobble_id"],
            ascending=[True, True, True],
        )
        .reset_index(drop=True)
    )


def assign_sessions(
    df: pd.DataFrame,
    threshold_minutes: int,
) -> pd.DataFrame:
    work = df.copy()

    grouped = work.groupby("user_id", sort=False)

    work["previous_scrobble_id"] = grouped["scrobble_id"].shift(1)

    work["gap_from_previous_minutes"] = (
        grouped["scrobble_time_utc"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    work["starts_new_session"] = (
        work["gap_from_previous_minutes"].isna()
        | work["gap_from_previous_minutes"].gt(
            threshold_minutes
        )
    )

    work["user_session_number"] = (
        work.groupby("user_id", sort=False)["starts_new_session"]
        .cumsum()
        .astype(int)
    )

    work["session_number"] = (
        work["starts_new_session"]
        .cumsum()
        .astype(int)
    )

    work["session_id"] = (
        work["user_id"].astype(str)
        + ":session_"
        + work["user_session_number"].astype(str).str.zfill(6)
    )

    work["position_in_session"] = (
        work.groupby("session_id")
        .cumcount()
        .add(1)
    )

    return work


def aggregate_sessions(
    df: pd.DataFrame,
    long_session_minutes: int,
) -> pd.DataFrame:
    sessions = (
        df.groupby(["user_id", "session_id"])
        .agg(
            first_scrobble_utc=("scrobble_time_utc", "min"),
            last_scrobble_utc=("scrobble_time_utc", "max"),
            scrobble_count=("scrobble_id", "size"),
            unique_artist_count=("artist_id", "nunique"),
            unique_track_count=("track", "nunique"),
            unique_album_count=("album", "nunique"),
        )
        .reset_index()
    )

    sessions["duration_minutes"] = (
        sessions["last_scrobble_utc"]
        .sub(sessions["first_scrobble_utc"])
        .dt.total_seconds()
        .div(60)
        .round(6)
    )

    sessions["is_singleton"] = (
        sessions["scrobble_count"] == 1
    )

    sessions["has_multiple_artists"] = (
        sessions["unique_artist_count"] >= 2
    )

    sessions["is_long_session"] = (
        sessions["duration_minutes"] > long_session_minutes
    )

    # Maximum possible pairs if every artist in a session were
    # connected to every other artist. This is diagnostic only.
    sessions["full_clique_artist_pair_upper_bound"] = (
        sessions["unique_artist_count"]
        .mul(sessions["unique_artist_count"] - 1)
        .floordiv(2)
    )

    return sessions


def build_sessions(
    input_csv: Path,
    output_root: Path,
    threshold_minutes: int,
    long_session_minutes: int,
) -> None:
    output_dir = (
        output_root
        / f"threshold_{threshold_minutes}m"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading canonical scrobbles: {input_csv}")
    print(
        "Session inactivity threshold: "
        f"{threshold_minutes} minutes"
    )

    scrobbles = load_scrobbles(input_csv)

    scrobbles_with_sessions = assign_sessions(
        scrobbles,
        threshold_minutes=threshold_minutes,
    )

    sessions = aggregate_sessions(
        scrobbles_with_sessions,
        long_session_minutes=long_session_minutes,
    )

    zero_gap_rows = scrobbles_with_sessions.loc[
        scrobbles_with_sessions[
            "gap_from_previous_minutes"
        ]
        .eq(0)
    ].copy()

    long_sessions = sessions.loc[
        sessions["is_long_session"]
    ].copy()

    scrobbles_output = (
        output_dir
        / "scrobbles_with_sessions.csv"
    )

    sessions_output = (
        output_dir
        / "listening_sessions.csv"
    )

    scrobbles_with_sessions.to_csv(
        scrobbles_output,
        index=False,
        encoding="utf-8",
    )

    sessions.to_csv(
        sessions_output,
        index=False,
        encoding="utf-8",
    )

    zero_gap_rows.to_csv(
        output_dir / "zero_gap_scrobbles.csv",
        index=False,
        encoding="utf-8",
    )

    long_sessions.to_csv(
        output_dir / "long_sessions.csv",
        index=False,
        encoding="utf-8",
    )

    summary = {
        "input_file": str(input_csv),
        "session_inactivity_threshold_minutes": int(
            threshold_minutes
        ),
        "long_session_audit_threshold_minutes": int(
            long_session_minutes
        ),
        "scrobble_count": int(
            len(scrobbles_with_sessions)
        ),
        "user_count": int(
            scrobbles_with_sessions["user_id"].nunique()
        ),
        "session_count": int(len(sessions)),
        "singleton_session_count": int(
            sessions["is_singleton"].sum()
        ),
        "singleton_session_percentage": round(
            float(sessions["is_singleton"].mean() * 100),
            6,
        ),
        "sessions_with_multiple_artists": int(
            sessions["has_multiple_artists"].sum()
        ),
        "sessions_with_multiple_artists_percentage": round(
            float(
                sessions["has_multiple_artists"].mean()
                * 100
            ),
            6,
        ),
        "long_session_count": int(
            sessions["is_long_session"].sum()
        ),
        "zero_gap_adjacent_scrobble_count": int(
            len(zero_gap_rows)
        ),
        "mean_scrobbles_per_session": round(
            float(sessions["scrobble_count"].mean()),
            6,
        ),
        "median_scrobbles_per_session": round(
            float(sessions["scrobble_count"].median()),
            6,
        ),
        "mean_unique_artists_per_session": round(
            float(
                sessions["unique_artist_count"].mean()
            ),
            6,
        ),
        "median_unique_artists_per_session": round(
            float(
                sessions["unique_artist_count"].median()
            ),
            6,
        ),
        "median_session_duration_minutes": round(
            float(
                sessions["duration_minutes"].median()
            ),
            6,
        ),
        "p90_session_duration_minutes": round(
            float(
                sessions["duration_minutes"].quantile(0.90)
            ),
            6,
        ),
        "max_session_duration_minutes": round(
            float(
                sessions["duration_minutes"].max()
            ),
            6,
        ),
        "full_clique_artist_pair_upper_bound_total": int(
            sessions[
                "full_clique_artist_pair_upper_bound"
            ].sum()
        ),
        "outputs": {
            "scrobbles_with_sessions": str(
                scrobbles_output
            ),
            "listening_sessions": str(
                sessions_output
            ),
            "zero_gap_scrobbles": str(
                output_dir / "zero_gap_scrobbles.csv"
            ),
            "long_sessions": str(
                output_dir / "long_sessions.csv"
            ),
        },
    }

    with (
        output_dir / "sessionization_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    print(f"Session files written to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign canonical Last.fm scrobbles to listening "
            "sessions using an inactivity threshold."
        )
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to canonical_scrobbles.csv.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/interim/sessions"),
        help="Root directory for generated session tables.",
    )

    parser.add_argument(
        "--threshold-minutes",
        type=int,
        default=60,
        help=(
            "Maximum inactivity gap inside one listening "
            "session. Default: 60."
        ),
    )

    parser.add_argument(
        "--long-session-minutes",
        type=int,
        default=480,
        help=(
            "Audit threshold for unusually long sessions. "
            "Default: 480 minutes."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    build_sessions(
        input_csv=args.input_csv,
        output_root=args.output_root,
        threshold_minutes=args.threshold_minutes,
        long_session_minutes=args.long_session_minutes,
    )
