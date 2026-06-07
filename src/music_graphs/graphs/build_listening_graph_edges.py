from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd


DEFAULT_MIN_SHARED_SESSION_THRESHOLDS = [1, 2, 3, 5, 10]


def most_common_non_null(series: pd.Series) -> str | None:
    clean = series.dropna()

    if clean.empty:
        return None

    return str(clean.value_counts().index[0])


def load_scrobbles(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_csv}"
        )

    df = pd.read_csv(input_csv, low_memory=False)

    required_columns = {
        "scrobble_id",
        "scrobble_time_utc",
        "session_id",
        "position_in_session",
        "artist_id",
        "artist",
    }

    missing_columns = sorted(
        required_columns - set(df.columns)
    )

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

    if df["artist_id"].isna().any():
        raise ValueError("Found rows without artist_id.")

    return (
        df.sort_values(
            [
                "session_id",
                "position_in_session",
                "scrobble_time_utc",
                "scrobble_id",
            ],
            ascending=[True, True, True, True],
        )
        .reset_index(drop=True)
    )


def build_artist_activity(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return (
        df.groupby("artist_id")
        .agg(
            artist_name=(
                "artist",
                most_common_non_null,
            ),
            scrobble_count=("scrobble_id", "size"),
            session_count=("session_id", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["scrobble_count", "artist_name"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def canonical_pair(
    artist_a: str,
    artist_b: str,
) -> tuple[str, str] | None:
    """
    Return an ordered pair for an undirected graph.

    Self-loops are excluded because they do not represent
    relations between different artists.
    """
    if artist_a == artist_b:
        return None

    if artist_a < artist_b:
        return artist_a, artist_b

    return artist_b, artist_a


def iter_candidate_pairs(
    artist_ids: list[str],
    timestamps: list[pd.Timestamp],
    mode: str,
    window_size: int,
    window_minutes: float,
):
    session_length = len(artist_ids)

    for index_a in range(session_length):
        if mode == "sequential":
            stop = min(
                session_length,
                index_a + window_size + 1,
            )

            candidate_indexes = range(
                index_a + 1,
                stop,
            )

        else:
            candidate_indexes = range(
                index_a + 1,
                session_length,
            )

        for index_b in candidate_indexes:
            gap_minutes = (
                timestamps[index_b]
                - timestamps[index_a]
            ).total_seconds() / 60

            if (
                mode == "temporal"
                and gap_minutes > window_minutes
            ):
                break

            yield (
                artist_ids[index_a],
                artist_ids[index_b],
                float(gap_minutes),
            )


def empty_stats() -> dict[str, Any]:
    return {
        "proximity_count": 0,
        "shared_session_count": 0,
        "gap_sum_minutes": 0.0,
        "min_gap_minutes": math.inf,
        "max_gap_minutes": 0.0,
        "same_timestamp_pair_count": 0,
    }


def build_edge_table(
    df: pd.DataFrame,
    mode: str,
    window_size: int,
    window_minutes: float,
    progress_every: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    global_edges: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    candidate_pair_count = 0
    removed_self_loop_candidate_count = 0

    grouped = df.groupby(
        "session_id",
        sort=False,
    )

    session_count = grouped.ngroups

    for session_index, (_, group) in enumerate(
        grouped,
        start=1,
    ):
        artist_ids = (
            group["artist_id"]
            .astype(str)
            .tolist()
        )

        timestamps = (
            group["scrobble_time_utc"]
            .tolist()
        )

        local_edges: dict[
            tuple[str, str],
            dict[str, Any],
        ] = {}

        for artist_a, artist_b, gap_minutes in (
            iter_candidate_pairs(
                artist_ids=artist_ids,
                timestamps=timestamps,
                mode=mode,
                window_size=window_size,
                window_minutes=window_minutes,
            )
        ):
            candidate_pair_count += 1

            pair = canonical_pair(
                artist_a,
                artist_b,
            )

            if pair is None:
                removed_self_loop_candidate_count += 1
                continue

            stats = local_edges.setdefault(
                pair,
                empty_stats(),
            )

            stats["proximity_count"] += 1
            stats["gap_sum_minutes"] += gap_minutes

            stats["min_gap_minutes"] = min(
                stats["min_gap_minutes"],
                gap_minutes,
            )

            stats["max_gap_minutes"] = max(
                stats["max_gap_minutes"],
                gap_minutes,
            )

            if gap_minutes == 0:
                stats[
                    "same_timestamp_pair_count"
                ] += 1

        # A relation is counted at most once per session
        # when computing shared_session_count.
        for pair, local_stats in local_edges.items():
            global_stats = global_edges.setdefault(
                pair,
                empty_stats(),
            )

            global_stats["proximity_count"] += (
                local_stats["proximity_count"]
            )

            global_stats["shared_session_count"] += 1

            global_stats["gap_sum_minutes"] += (
                local_stats["gap_sum_minutes"]
            )

            global_stats["min_gap_minutes"] = min(
                global_stats["min_gap_minutes"],
                local_stats["min_gap_minutes"],
            )

            global_stats["max_gap_minutes"] = max(
                global_stats["max_gap_minutes"],
                local_stats["max_gap_minutes"],
            )

            global_stats[
                "same_timestamp_pair_count"
            ] += local_stats[
                "same_timestamp_pair_count"
            ]

        if (
            progress_every > 0
            and session_index % progress_every == 0
        ):
            print(
                f"Processed {session_index:,}/"
                f"{session_count:,} sessions; "
                f"current unique edges: "
                f"{len(global_edges):,}"
            )

    records = []

    for (
        artist_a_id,
        artist_b_id,
    ), stats in global_edges.items():
        proximity_count = int(
            stats["proximity_count"]
        )

        records.append(
            {
                "artist_a_id": artist_a_id,
                "artist_b_id": artist_b_id,
                "proximity_count": proximity_count,
                "shared_session_count": int(
                    stats["shared_session_count"]
                ),
                "mean_gap_minutes": round(
                    float(
                        stats["gap_sum_minutes"]
                        / proximity_count
                    ),
                    6,
                ),
                "min_gap_minutes": round(
                    float(stats["min_gap_minutes"]),
                    6,
                ),
                "max_gap_minutes": round(
                    float(stats["max_gap_minutes"]),
                    6,
                ),
                "same_timestamp_pair_count": int(
                    stats[
                        "same_timestamp_pair_count"
                    ]
                ),
            }
        )

    edges = pd.DataFrame.from_records(records)

    counters = {
        "candidate_pair_count_before_self_loop_removal": int(
            candidate_pair_count
        ),
        "removed_self_loop_candidate_count": int(
            removed_self_loop_candidate_count
        ),
        "raw_unique_edge_count": int(len(edges)),
    }

    return edges, counters


def add_normalized_weights(
    edges: pd.DataFrame,
    artist_activity: pd.DataFrame,
) -> pd.DataFrame:
    if edges.empty:
        raise ValueError(
            "No relations were generated. "
            "Review the selected window."
        )

    activity = (
        artist_activity
        .set_index("artist_id")
        .to_dict("index")
    )

    def artist_stat(
        artist_id: str,
        field: str,
    ) -> int:
        return int(activity[artist_id][field])

    edges = edges.copy()

    edges["artist_a_name"] = (
        edges["artist_a_id"].map(
            lambda artist_id: activity[
                artist_id
            ]["artist_name"]
        )
    )

    edges["artist_b_name"] = (
        edges["artist_b_id"].map(
            lambda artist_id: activity[
                artist_id
            ]["artist_name"]
        )
    )

    edges["artist_a_scrobble_count"] = (
        edges["artist_a_id"].map(
            lambda artist_id: artist_stat(
                artist_id,
                "scrobble_count",
            )
        )
    )

    edges["artist_b_scrobble_count"] = (
        edges["artist_b_id"].map(
            lambda artist_id: artist_stat(
                artist_id,
                "scrobble_count",
            )
        )
    )

    edges["artist_a_session_count"] = (
        edges["artist_a_id"].map(
            lambda artist_id: artist_stat(
                artist_id,
                "session_count",
            )
        )
    )

    edges["artist_b_session_count"] = (
        edges["artist_b_id"].map(
            lambda artist_id: artist_stat(
                artist_id,
                "session_count",
            )
        )
    )

    edges["proximity_cosine"] = (
        edges["proximity_count"]
        / (
            edges["artist_a_scrobble_count"]
            .mul(
                edges["artist_b_scrobble_count"]
            )
            .pow(0.5)
        )
    )

    edges["shared_session_cosine"] = (
        edges["shared_session_count"]
        / (
            edges["artist_a_session_count"]
            .mul(
                edges["artist_b_session_count"]
            )
            .pow(0.5)
        )
    )

    edges["shared_session_jaccard"] = (
        edges["shared_session_count"]
        / (
            edges["artist_a_session_count"]
            + edges["artist_b_session_count"]
            - edges["shared_session_count"]
        )
    )

    for column in [
        "proximity_cosine",
        "shared_session_cosine",
        "shared_session_jaccard",
    ]:
        edges[column] = edges[column].round(10)

    preferred_column_order = [
        "artist_a_id",
        "artist_b_id",
        "artist_a_name",
        "artist_b_name",
        "proximity_count",
        "shared_session_count",
        "proximity_cosine",
        "shared_session_cosine",
        "shared_session_jaccard",
        "mean_gap_minutes",
        "min_gap_minutes",
        "max_gap_minutes",
        "same_timestamp_pair_count",
        "artist_a_scrobble_count",
        "artist_b_scrobble_count",
        "artist_a_session_count",
        "artist_b_session_count",
    ]

    return (
        edges[preferred_column_order]
        .sort_values(
            [
                "shared_session_cosine",
                "shared_session_count",
                "proximity_count",
            ],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )


def summarize_filtered_graph(
    artist_activity: pd.DataFrame,
    edges: pd.DataFrame,
    min_shared_sessions: int,
) -> dict[str, Any]:
    filtered_edges = edges.loc[
        edges["shared_session_count"]
        >= min_shared_sessions
    ].copy()

    graph = nx.Graph()

    graph.add_nodes_from(
        artist_activity["artist_id"].tolist()
    )

    graph.add_weighted_edges_from(
        filtered_edges[
            [
                "artist_a_id",
                "artist_b_id",
                "shared_session_cosine",
            ]
        ].itertuples(
            index=False,
            name=None,
        ),
        weight="shared_session_cosine",
    )

    active_nodes = [
        node
        for node, degree in graph.degree()
        if degree > 0
    ]

    active_graph = graph.subgraph(
        active_nodes
    ).copy()

    if active_graph.number_of_nodes() > 0:
        active_components = list(
            nx.connected_components(
                active_graph
            )
        )

        largest_component_size = max(
            len(component)
            for component in active_components
        )

        density_active = nx.density(
            active_graph
        )

        mean_degree_active = (
            sum(
                dict(
                    active_graph.degree()
                ).values()
            )
            / active_graph.number_of_nodes()
        )

    else:
        active_components = []
        largest_component_size = 0
        density_active = 0.0
        mean_degree_active = 0.0

    return {
        "min_shared_sessions": int(
            min_shared_sessions
        ),
        "node_count_total": int(
            graph.number_of_nodes()
        ),
        "node_count_with_edges": int(
            len(active_nodes)
        ),
        "isolated_node_count": int(
            graph.number_of_nodes()
            - len(active_nodes)
        ),
        "edge_count": int(
            graph.number_of_edges()
        ),
        "density_all_nodes": round(
            float(nx.density(graph)),
            10,
        ),
        "density_active_nodes": round(
            float(density_active),
            10,
        ),
        "connected_component_count_including_isolates": int(
            nx.number_connected_components(
                graph
            )
        ),
        "connected_component_count_active_nodes": int(
            len(active_components)
        ),
        "largest_component_node_count": int(
            largest_component_size
        ),
        "largest_component_percentage_of_active_nodes": round(
            (
                largest_component_size
                / len(active_nodes)
                * 100
                if active_nodes
                else 0.0
            ),
            6,
        ),
        "mean_degree_active_nodes": round(
            float(mean_degree_active),
            6,
        ),
        "median_shared_session_cosine": round(
            float(
                filtered_edges[
                    "shared_session_cosine"
                ].median()
            )
            if not filtered_edges.empty
            else 0.0,
            10,
        ),
    }


def variant_name(
    mode: str,
    window_size: int,
    window_minutes: float,
) -> str:
    if mode == "sequential":
        return f"sequential_k{window_size}"

    formatted_minutes = (
        str(int(window_minutes))
        if float(window_minutes).is_integer()
        else str(window_minutes).replace(
            ".",
            "_",
        )
    )

    return (
        f"temporal_{formatted_minutes}m"
    )


def build_listening_graph_edges(
    input_csv: Path,
    output_root: Path,
    mode: str,
    window_size: int,
    window_minutes: float,
    min_shared_session_thresholds: list[int],
    progress_every: int,
) -> None:
    variant = variant_name(
        mode=mode,
        window_size=window_size,
        window_minutes=window_minutes,
    )

    output_dir = output_root / variant

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Reading sessionized scrobbles: "
        f"{input_csv}"
    )

    print(
        f"Building listening relation "
        f"variant: {variant}"
    )

    scrobbles = load_scrobbles(
        input_csv
    )

    artist_activity = build_artist_activity(
        scrobbles
    )

    edges, counters = build_edge_table(
        df=scrobbles,
        mode=mode,
        window_size=window_size,
        window_minutes=window_minutes,
        progress_every=progress_every,
    )

    edges = add_normalized_weights(
        edges=edges,
        artist_activity=artist_activity,
    )

    edges_output = (
        output_dir / "edges_raw.csv"
    )

    nodes_output = (
        output_dir / "nodes_activity.csv"
    )

    edges.to_csv(
        edges_output,
        index=False,
        encoding="utf-8",
    )

    artist_activity.to_csv(
        nodes_output,
        index=False,
        encoding="utf-8",
    )

    sensitivity = pd.DataFrame(
        [
            summarize_filtered_graph(
                artist_activity=artist_activity,
                edges=edges,
                min_shared_sessions=threshold,
            )
            for threshold in (
                min_shared_session_thresholds
            )
        ]
    )

    sensitivity_output = (
        output_dir
        / "min_shared_sessions_sensitivity.csv"
    )

    sensitivity.to_csv(
        sensitivity_output,
        index=False,
        encoding="utf-8",
    )

    summary = {
        "input_file": str(input_csv),
        "variant": variant,
        "mode": mode,
        "window_size_scrobbles": (
            int(window_size)
            if mode == "sequential"
            else None
        ),
        "window_minutes": (
            float(window_minutes)
            if mode == "temporal"
            else None
        ),
        "scrobble_count": int(
            len(scrobbles)
        ),
        "session_count": int(
            scrobbles[
                "session_id"
            ].nunique()
        ),
        "artist_count": int(
            len(artist_activity)
        ),
        **counters,
        "tested_min_shared_session_thresholds": (
            min_shared_session_thresholds
        ),
        "default_recommended_edge_weight_for_audit": (
            "shared_session_cosine"
        ),
        "outputs": {
            "edges_raw": str(
                edges_output
            ),
            "nodes_activity": str(
                nodes_output
            ),
            "min_shared_sessions_sensitivity": str(
                sensitivity_output
            ),
        },
    }

    with (
        output_dir
        / "graph_edges_summary.json"
    ).open(
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
        "Minimum shared-session sensitivity"
    )
    print(
        "----------------------------------"
    )

    print(
        sensitivity.to_string(
            index=False
        )
    )

    print()
    print(
        f"Graph edge files written to: "
        f"{output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build genre-free artist relation tables "
            "from sessionized Last.fm scrobbles."
        )
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help=(
            "Path to "
            "scrobbles_with_sessions.csv."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/interim/listening_graphs"
        ),
        help=(
            "Root directory for generated "
            "graph variants."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "sequential",
            "temporal",
        ],
        required=True,
        help=(
            "How candidate artist pairs "
            "are generated."
        ),
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help=(
            "Number of subsequent scrobbles "
            "considered in sequential mode. "
            "Default: 5."
        ),
    )

    parser.add_argument(
        "--window-minutes",
        type=float,
        default=30.0,
        help=(
            "Maximum forward time gap considered "
            "in temporal mode. Default: 30."
        ),
    )

    parser.add_argument(
        "--min-shared-session-thresholds",
        type=int,
        nargs="+",
        default=(
            DEFAULT_MIN_SHARED_SESSION_THRESHOLDS
        ),
        help=(
            "Thresholds used only for graph-audit "
            "sensitivity. Default: 1 2 3 5 10."
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help=(
            "Print progress every N sessions. "
            "Use 0 to disable."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.window_size < 1:
        raise ValueError(
            "--window-size must be at least 1."
        )

    if args.window_minutes <= 0:
        raise ValueError(
            "--window-minutes must be positive."
        )

    build_listening_graph_edges(
        input_csv=args.input_csv,
        output_root=args.output_root,
        mode=args.mode,
        window_size=args.window_size,
        window_minutes=args.window_minutes,
        min_shared_session_thresholds=(
            args.min_shared_session_thresholds
        ),
        progress_every=args.progress_every,
    )