from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd


def is_missing_scalar(value: Any) -> bool:
    """
    Return True when a scalar value is missing.

    This helper assumes that the received value is a scalar rather than
    a list, Series, or NumPy array.
    """
    return bool(pd.isna(value))


def python_scalar(value: Any) -> Any:
    """
    Convert pandas and NumPy scalar values into values that GraphML
    can serialize safely.
    """
    if is_missing_scalar(value):
        return ""

    item_method = getattr(value, "item", None)

    if callable(item_method):
        return item_method()

    return value


def required_int(value: Any, field_name: str) -> int:
    """
    Convert a required scalar value to int.

    Fail with a descriptive message when the value is missing or
    malformed. Using this wrapper also prevents Pylance from treating
    pandas Scalar values as potentially incompatible with int().
    """
    if is_missing_scalar(value):
        raise ValueError(
            f"Missing required integer value for: {field_name}"
        )

    try:
        return int(value)

    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Invalid integer value for {field_name}: {value!r}"
        ) from error


def required_float(value: Any, field_name: str) -> float:
    """
    Convert a required scalar value to float and fail clearly when
    the value is missing or malformed.
    """
    if is_missing_scalar(value):
        raise ValueError(
            f"Missing required numeric value for: {field_name}"
        )

    try:
        return float(value)

    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Invalid numeric value for {field_name}: {value!r}"
        ) from error


def optional_float(
    value: Any,
    field_name: str,
    default: float = 0.0,
) -> float:
    """
    Convert an optional scalar value to float.

    Return the supplied default when the value is absent.
    """
    if is_missing_scalar(value):
        return default

    try:
        return float(value)

    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Invalid numeric value for {field_name}: {value!r}"
        ) from error


def normalize_identifier_column(
    series: pd.Series,
    field_name: str,
) -> pd.Series:
    """
    Normalize graph identifiers as non-empty strings.
    """
    normalized = (
        series.astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    if normalized.isna().any():
        missing_count = int(normalized.isna().sum())

        raise ValueError(
            f"Found {missing_count} missing or blank values in "
            f"{field_name}."
        )

    return normalized.astype(str)


def load_nodes(nodes_csv: Path) -> pd.DataFrame:
    if not nodes_csv.exists():
        raise FileNotFoundError(
            f"Nodes file does not exist: {nodes_csv}"
        )

    nodes = pd.read_csv(
        nodes_csv,
        low_memory=False,
    )

    required_columns = {
        "artist_id",
        "artist_name",
        "scrobble_count",
        "session_count",
    }

    missing = sorted(
        required_columns - set(nodes.columns)
    )

    if missing:
        raise ValueError(
            "Nodes file is missing required columns: "
            + ", ".join(missing)
        )

    nodes["artist_id"] = normalize_identifier_column(
        nodes["artist_id"],
        "artist_id",
    )

    if nodes["artist_id"].duplicated().any():
        raise ValueError(
            "Found duplicate artist_id values in nodes file."
        )

    for column in [
        "scrobble_count",
        "session_count",
        *(
            ["user_count"]
            if "user_count" in nodes.columns
            else []
        ),
    ]:
        nodes[column] = pd.to_numeric(
            nodes[column],
            errors="raise",
        )

        if nodes[column].isna().any():
            raise ValueError(
                f"Found missing values in node column: {column}"
            )

    return nodes


def load_edges(edges_csv: Path) -> pd.DataFrame:
    if not edges_csv.exists():
        raise FileNotFoundError(
            f"Edges file does not exist: {edges_csv}"
        )

    edges = pd.read_csv(
        edges_csv,
        low_memory=False,
    )

    required_columns = {
        "artist_a_id",
        "artist_b_id",
        "proximity_count",
        "shared_session_count",
        "shared_session_cosine",
    }

    missing = sorted(
        required_columns - set(edges.columns)
    )

    if missing:
        raise ValueError(
            "Edges file is missing required columns: "
            + ", ".join(missing)
        )

    edges["artist_a_id"] = normalize_identifier_column(
        edges["artist_a_id"],
        "artist_a_id",
    )

    edges["artist_b_id"] = normalize_identifier_column(
        edges["artist_b_id"],
        "artist_b_id",
    )

    for column in [
        "proximity_count",
        "shared_session_count",
        "shared_session_cosine",
    ]:
        edges[column] = pd.to_numeric(
            edges[column],
            errors="raise",
        )

        if edges[column].isna().any():
            raise ValueError(
                f"Found missing values in edge column: {column}"
            )

    if "mean_gap_minutes" in edges.columns:
        edges["mean_gap_minutes"] = pd.to_numeric(
            edges["mean_gap_minutes"],
            errors="coerce",
        ).fillna(0.0)

    return edges


def build_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    weight_column: str,
) -> nx.Graph:
    if weight_column not in edges.columns:
        raise ValueError(
            "Weight column does not exist in edges table: "
            f"{weight_column}"
        )

    graph = nx.Graph()

    for row in nodes.itertuples(index=False):
        graph.add_node(
            str(row.artist_id),
            artist_name=python_scalar(
                row.artist_name
            ),
            scrobble_count=required_int(
                row.scrobble_count,
                "scrobble_count",
            ),
            session_count=required_int(
                row.session_count,
                "session_count",
            ),
            user_count=required_int(
                getattr(row, "user_count", 1),
                "user_count",
            ),
        )

    for row in edges.to_dict(orient="records"):
        artist_a_id = str(
            row["artist_a_id"]
        )

        artist_b_id = str(
            row["artist_b_id"]
        )

        graph.add_edge(
            artist_a_id,
            artist_b_id,
            weight=required_float(
                row[weight_column],
                weight_column,
            ),
            shared_session_count=required_int(
                row["shared_session_count"],
                "shared_session_count",
            ),
            shared_user_count=required_int(
                row.get("shared_user_count", 1),
                "shared_user_count",
            ),
            proximity_count=required_int(
                row["proximity_count"],
                "proximity_count",
            ),
            mean_gap_minutes=optional_float(
                row.get(
                    "mean_gap_minutes",
                    0.0,
                ),
                "mean_gap_minutes",
                default=0.0,
            ),
        )

    return graph


def filter_edges_by_node_set(
    edges: pd.DataFrame,
    node_ids: set[str],
) -> pd.DataFrame:
    return (
        edges.loc[
            edges["artist_a_id"].isin(node_ids)
            & edges["artist_b_id"].isin(node_ids)
        ]
        .copy()
        .reset_index(drop=True)
    )


def save_graphml(
    graph: nx.Graph,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nx.write_graphml(
        graph,
        output_path,
    )


def materialize_graph(
    nodes_csv: Path,
    edges_csv: Path,
    output_dir: Path,
    min_scrobbles: int,
    min_shared_sessions: int,
    weight_column: str,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Reading nodes: {nodes_csv}")
    print(f"Reading edges: {edges_csv}")

    raw_nodes = load_nodes(
        nodes_csv
    )

    raw_edges = load_edges(
        edges_csv
    )

    eligible_nodes = (
        raw_nodes.loc[
            raw_nodes["scrobble_count"]
            >= min_scrobbles
        ]
        .copy()
        .reset_index(drop=True)
    )

    eligible_node_ids = set(
        eligible_nodes[
            "artist_id"
        ].astype(str)
    )

    filtered_edges = (
        raw_edges.loc[
            (
                raw_edges["shared_session_count"]
                >= min_shared_sessions
            )
            & raw_edges["artist_a_id"].isin(
                eligible_node_ids
            )
            & raw_edges["artist_b_id"].isin(
                eligible_node_ids
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    graph = build_graph(
        nodes=eligible_nodes,
        edges=filtered_edges,
        weight_column=weight_column,
    )

    active_node_ids = {
        str(node)
        for node, degree in graph.degree()
        if degree > 0
    }

    active_graph = graph.subgraph(
        active_node_ids
    ).copy()

    if active_graph.number_of_nodes() == 0:
        raise ValueError(
            "No connected nodes remain after filtering. "
            "Review the selected thresholds."
        )

    components = sorted(
        nx.connected_components(
            active_graph
        ),
        key=len,
        reverse=True,
    )

    largest_component_node_ids = {
        str(node)
        for node in components[0]
    }

    largest_component_graph = (
        active_graph.subgraph(
            largest_component_node_ids
        )
        .copy()
    )

    active_nodes = (
        eligible_nodes.loc[
            eligible_nodes["artist_id"].isin(
                active_node_ids
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    active_edges = filter_edges_by_node_set(
        filtered_edges,
        active_node_ids,
    )

    largest_component_nodes = (
        eligible_nodes.loc[
            eligible_nodes["artist_id"].isin(
                largest_component_node_ids
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    largest_component_edges = (
        filter_edges_by_node_set(
            filtered_edges,
            largest_component_node_ids,
        )
    )

    excluded_nodes = raw_nodes.copy()

    excluded_nodes[
        "exclusion_reason"
    ] = "included_in_active_graph"

    excluded_nodes.loc[
        excluded_nodes["scrobble_count"]
        < min_scrobbles,
        "exclusion_reason",
    ] = "below_min_scrobble_count"

    excluded_nodes.loc[
        (
            excluded_nodes["scrobble_count"]
            >= min_scrobbles
        )
        & ~excluded_nodes["artist_id"].isin(
            active_node_ids
        ),
        "exclusion_reason",
    ] = "isolated_after_edge_filter"

    excluded_nodes.loc[
        excluded_nodes["artist_id"].isin(
            active_node_ids
        )
        & ~excluded_nodes["artist_id"].isin(
            largest_component_node_ids
        ),
        "exclusion_reason",
    ] = "outside_largest_component"

    # Save CSV tables.
    eligible_nodes.to_csv(
        output_dir / "nodes_eligible.csv",
        index=False,
        encoding="utf-8",
    )

    filtered_edges.to_csv(
        output_dir / "edges_filtered.csv",
        index=False,
        encoding="utf-8",
    )

    active_nodes.to_csv(
        output_dir / "nodes_active.csv",
        index=False,
        encoding="utf-8",
    )

    active_edges.to_csv(
        output_dir / "edges_active.csv",
        index=False,
        encoding="utf-8",
    )

    largest_component_nodes.to_csv(
        output_dir
        / "nodes_largest_component.csv",
        index=False,
        encoding="utf-8",
    )

    largest_component_edges.to_csv(
        output_dir
        / "edges_largest_component.csv",
        index=False,
        encoding="utf-8",
    )

    excluded_nodes.to_csv(
        output_dir
        / "node_inclusion_audit.csv",
        index=False,
        encoding="utf-8",
    )

    # Save interoperable graph files.
    save_graphml(
        active_graph,
        output_dir
        / "graph_active.graphml",
    )

    save_graphml(
        largest_component_graph,
        output_dir
        / "graph_largest_component.graphml",
    )

    summary = {
        "inputs": {
            "nodes_csv": str(nodes_csv),
            "edges_csv": str(edges_csv),
        },
        "configuration": {
            "min_scrobble_count": int(
                min_scrobbles
            ),
            "min_shared_session_count": int(
                min_shared_sessions
            ),
            "edge_weight_column": (
                weight_column
            ),
        },
        "raw": {
            "node_count": int(
                len(raw_nodes)
            ),
            "edge_count": int(
                len(raw_edges)
            ),
        },
        "eligible_after_min_scrobble_filter": {
            "node_count": int(
                len(eligible_nodes)
            ),
        },
        "after_edge_filter": {
            "edge_count": int(
                len(filtered_edges)
            ),
            "active_node_count": int(
                len(active_nodes)
            ),
            "isolated_eligible_node_count": int(
                len(eligible_nodes)
                - len(active_nodes)
            ),
            "connected_component_count": int(
                len(components)
            ),
            "mean_degree_active_nodes": round(
                float(
                    sum(
                        dict(
                            active_graph.degree()
                        ).values()
                    )
                    / active_graph.number_of_nodes()
                ),
                6,
            ),
            "density_active_nodes": round(
                float(
                    nx.density(
                        active_graph
                    )
                ),
                10,
            ),
        },
        "largest_component": {
            "node_count": int(
                largest_component_graph.number_of_nodes()
            ),
            "edge_count": int(
                largest_component_graph.number_of_edges()
            ),
            "percentage_of_active_nodes": round(
                float(
                    largest_component_graph.number_of_nodes()
                    / active_graph.number_of_nodes()
                    * 100
                ),
                6,
            ),
            "mean_degree": round(
                float(
                    sum(
                        dict(
                            largest_component_graph.degree()
                        ).values()
                    )
                    / largest_component_graph.number_of_nodes()
                ),
                6,
            ),
            "density": round(
                float(
                    nx.density(
                        largest_component_graph
                    )
                ),
                10,
            ),
        },
        "outputs": {
            "nodes_active": str(
                output_dir
                / "nodes_active.csv"
            ),
            "edges_active": str(
                output_dir
                / "edges_active.csv"
            ),
            "nodes_largest_component": str(
                output_dir
                / "nodes_largest_component.csv"
            ),
            "edges_largest_component": str(
                output_dir
                / "edges_largest_component.csv"
            ),
            "graph_active": str(
                output_dir
                / "graph_active.graphml"
            ),
            "graph_largest_component": str(
                output_dir
                / "graph_largest_component.graphml"
            ),
            "node_inclusion_audit": str(
                output_dir
                / "node_inclusion_audit.csv"
            ),
        },
    }

    with (
        output_dir
        / "materialization_summary.json"
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
        f"Processed graph written to: "
        f"{output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter and materialize a genre-free "
            "Last.fm listening graph."
        )
    )

    parser.add_argument(
        "nodes_csv",
        type=Path,
        help="Path to nodes_activity.csv.",
    )

    parser.add_argument(
        "edges_csv",
        type=Path,
        help="Path to edges_raw.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory for the processed graph "
            "files."
        ),
    )

    parser.add_argument(
        "--min-scrobbles",
        type=int,
        default=3,
        help=(
            "Minimum artist scrobble count. "
            "Default: 3."
        ),
    )

    parser.add_argument(
        "--min-shared-sessions",
        type=int,
        default=2,
        help=(
            "Minimum number of sessions "
            "supporting an edge. Default: 2."
        ),
    )

    parser.add_argument(
        "--weight-column",
        type=str,
        default="multi_user_shared_session_cosine",
        help=(
            "Behavioral edge-weight column. "
            "Default: multi_user_shared_session_cosine."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.min_scrobbles < 1:
        raise ValueError(
            "--min-scrobbles must be at least 1."
        )

    if args.min_shared_sessions < 1:
        raise ValueError(
            "--min-shared-sessions must be "
            "at least 1."
        )

    materialize_graph(
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        output_dir=args.output_dir,
        min_scrobbles=args.min_scrobbles,
        min_shared_sessions=(
            args.min_shared_sessions
        ),
        weight_column=args.weight_column,
    )
