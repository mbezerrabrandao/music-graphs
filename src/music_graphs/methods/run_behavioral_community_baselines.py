from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import igraph as ig
import leidenalg
import networkx as nx
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)


def load_graph(graphml_path: Path) -> nx.Graph:
    if not graphml_path.exists():
        raise FileNotFoundError(
            f"GraphML file does not exist: {graphml_path}"
        )

    graph = nx.read_graphml(graphml_path)

    if graph.is_directed():
        graph = nx.Graph(graph)

    if graph.number_of_nodes() == 0:
        raise ValueError("The input graph contains no nodes.")

    if graph.number_of_edges() == 0:
        raise ValueError("The input graph contains no edges.")

    for _, _, attributes in graph.edges(data=True):
        try:
            attributes["weight"] = float(
                attributes.get("weight", 1.0)
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as error:
            raise ValueError(
                "Found an invalid edge weight in the GraphML file."
            ) from error

    return graph


def load_label_table(
    coverage_csv: Path,
) -> pd.DataFrame:
    if not coverage_csv.exists():
        raise FileNotFoundError(
            f"Genre coverage table does not exist: {coverage_csv}"
        )

    labels = pd.read_csv(
        coverage_csv,
        dtype={
            "artist_id": "string",
            "artist_name": "string",
            "dominant_genre": "string",
            "dominant_genre_normalized": "string",
            "metadata_group": "string",
        },
        low_memory=False,
    )

    required_columns = {
        "artist_id",
        "artist_name",
        "metadata_group",
        "dominant_genre",
        "dominant_genre_normalized",
    }

    missing_columns = sorted(
        required_columns
        - set(labels.columns)
    )

    if missing_columns:
        raise ValueError(
            "Genre coverage table is missing required columns: "
            + ", ".join(missing_columns)
        )

    labels["artist_id"] = (
        labels["artist_id"]
        .astype("string")
        .str.strip()
    )

    if labels["artist_id"].isna().any():
        raise ValueError(
            "Found missing artist_id values in coverage table."
        )

    if labels["artist_id"].duplicated().any():
        raise ValueError(
            "Found duplicate artist_id values in coverage table."
        )

    return labels


def build_known_label_map(
    graph: nx.Graph,
    labels: pd.DataFrame,
) -> dict[str, str]:
    labels_by_artist = labels.set_index(
        "artist_id"
    )

    known_labels: dict[str, str] = {}

    for node in graph.nodes:
        node_id = str(node)

        if node_id not in labels_by_artist.index:
            raise ValueError(
                f"Graph node not found in genre coverage table: {node_id}"
            )

        genre = labels_by_artist.at[
            node_id,
            "dominant_genre_normalized",
        ]

        if pd.notna(genre):
            genre_text = str(genre).strip()

            if genre_text:
                known_labels[node_id] = genre_text

    return known_labels


def graph_to_igraph(
    graph: nx.Graph,
) -> tuple[ig.Graph, list[str], float]:
    started_at = time.perf_counter()

    node_ids = [
        str(node)
        for node in graph.nodes
    ]

    node_index = {
        node_id: index
        for index, node_id in enumerate(
            node_ids
        )
    }

    igraph_edges = [
        (
            node_index[str(source)],
            node_index[str(target)],
        )
        for source, target in graph.edges
    ]

    weights = [
        float(
            attributes.get(
                "weight",
                1.0,
            )
        )
        for _, _, attributes in graph.edges(
            data=True
        )
    ]

    igraph_graph = ig.Graph(
        n=len(node_ids),
        edges=igraph_edges,
        directed=False,
    )

    igraph_graph.vs["name"] = (
        node_ids
    )

    igraph_graph.es["weight"] = (
        weights
    )

    elapsed_seconds = (
        time.perf_counter()
        - started_at
    )

    return (
        igraph_graph,
        node_ids,
        elapsed_seconds,
    )


def communities_to_partition(
    communities: list[set[str]],
) -> dict[str, int]:
    partition: dict[str, int] = {}

    for community_id, members in enumerate(
        communities
    ):
        for node in members:
            node_id = str(node)

            if node_id in partition:
                raise ValueError(
                    f"Node assigned twice: {node_id}"
                )

            partition[node_id] = (
                community_id
            )

    return partition


def weighted_purity(
    truth: list[str],
    predictions: list[int],
) -> float:
    if len(truth) != len(predictions):
        raise ValueError(
            "Truth and prediction lists have different lengths."
        )

    if not truth:
        raise ValueError(
            "Cannot calculate purity without labeled nodes."
        )

    community_genres: dict[
        int,
        list[str],
    ] = {}

    for genre, community_id in zip(
        truth,
        predictions,
    ):
        community_genres.setdefault(
            community_id,
            [],
        ).append(
            genre
        )

    correct_count = 0

    for genres in community_genres.values():
        most_common_count = (
            Counter(
                genres
            )
            .most_common(1)[0][1]
        )

        correct_count += (
            most_common_count
        )

    return (
        correct_count
        / len(truth)
    )


def summarize_community_sizes(
    communities: list[set[str]],
) -> dict[str, Any]:
    sizes = [
        len(community)
        for community in communities
    ]

    if not sizes:
        raise ValueError(
            "The detected partition contains no communities."
        )

    return {
        "community_count": int(
            len(sizes)
        ),
        "singleton_community_count": int(
            sum(
                size == 1
                for size in sizes
            )
        ),
        "minimum_community_size": int(
            min(sizes)
        ),
        "median_community_size": float(
            statistics.median(
                sizes
            )
        ),
        "maximum_community_size": int(
            max(sizes)
        ),
    }


def evaluate_partition(
    graph: nx.Graph,
    communities: list[set[str]],
    known_labels: dict[str, str],
    resolution: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    partition = communities_to_partition(
        communities
    )

    missing_nodes = sorted(
        set(
            str(node)
            for node in graph.nodes
        )
        - set(partition)
    )

    if missing_nodes:
        raise ValueError(
            "Some graph nodes were not assigned to a community. "
            f"Example: {missing_nodes[0]}"
        )

    labeled_nodes = sorted(
        set(partition)
        & set(known_labels)
    )

    if not labeled_nodes:
        raise ValueError(
            "No labeled graph nodes are available for evaluation."
        )

    truth = [
        known_labels[node]
        for node in labeled_nodes
    ]

    predictions = [
        partition[node]
        for node in labeled_nodes
    ]

    metrics: dict[str, Any] = {
        "modularity": round(
            float(
                nx.community.modularity(
                    graph,
                    communities,
                    weight="weight",
                    resolution=resolution,
                )
            ),
            10,
        ),
        "nmi": round(
            float(
                normalized_mutual_info_score(
                    truth,
                    predictions,
                    average_method="arithmetic",
                )
            ),
            10,
        ),
        "ami": round(
            float(
                adjusted_mutual_info_score(
                    truth,
                    predictions,
                    average_method="arithmetic",
                )
            ),
            10,
        ),
        "ari": round(
            float(
                adjusted_rand_score(
                    truth,
                    predictions,
                )
            ),
            10,
        ),
        "purity_labeled_nodes": round(
            float(
                weighted_purity(
                    truth,
                    predictions,
                )
            ),
            10,
        ),
        "homogeneity": round(
            float(
                homogeneity_score(
                    truth,
                    predictions,
                )
            ),
            10,
        ),
        "completeness": round(
            float(
                completeness_score(
                    truth,
                    predictions,
                )
            ),
            10,
        ),
        "v_measure": round(
            float(
                v_measure_score(
                    truth,
                    predictions,
                )
            ),
            10,
        ),
        "evaluated_labeled_node_count": int(
            len(labeled_nodes)
        ),
        "known_genre_count_in_evaluation": int(
            len(
                set(
                    truth
                )
            )
        ),
    }

    metrics.update(
        summarize_community_sizes(
            communities
        )
    )

    return metrics, partition


def run_louvain(
    graph: nx.Graph,
    resolution: float,
    seed: int,
) -> tuple[
    list[set[str]],
    float,
]:
    louvain_function = getattr(
        nx.community,
        "louvain_communities",
        None,
    )

    if louvain_function is None:
        raise RuntimeError(
            "This NetworkX version does not provide "
            "louvain_communities(). Upgrade NetworkX."
        )

    started_at = (
        time.perf_counter()
    )

    communities = (
        louvain_function(
            graph,
            weight="weight",
            resolution=resolution,
            seed=seed,
        )
    )

    elapsed_seconds = (
        time.perf_counter()
        - started_at
    )

    normalized_communities = [
        {
            str(node)
            for node in community
        }
        for community in communities
    ]

    return (
        normalized_communities,
        elapsed_seconds,
    )


def run_leiden(
    igraph_graph: ig.Graph,
    node_ids: list[str],
    resolution: float,
    seed: int,
) -> tuple[
    list[set[str]],
    float,
]:
    started_at = (
        time.perf_counter()
    )

    partition = (
        leidenalg.find_partition(
            igraph_graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=igraph_graph.es[
                "weight"
            ],
            resolution_parameter=resolution,
            seed=seed,
        )
    )

    elapsed_seconds = (
        time.perf_counter()
        - started_at
    )

    communities = [
        {
            node_ids[
                vertex_index
            ]
            for vertex_index in community
        }
        for community in partition
    ]

    return (
        communities,
        elapsed_seconds,
    )


def safe_resolution_text(
    resolution: float,
) -> str:
    return (
        str(
            resolution
        )
        .replace(
            ".",
            "_",
        )
    )


def run_id(
    algorithm: str,
    resolution: float,
    seed: int,
) -> str:
    return (
        f"{algorithm.lower()}"
        f"_resolution_{safe_resolution_text(resolution)}"
        f"_seed_{seed}"
    )


def save_partition(
    partition: dict[str, int],
    labels: pd.DataFrame,
    output_csv: Path,
) -> None:
    labels_by_artist = (
        labels.set_index(
            "artist_id"
        )
    )

    records = []

    for artist_id, community_id in sorted(
        partition.items()
    ):
        row = labels_by_artist.loc[
            artist_id
        ]

        dominant_genre = row[
            "dominant_genre"
        ]

        dominant_genre_normalized = row[
            "dominant_genre_normalized"
        ]

        records.append(
            {
                "artist_id": artist_id,
                "artist_name": row[
                    "artist_name"
                ],
                "community_id": int(
                    community_id
                ),
                "metadata_group": row[
                    "metadata_group"
                ],
                "dominant_genre": (
                    dominant_genre
                    if pd.notna(
                        dominant_genre
                    )
                    else ""
                ),
                "dominant_genre_normalized": (
                    dominant_genre_normalized
                    if pd.notna(
                        dominant_genre_normalized
                    )
                    else ""
                ),
            }
        )

    pd.DataFrame.from_records(
        records
    ).to_csv(
        output_csv,
        index=False,
        encoding="utf-8",
    )


def build_label_distribution(
    known_labels: dict[str, str],
) -> pd.DataFrame:
    counts = Counter(
        known_labels.values()
    )

    return (
        pd.DataFrame(
            [
                {
                    "dominant_genre_normalized": genre,
                    "artist_count": int(
                        count
                    ),
                }
                for genre, count in counts.items()
            ]
        )
        .sort_values(
            [
                "artist_count",
                "dominant_genre_normalized",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def execute_experiments(
    graphml_path: Path,
    labels_csv: Path,
    output_dir: Path,
    resolutions: list[float],
    seeds: list[int],
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    partitions_dir = (
        output_dir
        / "partitions"
    )

    partitions_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Reading behavioral graph: "
        f"{graphml_path}"
    )

    graph = load_graph(
        graphml_path
    )

    print(
        f"Reading external genre labels: "
        f"{labels_csv}"
    )

    labels = load_label_table(
        labels_csv
    )

    known_labels = (
        build_known_label_map(
            graph=graph,
            labels=labels,
        )
    )

    label_distribution = (
        build_label_distribution(
            known_labels
        )
    )

    label_distribution.to_csv(
        output_dir
        / "known_genre_distribution.csv",
        index=False,
        encoding="utf-8",
    )

    print(
        "Converting NetworkX graph to igraph "
        "for Leiden."
    )

    (
        igraph_graph,
        node_ids,
        conversion_seconds,
    ) = graph_to_igraph(
        graph
    )

    records: list[
        dict[str, Any]
    ] = []

    total_run_count = (
        len(
            resolutions
        )
        * len(
            seeds
        )
        * 2
    )

    current_run = 0

    for resolution in resolutions:
        for seed in seeds:
            for algorithm in [
                "Louvain",
                "Leiden",
            ]:
                current_run += 1

                identifier = run_id(
                    algorithm=algorithm,
                    resolution=resolution,
                    seed=seed,
                )

                print(
                    f"[{current_run}/{total_run_count}] "
                    f"{identifier}"
                )

                if algorithm == "Louvain":
                    (
                        communities,
                        elapsed_seconds,
                    ) = run_louvain(
                        graph=graph,
                        resolution=resolution,
                        seed=seed,
                    )

                else:
                    (
                        communities,
                        elapsed_seconds,
                    ) = run_leiden(
                        igraph_graph=igraph_graph,
                        node_ids=node_ids,
                        resolution=resolution,
                        seed=seed,
                    )

                metrics, partition = (
                    evaluate_partition(
                        graph=graph,
                        communities=communities,
                        known_labels=known_labels,
                        resolution=resolution,
                    )
                )

                save_partition(
                    partition=partition,
                    labels=labels,
                    output_csv=(
                        partitions_dir
                        / f"{identifier}.csv"
                    ),
                )

                records.append(
                    {
                        "run_id": identifier,
                        "algorithm": algorithm,
                        "resolution": float(
                            resolution
                        ),
                        "seed": int(
                            seed
                        ),
                        "algorithm_elapsed_seconds": round(
                            float(
                                elapsed_seconds
                            ),
                            10,
                        ),
                        **metrics,
                    }
                )

    results = (
        pd.DataFrame.from_records(
            records
        )
        .sort_values(
            [
                "algorithm",
                "resolution",
                "seed",
            ]
        )
        .reset_index(drop=True)
    )

    results_output = (
        output_dir
        / "behavioral_baseline_runs.csv"
    )

    results.to_csv(
        results_output,
        index=False,
        encoding="utf-8",
    )

    best_nmi_rows = (
        results.sort_values(
            [
                "nmi",
                "ami",
                "modularity",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .groupby(
            "algorithm",
            as_index=False,
        )
        .first()
    )

    best_nmi_output = (
        output_dir
        / "best_behavioral_baselines_by_nmi.csv"
    )

    best_nmi_rows.to_csv(
        best_nmi_output,
        index=False,
        encoding="utf-8",
    )

    summary = {
        "inputs": {
            "graphml_path": str(
                graphml_path
            ),
            "labels_csv": str(
                labels_csv
            ),
        },
        "graph": {
            "node_count": int(
                graph.number_of_nodes()
            ),
            "edge_count": int(
                graph.number_of_edges()
            ),
            "density": round(
                float(
                    nx.density(
                        graph
                    )
                ),
                10,
            ),
        },
        "external_label_coverage": {
            "labeled_node_count": int(
                len(
                    known_labels
                )
            ),
            "labeled_node_percentage": round(
                float(
                    len(
                        known_labels
                    )
                    / graph.number_of_nodes()
                    * 100
                ),
                6,
            ),
            "known_dominant_genre_count": int(
                label_distribution[
                    "dominant_genre_normalized"
                ].nunique()
            ),
            "single_artist_genre_count": int(
                (
                    label_distribution[
                        "artist_count"
                    ]
                    == 1
                )
                .sum()
            ),
        },
        "configuration": {
            "resolutions": resolutions,
            "seeds": seeds,
            "run_count": int(
                total_run_count
            ),
            "edge_weight": "weight",
        },
        "timing": {
            "networkx_to_igraph_conversion_seconds": round(
                float(
                    conversion_seconds
                ),
                10,
            ),
            "note": (
                "Algorithm timings are informative but not "
                "strictly comparable because Louvain runs in "
                "NetworkX while Leiden runs in igraph/leidenalg. "
                "The NetworkX-to-igraph conversion time is "
                "reported separately."
            ),
        },
        "outputs": {
            "behavioral_baseline_runs": str(
                results_output
            ),
            "best_behavioral_baselines_by_nmi": str(
                best_nmi_output
            ),
            "known_genre_distribution": str(
                output_dir
                / "known_genre_distribution.csv"
            ),
            "partitions_directory": str(
                partitions_dir
            ),
        },
    }

    with (
        output_dir
        / "behavioral_baselines_summary.json"
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
        "Best behavioral runs by NMI"
    )

    print(
        "---------------------------"
    )

    print(
        best_nmi_rows.to_string(
            index=False
        )
    )

    print()

    print(
        f"Behavioral baseline results "
        f"written to: {output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Louvain and Leiden baselines on a genre-free "
            "behavioral artist graph and evaluate the resulting "
            "communities against external MusicBrainz labels."
        )
    )

    parser.add_argument(
        "graphml_path",
        type=Path,
        help=(
            "Path to graph_largest_component.graphml."
        ),
    )

    parser.add_argument(
        "labels_csv",
        type=Path,
        help=(
            "Path to largest_component_genre_coverage.csv."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/behavioral_baselines"
        ),
        help=(
            "Directory for metrics and partitions."
        ),
    )

    parser.add_argument(
        "--resolutions",
        type=float,
        nargs="+",
        default=[
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
        ],
        help=(
            "Community resolution values. "
            "Default: 0.5 0.75 1.0 1.25 1.5"
        ),
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[
            42,
            43,
            44,
            45,
            46,
        ],
        help=(
            "Random seeds. Default: 42 43 44 45 46"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if any(
        resolution <= 0
        for resolution in args.resolutions
    ):
        raise ValueError(
            "All resolution values must be positive."
        )

    execute_experiments(
        graphml_path=args.graphml_path,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        resolutions=args.resolutions,
        seeds=args.seeds,
    )