from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import normalize


def required_text(
    value: Any,
    field_name: str,
) -> str:
    """
    Convert a required scalar value to non-empty text.
    """
    if value is None or pd.isna(value):
        raise ValueError(
            f"Missing required text value for: {field_name}"
        )

    text = str(value).strip()

    if not text:
        raise ValueError(
            f"Blank required text value for: {field_name}"
        )

    return text


def required_float(
    value: Any,
    field_name: str,
) -> float:
    """
    Convert a required scalar value to float.
    """
    if value is None or pd.isna(value):
        raise ValueError(
            f"Missing required numeric value for: {field_name}"
        )

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise ValueError(
            f"Invalid numeric value for {field_name}: {value!r}"
        ) from error


def safe_float_text(
    value: float,
) -> str:
    """
    Convert a float to a filename-safe representation.
    """
    return (
        str(value)
        .replace(".", "_")
    )


def load_graph(
    graphml_path: Path,
) -> nx.Graph:
    """
    Load an undirected weighted GraphML graph.
    """
    if not graphml_path.exists():
        raise FileNotFoundError(
            f"GraphML file does not exist: {graphml_path}"
        )

    graph = nx.read_graphml(
        graphml_path
    )

    if graph.is_directed():
        graph = nx.Graph(
            graph
        )

    if graph.number_of_nodes() == 0:
        raise ValueError(
            "The graph contains no nodes."
        )

    if graph.number_of_edges() == 0:
        raise ValueError(
            "The graph contains no edges."
        )

    for _, _, attributes in graph.edges(
        data=True
    ):
        weight = required_float(
            attributes.get(
                "weight",
                1.0,
            ),
            "weight",
        )

        if weight <= 0:
            raise ValueError(
                "Node2Vec requires strictly positive edge weights."
            )

        attributes["weight"] = weight

    isolated_nodes = list(
        nx.isolates(
            graph
        )
    )

    if isolated_nodes:
        raise ValueError(
            "The Node2Vec input graph contains isolated nodes. "
            f"Example: {isolated_nodes[0]}"
        )

    return graph


def load_external_labels(
    coverage_csv: Path,
    graph: nx.Graph,
) -> pd.DataFrame:
    """
    Load MusicBrainz labels for graph nodes.

    Genres are used only after embedding training for external
    evaluation.
    """
    if not coverage_csv.exists():
        raise FileNotFoundError(
            f"Coverage table does not exist: {coverage_csv}"
        )

    labels = pd.read_csv(
        coverage_csv,
        dtype={
            "artist_id": "string",
            "artist_name": "string",
            "metadata_group": "string",
            "dominant_genre": "string",
            "dominant_genre_normalized": "string",
        },
        low_memory=False,
    )

    required_columns = {
        "artist_id",
        "artist_name",
        "metadata_group",
        "dominant_genre_normalized",
    }

    missing_columns = sorted(
        required_columns
        - set(labels.columns)
    )

    if missing_columns:
        raise ValueError(
            "Coverage table is missing required columns: "
            + ", ".join(missing_columns)
        )

    labels["artist_id"] = (
        labels["artist_id"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    if labels["artist_id"].isna().any():
        raise ValueError(
            "Coverage table contains missing artist IDs."
        )

    if labels["artist_id"].duplicated().any():
        raise ValueError(
            "Coverage table contains duplicated artist IDs."
        )

    graph_node_ids = {
        str(node)
        for node in graph.nodes
    }

    label_node_ids = set(
        labels["artist_id"]
        .astype(str)
    )

    missing_from_labels = sorted(
        graph_node_ids
        - label_node_ids
    )

    if missing_from_labels:
        raise ValueError(
            "Some graph nodes are absent from the coverage table. "
            f"Example: {missing_from_labels[0]}"
        )

    return labels


def build_known_label_map(
    labels: pd.DataFrame,
) -> dict[str, str]:
    """
    Return external labels for artists with a known dominant genre.
    """
    known: dict[
        str,
        str,
    ] = {}

    for row in labels.to_dict(
        orient="records"
    ):
        artist_id = required_text(
            row["artist_id"],
            "artist_id",
        )

        genre = row[
            "dominant_genre_normalized"
        ]

        if genre is None or pd.isna(
            genre
        ):
            continue

        genre_text = str(
            genre
        ).strip()

        if genre_text:
            known[
                artist_id
            ] = genre_text

    return known


def weighted_choice(
    rng: random.Random,
    values: list[str],
    weights: list[float],
) -> str:
    """
    Select one value proportionally to positive weights.
    """
    if len(values) != len(weights):
        raise ValueError(
            "Candidate values and weights have different lengths."
        )

    if not values:
        raise ValueError(
            "Cannot sample from an empty candidate list."
        )

    total_weight = sum(
        weights
    )

    if total_weight <= 0:
        return rng.choice(
            values
        )

    threshold = (
        rng.random()
        * total_weight
    )

    cumulative = 0.0

    for value, weight in zip(
        values,
        weights,
    ):
        cumulative += weight

        if cumulative >= threshold:
            return value

    return values[-1]


def choose_next_node(
    graph: nx.Graph,
    rng: random.Random,
    previous_node: str | None,
    current_node: str,
    p: float,
    q: float,
) -> str:
    """
    Select the next node using weighted Node2Vec transition biases.
    """
    neighbors = [
        str(node)
        for node in graph.neighbors(
            current_node
        )
    ]

    if not neighbors:
        return current_node

    transition_weights: list[
        float
    ] = []

    for candidate in neighbors:
        edge_weight = required_float(
            graph[
                current_node
            ][
                candidate
            ].get(
                "weight",
                1.0,
            ),
            "weight",
        )

        if previous_node is None:
            bias = 1.0

        elif candidate == previous_node:
            bias = (
                1.0
                / p
            )

        elif graph.has_edge(
            previous_node,
            candidate,
        ):
            bias = 1.0

        else:
            bias = (
                1.0
                / q
            )

        transition_weights.append(
            edge_weight
            * bias
        )

    return weighted_choice(
        rng=rng,
        values=neighbors,
        weights=transition_weights,
    )


def generate_random_walk(
    graph: nx.Graph,
    start_node: str,
    walk_length: int,
    rng: random.Random,
    p: float,
    q: float,
) -> list[str]:
    """
    Generate one biased random walk.
    """
    walk = [
        start_node
    ]

    while len(walk) < walk_length:
        current_node = (
            walk[-1]
        )

        previous_node = (
            walk[-2]
            if len(walk) >= 2
            else None
        )

        next_node = choose_next_node(
            graph=graph,
            rng=rng,
            previous_node=previous_node,
            current_node=current_node,
            p=p,
            q=q,
        )

        walk.append(
            next_node
        )

    return walk


def generate_walk_corpus(
    graph: nx.Graph,
    walk_length: int,
    walks_per_node: int,
    seed: int,
    p: float,
    q: float,
) -> list[list[str]]:
    """
    Generate the complete restartable walk corpus for Word2Vec.
    """
    rng = random.Random(
        seed
    )

    node_ids = sorted(
        str(node)
        for node in graph.nodes
    )

    walks: list[
        list[str]
    ] = []

    for _ in range(
        walks_per_node
    ):
        shuffled_nodes = (
            node_ids.copy()
        )

        rng.shuffle(
            shuffled_nodes
        )

        for node_id in shuffled_nodes:
            walks.append(
                generate_random_walk(
                    graph=graph,
                    start_node=node_id,
                    walk_length=walk_length,
                    rng=rng,
                    p=p,
                    q=q,
                )
            )

    return walks


def train_embeddings(
    walks: list[list[str]],
    node_ids: list[str],
    dimensions: int,
    context_window: int,
    epochs: int,
    seed: int,
) -> np.ndarray:
    """
    Train deterministic skip-gram embeddings from random walks.
    """
    model = Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=context_window,
        min_count=1,
        workers=1,
        sg=1,
        negative=5,
        epochs=epochs,
        seed=seed,
    )

    embeddings = np.vstack(
        [
            model.wv[
                node_id
            ]
            for node_id in node_ids
        ]
    )

    return normalize(
        embeddings,
        norm="l2",
    )


def weighted_purity(
    truth: list[str],
    predictions: list[int],
) -> float:
    """
    Calculate node-weighted community purity.
    """
    if len(truth) != len(
        predictions
    ):
        raise ValueError(
            "Truth and prediction lists have different lengths."
        )

    cluster_labels: dict[
        int,
        list[str],
    ] = {}

    for genre, cluster_id in zip(
        truth,
        predictions,
    ):
        cluster_labels.setdefault(
            int(cluster_id),
            [],
        ).append(
            genre
        )

    correct_count = 0

    for genres in (
        cluster_labels.values()
    ):
        correct_count += (
            Counter(
                genres
            )
            .most_common(1)[0][1]
        )

    return (
        correct_count
        / len(
            truth
        )
    )


def evaluate_clusters(
    node_ids: list[str],
    cluster_ids: np.ndarray,
    known_labels: dict[str, str],
) -> dict[str, Any]:
    """
    Evaluate clusters against external labels after training.
    """
    cluster_by_node = {
        node_id: int(
            cluster_id
        )
        for node_id, cluster_id in zip(
            node_ids,
            cluster_ids,
        )
    }

    labeled_nodes = sorted(
        set(
            node_ids
        )
        & set(
            known_labels
        )
    )

    truth = [
        known_labels[
            node_id
        ]
        for node_id in labeled_nodes
    ]

    predictions = [
        cluster_by_node[
            node_id
        ]
        for node_id in labeled_nodes
    ]

    if not truth:
        raise ValueError(
            "No externally labeled nodes are available for evaluation."
        )

    cluster_sizes = Counter(
        int(
            cluster_id
        )
        for cluster_id in cluster_ids
    )

    return {
        "evaluated_labeled_node_count": int(
            len(
                labeled_nodes
            )
        ),
        "known_genre_count": int(
            len(
                set(
                    truth
                )
            )
        ),
        "nmi": float(
            normalized_mutual_info_score(
                truth,
                predictions,
                average_method="arithmetic",
            )
        ),
        "ami": float(
            adjusted_mutual_info_score(
                truth,
                predictions,
                average_method="arithmetic",
            )
        ),
        "ari": float(
            adjusted_rand_score(
                truth,
                predictions,
            )
        ),
        "purity_labeled_nodes": float(
            weighted_purity(
                truth,
                predictions,
            )
        ),
        "homogeneity": float(
            homogeneity_score(
                truth,
                predictions,
            )
        ),
        "completeness": float(
            completeness_score(
                truth,
                predictions,
            )
        ),
        "v_measure": float(
            v_measure_score(
                truth,
                predictions,
            )
        ),
        "cluster_count": int(
            len(
                cluster_sizes
            )
        ),
        "minimum_cluster_size": int(
            min(
                cluster_sizes.values()
            )
        ),
        "median_cluster_size": float(
            np.median(
                list(
                    cluster_sizes.values()
                )
            )
        ),
        "maximum_cluster_size": int(
            max(
                cluster_sizes.values()
            )
        ),
    }


def save_embeddings(
    node_ids: list[str],
    embeddings: np.ndarray,
    output_csv: Path,
) -> None:
    """
    Save one normalized embedding vector per artist.
    """
    embedding_columns = [
        f"embedding_{index:03d}"
        for index in range(
            embeddings.shape[1]
        )
    ]

    table = pd.DataFrame(
        embeddings,
        columns=embedding_columns,
    )

    table.insert(
        0,
        "artist_id",
        node_ids,
    )

    table.to_csv(
        output_csv,
        index=False,
        encoding="utf-8",
    )


def save_partition(
    node_ids: list[str],
    cluster_ids: np.ndarray,
    labels: pd.DataFrame,
    output_csv: Path,
) -> None:
    """
    Save one complete Node2Vec clustering assignment.
    """
    metadata = (
        labels.set_index(
            "artist_id"
        )
    )

    records: list[
        dict[str, Any]
    ] = []

    for node_id, cluster_id in zip(
        node_ids,
        cluster_ids,
    ):
        row = metadata.loc[
            node_id
        ]

        genre = row[
            "dominant_genre_normalized"
        ]

        records.append(
            {
                "artist_id": node_id,
                "artist_name": row[
                    "artist_name"
                ],
                "metadata_group": row[
                    "metadata_group"
                ],
                "dominant_genre_normalized": (
                    ""
                    if pd.isna(
                        genre
                    )
                    else str(
                        genre
                    )
                ),
                "cluster_id": int(
                    cluster_id
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


def run_experiments(
    graphml_path: Path,
    labels_csv: Path,
    output_dir: Path,
    p_values: list[float],
    q_values: list[float],
    seeds: list[int],
    k_values: list[int],
    dimensions: int,
    walk_length: int,
    walks_per_node: int,
    context_window: int,
    epochs: int,
) -> None:
    """
    Run Node2Vec and K-Means experiments.

    K is selected without genre labels by maximizing silhouette score.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_dir = (
        output_dir
        / "embeddings"
    )

    partitions_dir = (
        output_dir
        / "partitions"
    )

    embeddings_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    partitions_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Reading graph: "
        f"{graphml_path}"
    )

    graph = load_graph(
        graphml_path
    )

    print(
        f"Reading external labels: "
        f"{labels_csv}"
    )

    labels = load_external_labels(
        coverage_csv=labels_csv,
        graph=graph,
    )

    known_labels = build_known_label_map(
        labels
    )

    node_ids = sorted(
        str(
            node
        )
        for node in graph.nodes
    )

    records: list[
        dict[str, Any]
    ] = []

    selected_records: list[
        dict[str, Any]
    ] = []

    embedding_run_count = (
        len(
            p_values
        )
        * len(
            q_values
        )
        * len(
            seeds
        )
    )

    current_embedding_run = 0

    for p in p_values:
        for q in q_values:
            for seed in seeds:
                current_embedding_run += 1

                prefix = (
                    f"node2vec"
                    f"_p_{safe_float_text(p)}"
                    f"_q_{safe_float_text(q)}"
                    f"_seed_{seed}"
                )

                print()
                print(
                    f"[{current_embedding_run}/"
                    f"{embedding_run_count}] "
                    f"{prefix}"
                )

                started_at = (
                    time.perf_counter()
                )

                walks = generate_walk_corpus(
                    graph=graph,
                    walk_length=walk_length,
                    walks_per_node=walks_per_node,
                    seed=seed,
                    p=p,
                    q=q,
                )

                walk_seconds = (
                    time.perf_counter()
                    - started_at
                )

                embedding_started_at = (
                    time.perf_counter()
                )

                embeddings = train_embeddings(
                    walks=walks,
                    node_ids=node_ids,
                    dimensions=dimensions,
                    context_window=context_window,
                    epochs=epochs,
                    seed=seed,
                )

                embedding_seconds = (
                    time.perf_counter()
                    - embedding_started_at
                )

                save_embeddings(
                    node_ids=node_ids,
                    embeddings=embeddings,
                    output_csv=(
                        embeddings_dir
                        / f"{prefix}.csv"
                    ),
                )

                run_rows: list[
                    dict[str, Any]
                ] = []

                for k in k_values:
                    if k < 2 or k >= len(
                        node_ids
                    ):
                        raise ValueError(
                            f"Invalid K-Means cluster count: {k}"
                        )

                    kmeans_started_at = (
                        time.perf_counter()
                    )

                    kmeans = KMeans(
                        n_clusters=k,
                        n_init=20,
                        random_state=seed,
                    )

                    cluster_ids = (
                        kmeans.fit_predict(
                            embeddings
                        )
                    )

                    kmeans_seconds = (
                        time.perf_counter()
                        - kmeans_started_at
                    )

                    silhouette = float(
                        silhouette_score(
                            embeddings,
                            cluster_ids,
                            metric="euclidean",
                        )
                    )

                    metrics = evaluate_clusters(
                        node_ids=node_ids,
                        cluster_ids=cluster_ids,
                        known_labels=known_labels,
                    )

                    run_id = (
                        f"{prefix}"
                        f"_k_{k}"
                    )

                    partition_output = (
                        partitions_dir
                        / f"{run_id}.csv"
                    )

                    save_partition(
                        node_ids=node_ids,
                        cluster_ids=cluster_ids,
                        labels=labels,
                        output_csv=partition_output,
                    )

                    row = {
                        "run_id": run_id,
                        "p": float(
                            p
                        ),
                        "q": float(
                            q
                        ),
                        "seed": int(
                            seed
                        ),
                        "k": int(
                            k
                        ),
                        "silhouette_score": silhouette,
                        "walk_generation_seconds": float(
                            walk_seconds
                        ),
                        "embedding_training_seconds": float(
                            embedding_seconds
                        ),
                        "kmeans_seconds": float(
                            kmeans_seconds
                        ),
                        **metrics,
                    }

                    records.append(
                        row
                    )

                    run_rows.append(
                        row
                    )

                    print(
                        f"  k={k:>3} "
                        f"silhouette={silhouette:.4f} "
                        f"AMI={metrics['ami']:.4f} "
                        f"NMI={metrics['nmi']:.4f}"
                    )

                selected = max(
                    run_rows,
                    key=lambda row: (
                        row[
                            "silhouette_score"
                        ],
                        -row[
                            "k"
                        ],
                    ),
                )

                selected_records.append(
                    {
                        **selected,
                        "selection_rule": (
                            "maximum_silhouette_score"
                        ),
                    }
                )

                print(
                    "  selected without genre labels: "
                    f"k={selected['k']} "
                    f"silhouette="
                    f"{selected['silhouette_score']:.4f}"
                )

    all_runs = (
        pd.DataFrame.from_records(
            records
        )
        .sort_values(
            [
                "p",
                "q",
                "seed",
                "k",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    selected_runs = (
        pd.DataFrame.from_records(
            selected_records
        )
        .sort_values(
            [
                "p",
                "q",
                "seed",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    all_runs_output = (
        output_dir
        / "node2vec_clustering_runs.csv"
    )

    selected_runs_output = (
        output_dir
        / "node2vec_selected_unsupervised_runs.csv"
    )

    all_runs.to_csv(
        all_runs_output,
        index=False,
        encoding="utf-8",
    )

    selected_runs.to_csv(
        selected_runs_output,
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
            "labeled_node_count": int(
                len(
                    known_labels
                )
            ),
        },
        "configuration": {
            "p_values": p_values,
            "q_values": q_values,
            "seeds": seeds,
            "k_values": k_values,
            "dimensions": int(
                dimensions
            ),
            "walk_length": int(
                walk_length
            ),
            "walks_per_node": int(
                walks_per_node
            ),
            "context_window": int(
                context_window
            ),
            "epochs": int(
                epochs
            ),
            "word2vec_workers": 1,
            "kmeans_n_init": 20,
            "cluster_selection_rule": (
                "maximum silhouette score; "
                "external MusicBrainz genres are used only "
                "after unsupervised selection"
            ),
        },
        "outputs": {
            "node2vec_clustering_runs": str(
                all_runs_output
            ),
            "node2vec_selected_unsupervised_runs": str(
                selected_runs_output
            ),
            "embeddings_directory": str(
                embeddings_dir
            ),
            "partitions_directory": str(
                partitions_dir
            ),
        },
    }

    with (
        output_dir
        / "node2vec_summary.json"
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
        "Node2Vec experiment results written to:"
    )

    print(
        output_dir
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train weighted Node2Vec embeddings on the genre-free "
            "behavioral artist graph and cluster them with K-Means."
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
            "results/node2vec_baseline"
        ),
    )

    parser.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        default=[
            1.0
        ],
    )

    parser.add_argument(
        "--q-values",
        type=float,
        nargs="+",
        default=[
            1.0
        ],
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[
            42
        ],
    )

    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[
            40,
            50,
            54,
            60,
            70,
        ],
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--walk-length",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--walks-per-node",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--context-window",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if any(
        value <= 0
        for value in args.p_values
    ):
        raise ValueError(
            "All p values must be positive."
        )

    if any(
        value <= 0
        for value in args.q_values
    ):
        raise ValueError(
            "All q values must be positive."
        )

    run_experiments(
        graphml_path=args.graphml_path,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        p_values=args.p_values,
        q_values=args.q_values,
        seeds=args.seeds,
        k_values=args.k_values,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        walks_per_node=args.walks_per_node,
        context_window=args.context_window,
        epochs=args.epochs,
    )