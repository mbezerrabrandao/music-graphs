from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
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
from sklearn.preprocessing import StandardScaler, normalize


def required_text(value: Any, field_name: str) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"Missing required text value for: {field_name}")

    text = str(value).strip()

    if not text:
        raise ValueError(f"Blank required text value for: {field_name}")

    return text


def required_float(value: Any, field_name: str) -> float:
    if value is None or pd.isna(value):
        raise ValueError(f"Missing required numeric value for: {field_name}")

    try:
        return float(value)

    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Invalid numeric value for {field_name}: {value!r}"
        ) from error


def load_graph(graphml_path: Path) -> nx.Graph:
    if not graphml_path.exists():
        raise FileNotFoundError(f"GraphML file does not exist: {graphml_path}")

    graph = nx.read_graphml(graphml_path)

    if graph.is_directed():
        graph = nx.Graph(graph)

    if graph.number_of_nodes() == 0:
        raise ValueError("The input graph contains no nodes.")

    if graph.number_of_edges() == 0:
        raise ValueError("The input graph contains no edges.")

    if not nx.is_connected(graph):
        raise ValueError(
            "The behavioral-feature control expects the connected "
            "largest component graph."
        )

    for _, _, attributes in graph.edges(data=True):
        weight = required_float(attributes.get("weight", 1.0), "weight")

        if weight <= 0:
            raise ValueError("All graph weights must be strictly positive.")

        attributes["weight"] = weight

    return graph


def load_external_labels(labels_csv: Path, graph: nx.Graph) -> pd.DataFrame:
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Genre coverage table does not exist: {labels_csv}"
        )

    labels = pd.read_csv(
        labels_csv,
        dtype={
            "artist_id": "string",
            "artist_name": "string",
            "metadata_group": "string",
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

    missing_columns = sorted(required_columns - set(labels.columns))

    if missing_columns:
        raise ValueError(
            "Genre coverage table is missing columns: "
            + ", ".join(missing_columns)
        )

    labels["artist_id"] = (
        labels["artist_id"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    if labels["artist_id"].isna().any():
        raise ValueError("Found missing artist IDs in genre coverage table.")

    if labels["artist_id"].duplicated().any():
        raise ValueError("Found duplicated artist IDs in genre coverage table.")

    graph_node_ids = {str(node) for node in graph.nodes}
    label_node_ids = set(labels["artist_id"].astype(str))

    missing_labels = sorted(graph_node_ids - label_node_ids)

    if missing_labels:
        raise ValueError(
            "Some graph nodes are absent from genre coverage table. "
            f"Example: {missing_labels[0]}"
        )

    return labels


def build_known_label_map(labels: pd.DataFrame) -> dict[str, str]:
    known_labels: dict[str, str] = {}

    for row in labels.to_dict(orient="records"):
        artist_id = required_text(row["artist_id"], "artist_id")
        genre = row["dominant_genre_normalized"]

        if genre is None or pd.isna(genre):
            continue

        genre_text = str(genre).strip()

        if genre_text:
            known_labels[artist_id] = genre_text

    return known_labels


def build_weighted_normalized_adjacency(
    graph: nx.Graph,
    node_ids: list[str],
) -> csr_matrix:
    node_to_index = {
        node_id: index
        for index, node_id in enumerate(node_ids)
    }

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    for source, target, attributes in graph.edges(data=True):
        index_a = node_to_index[str(source)]
        index_b = node_to_index[str(target)]
        weight = required_float(attributes.get("weight", 1.0), "weight")

        rows.extend([index_a, index_b])
        columns.extend([index_b, index_a])
        values.extend([weight, weight])

    for node_index in range(len(node_ids)):
        rows.append(node_index)
        columns.append(node_index)
        values.append(1.0)

    adjacency = coo_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (
                np.asarray(rows, dtype=np.int64),
                np.asarray(columns, dtype=np.int64),
            ),
        ),
        shape=(len(node_ids), len(node_ids)),
        dtype=np.float32,
    ).tocsr()

    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)

    if np.any(degrees <= 0):
        raise ValueError("Normalized adjacency contains a non-positive degree.")

    inverse_sqrt_degree = np.power(degrees, -0.5).astype(np.float32)

    return adjacency.multiply(
        inverse_sqrt_degree[:, None]
    ).multiply(
        inverse_sqrt_degree[None, :]
    ).tocsr()


def build_behavioral_feature_matrix(
    graph: nx.Graph,
    node_ids: list[str],
) -> np.ndarray:
    weighted_degree = dict(graph.degree(weight="weight"))

    rows: list[list[float]] = []

    for node_id in node_ids:
        attributes = graph.nodes[node_id]

        scrobble_count = required_float(
            attributes.get("scrobble_count", 0.0),
            "scrobble_count",
        )

        session_count = required_float(
            attributes.get("session_count", 0.0),
            "session_count",
        )

        degree = float(graph.degree(node_id))

        node_weighted_degree = required_float(
            weighted_degree[node_id],
            "weighted_degree",
        )

        rows.append(
            [
                np.log1p(scrobble_count),
                np.log1p(session_count),
                np.log1p(degree),
                np.log1p(node_weighted_degree),
            ]
        )

    return np.asarray(rows, dtype=np.float32)


def prepare_embedding(
    base_features: np.ndarray,
    normalized_adjacency: csr_matrix,
    mode: str,
) -> np.ndarray:
    if mode == "raw":
        representation = base_features

    elif mode == "smooth1":
        representation = normalized_adjacency @ base_features

    elif mode == "smooth2":
        representation = normalized_adjacency @ (
            normalized_adjacency @ base_features
        )

    else:
        raise ValueError(f"Unsupported control mode: {mode}")

    standardized = StandardScaler().fit_transform(
        np.asarray(representation, dtype=np.float32)
    )

    return normalize(
        standardized,
        norm="l2",
    )


def weighted_purity(
    truth: list[str],
    predictions: list[int],
) -> float:
    if len(truth) != len(predictions):
        raise ValueError("Truth and prediction lengths do not match.")

    if not truth:
        raise ValueError("Cannot calculate purity without labeled nodes.")

    cluster_labels: dict[int, list[str]] = {}

    for genre, cluster_id in zip(truth, predictions):
        cluster_labels.setdefault(int(cluster_id), []).append(genre)

    correct_count = 0

    for genres in cluster_labels.values():
        correct_count += Counter(genres).most_common(1)[0][1]

    return correct_count / len(truth)


def evaluate_clusters(
    node_ids: list[str],
    cluster_ids: np.ndarray,
    known_labels: dict[str, str],
) -> dict[str, Any]:
    cluster_by_node = {
        node_id: int(cluster_id)
        for node_id, cluster_id in zip(node_ids, cluster_ids)
    }

    labeled_nodes = sorted(set(node_ids) & set(known_labels))

    if not labeled_nodes:
        raise ValueError("No labeled graph nodes are available for evaluation.")

    truth = [
        known_labels[node_id]
        for node_id in labeled_nodes
    ]

    predictions = [
        cluster_by_node[node_id]
        for node_id in labeled_nodes
    ]

    cluster_sizes = Counter(
        int(cluster_id)
        for cluster_id in cluster_ids
    )

    return {
        "evaluated_labeled_node_count": int(len(labeled_nodes)),
        "known_genre_count": int(len(set(truth))),
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
        "cluster_count": int(len(cluster_sizes)),
        "minimum_cluster_size": int(min(cluster_sizes.values())),
        "median_cluster_size": float(
            np.median(list(cluster_sizes.values()))
        ),
        "maximum_cluster_size": int(max(cluster_sizes.values())),
    }


def save_embeddings(
    node_ids: list[str],
    embeddings: np.ndarray,
    output_csv: Path,
) -> None:
    columns = [
        f"embedding_{index:03d}"
        for index in range(embeddings.shape[1])
    ]

    table = pd.DataFrame(
        embeddings,
        columns=columns,
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
    metadata = labels.set_index("artist_id")

    records: list[dict[str, Any]] = []

    for node_id, cluster_id in zip(node_ids, cluster_ids):
        row = metadata.loc[node_id]
        genre = row["dominant_genre_normalized"]

        records.append(
            {
                "artist_id": node_id,
                "artist_name": row["artist_name"],
                "metadata_group": row["metadata_group"],
                "dominant_genre_normalized": (
                    ""
                    if pd.isna(genre)
                    else str(genre)
                ),
                "cluster_id": int(cluster_id),
            }
        )

    pd.DataFrame.from_records(
        records
    ).to_csv(
        output_csv,
        index=False,
        encoding="utf-8",
    )


def run_controls(
    graphml_path: Path,
    labels_csv: Path,
    output_dir: Path,
    modes: list[str],
    seeds: list[int],
    k_values: list[int],
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_dir = output_dir / "embeddings"
    partitions_dir = output_dir / "partitions"

    embeddings_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    partitions_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Reading graph: {graphml_path}")

    graph = load_graph(graphml_path)

    print(f"Reading labels: {labels_csv}")

    labels = load_external_labels(
        labels_csv=labels_csv,
        graph=graph,
    )

    known_labels = build_known_label_map(labels)

    node_ids = sorted(
        str(node)
        for node in graph.nodes
    )

    normalized_adjacency = build_weighted_normalized_adjacency(
        graph=graph,
        node_ids=node_ids,
    )

    base_features = build_behavioral_feature_matrix(
        graph=graph,
        node_ids=node_ids,
    )

    records: list[dict[str, Any]] = []

    for mode in modes:
        print()
        print(f"Preparing control representation: {mode}")

        embeddings = prepare_embedding(
            base_features=base_features,
            normalized_adjacency=normalized_adjacency,
            mode=mode,
        )

        save_embeddings(
            node_ids=node_ids,
            embeddings=embeddings,
            output_csv=(
                embeddings_dir
                / f"behavioral_features_{mode}.csv"
            ),
        )

        for seed in seeds:
            for k in k_values:
                if k < 2 or k >= len(node_ids):
                    raise ValueError(f"Invalid K-Means cluster count: {k}")

                kmeans = KMeans(
                    n_clusters=k,
                    n_init=20,
                    random_state=seed,
                )

                cluster_ids = kmeans.fit_predict(embeddings)

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
                    f"behavioral_features_{mode}"
                    f"_seed_{seed}"
                    f"_k_{k}"
                )

                save_partition(
                    node_ids=node_ids,
                    cluster_ids=cluster_ids,
                    labels=labels,
                    output_csv=(
                        partitions_dir
                        / f"{run_id}.csv"
                    ),
                )

                records.append(
                    {
                        "run_id": run_id,
                        "control_mode": mode,
                        "seed": int(seed),
                        "k": int(k),
                        "silhouette_score": silhouette,
                        **metrics,
                    }
                )

                print(
                    f"  seed={seed} "
                    f"k={k} "
                    f"silhouette={silhouette:.4f} "
                    f"NMI={metrics['nmi']:.4f} "
                    f"AMI={metrics['ami']:.4f} "
                    f"ARI={metrics['ari']:.4f}"
                )

    results = (
        pd.DataFrame.from_records(records)
        .sort_values(
            [
                "control_mode",
                "seed",
                "k",
            ]
        )
        .reset_index(drop=True)
    )

    results_output = (
        output_dir
        / "behavioral_feature_control_runs.csv"
    )

    results.to_csv(
        results_output,
        index=False,
        encoding="utf-8",
    )

    summary = (
        results.groupby(
            [
                "control_mode",
                "k",
            ],
            as_index=False,
        )
        .agg(
            run_count=(
                "seed",
                "size",
            ),
            silhouette_mean=(
                "silhouette_score",
                "mean",
            ),
            silhouette_std=(
                "silhouette_score",
                "std",
            ),
            nmi_mean=(
                "nmi",
                "mean",
            ),
            nmi_std=(
                "nmi",
                "std",
            ),
            ami_mean=(
                "ami",
                "mean",
            ),
            ami_std=(
                "ami",
                "std",
            ),
            ari_mean=(
                "ari",
                "mean",
            ),
            ari_std=(
                "ari",
                "std",
            ),
            purity_mean=(
                "purity_labeled_nodes",
                "mean",
            ),
            minimum_cluster_size_mean=(
                "minimum_cluster_size",
                "mean",
            ),
            median_cluster_size_mean=(
                "median_cluster_size",
                "mean",
            ),
            maximum_cluster_size_mean=(
                "maximum_cluster_size",
                "mean",
            ),
        )
        .fillna(0.0)
        .sort_values(
            [
                "k",
                "silhouette_mean",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    summary_output = (
        output_dir
        / "behavioral_feature_control_summary.csv"
    )

    summary.to_csv(
        summary_output,
        index=False,
        encoding="utf-8",
    )

    metadata = {
        "inputs": {
            "graphml_path": str(graphml_path),
            "labels_csv": str(labels_csv),
        },
        "graph": {
            "node_count": int(graph.number_of_nodes()),
            "edge_count": int(graph.number_of_edges()),
            "labeled_node_count": int(len(known_labels)),
        },
        "configuration": {
            "modes": modes,
            "seeds": seeds,
            "k_values": k_values,
            "base_features": [
                "log1p(scrobble_count)",
                "log1p(session_count)",
                "log1p(degree)",
                "log1p(weighted_degree)",
            ],
            "adjacency": (
                "full weighted normalized adjacency with self-loops"
            ),
            "note": (
                "This is a clustering-only control. It does not perform "
                "link-prediction validation, so it intentionally uses the "
                "complete behavioral graph."
            ),
            "external_genres_used_during_representation_building": False,
            "external_genres_used_during_clustering": False,
        },
        "outputs": {
            "runs": str(results_output),
            "summary": str(summary_output),
            "embeddings_directory": str(embeddings_dir),
            "partitions_directory": str(partitions_dir),
        },
    }

    with (
        output_dir
        / "behavioral_feature_control_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Behavioral-feature control summary")
    print("----------------------------------")
    print(summary.to_string(index=False))
    print()
    print(f"Control outputs written to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run clustering controls using behavioral node features "
            "with zero, one, or two graph-smoothing steps."
        )
    )

    parser.add_argument(
        "graphml_path",
        type=Path,
    )

    parser.add_argument(
        "labels_csv",
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/behavioral_feature_controls"
        ),
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "raw",
            "smooth1",
            "smooth2",
        ],
        default=[
            "raw",
            "smooth1",
            "smooth2",
        ],
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
    )

    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[
            54,
            70,
        ],
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_controls(
        graphml_path=args.graphml_path,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        modes=args.modes,
        seeds=args.seeds,
        k_values=args.k_values,
    )
