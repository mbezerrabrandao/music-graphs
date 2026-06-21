from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    average_precision_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler, normalize


def require_torch():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Graph Autoencoder requires PyTorch. Install the optional GAE "
            "dependency in this environment, e.g. `python -m pip install -e \".[gae]\"`, "
            "then rerun the graph_autoencoder stage."
        ) from error

    return torch, nn, F


def format_float(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "_")


def run_id_for(
    *,
    hidden_dim: int,
    latent_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    k: int,
) -> str:
    return (
        "gae_behavioral"
        f"_hidden_{hidden_dim}"
        f"_latent_{latent_dim}"
        f"_dropout_{format_float(dropout)}"
        f"_lr_{format_float(learning_rate)}"
        f"_wd_{format_float(weight_decay)}"
        f"_seed_{seed}"
        f"_k_{k}"
    )


def config_id_for(
    *,
    hidden_dim: int,
    latent_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
) -> str:
    return (
        f"hidden_{hidden_dim}"
        f"_latent_{latent_dim}"
        f"_dropout_{format_float(dropout)}"
        f"_lr_{format_float(learning_rate)}"
        f"_wd_{format_float(weight_decay)}"
    )


def selected_legacy_dir_name(latent_dim: int, learning_rate: float) -> str:
    return f"latent_{latent_dim}_lr_{format_float(learning_rate)}"


def required_text(value: Any, field_name: str) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"Missing required text value for: {field_name}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Blank required text value for: {field_name}")
    return text


def load_graph(graphml_path: Path) -> nx.Graph:
    if not graphml_path.exists():
        raise FileNotFoundError(graphml_path)
    graph = nx.read_graphml(graphml_path)
    return nx.relabel_nodes(graph, {node: str(node) for node in graph.nodes}, copy=True)


def load_labels(labels_csv: Path) -> pd.DataFrame:
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
    required = {"artist_id", "artist_name", "metadata_group", "dominant_genre_normalized"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError("Label table is missing columns: " + ", ".join(missing))
    for column in required:
        labels[column] = labels[column].astype("string").str.strip().replace("", pd.NA)
    if labels["artist_id"].isna().any() or labels["artist_id"].duplicated().any():
        raise ValueError("Label table contains missing or duplicate artist IDs.")
    return labels


def build_known_label_map(labels: pd.DataFrame) -> dict[str, str]:
    known: dict[str, str] = {}
    for row in labels.to_dict(orient="records"):
        genre = row["dominant_genre_normalized"]
        if genre is None or pd.isna(genre):
            continue
        genre_text = str(genre).strip()
        if genre_text:
            known[required_text(row["artist_id"], "artist_id")] = genre_text
    return known


def weighted_purity(truth: list[str], predictions: list[int]) -> float:
    cluster_labels: dict[int, list[str]] = {}
    for genre, cluster_id in zip(truth, predictions):
        cluster_labels.setdefault(int(cluster_id), []).append(genre)
    correct = sum(Counter(values).most_common(1)[0][1] for values in cluster_labels.values())
    return correct / len(truth)


def evaluate_clusters(
    node_ids: list[str],
    cluster_ids: np.ndarray,
    known_labels: dict[str, str],
) -> dict[str, Any]:
    cluster_by_node = {node_id: int(cluster_id) for node_id, cluster_id in zip(node_ids, cluster_ids)}
    labeled_nodes = sorted(set(node_ids) & set(known_labels))
    truth = [known_labels[node_id] for node_id in labeled_nodes]
    predictions = [cluster_by_node[node_id] for node_id in labeled_nodes]
    if not truth:
        raise ValueError("No externally labeled nodes are available for GAE evaluation.")
    cluster_sizes = Counter(int(cluster_id) for cluster_id in cluster_ids)
    return {
        "evaluated_labeled_node_count": int(len(labeled_nodes)),
        "known_genre_count": int(len(set(truth))),
        "nmi": float(normalized_mutual_info_score(truth, predictions, average_method="arithmetic")),
        "ami": float(adjusted_mutual_info_score(truth, predictions, average_method="arithmetic")),
        "ari": float(adjusted_rand_score(truth, predictions)),
        "purity_labeled_nodes": float(weighted_purity(truth, predictions)),
        "homogeneity": float(homogeneity_score(truth, predictions)),
        "completeness": float(completeness_score(truth, predictions)),
        "v_measure": float(v_measure_score(truth, predictions)),
        "cluster_count": int(len(cluster_sizes)),
        "minimum_cluster_size": int(min(cluster_sizes.values())),
        "median_cluster_size": float(np.median(list(cluster_sizes.values()))),
        "maximum_cluster_size": int(max(cluster_sizes.values())),
    }


def save_embeddings(node_ids: list[str], embeddings: np.ndarray, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [f"embedding_{index:03d}" for index in range(embeddings.shape[1])]
    table = pd.DataFrame(embeddings, columns=columns)
    table.insert(0, "artist_id", node_ids)
    table.to_csv(output_csv, index=False, encoding="utf-8")


def save_partition(
    node_ids: list[str],
    cluster_ids: np.ndarray,
    labels: pd.DataFrame,
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
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
                "dominant_genre_normalized": "" if pd.isna(genre) else str(genre),
                "cluster_id": int(cluster_id),
            }
        )
    pd.DataFrame.from_records(records).to_csv(output_csv, index=False, encoding="utf-8")


def graph_feature_matrix(graph: nx.Graph, node_ids: list[str]) -> np.ndarray:
    degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="weight"))
    rows = []
    for node_id in node_ids:
        attrs = graph.nodes[node_id]
        rows.append(
            [
                math.log1p(float(attrs.get("scrobble_count", 0.0))),
                math.log1p(float(attrs.get("session_count", 0.0))),
                math.log1p(float(attrs.get("user_count", 1.0))),
                math.log1p(float(degree.get(node_id, 0.0))),
                math.log1p(float(weighted_degree.get(node_id, 0.0))),
            ]
        )
    return StandardScaler().fit_transform(np.asarray(rows, dtype=np.float32))


def split_edges(
    edges: list[tuple[int, int]],
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    edge_array = np.asarray(edges, dtype=np.int64)
    order = rng.permutation(len(edge_array))
    validation_count = max(1, int(round(len(edge_array) * validation_fraction)))
    validation_index = order[:validation_count]
    train_index = order[validation_count:]
    return edge_array[train_index], edge_array[validation_index]


def sample_negative_edges(
    *,
    node_count: int,
    count: int,
    forbidden: set[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    samples: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(samples) < count:
        remaining = count - len(samples)
        left = rng.integers(0, node_count, size=remaining * 3)
        right = rng.integers(0, node_count, size=remaining * 3)
        for a_raw, b_raw in zip(left, right):
            a = int(a_raw)
            b = int(b_raw)
            if a == b:
                continue
            edge = (a, b) if a < b else (b, a)
            if edge in forbidden or edge in seen:
                continue
            seen.add(edge)
            samples.append(edge)
            if len(samples) == count:
                break
    return np.asarray(samples, dtype=np.int64)


def normalized_sparse_adjacency(torch, node_count: int, train_edges: np.ndarray):
    loops = np.arange(node_count, dtype=np.int64)
    row = np.concatenate([train_edges[:, 0], train_edges[:, 1], loops])
    col = np.concatenate([train_edges[:, 1], train_edges[:, 0], loops])
    values = np.ones(len(row), dtype=np.float32)
    degree = np.bincount(row, weights=values, minlength=node_count).astype(np.float32)
    norm_values = values / np.sqrt(degree[row] * degree[col])
    indices = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    values_tensor = torch.tensor(norm_values, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values_tensor, (node_count, node_count)).coalesce()


@dataclass(frozen=True)
class TrainingResult:
    embeddings: np.ndarray
    validation_auc: float
    validation_average_precision: float
    training_seconds: float
    final_loss: float


def train_graph_autoencoder(
    *,
    graph: nx.Graph,
    node_ids: list[str],
    features: np.ndarray,
    hidden_dim: int,
    latent_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    validation_fraction: float,
    seed: int,
) -> TrainingResult:
    torch, nn, F = require_torch()
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    random.seed(seed)

    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    edges = [
        (min(node_index[str(a)], node_index[str(b)]), max(node_index[str(a)], node_index[str(b)]))
        for a, b in graph.edges
    ]
    edges = sorted(set(edges))
    train_edges, validation_edges = split_edges(edges, validation_fraction, seed)
    forbidden = set(edges)
    validation_negative_edges = sample_negative_edges(
        node_count=len(node_ids),
        count=len(validation_edges),
        forbidden=forbidden,
        rng=np_rng,
    )

    adjacency = normalized_sparse_adjacency(torch, len(node_ids), train_edges)
    x = torch.tensor(features, dtype=torch.float32)
    train_positive = torch.tensor(train_edges, dtype=torch.long)
    validation_positive = torch.tensor(validation_edges, dtype=torch.long)
    validation_negative = torch.tensor(validation_negative_edges, dtype=torch.long)

    class GraphAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_1 = nn.Linear(x.shape[1], hidden_dim, bias=False)
            self.encoder_2 = nn.Linear(hidden_dim, latent_dim, bias=False)
            self.dropout = float(dropout)

        def forward(self, features_tensor):
            hidden = torch.sparse.mm(adjacency, features_tensor)
            hidden = self.encoder_1(hidden)
            hidden = F.relu(hidden)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            latent = torch.sparse.mm(adjacency, hidden)
            latent = self.encoder_2(latent)
            return F.normalize(latent, p=2, dim=1)

    def edge_logits(z, edge_tensor):
        return (z[edge_tensor[:, 0]] * z[edge_tensor[:, 1]]).sum(dim=1)

    model = GraphAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    started = time.perf_counter()
    final_loss = float("nan")

    for _ in range(epochs):
        model.train()
        negative_edges = torch.tensor(
            sample_negative_edges(
                node_count=len(node_ids),
                count=len(train_edges),
                forbidden=forbidden,
                rng=np_rng,
            ),
            dtype=torch.long,
        )
        optimizer.zero_grad()
        z = model(x)
        logits = torch.cat([edge_logits(z, train_positive), edge_logits(z, negative_edges)])
        targets = torch.cat([
            torch.ones(len(train_positive), dtype=torch.float32),
            torch.zeros(len(negative_edges), dtype=torch.float32),
        ])
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())

    training_seconds = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        z = model(x)
        validation_logits = torch.cat([
            edge_logits(z, validation_positive),
            edge_logits(z, validation_negative),
        ]).detach().cpu().numpy()
        validation_targets = np.concatenate([
            np.ones(len(validation_positive), dtype=np.int8),
            np.zeros(len(validation_negative), dtype=np.int8),
        ])
        embeddings = z.detach().cpu().numpy()

    return TrainingResult(
        embeddings=normalize(embeddings, norm="l2"),
        validation_auc=float(roc_auc_score(validation_targets, validation_logits)),
        validation_average_precision=float(average_precision_score(validation_targets, validation_logits)),
        training_seconds=float(training_seconds),
        final_loss=final_loss,
    )


def run_experiments(
    *,
    graphml_path: Path,
    labels_csv: Path,
    output_dir: Path,
    summary_output_dir: Path,
    hidden_dims: list[int],
    latent_dims: list[int],
    dropout_values: list[float],
    learning_rates: list[float],
    weight_decays: list[float],
    seeds: list[int],
    k_values: list[int],
    epochs: int,
    validation_fraction: float,
) -> None:
    graph = load_graph(graphml_path)
    labels = load_labels(labels_csv)
    known_labels = build_known_label_map(labels)
    node_ids = sorted(str(node) for node in graph.nodes)
    label_ids = set(labels["artist_id"].astype(str))
    missing = sorted(set(node_ids) - label_ids)
    if missing:
        raise ValueError(f"Labels are missing graph node IDs. Example: {missing[0]}")
    features = graph_feature_matrix(graph, node_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    embedding_paths: dict[tuple[int, int, float, float, float, int], Path] = {}

    total = (
        len(hidden_dims)
        * len(latent_dims)
        * len(dropout_values)
        * len(learning_rates)
        * len(weight_decays)
        * len(seeds)
    )
    run_index = 0

    for hidden_dim in hidden_dims:
        for latent_dim in latent_dims:
            for dropout in dropout_values:
                for learning_rate in learning_rates:
                    for weight_decay in weight_decays:
                        config_id = config_id_for(
                            hidden_dim=hidden_dim,
                            latent_dim=latent_dim,
                            dropout=dropout,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                        )
                        config_dir = output_dir / config_id
                        embeddings_dir = config_dir / "embeddings"
                        partitions_dir = config_dir / "partitions"
                        embeddings_dir.mkdir(parents=True, exist_ok=True)
                        partitions_dir.mkdir(parents=True, exist_ok=True)

                        for seed in seeds:
                            run_index += 1
                            print(
                                f"[{run_index}/{total}] GAE {config_id} seed={seed}",
                                flush=True,
                            )
                            result = train_graph_autoencoder(
                                graph=graph,
                                node_ids=node_ids,
                                features=features,
                                hidden_dim=hidden_dim,
                                latent_dim=latent_dim,
                                dropout=dropout,
                                learning_rate=learning_rate,
                                weight_decay=weight_decay,
                                epochs=epochs,
                                validation_fraction=validation_fraction,
                                seed=seed,
                            )
                            embedding_output = embeddings_dir / f"gae_behavioral_seed_{seed}.csv"
                            save_embeddings(node_ids, result.embeddings, embedding_output)
                            embedding_paths[(hidden_dim, latent_dim, dropout, learning_rate, weight_decay, seed)] = embedding_output

                            for k in k_values:
                                run_id = run_id_for(
                                    hidden_dim=hidden_dim,
                                    latent_dim=latent_dim,
                                    dropout=dropout,
                                    learning_rate=learning_rate,
                                    weight_decay=weight_decay,
                                    seed=seed,
                                    k=k,
                                )
                                kmeans_started = time.perf_counter()
                                kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
                                cluster_ids = kmeans.fit_predict(result.embeddings)
                                kmeans_seconds = time.perf_counter() - kmeans_started
                                silhouette = float(silhouette_score(result.embeddings, cluster_ids))
                                metrics = evaluate_clusters(node_ids, cluster_ids, known_labels)
                                save_partition(
                                    node_ids=node_ids,
                                    cluster_ids=cluster_ids,
                                    labels=labels,
                                    output_csv=partitions_dir / f"{run_id}.csv",
                                )
                                records.append(
                                    {
                                        "run_id": run_id,
                                        "feature_mode": "behavioral",
                                        "hidden_dim": int(hidden_dim),
                                        "latent_dim": int(latent_dim),
                                        "dropout": float(dropout),
                                        "learning_rate": float(learning_rate),
                                        "weight_decay": float(weight_decay),
                                        "seed": int(seed),
                                        "k": int(k),
                                        "epoch_count": int(epochs),
                                        "validation_fraction": float(validation_fraction),
                                        "validation_auc": result.validation_auc,
                                        "validation_average_precision": result.validation_average_precision,
                                        "final_training_loss": result.final_loss,
                                        "training_seconds": result.training_seconds,
                                        "kmeans_seconds": float(kmeans_seconds),
                                        "silhouette_score": silhouette,
                                        **metrics,
                                    }
                                )
                                print(
                                    f"  k={k:>3} val_auc={result.validation_auc:.4f} "
                                    f"silhouette={silhouette:.4f} AMI={metrics['ami']:.4f}",
                                    flush=True,
                                )

    runs = pd.DataFrame.from_records(records)
    config_columns = ["hidden_dim", "latent_dim", "dropout", "learning_rate", "weight_decay"]
    config_summary = (
        runs.groupby(config_columns, as_index=False)
        .agg(
            validation_auc_mean=("validation_auc", "mean"),
            validation_auc_std=("validation_auc", "std"),
            validation_average_precision_mean=("validation_average_precision", "mean"),
            silhouette_mean=("silhouette_score", "mean"),
            ami_mean=("ami", "mean"),
        )
        .fillna(0.0)
        .sort_values(
            ["validation_auc_mean", "validation_average_precision_mean", "silhouette_mean"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    k_summary = (
        runs.groupby("k", as_index=False)
        .agg(
            run_count=("run_id", "count"),
            validation_auc_mean=("validation_auc", "mean"),
            validation_average_precision_mean=("validation_average_precision", "mean"),
            silhouette_mean=("silhouette_score", "mean"),
            silhouette_std=("silhouette_score", "std"),
            ami_mean=("ami", "mean"),
            ari_mean=("ari", "mean"),
            purity_labeled_nodes_mean=("purity_labeled_nodes", "mean"),
            minimum_cluster_size_mean=("minimum_cluster_size", "mean"),
            median_cluster_size_mean=("median_cluster_size", "mean"),
            maximum_cluster_size_mean=("maximum_cluster_size", "mean"),
        )
        .fillna(0.0)
        .sort_values("k")
        .reset_index(drop=True)
    )
    selected_config = config_summary.iloc[0].to_dict()
    selected_mask = np.ones(len(runs), dtype=bool)
    for column in config_columns:
        selected_mask &= runs[column].map(lambda value, c=column: math.isclose(float(value), float(selected_config[c]), rel_tol=0.0, abs_tol=1e-12))
    runs["selected_for_balanced_evaluation"] = selected_mask

    combined_output = summary_output_dir / "graph_autoencoder_combined_runs.csv"
    config_summary_output = summary_output_dir / "graph_autoencoder_hyperparameter_summary.csv"
    k_summary_output = summary_output_dir / "graph_autoencoder_k_sensitivity.csv"
    selection_output = summary_output_dir / "graph_autoencoder_selected_hyperparameters.json"
    runs.to_csv(combined_output, index=False, encoding="utf-8")
    config_summary.to_csv(config_summary_output, index=False, encoding="utf-8")
    k_summary.to_csv(k_summary_output, index=False, encoding="utf-8")
    selection_output.write_text(json.dumps(selected_config, indent=2, ensure_ascii=False), encoding="utf-8")

    selected_embeddings_dir = output_dir / "selected" / "embeddings"
    selected_embeddings_dir.mkdir(parents=True, exist_ok=True)
    selected_hidden_dim = int(selected_config["hidden_dim"])
    selected_latent_dim = int(selected_config["latent_dim"])
    selected_dropout = float(selected_config["dropout"])
    selected_learning_rate = float(selected_config["learning_rate"])
    selected_weight_decay = float(selected_config["weight_decay"])
    for seed in seeds:
        source = embedding_paths[
            (
                selected_hidden_dim,
                selected_latent_dim,
                selected_dropout,
                selected_learning_rate,
                selected_weight_decay,
                seed,
            )
        ]
        shutil.copyfile(source, selected_embeddings_dir / f"gae_behavioral_seed_{seed}.csv")

    legacy_dir = output_dir / selected_legacy_dir_name(
        int(selected_config["latent_dim"]),
        float(selected_config["learning_rate"]),
    ) / "embeddings"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        shutil.copyfile(
            selected_embeddings_dir / f"gae_behavioral_seed_{seed}.csv",
            legacy_dir / f"gae_behavioral_seed_{seed}.csv",
        )

    summary = {
        "inputs": {
            "graphml_path": str(graphml_path),
            "labels_csv": str(labels_csv),
        },
        "graph": {
            "node_count": int(graph.number_of_nodes()),
            "edge_count": int(graph.number_of_edges()),
        },
        "candidate_grid": {
            "hidden_dims": hidden_dims,
            "latent_dims": latent_dims,
            "dropout_values": dropout_values,
            "learning_rates": learning_rates,
            "weight_decays": weight_decays,
            "seeds": seeds,
            "k_values": k_values,
            "epochs": int(epochs),
        },
        "selected_hyperparameters": selected_config,
        "outputs": {
            "combined_runs": str(combined_output),
            "hyperparameter_summary": str(config_summary_output),
            "k_sensitivity": str(k_summary_output),
            "selected_embeddings_dir": str(selected_embeddings_dir),
        },
    }
    (summary_output_dir / "graph_autoencoder_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch graph autoencoder and evaluate K-Means communities."
    )
    parser.add_argument("graphml_path", type=Path)
    parser.add_argument("labels_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/graph_autoencoder_final_candidates_no_leakage"))
    parser.add_argument("--summary-output-dir", type=Path, default=Path("results/graph_autoencoder_final_candidates_no_leakage_summary"))
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128])
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[16])
    parser.add_argument("--dropout-values", type=float, nargs="+", default=[0.1])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.005])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.0001])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--k-values", type=int, nargs="+", default=[54])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5.")
    run_experiments(
        graphml_path=args.graphml_path,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        summary_output_dir=args.summary_output_dir,
        hidden_dims=args.hidden_dims,
        latent_dims=args.latent_dims,
        dropout_values=args.dropout_values,
        learning_rates=args.learning_rates,
        weight_decays=args.weight_decays,
        seeds=args.seeds,
        k_values=args.k_values,
        epochs=args.epochs,
        validation_fraction=args.validation_fraction,
    )
