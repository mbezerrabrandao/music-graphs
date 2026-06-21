from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

API_ROOT = "https://acousticbrainz.org/api/v1"
DEFAULT_FEATURE_PATHS = [
    "metadata.audio_properties.length",
    "metadata.audio_properties.replay_gain",
    "lowlevel.average_loudness",
    "lowlevel.dynamic_complexity",
    "lowlevel.barkbands.mean",
    "lowlevel.erbbands.mean",
    "lowlevel.mfcc.mean",
    "lowlevel.spectral_centroid.mean",
    "lowlevel.spectral_complexity.mean",
    "lowlevel.spectral_flux.mean",
    "lowlevel.spectral_kurtosis.mean",
    "lowlevel.spectral_rms.mean",
    "lowlevel.spectral_rolloff.mean",
    "lowlevel.spectral_skewness.mean",
    "lowlevel.spectral_spread.mean",
    "lowlevel.zerocrossingrate.mean",
    "rhythm.bpm",
    "rhythm.beats_loudness.mean",
    "rhythm.onset_rate",
    "tonal.chords_changes_rate",
    "tonal.chords_strength.mean",
    "tonal.hpcp.mean",
]


def required_text(value: Any, field_name: str) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"Missing required value for {field_name}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Blank required value for {field_name}")
    return text


def normalize_mbid(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    return text or None


def nested_get(document: dict[str, Any], path: str) -> Any:
    value: Any = document
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def flatten_numeric(value: Any, prefix: str) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return {prefix: float(value)}
    if isinstance(value, list):
        out: dict[str, float] = {}
        for index, item in enumerate(value):
            if isinstance(item, (int, float)) and math.isfinite(float(item)):
                out[f"{prefix}_{index:03d}"] = float(item)
        return out
    return {}


def extract_feature_vector(document: dict[str, Any], feature_paths: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for path in feature_paths:
        safe = path.replace(".", "_")
        values.update(flatten_numeric(nested_get(document, path), safe))
    return values


def load_label_artist_ids(labels_csv: Path) -> set[str]:
    labels = pd.read_csv(labels_csv, usecols=["artist_id"], dtype="string")
    return set(labels["artist_id"].dropna().astype(str))


def select_top_recordings(
    *,
    canonical_scrobbles_csv: Path,
    labels_csv: Path,
    output_csv: Path,
    top_n: int,
    min_track_scrobbles: int,
) -> None:
    label_artist_ids = load_label_artist_ids(labels_csv)
    usecols = ["artist_id", "artist", "track", "track_mbid", "scrobble_id"]
    scrobbles = pd.read_csv(canonical_scrobbles_csv, usecols=usecols, dtype="string")
    scrobbles = scrobbles.loc[scrobbles["artist_id"].isin(label_artist_ids)].copy()
    scrobbles["track_mbid"] = scrobbles["track_mbid"].map(normalize_mbid)
    scrobbles = scrobbles.dropna(subset=["artist_id", "track_mbid"])

    grouped = (
        scrobbles.groupby(["artist_id", "artist", "track", "track_mbid"], dropna=False)
        .agg(scrobble_count=("scrobble_id", "size"))
        .reset_index()
    )
    grouped = grouped.loc[grouped["scrobble_count"] >= min_track_scrobbles].copy()
    grouped = grouped.sort_values(["artist_id", "scrobble_count", "track"], ascending=[True, False, True])
    grouped["artist_recording_rank"] = grouped.groupby("artist_id").cumcount() + 1
    selected = grouped.loc[grouped["artist_recording_rank"] <= top_n].reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_csv, index=False, encoding="utf-8")

    summary = {
        "canonical_scrobbles_csv": str(canonical_scrobbles_csv),
        "labels_csv": str(labels_csv),
        "top_n": int(top_n),
        "min_track_scrobbles": int(min_track_scrobbles),
        "artist_count_with_selected_recordings": int(selected["artist_id"].nunique()),
        "selected_recording_count": int(selected["track_mbid"].nunique()),
        "row_count": int(len(selected)),
    }
    (output_csv.parent / "top_acoustic_recordings_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def cache_path_for(cache_dir: Path, mbid: str) -> Path:
    return cache_dir / f"{mbid}.json"


def fetch_json(url: str, timeout: float, user_agent: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def default_negative_cache_csv(cache_dir: Path) -> Path:
    return cache_dir.parent / "low_level_negative_cache.csv"


def load_negative_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    table = pd.read_csv(path, dtype={"track_mbid": "string", "status": "string"}, low_memory=False)
    if "track_mbid" not in table.columns or "status" not in table.columns:
        raise ValueError(f"Negative cache is missing required columns: {path}")
    cache: dict[str, str] = {}
    for row in table.to_dict(orient="records"):
        mbid = normalize_mbid(row.get("track_mbid"))
        status = str(row.get("status", "")).strip()
        if mbid and status:
            cache[mbid] = status
    return cache


def write_negative_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame.from_records(
        [{"track_mbid": mbid, "status": status} for mbid, status in sorted(cache.items())]
    )
    table.to_csv(path, index=False, encoding="utf-8")


def write_fetch_manifest(manifest_csv: Path, rows: list[dict[str, Any]]) -> None:
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame.from_records(rows)
    manifest.to_csv(manifest_csv, index=False, encoding="utf-8")
    if "status" in manifest.columns:
        summary = manifest["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="count")
        summary.to_csv(manifest_csv.parent / "acousticbrainz_fetch_status_summary.csv", index=False, encoding="utf-8")


def fetch_acousticbrainz(
    *,
    recordings_csv: Path,
    cache_dir: Path,
    manifest_csv: Path,
    negative_cache_csv: Path | None,
    batch_size: int,
    sleep_seconds: float,
    timeout_seconds: float,
    user_agent: str,
) -> None:
    recordings = pd.read_csv(recordings_csv, usecols=["track_mbid"], dtype="string")
    mbids = sorted({normalize_mbid(value) for value in recordings["track_mbid"] if normalize_mbid(value)})
    cache_dir.mkdir(parents=True, exist_ok=True)
    negative_cache_path = negative_cache_csv or default_negative_cache_csv(cache_dir)
    negative_cache = load_negative_cache(negative_cache_path)
    manifest_rows: list[dict[str, Any]] = []

    print(
        f"Loaded {len(negative_cache):,} AcousticBrainz negative-cache entries from {negative_cache_path}",
        flush=True,
    )

    for start in range(0, len(mbids), batch_size):
        batch = mbids[start : start + batch_size]
        missing: list[str] = []
        for mbid in batch:
            if cache_path_for(cache_dir, mbid).exists():
                manifest_rows.append({"track_mbid": mbid, "status": "cached"})
            elif mbid in negative_cache:
                manifest_rows.append(
                    {
                        "track_mbid": mbid,
                        "status": f"{negative_cache[mbid]}_cached",
                    }
                )
            else:
                missing.append(mbid)
        if not missing:
            write_fetch_manifest(manifest_csv, manifest_rows)
            continue
        query = urllib.parse.urlencode({"recording_ids": ";".join(missing)})
        url = f"{API_ROOT}/low-level?{query}"
        print(f"[{start + 1:,}/{len(mbids):,}] fetching {len(missing)} recordings", flush=True)
        try:
            payload = fetch_json(url, timeout_seconds, user_agent)
        except Exception as error:  # noqa: BLE001 - preserve per-recording progress in manifest.
            for mbid in missing:
                manifest_rows.append({"track_mbid": mbid, "status": "request_failed", "error": str(error)})
            write_fetch_manifest(manifest_csv, manifest_rows)
            time.sleep(sleep_seconds)
            continue

        negative_cache_changed = False
        returned = {key for key in payload if key != "mbid_mapping"}
        for mbid in missing:
            document_by_offset = payload.get(mbid)
            if document_by_offset is None:
                negative_cache[mbid] = "not_found"
                negative_cache_changed = True
                manifest_rows.append({"track_mbid": mbid, "status": "not_found"})
                continue
            document = document_by_offset.get("0") if isinstance(document_by_offset, dict) else None
            if not isinstance(document, dict):
                negative_cache[mbid] = "invalid_document"
                negative_cache_changed = True
                manifest_rows.append({"track_mbid": mbid, "status": "invalid_document"})
                continue
            cache_path_for(cache_dir, mbid).write_text(json.dumps(document), encoding="utf-8")
            manifest_rows.append({"track_mbid": mbid, "status": "fetched"})
        for mbid in returned - set(missing):
            manifest_rows.append({"track_mbid": mbid, "status": "returned_unrequested"})
        if negative_cache_changed:
            write_negative_cache(negative_cache_path, negative_cache)
        write_fetch_manifest(manifest_csv, manifest_rows)
        time.sleep(sleep_seconds)

    write_negative_cache(negative_cache_path, negative_cache)
    write_fetch_manifest(manifest_csv, manifest_rows)
    manifest = pd.DataFrame.from_records(manifest_rows)
    summary = manifest["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="count")
    print(summary.to_string(index=False))


def build_artist_feature_table(
    *,
    recordings_csv: Path,
    cache_dir: Path,
    labels_csv: Path,
    output_csv: Path,
    feature_paths: list[str],
    min_recordings_per_artist: int,
) -> pd.DataFrame:
    recordings = pd.read_csv(recordings_csv, dtype={"artist_id": "string", "artist": "string", "track_mbid": "string"})
    labels = pd.read_csv(labels_csv, dtype={"artist_id": "string", "artist_name": "string"})
    label_lookup = labels.set_index("artist_id")

    rows: list[dict[str, Any]] = []
    feature_keys: set[str] = set()
    for row in recordings.to_dict(orient="records"):
        mbid = normalize_mbid(row["track_mbid"])
        if not mbid:
            continue
        path = cache_path_for(cache_dir, mbid)
        if not path.exists():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        features = extract_feature_vector(document, feature_paths)
        if not features:
            continue
        feature_keys.update(features)
        rows.append({
            "artist_id": required_text(row["artist_id"], "artist_id"),
            "artist_name": str(row.get("artist") or label_lookup.loc[row["artist_id"], "artist_name"]),
            "track_mbid": mbid,
            "track_scrobble_count": int(row["scrobble_count"]),
            **features,
        })

    if not rows:
        raise ValueError("No AcousticBrainz feature documents were available for selected recordings.")

    recording_features = pd.DataFrame.from_records(rows).fillna(0.0)
    feature_columns = sorted(feature_keys)
    artist_rows: list[dict[str, Any]] = []
    for artist_id, group in recording_features.groupby("artist_id"):
        if len(group) < min_recordings_per_artist:
            continue
        weights = group["track_scrobble_count"].to_numpy(dtype=float)
        values = group[feature_columns].to_numpy(dtype=float)
        weighted = np.average(values, axis=0, weights=weights)
        artist_rows.append({
            "artist_id": artist_id,
            "artist_name": group["artist_name"].iloc[0],
            "acoustic_recording_count": int(len(group)),
            "acoustic_recording_scrobble_count": int(group["track_scrobble_count"].sum()),
            **{column: float(value) for column, value in zip(feature_columns, weighted)},
        })

    if not artist_rows:
        raise ValueError("No artists met min_recordings_per_artist for acoustic graph construction.")

    artist_features = pd.DataFrame.from_records(artist_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    artist_features.to_csv(output_csv, index=False, encoding="utf-8")
    return artist_features


def build_graph(
    *,
    artist_features: pd.DataFrame,
    output_dir: Path,
    nearest_neighbors: int,
    min_similarity: float,
    sensitivity_thresholds: list[float],
    sensitivity_neighbors: list[int],
) -> None:
    feature_columns = [
        column
        for column in artist_features.columns
        if column not in {"artist_id", "artist_name", "acoustic_recording_count", "acoustic_recording_scrobble_count"}
    ]
    matrix = artist_features[feature_columns].to_numpy(dtype=float)
    matrix = StandardScaler().fit_transform(matrix)
    matrix = normalize(matrix, norm="l2")

    neighbor_grid = sorted(
        {
            int(value)
            for value in sensitivity_neighbors + [nearest_neighbors]
            if int(value) > 0
        }
    )
    max_neighbors = max(neighbor_grid)
    n_neighbors = min(max_neighbors + 1, len(artist_features))
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    neighbors.fit(matrix)
    distances, indices = neighbors.kneighbors(matrix)

    artist_ids = artist_features["artist_id"].astype(str).tolist()

    def candidate_weights_for_k(k_neighbors: int) -> dict[tuple[str, str], float]:
        candidate_weights: dict[tuple[str, str], float] = {}
        for source_index, (row_distances, row_indices) in enumerate(zip(distances, indices)):
            source = artist_ids[source_index]
            neighbor_rank = 0
            for distance, target_index in zip(row_distances, row_indices):
                if source_index == int(target_index):
                    continue
                neighbor_rank += 1
                if neighbor_rank > k_neighbors:
                    break
                similarity = float(1.0 - distance)
                target = artist_ids[int(target_index)]
                a, b = sorted([source, target])
                candidate_weights[(a, b)] = max(candidate_weights.get((a, b), -1.0), similarity)
        return candidate_weights

    sensitivity_rows: list[dict[str, Any]] = []
    threshold_grid = sorted(set(sensitivity_thresholds + [min_similarity]))
    candidate_weights_by_k = {
        k_neighbors: candidate_weights_for_k(k_neighbors)
        for k_neighbors in neighbor_grid
    }
    for k_neighbors, candidate_weights_at_k in candidate_weights_by_k.items():
        for threshold in threshold_grid:
            graph_at_threshold = nx.Graph()
            graph_at_threshold.add_nodes_from(artist_ids)
            graph_at_threshold.add_edges_from(
                edge for edge, similarity in candidate_weights_at_k.items() if similarity >= threshold
            )
            components = sorted(nx.connected_components(graph_at_threshold), key=len, reverse=True)
            largest_node_count = len(components[0]) if components else 0
            active_nodes = [node for node, degree in graph_at_threshold.degree() if degree > 0]
            sensitivity_rows.append(
                {
                    "nearest_neighbors": int(k_neighbors),
                    "min_similarity": float(threshold),
                    "node_count": int(graph_at_threshold.number_of_nodes()),
                    "edge_count": int(graph_at_threshold.number_of_edges()),
                    "active_node_count": int(len(active_nodes)),
                    "isolated_node_count": int(graph_at_threshold.number_of_nodes() - len(active_nodes)),
                    "largest_component_node_count": int(largest_node_count),
                    "largest_component_percentage": float(100.0 * largest_node_count / graph_at_threshold.number_of_nodes()),
                    "mean_degree": float(np.mean([degree for _, degree in graph_at_threshold.degree()])),
                    "density": float(nx.density(graph_at_threshold)),
                }
            )

    candidate_weights = candidate_weights_by_k[int(nearest_neighbors)]
    graph = nx.Graph(graph_type="acoustic_artist_similarity", similarity_metric="cosine")
    for row in artist_features.to_dict(orient="records"):
        graph.add_node(
            str(row["artist_id"]),
            artist_name=str(row["artist_name"]),
            acoustic_recording_count=int(row["acoustic_recording_count"]),
            scrobble_count=int(row["acoustic_recording_scrobble_count"]),
            session_count=0,
            user_count=0,
        )

    for (source, target), similarity in candidate_weights.items():
        if similarity < min_similarity:
            continue
        graph.add_edge(
            source,
            target,
            weight=similarity,
            acoustic_cosine_similarity=similarity,
        )

    if graph.number_of_edges() == 0:
        raise ValueError("Acoustic graph has no edges; lower min_similarity or increase nearest_neighbors.")

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    largest = graph.subgraph(components[0]).copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, output_dir / "graph_all_components.graphml")
    nx.write_graphml(largest, output_dir / "graph_largest_component.graphml")

    nodes = pd.DataFrame.from_records([
        {"artist_id": node, **attrs} for node, attrs in graph.nodes(data=True)
    ])
    edges = pd.DataFrame.from_records([
        {"source": source, "target": target, **attrs} for source, target, attrs in graph.edges(data=True)
    ])
    nodes.to_csv(output_dir / "nodes_acoustic_features.csv", index=False, encoding="utf-8")
    edges.to_csv(output_dir / "edges_acoustic_similarity.csv", index=False, encoding="utf-8")
    sensitivity = pd.DataFrame.from_records(sensitivity_rows)
    sensitivity.to_csv(
        output_dir / "graph_acoustic_knn_threshold_sensitivity.csv",
        index=False,
        encoding="utf-8",
    )
    sensitivity.to_csv(
        output_dir / "graph_acoustic_threshold_sensitivity.csv",
        index=False,
        encoding="utf-8",
    )
    summary = {
        "node_count": int(graph.number_of_nodes()),
        "edge_count": int(graph.number_of_edges()),
        "largest_component_node_count": int(largest.number_of_nodes()),
        "largest_component_edge_count": int(largest.number_of_edges()),
        "largest_component_percentage": float(100.0 * largest.number_of_nodes() / graph.number_of_nodes()),
        "nearest_neighbors": int(nearest_neighbors),
        "min_similarity": float(min_similarity),
        "sensitivity_neighbors": neighbor_grid,
        "sensitivity_thresholds": threshold_grid,
        "feature_count": int(len(feature_columns)),
        "degree_summary": dict(Counter(dict(graph.degree()).values()).most_common(10)),
    }
    (output_dir / "graph_acoustic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

def run_all(args: argparse.Namespace) -> None:
    select_top_recordings(
        canonical_scrobbles_csv=args.canonical_scrobbles_csv,
        labels_csv=args.labels_csv,
        output_csv=args.recordings_csv,
        top_n=args.top_n,
        min_track_scrobbles=args.min_track_scrobbles,
    )
    if args.skip_fetch:
        print("Skipping AcousticBrainz fetch and rebuilding from existing cache.")
    else:
        fetch_acousticbrainz(
            recordings_csv=args.recordings_csv,
            cache_dir=args.cache_dir,
            manifest_csv=args.fetch_manifest_csv,
            negative_cache_csv=args.negative_cache_csv,
            batch_size=args.batch_size,
            sleep_seconds=args.sleep_seconds,
            timeout_seconds=args.timeout_seconds,
            user_agent=args.user_agent,
        )
    artist_features = build_artist_feature_table(
        recordings_csv=args.recordings_csv,
        cache_dir=args.cache_dir,
        labels_csv=args.labels_csv,
        output_csv=args.artist_features_csv,
        feature_paths=args.feature_paths,
        min_recordings_per_artist=args.min_recordings_per_artist,
    )
    build_graph(
        artist_features=artist_features,
        output_dir=args.output_dir,
        nearest_neighbors=args.nearest_neighbors,
        min_similarity=args.min_similarity,
        sensitivity_thresholds=args.sensitivity_thresholds,
        sensitivity_neighbors=args.sensitivity_neighbors,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an artist similarity graph from AcousticBrainz low-level recording features.")
    parser.add_argument("canonical_scrobbles_csv", type=Path)
    parser.add_argument("labels_csv", type=Path)
    parser.add_argument("--recordings-csv", type=Path, default=Path("data/interim/acoustic/top_artist_recordings_top10.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/acousticbrainz/low_level"))
    parser.add_argument("--fetch-manifest-csv", type=Path, default=Path("data/interim/acoustic/acousticbrainz_fetch_manifest.csv"))
    parser.add_argument("--negative-cache-csv", type=Path, default=None)
    parser.add_argument("--artist-features-csv", type=Path, default=Path("data/interim/acoustic/artist_acoustic_features_top10.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/acoustic_graphs/top10_knn20_min0_40"))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-track-scrobbles", type=int, default=2)
    parser.add_argument("--min-recordings-per-artist", type=int, default=1)
    parser.add_argument("--nearest-neighbors", type=int, default=20)
    parser.add_argument("--min-similarity", type=float, default=0.40)
    parser.add_argument(
        "--sensitivity-thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.35, 0.4, 0.45, 0.5, 0.6],
    )
    parser.add_argument(
        "--sensitivity-neighbors",
        type=int,
        nargs="+",
        default=[10, 20, 30, 50],
    )
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--user-agent", default="music-graphs acoustic comparison (research; contact: local)")
    parser.add_argument("--feature-paths", nargs="+", default=DEFAULT_FEATURE_PATHS)
    return parser.parse_args()


if __name__ == "__main__":
    run_all(parse_args())
