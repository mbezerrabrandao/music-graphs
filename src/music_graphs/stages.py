from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    description: str
    optional_group: str | None = None


CORE_STAGES = [
    StageSpec("audit_export", "Audit the immutable Last.fm export."),
    StageSpec("canonicalize", "Canonicalize scrobbles and artist identifiers."),
    StageSpec("session_audit", "Audit inter-scrobble gaps and candidate session thresholds."),
    StageSpec("select_session_threshold", "Select and validate the session threshold from the gap audit."),
    StageSpec("build_sessions", "Build listening sessions."),
    StageSpec("build_behavioral_graph", "Construct and materialize the behavioral graph."),
    StageSpec("summarize_graph_support", "Summarize the repeated-session edge-support decision."),
    StageSpec("fetch_musicbrainz", "Fetch and cache MusicBrainz artist genres."),
    StageSpec("prepare_genre_labels", "Prepare external MusicBrainz labels."),
    StageSpec("run_modularity_baselines", "Run the Louvain and Leiden resolution grid."),
    StageSpec("summarize_modularity_grid", "Summarize the structural modularity-resolution analysis."),
    StageSpec("run_node2vec", "Train the balanced Node2Vec p/q grid and cluster embeddings."),
    StageSpec("select_node2vec", "Select and validate Node2Vec walk-bias parameters by mean silhouette."),
    StageSpec("run_smooth2_control", "Run raw, smooth1, and smooth2 fixed-propagation controls."),
    StageSpec("uniform_evaluation", "Evaluate frozen balanced core methods uniformly."),
    StageSpec("genre_inference", "Run exploratory downstream genre inference."),
    StageSpec("selection_report", "Compile methodological choices and generated evidence."),
]

OPTIONAL_STAGES = [
    StageSpec(
        "semantic_reference",
        "Build and evaluate the genre-informed semantic reference.",
        optional_group="semantic_reference",
    ),
    StageSpec(
        "graph_autoencoder",
        "Run the optional Graph Autoencoder benchmark.",
        optional_group="gae",
    ),
    StageSpec(
        "ablations",
        "Run optional ablation studies.",
        optional_group="ablations",
    ),
]

ALL_STAGES = CORE_STAGES + OPTIONAL_STAGES
