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
    StageSpec(
        "build_acoustic_graph",
        "Build an artist graph from AcousticBrainz low-level recording features.",
        optional_group="acoustic_reference",
    ),
    StageSpec("run_modularity_baselines", "Run Louvain and Leiden on the behavioral artist graph."),
    StageSpec("summarize_modularity_grid", "Summarize the behavioral modularity-resolution analysis."),
    StageSpec(
        "run_acoustic_modularity_baselines",
        "Run Louvain and Leiden on the acoustic artist graph.",
        optional_group="acoustic_reference",
    ),
    StageSpec("run_node2vec", "Train the behavioral Node2Vec p/q grid and cluster embeddings."),
    StageSpec("select_node2vec", "Select and validate behavioral Node2Vec walk-bias parameters by mean silhouette."),
    StageSpec(
        "run_acoustic_node2vec",
        "Train Node2Vec embeddings on the acoustic artist graph.",
        optional_group="acoustic_reference",
    ),
    StageSpec(
        "select_acoustic_node2vec",
        "Select acoustic Node2Vec walk-bias parameters by mean silhouette.",
        optional_group="acoustic_reference",
    ),
    StageSpec("run_smooth2_control", "Run raw, smooth1, and smooth2 controls on the behavioral graph."),
    StageSpec(
        "run_acoustic_smooth2_control",
        "Run raw, smooth1, and smooth2 controls on the acoustic artist graph.",
        optional_group="acoustic_reference",
    ),
    StageSpec(
        "graph_autoencoder",
        "Run the behavioral Graph Autoencoder embedding benchmark.",
        optional_group="gae",
    ),
    StageSpec(
        "run_acoustic_gae",
        "Train Graph Autoencoder embeddings on the acoustic artist graph.",
        optional_group="acoustic_reference",
    ),
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
        "ablations",
        "Run optional ablation studies.",
        optional_group="ablations",
    ),
]

ALL_STAGES = CORE_STAGES + OPTIONAL_STAGES
