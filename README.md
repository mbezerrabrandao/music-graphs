# Music Artist Graphs from Last.fm

Reproducible pipeline for building, evaluating, and reusing music-artist graphs derived from a Last.fm listening history.

## Core research path

```text
Last.fm export
  ↓
audit and canonicalization
  ↓
listening sessions
  ↓
behavioral artist graph
  ↓
Louvain + Leiden
  ↓
Node2Vec embeddings
  ↓
smooth2 structural control
  ↓
uniform evaluation
  ↓
exploratory missing-genre inference
```

The default public pipeline centers on Node2Vec as the main reusable graph representation. Louvain and Leiden remain structural community-detection baselines. The fixed two-step smoothing control documents how much signal can be explained without representation learning.

The Graph Autoencoder and the genre-informed semantic reference are optional research extensions and are not required for the default reproduction path.

## Methodological separation

The behavioral graph and unsupervised graph representations are constructed without MusicBrainz genre labels.

MusicBrainz genres are used only:

1. as incomplete external metadata for evaluation;
2. as supervised labels in the explicitly downstream exploratory inference task.

## Installation

Python 3.10 or newer is supported.

```powershell
py -m pip install -e ".[community]"
```

## Private local data

Place a Last.fm export at:

```text
data/raw/lastfm/recenttracks.csv
```

Do not commit personal exports, canonical tables, MusicBrainz caches, personal graph files, generated results, or inferred metadata by default.

See `data/README.md`.

## Inspect the execution plan

```powershell
py -m music_graphs.cli `
  --config "configs\paper.yaml" `
  --plan
```

## Run the pipeline

```powershell
$env:MUSICBRAINZ_CONTACT = "your-email@example.com"

py -m music_graphs.cli `
  --config "configs\paper.yaml"
```

Stages are resumable. Completed stages are skipped when their inputs, parameters, and declared outputs still match the saved fingerprint.

To restart from one stage:

```powershell
py -m music_graphs.cli `
  --config "configs\paper.yaml" `
  --from-stage run_node2vec `
  --force-stage run_node2vec
```

## Automatic methodological report

The pipeline writes:

```text
results/selection_report.md
```

This report records:

- tested candidates;
- selected parameters;
- selection rules;
- evidence files;
- whether external genres were used.

## Frozen manuscript configuration

The default `configs/paper.yaml` reproduces the manuscript path:

```text
session threshold = 60 minutes
behavioral relation mode = sequential_k5
minimum shared sessions = 2
edge weight = shared_session_cosine
minimum artist scrobbles = 3
Node2Vec balanced scale = k54
Node2Vec walk bias = p2.0, q2.0
downstream representation = Node2Vec
```

