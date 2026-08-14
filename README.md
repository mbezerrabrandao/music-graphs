# Music Artist Graphs from Last.fm

Reproducible pipeline for constructing and evaluating music-artist graphs from Last.fm listening histories. The final paper configuration compares two genre-free graph views:

- a **behavioral graph** built from sequential listening transitions;
- an **acoustic graph** built from AcousticBrainz low-level descriptor similarity.

MusicBrainz genres are not used to build either graph. They are used only after graph construction as external reference labels for community evaluation and as supervised labels in the downstream genre-inference benchmark.

## Repository Contents

This public version keeps only the materials needed to inspect and rerun the method without exposing private data or generated artifacts:

```text
configs/local_large.yaml                         final paper configuration
data/sample/recenttracks_sample.csv              tiny Last.fm sample input
sensitivity_freeze_overview.ipynb                 compact sensitivity/freezing notebook
reviewer_01_manual_validation_top3_sample_seed_42.csv
reviewer_02_manual_validation_top3_sample_seed_42.csv
src/                                              pipeline source code
scripts/                                          command helpers
tests/                                            lightweight tests
```

Generated outputs, raw/private data, caches, figures, tables, and exploratory notebooks are intentionally ignored by Git.

## Pipeline Overview

```text
Last.fm exports
  -> audit and canonicalization
  -> listening sessions
  -> behavioral artist graph
  -> MusicBrainz external labels
  -> AcousticBrainz low-level descriptors
  -> acoustic artist graph
  -> Louvain / Leiden / Node2Vec / GAE comparisons
  -> raw and smoothed graph-feature controls
  -> genre-inference benchmark
  -> manual validation candidates
```

The final configuration freezes the following main choices:

```text
session threshold = 60 minutes
behavioral relation mode = sequential window
behavioral window = 5 scrobbles
behavioral edge weight = multi_user_decayed_shared_session_cosine
minimum shared sessions = 2
minimum artist scrobbles = 3
acoustic graph = top10 recordings, kNN=20, minimum cosine similarity=0.40
Node2Vec K-Means sensitivity = k25, k40, k54, k75, k100, k150
GAE K-Means sensitivity = k25, k40, k54, k75, k100, k150
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m pip install -e ".[community]"
```

For graph autoencoder stages, install a PyTorch build compatible with your machine before running the full pipeline.

## Data

A tiny sample file is provided in `data/sample/` only to illustrate the input shape. To run the full pipeline, place private Last.fm exports under:

```text
data/raw/lastfm/
```

Multi-user exports named like `recenttracks-username-exportid.csv` are loaded together. The pipeline infers `user_id` from the file name, sessionizes each user independently, and strengthens artist relations independently observed across multiple users.

Do not commit personal exports, canonical tables, API caches, generated graphs, embeddings, figures, tables, or inferred metadata.

## Inspect the Execution Plan

```bash
python -m music_graphs.cli \
  --config configs/local_large.yaml \
  --plan
```

## Run the Pipeline

MusicBrainz requests require a contact string.

```bash
export MUSICBRAINZ_CONTACT="your-email@example.com"

python -m music_graphs.cli \
  --config configs/local_large.yaml
```

Stages are resumable. Completed stages are skipped when inputs, parameters, and declared outputs still match the saved fingerprint.

To restart from one stage:

```bash
python -m music_graphs.cli \
  --config configs/local_large.yaml \
  --from-stage run_node2vec \
  --force-stage run_node2vec
```

## Sensitivity Notebook

`sensitivity_freeze_overview.ipynb` summarizes the sensitivity evidence used to freeze graph-construction parameters, clustering hyperparameters, and genre-inference settings for the paper.

## Manual Validation CSVs

The completed reviewer worksheets are copied to the repository root:

```text
reviewer_01_manual_validation_top3_sample_seed_42.csv
reviewer_02_manual_validation_top3_sample_seed_42.csv
reviewer_03_manual_validation_top3_sample_seed_42.csv
```

They correspond to the qualitative top-3 genre-suggestion audit. 
