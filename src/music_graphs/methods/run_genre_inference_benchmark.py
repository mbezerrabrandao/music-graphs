from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_EXPECTED_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_CV_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_FAMILIES = [
    "raw",
    "smooth1",
    "smooth2",
    "node2vec",
    "gae",
]

FAMILY_ORDER = {
    "raw": 0,
    "smooth1": 1,
    "smooth2": 2,
    "node2vec": 3,
    "gae": 4,
}

FAMILY_DISPLAY_NAMES = {
    "raw": "Behavioral raw",
    "smooth1": "Behavioral smooth1",
    "smooth2": "Behavioral smooth2",
    "node2vec": "Node2Vec",
    "gae": "GAE behavioral",
}


@dataclass(frozen=True)
class RepresentationFile:
    family: str
    representation_id: str
    path: Path
    representation_seed: int | None


def required_text(
    value: Any,
    field_name: str,
) -> str:
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


def required_int(
    value: Any,
    field_name: str,
) -> int:
    if value is None or pd.isna(value):
        raise ValueError(
            f"Missing required integer value for: {field_name}"
        )

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise ValueError(
            f"Invalid integer value for {field_name}: {value!r}"
        ) from error


def read_labels(
    labels_csv: Path,
) -> pd.DataFrame:
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Label table does not exist: {labels_csv}"
        )

    labels = pd.read_csv(
        labels_csv,
        dtype={
            "artist_id": "string",
            "artist_name": "string",
            "artist_mbid": "string",
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

    missing_columns = sorted(
        required_columns - set(
            labels.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Label table is missing required columns: "
            + ", ".join(
                missing_columns
            )
        )

    if "artist_mbid" not in labels.columns:
        labels[
            "artist_mbid"
        ] = pd.NA

    for column in [
        "artist_id",
        "artist_name",
        "artist_mbid",
        "metadata_group",
        "dominant_genre_normalized",
    ]:
        labels[
            column
        ] = (
            labels[
                column
            ]
            .astype(
                "string"
            )
            .str.strip()
            .replace(
                "",
                pd.NA,
            )
        )

    if labels[
        "artist_id"
    ].isna().any():
        raise ValueError(
            "Label table contains missing artist IDs."
        )

    if labels[
        "artist_id"
    ].duplicated().any():
        raise ValueError(
            "Label table contains duplicate artist IDs."
        )

    return (
        labels.sort_values(
            "artist_id"
        )
        .reset_index(
            drop=True
        )
    )


def parse_seed_from_filename(
    path: Path,
) -> int:
    match = re.search(
        r"_seed_(\d+)\.csv$",
        path.name,
    )

    if match is None:
        raise ValueError(
            "Could not parse representation seed from filename: "
            f"{path}"
        )

    return int(
        match.group(
            1
        )
    )


def require_expected_seed_files(
    paths: list[Path],
    expected_seeds: list[int],
    family: str,
) -> list[
    tuple[
        Path,
        int,
    ]
]:
    discovered: dict[
        int,
        Path,
    ] = {}

    for path in paths:
        seed = parse_seed_from_filename(
            path
        )

        if seed in discovered:
            raise ValueError(
                f"Found multiple {family} embedding files for seed {seed}: "
                f"{discovered[seed]} and {path}"
            )

        discovered[
            seed
        ] = path

    missing_seeds = sorted(
        set(
            expected_seeds
        )
        - set(
            discovered
        )
    )

    unexpected_seeds = sorted(
        set(
            discovered
        )
        - set(
            expected_seeds
        )
    )

    if missing_seeds:
        raise FileNotFoundError(
            f"Missing {family} embeddings for seeds: {missing_seeds}"
        )

    if unexpected_seeds:
        raise ValueError(
            f"Unexpected {family} embedding seeds: {unexpected_seeds}"
        )

    return [
        (
            discovered[
                seed
            ],
            seed,
        )
        for seed in sorted(
            expected_seeds
        )
    ]


def discover_representations(
    *,
    families: list[str],
    behavioral_control_embeddings_dir: Path,
    node2vec_embeddings_dir: Path,
    gae_embeddings_dir: Path,
    expected_seeds: list[int],
) -> list[
    RepresentationFile
]:
    representations: list[
        RepresentationFile
    ] = []

    for family in families:
        if family in {
            "raw",
            "smooth1",
            "smooth2",
        }:
            path = (
                behavioral_control_embeddings_dir
                / f"behavioral_features_{family}.csv"
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing behavioral-control embeddings: {path}"
                )

            representations.append(
                RepresentationFile(
                    family=family,
                    representation_id=(
                        f"behavioral_features_{family}"
                    ),
                    path=path,
                    representation_seed=None,
                )
            )

        elif family == "node2vec":
            if not node2vec_embeddings_dir.exists():
                raise FileNotFoundError(
                    "Node2Vec embeddings directory does not exist: "
                    f"{node2vec_embeddings_dir}"
                )

            paths = sorted(
                node2vec_embeddings_dir.glob(
                    "node2vec_p_2_0_q_2_0_seed_*.csv"
                )
            )

            if not paths:
                raise FileNotFoundError(
                    "No selected Node2Vec embedding files found under: "
                    f"{node2vec_embeddings_dir}"
                )

            for path, seed in require_expected_seed_files(
                paths=paths,
                expected_seeds=expected_seeds,
                family="Node2Vec",
            ):
                representations.append(
                    RepresentationFile(
                        family=family,
                        representation_id=(
                            f"node2vec_p_2_0_q_2_0_seed_{seed}"
                        ),
                        path=path,
                        representation_seed=seed,
                    )
                )

        elif family == "gae":
            if not gae_embeddings_dir.exists():
                raise FileNotFoundError(
                    "GAE embeddings directory does not exist: "
                    f"{gae_embeddings_dir}"
                )

            paths = sorted(
                gae_embeddings_dir.glob(
                    "*.csv"
                )
            )

            if not paths:
                raise FileNotFoundError(
                    "No selected GAE embedding files found under: "
                    f"{gae_embeddings_dir}"
                )

            for path, seed in require_expected_seed_files(
                paths=paths,
                expected_seeds=expected_seeds,
                family="GAE",
            ):
                representations.append(
                    RepresentationFile(
                        family=family,
                        representation_id=path.stem,
                        path=path,
                        representation_seed=seed,
                    )
                )

        else:
            raise ValueError(
                f"Unsupported representation family: {family}"
            )

    return representations


def load_embedding_table(
    representation: RepresentationFile,
    labels: pd.DataFrame,
) -> tuple[
    list[str],
    np.ndarray,
]:
    if not representation.path.exists():
        raise FileNotFoundError(
            f"Embedding file does not exist: {representation.path}"
        )

    table = pd.read_csv(
        representation.path,
        dtype={
            "artist_id": "string",
        },
        low_memory=False,
    )

    if "artist_id" not in table.columns:
        raise ValueError(
            f"Embedding table is missing artist_id: {representation.path}"
        )

    table[
        "artist_id"
    ] = (
        table[
            "artist_id"
        ]
        .astype(
            "string"
        )
        .str.strip()
        .replace(
            "",
            pd.NA,
        )
    )

    if table[
        "artist_id"
    ].isna().any():
        raise ValueError(
            f"Embedding table contains missing artist IDs: "
            f"{representation.path}"
        )

    if table[
        "artist_id"
    ].duplicated().any():
        raise ValueError(
            f"Embedding table contains duplicate artist IDs: "
            f"{representation.path}"
        )

    feature_columns = [
        column
        for column in table.columns
        if column
        != "artist_id"
    ]

    if not feature_columns:
        raise ValueError(
            f"Embedding table has no feature columns: {representation.path}"
        )

    for column in feature_columns:
        table[
            column
        ] = pd.to_numeric(
            table[
                column
            ],
            errors="raise",
        )

    ordered_ids = (
        labels[
            "artist_id"
        ]
        .astype(
            str
        )
        .tolist()
    )

    table_indexed = table.set_index(
        "artist_id"
    )

    missing_ids = sorted(
        set(
            ordered_ids
        )
        - set(
            table_indexed.index.astype(
                str
            )
        )
    )

    if missing_ids:
        raise ValueError(
            f"Embedding table {representation.path} is missing graph nodes. "
            f"Example: {missing_ids[0]}"
        )

    ordered_table = table_indexed.loc[
        ordered_ids,
        feature_columns,
    ]

    matrix = ordered_table.to_numpy(
        dtype=np.float64,
    )

    if not np.isfinite(
        matrix
    ).all():
        raise ValueError(
            f"Embedding table contains non-finite values: "
            f"{representation.path}"
        )

    return (
        ordered_ids,
        matrix,
    )


def retained_training_labels(
    labels: pd.DataFrame,
    minimum_genre_support: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    pd.DataFrame,
]:
    known = (
        labels.loc[
            labels[
                "dominant_genre_normalized"
            ].notna()
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    support_table = (
        known[
            "dominant_genre_normalized"
        ]
        .value_counts()
        .rename_axis(
            "genre"
        )
        .reset_index(
            name="artist_count",
        )
    )

    support_table[
        "retained_for_supervised_benchmark"
    ] = (
        support_table[
            "artist_count"
        ]
        >= minimum_genre_support
    )

    retained_genres = (
        support_table.loc[
            support_table[
                "retained_for_supervised_benchmark"
            ],
            "genre",
        ]
        .astype(
            str
        )
        .sort_values()
        .tolist()
    )

    if len(
        retained_genres
    ) < 2:
        raise ValueError(
            "Supervised benchmark requires at least two retained genres."
        )

    retained_mask = (
        labels[
            "dominant_genre_normalized"
        ]
        .astype(
            "string"
        )
        .isin(
            retained_genres
        )
        .to_numpy()
    )

    retained_indices = np.flatnonzero(
        retained_mask
    )

    targets = (
        labels.iloc[
            retained_indices
        ][
            "dominant_genre_normalized"
        ]
        .astype(
            str
        )
        .to_numpy()
    )

    minimum_count = int(
        pd.Series(
            targets
        )
        .value_counts()
        .min()
    )

    if minimum_count < 2:
        raise ValueError(
            "At least one retained genre has fewer than two artists."
        )

    return (
        retained_indices,
        targets,
        retained_genres,
        support_table,
    )


def build_classifier(
    *,
    regularization_c: float,
    max_iter: int,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=regularization_c,
                    class_weight="balanced",
                    max_iter=max_iter,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def safe_top_k(
    requested_k: int,
    class_count: int,
) -> int:
    return max(
        1,
        min(
            requested_k,
            class_count,
        ),
    )


def classification_metrics(
    *,
    truth: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    return {
        "accuracy": float(
            accuracy_score(
                truth,
                predicted,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                truth,
                predicted,
            )
        ),
        "macro_f1": float(
            f1_score(
                truth,
                predicted,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                truth,
                predicted,
                average="weighted",
                zero_division=0,
            )
        ),
        "top_k_accuracy": float(
            top_k_accuracy_score(
                truth,
                probabilities,
                k=safe_top_k(
                    top_k,
                    len(
                        classes
                    ),
                ),
                labels=classes,
            )
        ),
        "log_loss": float(
            log_loss(
                truth,
                probabilities,
                labels=classes,
            )
        ),
    }


def evaluate_one_representation(
    *,
    representation: RepresentationFile,
    feature_matrix: np.ndarray,
    retained_indices: np.ndarray,
    targets: np.ndarray,
    cv_seeds: list[int],
    cv_folds: int,
    regularization_c: float,
    max_iter: int,
    top_k: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    x = feature_matrix[
        retained_indices
    ]

    target_counts = (
        pd.Series(
            targets
        )
        .value_counts()
    )

    if int(
        target_counts.min()
    ) < cv_folds:
        raise ValueError(
            "Cross-validation fold count exceeds the minimum retained "
            f"genre support for representation {representation.representation_id}."
        )

    fold_records: list[
        dict[str, Any]
    ] = []

    repeat_records: list[
        dict[str, Any]
    ] = []

    for cv_seed in cv_seeds:
        splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=cv_seed,
        )

        out_of_fold_predicted = np.empty(
            len(
                targets
            ),
            dtype=object,
        )

        global_classes = np.asarray(
            sorted(
                set(
                    targets.tolist()
                )
            ),
            dtype=object,
        )

        out_of_fold_probabilities = np.zeros(
            (
                len(
                    targets
                ),
                len(
                    global_classes
                ),
            ),
            dtype=np.float64,
        )

        global_class_to_index = {
            class_name: index
            for index, class_name in enumerate(
                global_classes
            )
        }

        for fold_index, (
            train_indices,
            test_indices,
        ) in enumerate(
            splitter.split(
                x,
                targets,
            ),
            start=1,
        ):
            classifier = build_classifier(
                regularization_c=regularization_c,
                max_iter=max_iter,
            )

            classifier.fit(
                x[
                    train_indices
                ],
                targets[
                    train_indices
                ],
            )

            predicted = classifier.predict(
                x[
                    test_indices
                ]
            )

            probabilities = classifier.predict_proba(
                x[
                    test_indices
                ]
            )

            classes = classifier.named_steps[
                "classifier"
            ].classes_

            aligned_probabilities = np.zeros(
                (
                    len(
                        test_indices
                    ),
                    len(
                        global_classes
                    ),
                ),
                dtype=np.float64,
            )

            for local_index, class_name in enumerate(
                classes
            ):
                aligned_probabilities[
                    :,
                    global_class_to_index[
                        str(
                            class_name
                        )
                    ],
                ] = probabilities[
                    :,
                    local_index
                ]

            out_of_fold_predicted[
                test_indices
            ] = predicted

            out_of_fold_probabilities[
                test_indices
            ] = aligned_probabilities

            metrics = classification_metrics(
                truth=targets[
                    test_indices
                ],
                predicted=predicted,
                probabilities=aligned_probabilities,
                classes=global_classes,
                top_k=top_k,
            )

            fold_records.append(
                {
                    "family_order": int(
                        FAMILY_ORDER[
                            representation.family
                        ]
                    ),
                    "family": (
                        representation.family
                    ),
                    "family_display_name": (
                        FAMILY_DISPLAY_NAMES[
                            representation.family
                        ]
                    ),
                    "representation_id": (
                        representation.representation_id
                    ),
                    "representation_path": str(
                        representation.path
                    ),
                    "representation_seed": (
                        representation.representation_seed
                    ),
                    "cv_seed": int(
                        cv_seed
                    ),
                    "fold_index": int(
                        fold_index
                    ),
                    "train_node_count": int(
                        len(
                            train_indices
                        )
                    ),
                    "test_node_count": int(
                        len(
                            test_indices
                        )
                    ),
                    **metrics,
                }
            )

        repeat_metrics = classification_metrics(
            truth=targets,
            predicted=out_of_fold_predicted,
            probabilities=out_of_fold_probabilities,
            classes=global_classes,
            top_k=top_k,
        )

        repeat_records.append(
            {
                "family_order": int(
                    FAMILY_ORDER[
                        representation.family
                    ]
                ),
                "family": representation.family,
                "family_display_name": (
                    FAMILY_DISPLAY_NAMES[
                        representation.family
                    ]
                ),
                "representation_id": (
                    representation.representation_id
                ),
                "representation_path": str(
                    representation.path
                ),
                "representation_seed": (
                    representation.representation_seed
                ),
                "cv_seed": int(
                    cv_seed
                ),
                "evaluated_node_count": int(
                    len(
                        targets
                    )
                ),
                "retained_genre_count": int(
                    len(
                        global_classes
                    )
                ),
                **repeat_metrics,
            }
        )

    return (
        pd.DataFrame.from_records(
            fold_records
        ),
        pd.DataFrame.from_records(
            repeat_records
        ),
    )


def summarize_cv_runs(
    cv_runs: pd.DataFrame,
) -> pd.DataFrame:
    return (
        cv_runs.groupby(
            [
                "family_order",
                "family",
                "family_display_name",
            ],
            as_index=False,
        )
        .agg(
            evaluation_repeat_count=(
                "representation_id",
                "size",
            ),
            representation_count=(
                "representation_id",
                "nunique",
            ),
            evaluated_node_count=(
                "evaluated_node_count",
                "mean",
            ),
            retained_genre_count=(
                "retained_genre_count",
                "mean",
            ),
            accuracy_mean=(
                "accuracy",
                "mean",
            ),
            accuracy_std=(
                "accuracy",
                "std",
            ),
            balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean",
            ),
            balanced_accuracy_std=(
                "balanced_accuracy",
                "std",
            ),
            macro_f1_mean=(
                "macro_f1",
                "mean",
            ),
            macro_f1_std=(
                "macro_f1",
                "std",
            ),
            weighted_f1_mean=(
                "weighted_f1",
                "mean",
            ),
            weighted_f1_std=(
                "weighted_f1",
                "std",
            ),
            top_k_accuracy_mean=(
                "top_k_accuracy",
                "mean",
            ),
            top_k_accuracy_std=(
                "top_k_accuracy",
                "std",
            ),
            log_loss_mean=(
                "log_loss",
                "mean",
            ),
            log_loss_std=(
                "log_loss",
                "std",
            ),
        )
        .fillna(
            0.0
        )
        .sort_values(
            [
                "macro_f1_mean",
                "balanced_accuracy_mean",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def normalize_entropy(
    probabilities: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    entropy = -np.sum(
        clipped
        * np.log(
            clipped
        ),
        axis=1,
    )

    maximum_entropy = math.log(
        probabilities.shape[
            1
        ]
    )

    if maximum_entropy <= 0:
        return np.zeros(
            len(
                probabilities
            ),
            dtype=np.float64,
        )

    return (
        entropy
        / maximum_entropy
    )


def predict_unknown_for_family(
    *,
    family: str,
    representations: list[
        RepresentationFile
    ],
    labels: pd.DataFrame,
    feature_matrices: dict[
        str,
        np.ndarray,
    ],
    retained_indices: np.ndarray,
    targets: np.ndarray,
    retained_genres: list[str],
    regularization_c: float,
    max_iter: int,
) -> pd.DataFrame:
    unknown_indices = np.flatnonzero(
        labels[
            "dominant_genre_normalized"
        ]
        .isna()
        .to_numpy()
    )

    if len(
        unknown_indices
    ) == 0:
        raise ValueError(
            "No unlabeled graph nodes are available for inference."
        )

    class_names = np.asarray(
        sorted(
            retained_genres
        ),
        dtype=object,
    )

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(
            class_names
        )
    }

    probability_sum = np.zeros(
        (
            len(
                unknown_indices
            ),
            len(
                class_names
            ),
        ),
        dtype=np.float64,
    )

    for representation in representations:
        feature_matrix = feature_matrices[
            representation.representation_id
        ]

        classifier = build_classifier(
            regularization_c=regularization_c,
            max_iter=max_iter,
        )

        classifier.fit(
            feature_matrix[
                retained_indices
            ],
            targets,
        )

        probabilities = classifier.predict_proba(
            feature_matrix[
                unknown_indices
            ]
        )

        local_classes = classifier.named_steps[
            "classifier"
        ].classes_

        aligned_probabilities = np.zeros_like(
            probability_sum
        )

        for local_index, class_name in enumerate(
            local_classes
        ):
            aligned_probabilities[
                :,
                class_to_index[
                    str(
                        class_name
                    )
                ],
            ] = probabilities[
                :,
                local_index
            ]

        probability_sum += aligned_probabilities

    probabilities = (
        probability_sum
        / len(
            representations
        )
    )

    sorted_indices = np.argsort(
        -probabilities,
        axis=1,
    )

    entropy = normalize_entropy(
        probabilities
    )

    rows: list[
        dict[str, Any]
    ] = []

    unknown_metadata = labels.iloc[
        unknown_indices
    ].reset_index(
        drop=True
    )

    for row_index, metadata in unknown_metadata.iterrows():
        top_indices = sorted_indices[
            row_index,
            :3,
        ]

        top_probabilities = probabilities[
            row_index,
            top_indices,
        ]

        top_one_probability = float(
            top_probabilities[
                0
            ]
        )

        margin = float(
            top_probabilities[
                0
            ]
            - top_probabilities[
                1
            ]
        )

        if (
            top_one_probability
            >= 0.50
            and margin
            >= 0.15
        ):
            confidence_band = "higher"

        elif (
            top_one_probability
            >= 0.30
            and margin
            >= 0.05
        ):
            confidence_band = "moderate"

        else:
            confidence_band = "lower"

        rows.append(
            {
                "family_order": int(
                    FAMILY_ORDER[
                        family
                    ]
                ),
                "family": family,
                "family_display_name": (
                    FAMILY_DISPLAY_NAMES[
                        family
                    ]
                ),
                "artist_id": required_text(
                    metadata[
                        "artist_id"
                    ],
                    "artist_id",
                ),
                "artist_name": required_text(
                    metadata[
                        "artist_name"
                    ],
                    "artist_name",
                ),
                "artist_mbid": (
                    ""
                    if pd.isna(
                        metadata[
                            "artist_mbid"
                        ]
                    )
                    else str(
                        metadata[
                            "artist_mbid"
                        ]
                    )
                ),
                "metadata_group": required_text(
                    metadata[
                        "metadata_group"
                    ],
                    "metadata_group",
                ),
                "prediction_top1_genre": str(
                    class_names[
                        top_indices[
                            0
                        ]
                    ]
                ),
                "prediction_top1_probability": (
                    top_one_probability
                ),
                "prediction_top2_genre": str(
                    class_names[
                        top_indices[
                            1
                        ]
                    ]
                ),
                "prediction_top2_probability": float(
                    top_probabilities[
                        1
                    ]
                ),
                "prediction_top3_genre": str(
                    class_names[
                        top_indices[
                            2
                        ]
                    ]
                ),
                "prediction_top3_probability": float(
                    top_probabilities[
                        2
                    ]
                ),
                "top1_top2_probability_margin": margin,
                "normalized_entropy": float(
                    entropy[
                        row_index
                    ]
                ),
                "confidence_band_heuristic": (
                    confidence_band
                ),
                "ensemble_representation_count": int(
                    len(
                        representations
                    )
                ),
            }
        )

    return (
        pd.DataFrame.from_records(
            rows
        )
        .sort_values(
            [
                "confidence_band_heuristic",
                "prediction_top1_probability",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def run_benchmark(
    *,
    labels_csv: Path,
    behavioral_control_embeddings_dir: Path,
    node2vec_embeddings_dir: Path,
    gae_embeddings_dir: Path,
    output_dir: Path,
    families: list[str],
    expected_seeds: list[int],
    cv_seeds: list[int],
    cv_folds: int,
    minimum_genre_support: int,
    regularization_c: float,
    max_iter: int,
    top_k: int,
) -> None:
    if minimum_genre_support < cv_folds:
        raise ValueError(
            "--minimum-genre-support must be at least --cv-folds "
            "to support stratified evaluation."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Reading label coverage table: {labels_csv}"
    )

    labels = read_labels(
        labels_csv
    )

    (
        retained_indices,
        targets,
        retained_genres,
        support_table,
    ) = retained_training_labels(
        labels=labels,
        minimum_genre_support=(
            minimum_genre_support
        ),
    )

    print(
        "Discovering frozen representation files."
    )

    representations = discover_representations(
        families=families,
        behavioral_control_embeddings_dir=(
            behavioral_control_embeddings_dir
        ),
        node2vec_embeddings_dir=(
            node2vec_embeddings_dir
        ),
        gae_embeddings_dir=(
            gae_embeddings_dir
        ),
        expected_seeds=(
            expected_seeds
        ),
    )

    representation_manifest = pd.DataFrame.from_records(
        [
            {
                "family_order": int(
                    FAMILY_ORDER[
                        representation.family
                    ]
                ),
                "family": representation.family,
                "family_display_name": (
                    FAMILY_DISPLAY_NAMES[
                        representation.family
                    ]
                ),
                "representation_id": (
                    representation.representation_id
                ),
                "representation_seed": (
                    representation.representation_seed
                ),
                "path": str(
                    representation.path
                ),
            }
            for representation in (
                representations
            )
        ]
    )

    feature_matrices: dict[
        str,
        np.ndarray,
    ] = {}

    all_fold_metrics: list[
        pd.DataFrame
    ] = []

    all_cv_runs: list[
        pd.DataFrame
    ] = []

    for representation in representations:
        print()
        print(
            "Evaluating representation: "
            f"{representation.representation_id}"
        )

        (
            _,
            feature_matrix,
        ) = load_embedding_table(
            representation=representation,
            labels=labels,
        )

        feature_matrices[
            representation.representation_id
        ] = feature_matrix

        (
            fold_metrics,
            cv_runs,
        ) = evaluate_one_representation(
            representation=representation,
            feature_matrix=feature_matrix,
            retained_indices=retained_indices,
            targets=targets,
            cv_seeds=cv_seeds,
            cv_folds=cv_folds,
            regularization_c=regularization_c,
            max_iter=max_iter,
            top_k=top_k,
        )

        all_fold_metrics.append(
            fold_metrics
        )

        all_cv_runs.append(
            cv_runs
        )

    fold_metrics = pd.concat(
        all_fold_metrics,
        ignore_index=True,
    )

    cv_runs = pd.concat(
        all_cv_runs,
        ignore_index=True,
    )

    cv_summary = summarize_cv_runs(
        cv_runs
    )

    selected_family = required_text(
        cv_summary.iloc[
            0
        ][
            "family"
        ],
        "selected_family",
    )

    print()
    print(
        f"Selected downstream representation family by mean macro-F1: "
        f"{selected_family}"
    )

    prediction_tables: list[
        pd.DataFrame
    ] = []

    for family in families:
        family_representations = [
            representation
            for representation in (
                representations
            )
            if representation.family
            == family
        ]

        predictions = predict_unknown_for_family(
            family=family,
            representations=(
                family_representations
            ),
            labels=labels,
            feature_matrices=(
                feature_matrices
            ),
            retained_indices=(
                retained_indices
            ),
            targets=targets,
            retained_genres=(
                retained_genres
            ),
            regularization_c=(
                regularization_c
            ),
            max_iter=max_iter,
        )

        prediction_tables.append(
            predictions
        )

        predictions.to_csv(
            (
                output_dir
                / f"predictions_unknown_{family}.csv"
            ),
            index=False,
            encoding="utf-8",
        )

    predictions_all = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    selected_predictions = (
        predictions_all.loc[
            predictions_all[
                "family"
            ]
            == selected_family
        ]
        .copy()
        .sort_values(
            [
                "prediction_top1_probability",
                "top1_top2_probability_margin",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    selected_predictions[
        "recommended_for_manual_review"
    ] = (
        selected_predictions[
            "confidence_band_heuristic"
        ]
        != "higher"
    )

    representation_manifest.to_csv(
        (
            output_dir
            / "representation_manifest.csv"
        ),
        index=False,
        encoding="utf-8",
    )

    support_table.to_csv(
        (
            output_dir
            / "genre_support_table.csv"
        ),
        index=False,
        encoding="utf-8",
    )

    fold_metrics.to_csv(
        (
            output_dir
            / "classification_cv_fold_metrics.csv"
        ),
        index=False,
        encoding="utf-8",
    )

    cv_runs.to_csv(
        (
            output_dir
            / "classification_cv_repeat_metrics.csv"
        ),
        index=False,
        encoding="utf-8",
    )

    cv_summary.to_csv(
        (
            output_dir
            / "classification_cv_summary.csv"
        ),
        index=False,
        encoding="utf-8",
    )

    predictions_all.to_csv(
        (
            output_dir
            / "predictions_unknown_all_families.csv"
        ),
        index=False,
        encoding="utf-8",
    )

    selected_predictions.to_csv(
        (
            output_dir
            / "predictions_unknown_selected_family.csv"
        ),
        index=False,
        encoding="utf-8",
    )

    selected_family_predictions_by_group = (
        selected_predictions[
            "metadata_group"
        ]
        .value_counts()
        .rename_axis(
            "metadata_group"
        )
        .reset_index(
            name="artist_count",
        )
    )

    selected_family_predictions_by_confidence = (
        selected_predictions[
            "confidence_band_heuristic"
        ]
        .value_counts()
        .rename_axis(
            "confidence_band_heuristic"
        )
        .reset_index(
            name="artist_count",
        )
    )

    selected_family_predictions_by_group.to_csv(
        (
            output_dir
            / "selected_family_unknown_counts_by_metadata_group.csv"
        ),
        index=False,
        encoding="utf-8",
    )

    selected_family_predictions_by_confidence.to_csv(
        (
            output_dir
            / "selected_family_unknown_counts_by_confidence.csv"
        ),
        index=False,
        encoding="utf-8",
    )

    metadata = {
        "inputs": {
            "labels_csv": str(
                labels_csv
            ),
            "behavioral_control_embeddings_dir": str(
                behavioral_control_embeddings_dir
            ),
            "node2vec_embeddings_dir": str(
                node2vec_embeddings_dir
            ),
            "gae_embeddings_dir": str(
                gae_embeddings_dir
            ),
        },
        "configuration": {
            "families": families,
            "expected_representation_seeds": (
                expected_seeds
            ),
            "cv_seeds": cv_seeds,
            "cv_folds": int(
                cv_folds
            ),
            "minimum_genre_support": int(
                minimum_genre_support
            ),
            "classifier": (
                "LogisticRegression with class_weight=balanced"
            ),
            "regularization_c": float(
                regularization_c
            ),
            "max_iter": int(
                max_iter
            ),
            "top_k": int(
                top_k
            ),
            "representation_learning_uses_genres": False,
            "downstream_classifier_uses_genres": True,
            "evaluation_setting": (
                "transductive: unsupervised graph representations may "
                "include evaluation nodes but do not use their genre labels"
            ),
            "confidence_band_note": (
                "Confidence bands are heuristic and based on uncalibrated "
                "classifier probabilities plus top1-top2 probability margin."
            ),
        },
        "label_coverage": {
            "graph_node_count": int(
                len(
                    labels
                )
            ),
            "known_genre_node_count": int(
                labels[
                    "dominant_genre_normalized"
                ]
                .notna()
                .sum()
            ),
            "unknown_genre_node_count": int(
                labels[
                    "dominant_genre_normalized"
                ]
                .isna()
                .sum()
            ),
            "retained_supervised_node_count": int(
                len(
                    retained_indices
                )
            ),
            "retained_supervised_genre_count": int(
                len(
                    retained_genres
                )
            ),
        },
        "selection": {
            "primary_downstream_representation_family": (
                selected_family
            ),
            "selection_metric": (
                "mean macro-F1 across repeated stratified CV"
            ),
        },
        "outputs": {
            "representation_manifest": str(
                output_dir
                / "representation_manifest.csv"
            ),
            "genre_support_table": str(
                output_dir
                / "genre_support_table.csv"
            ),
            "classification_cv_fold_metrics": str(
                output_dir
                / "classification_cv_fold_metrics.csv"
            ),
            "classification_cv_repeat_metrics": str(
                output_dir
                / "classification_cv_repeat_metrics.csv"
            ),
            "classification_cv_summary": str(
                output_dir
                / "classification_cv_summary.csv"
            ),
            "predictions_unknown_all_families": str(
                output_dir
                / "predictions_unknown_all_families.csv"
            ),
            "predictions_unknown_selected_family": str(
                output_dir
                / "predictions_unknown_selected_family.csv"
            ),
            "selected_family_unknown_counts_by_metadata_group": str(
                output_dir
                / "selected_family_unknown_counts_by_metadata_group.csv"
            ),
            "selected_family_unknown_counts_by_confidence": str(
                output_dir
                / "selected_family_unknown_counts_by_confidence.csv"
            ),
        },
    }

    with (
        output_dir
        / "genre_inference_benchmark_summary.json"
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
    print(
        "Classification CV summary"
    )

    print(
        "-------------------------"
    )

    print(
        cv_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Selected-family unknown predictions by metadata group"
    )

    print(
        "-----------------------------------------------------"
    )

    print(
        selected_family_predictions_by_group.to_string(
            index=False
        )
    )

    print()
    print(
        "Selected-family unknown predictions by heuristic confidence"
    )

    print(
        "-----------------------------------------------------------"
    )

    print(
        selected_family_predictions_by_confidence.to_string(
            index=False
        )
    )

    print()
    print(
        f"Genre inference benchmark written to: "
        f"{output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark genre inference from frozen genre-free graph "
            "representations and generate exploratory predictions for "
            "unlabeled artists."
        )
    )

    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path(
            "data/interim/musicbrainz/largest_component_genre_coverage.csv"
        ),
    )

    parser.add_argument(
        "--behavioral-control-embeddings-dir",
        type=Path,
        default=Path(
            "results/behavioral_feature_controls/embeddings"
        ),
    )

    parser.add_argument(
        "--node2vec-embeddings-dir",
        type=Path,
        default=Path(
            "results/node2vec_final_candidates/embeddings"
        ),
    )

    parser.add_argument(
        "--gae-embeddings-dir",
        type=Path,
        default=Path(
            "results/graph_autoencoder_final_candidates_no_leakage/"
            "latent_16_lr_0_005/embeddings"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/genre_inference_benchmark"
        ),
    )

    parser.add_argument(
        "--families",
        nargs="+",
        choices=sorted(
            FAMILY_ORDER
        ),
        default=DEFAULT_FAMILIES,
    )

    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_EXPECTED_SEEDS,
    )

    parser.add_argument(
        "--cv-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_CV_SEEDS,
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--minimum-genre-support",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--regularization-c",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.cv_folds < 2:
        raise ValueError(
            "--cv-folds must be at least 2."
        )

    if args.minimum_genre_support < 1:
        raise ValueError(
            "--minimum-genre-support must be positive."
        )

    if args.regularization_c <= 0:
        raise ValueError(
            "--regularization-c must be positive."
        )

    if args.max_iter < 1:
        raise ValueError(
            "--max-iter must be positive."
        )

    if args.top_k < 1:
        raise ValueError(
            "--top-k must be positive."
        )

    run_benchmark(
        labels_csv=args.labels_csv,
        behavioral_control_embeddings_dir=(
            args.behavioral_control_embeddings_dir
        ),
        node2vec_embeddings_dir=(
            args.node2vec_embeddings_dir
        ),
        gae_embeddings_dir=(
            args.gae_embeddings_dir
        ),
        output_dir=args.output_dir,
        families=args.families,
        expected_seeds=args.expected_seeds,
        cv_seeds=args.cv_seeds,
        cv_folds=args.cv_folds,
        minimum_genre_support=(
            args.minimum_genre_support
        ),
        regularization_c=(
            args.regularization_c
        ),
        max_iter=args.max_iter,
        top_k=args.top_k,
    )
