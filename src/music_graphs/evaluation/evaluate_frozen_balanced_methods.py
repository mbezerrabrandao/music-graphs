from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)


DEFAULT_SUPPORTS = [1, 2, 3, 5]
DEFAULT_EXPECTED_SEEDS = [42, 43, 44, 45, 46]

METHOD_ORDER = [
    "leiden_behavioral_balanced",
    "louvain_behavioral_balanced",
    "node2vec_behavioral_balanced",
    "smooth2_behavioral_control",
    "gae_behavioral_balanced",
    "leiden_acoustic_balanced",
    "louvain_acoustic_balanced",
    "node2vec_acoustic_balanced",
    "smooth2_acoustic_control",
    "gae_acoustic_balanced",
]

METHOD_DISPLAY_NAMES = {
    "leiden_behavioral_balanced": "Leiden behavioral",
    "louvain_behavioral_balanced": "Louvain behavioral",
    "node2vec_behavioral_balanced": "Node2Vec behavioral + K-Means",
    "smooth2_behavioral_control": "Behavioral smooth2 + K-Means",
    "gae_behavioral_balanced": "GAE behavioral + K-Means",
    "leiden_acoustic_balanced": "Leiden acoustic",
    "louvain_acoustic_balanced": "Louvain acoustic",
    "node2vec_acoustic_balanced": "Node2Vec acoustic + K-Means",
    "smooth2_acoustic_control": "Acoustic smooth2 + K-Means",
    "gae_acoustic_balanced": "GAE acoustic + K-Means",
}


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


def required_float(
    value: Any,
    field_name: str,
) -> float:
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


def float_close(
    left: Any,
    right: float,
    *,
    absolute_tolerance: float = 1e-12,
) -> bool:
    return math.isclose(
        required_float(
            left,
            "numeric filter value",
        ),
        float(
            right
        ),
        rel_tol=0.0,
        abs_tol=absolute_tolerance,
    )


def read_csv_required(
    path: Path,
    *,
    dtype: dict[str, str] | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required CSV file does not exist: {path}"
        )

    return pd.read_csv(
        path,
        dtype=dtype,
        low_memory=False,
    )


def discover_partition_map(
    root_dir: Path,
) -> dict[str, Path]:
    if not root_dir.exists():
        raise FileNotFoundError(
            f"Partition root directory does not exist: {root_dir}"
        )

    paths = sorted(
        root_dir.rglob(
            "partitions/*.csv"
        )
    )

    if not paths:
        raise FileNotFoundError(
            f"No partition CSV files were found under: {root_dir}"
        )

    partition_map: dict[
        str,
        Path,
    ] = {}

    for path in paths:
        run_id = path.stem

        if run_id in partition_map:
            raise ValueError(
                "Found duplicate partition filenames for run_id: "
                f"{run_id}"
            )

        partition_map[
            run_id
        ] = path

    return partition_map


def validate_selected_seeds(
    method_id: str,
    rows: pd.DataFrame,
    expected_seeds: list[int],
) -> None:
    actual_seeds = sorted(
        int(
            seed
        )
        for seed in (
            rows[
                "seed"
            ]
            .tolist()
        )
    )

    if actual_seeds != sorted(
        expected_seeds
    ):
        raise ValueError(
            f"{method_id} does not contain the expected seeds. "
            f"Expected {sorted(expected_seeds)}, got {actual_seeds}."
        )


def base_manifest_record(
    *,
    method_id: str,
    run_id: str,
    seed: int,
    partition_path: Path,
    assignment_column: str,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "method_order": int(
            METHOD_ORDER.index(
                method_id
            )
        ),
        "method_id": method_id,
        "method_display_name": (
            METHOD_DISPLAY_NAMES[
                method_id
            ]
        ),
        "run_id": run_id,
        "seed": int(
            seed
        ),
        "partition_path": str(
            partition_path
        ),
        "assignment_column": (
            assignment_column
        ),
        "configuration_json": json.dumps(
            configuration,
            sort_keys=True,
            ensure_ascii=False,
        ),
    }


def select_behavioral_modularity_runs(
    runs_csv: Path,
    partition_root: Path,
    expected_seeds: list[int],
) -> list[dict[str, Any]]:
    runs = read_csv_required(
        runs_csv,
        dtype={
            "run_id": "string",
            "algorithm": "string",
        },
    )

    required_columns = {
        "run_id",
        "algorithm",
        "resolution",
        "seed",
    }

    missing_columns = sorted(
        required_columns - set(
            runs.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Behavioral modularity run table is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    partition_map = discover_partition_map(
        partition_root
    )

    records: list[
        dict[str, Any]
    ] = []

    for algorithm, method_id in [
        (
            "Leiden",
            "leiden_behavioral_balanced",
        ),
        (
            "Louvain",
            "louvain_behavioral_balanced",
        ),
    ]:
        selected = runs.loc[
            runs[
                "algorithm"
            ].astype(
                str
            )
            == algorithm
        ].copy()

        selected = selected.loc[
            selected[
                "resolution"
            ].map(
                lambda value: float_close(
                    value,
                    2.25,
                )
            )
        ].copy()

        if selected.empty:
            raise ValueError(
                f"No selected behavioral baseline runs found for {algorithm}."
            )

        selected[
            "seed"
        ] = pd.to_numeric(
            selected[
                "seed"
            ],
            errors="raise",
        ).astype(
            int
        )

        validate_selected_seeds(
            method_id=method_id,
            rows=selected,
            expected_seeds=expected_seeds,
        )

        for row in selected.to_dict(
            orient="records"
        ):
            run_id = required_text(
                row[
                    "run_id"
                ],
                "run_id",
            )

            if run_id not in partition_map:
                raise FileNotFoundError(
                    f"Missing partition CSV for run_id: {run_id}"
                )

            records.append(
                base_manifest_record(
                    method_id=method_id,
                    run_id=run_id,
                    seed=required_int(
                        row[
                            "seed"
                        ],
                        "seed",
                    ),
                    partition_path=(
                        partition_map[
                            run_id
                        ]
                    ),
                    assignment_column=(
                        "community_id"
                    ),
                    configuration={
                        "graph": (
                            "behavioral_sequential_k5_support2_min3"
                        ),
                        "algorithm": algorithm,
                        "resolution": 2.25,
                    },
                )
            )

    return records


def select_node2vec_runs(
    runs_csv: Path,
    partition_root: Path,
    expected_seeds: list[int],
) -> list[dict[str, Any]]:
    runs = read_csv_required(
        runs_csv,
        dtype={
            "run_id": "string",
        },
    )

    required_columns = {
        "run_id",
        "p",
        "q",
        "seed",
        "k",
        "silhouette_score",
    }

    missing_columns = sorted(
        required_columns - set(
            runs.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Node2Vec run table is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    balanced_runs = runs.loc[
        pd.to_numeric(
            runs[
                "k"
            ],
            errors="raise",
        )
        == 54
    ].copy()

    if balanced_runs.empty:
        raise ValueError(
            "No Node2Vec runs were found at balanced k=54."
        )

    balanced_runs[
        "silhouette_score"
    ] = pd.to_numeric(
        balanced_runs[
            "silhouette_score"
        ],
        errors="raise",
    )

    selected_pq = (
        balanced_runs
        .groupby(
            [
                "p",
                "q",
            ],
            as_index=False,
        )
        .agg(
            silhouette_mean=(
                "silhouette_score",
                "mean",
            ),
            seed_count=(
                "seed",
                "nunique",
            ),
        )
        .sort_values(
            [
                "silhouette_mean",
                "seed_count",
                "p",
                "q",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        )
        .iloc[0]
    )

    selected = balanced_runs.loc[
        balanced_runs[
            "p"
        ].map(
            lambda value: float_close(
                value,
                float(selected_pq[
                    "p"
                ]),
            )
        )
        & balanced_runs[
            "q"
        ].map(
            lambda value: float_close(
                value,
                float(selected_pq[
                    "q"
                ]),
            )
        )
    ].copy()

    if selected.empty:
        raise ValueError(
            "No selected Node2Vec balanced runs were found."
        )

    selected[
        "seed"
    ] = pd.to_numeric(
        selected[
            "seed"
        ],
        errors="raise",
    ).astype(
        int
    )

    validate_selected_seeds(
        method_id=(
            "node2vec_behavioral_balanced"
        ),
        rows=selected,
        expected_seeds=expected_seeds,
    )

    partition_map = discover_partition_map(
        partition_root
    )

    records: list[
        dict[str, Any]
    ] = []

    for row in selected.to_dict(
        orient="records"
    ):
        run_id = required_text(
            row[
                "run_id"
            ],
            "run_id",
        )

        if run_id not in partition_map:
            raise FileNotFoundError(
                f"Missing partition CSV for run_id: {run_id}"
            )

        records.append(
            base_manifest_record(
                method_id=(
                    "node2vec_behavioral_balanced"
                ),
                run_id=run_id,
                seed=required_int(
                    row[
                        "seed"
                    ],
                    "seed",
                ),
                partition_path=(
                    partition_map[
                        run_id
                    ]
                ),
                assignment_column=(
                    "cluster_id"
                ),
                configuration={
                    "graph": (
                        "behavioral_sequential_k5_support2_min3"
                    ),
                    "p": 2.0,
                    "q": 2.0,
                    "k": 54,
                    "selection_rule": (
                        "maximum_mean_silhouette_within_balanced_scale"
                    ),
                },
            )
        )

    return records


def select_smooth2_runs(
    runs_csv: Path,
    partition_root: Path,
    expected_seeds: list[int],
) -> list[dict[str, Any]]:
    runs = read_csv_required(
        runs_csv,
        dtype={
            "run_id": "string",
            "control_mode": "string",
        },
    )

    required_columns = {
        "run_id",
        "control_mode",
        "seed",
        "k",
    }

    missing_columns = sorted(
        required_columns - set(
            runs.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Behavioral-feature control run table is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    selected = runs.loc[
        (
            runs[
                "control_mode"
            ].astype(
                str
            )
            == "smooth2"
        )
        & (
            pd.to_numeric(
                runs[
                    "k"
                ],
                errors="raise",
            )
            == 54
        )
    ].copy()

    if selected.empty:
        raise ValueError(
            "No selected smooth2 balanced control runs were found."
        )

    selected[
        "seed"
    ] = pd.to_numeric(
        selected[
            "seed"
        ],
        errors="raise",
    ).astype(
        int
    )

    validate_selected_seeds(
        method_id=(
            "smooth2_behavioral_control"
        ),
        rows=selected,
        expected_seeds=expected_seeds,
    )

    partition_map = discover_partition_map(
        partition_root
    )

    records: list[
        dict[str, Any]
    ] = []

    for row in selected.to_dict(
        orient="records"
    ):
        run_id = required_text(
            row[
                "run_id"
            ],
            "run_id",
        )

        if run_id not in partition_map:
            raise FileNotFoundError(
                f"Missing partition CSV for run_id: {run_id}"
            )

        records.append(
            base_manifest_record(
                method_id=(
                    "smooth2_behavioral_control"
                ),
                run_id=run_id,
                seed=required_int(
                    row[
                        "seed"
                    ],
                    "seed",
                ),
                partition_path=(
                    partition_map[
                        run_id
                    ]
                ),
                assignment_column=(
                    "cluster_id"
                ),
                configuration={
                    "graph": (
                        "behavioral_sequential_k5_support2_min3"
                    ),
                    "control_mode": (
                        "smooth2"
                    ),
                    "k": 54,
                    "features": [
                        "log1p(scrobble_count)",
                        "log1p(session_count)",
                        "log1p(degree)",
                        "log1p(weighted_degree)",
                    ],
                },
            )
        )

    return records


def _truthy_series(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: str(value).strip().lower()
        in {"1", "true", "t", "yes", "y"}
    )


def select_gae_runs(
    runs_csv: Path,
    partition_root: Path,
    expected_seeds: list[int],
) -> list[dict[str, Any]]:
    runs = read_csv_required(
        runs_csv,
        dtype={
            "run_id": "string",
            "feature_mode": "string",
        },
    )

    required_columns = {
        "run_id",
        "feature_mode",
        "hidden_dim",
        "latent_dim",
        "dropout",
        "learning_rate",
        "weight_decay",
        "seed",
        "k",
    }

    missing_columns = sorted(
        required_columns - set(
            runs.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "GAE combined run table is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    behavioral = runs.loc[
        runs[
            "feature_mode"
        ].astype(
            str
        )
        == "behavioral"
    ].copy()

    behavioral[
        "k"
    ] = pd.to_numeric(
        behavioral[
            "k"
        ],
        errors="raise",
    ).astype(
        int
    )

    if "selected_for_balanced_evaluation" in behavioral.columns:
        selected = behavioral.loc[
            _truthy_series(
                behavioral[
                    "selected_for_balanced_evaluation"
                ]
            )
            & (
                behavioral[
                    "k"
                ]
                == 54
            )
        ].copy()
    else:
        selected = behavioral.loc[
            (
                pd.to_numeric(
                    behavioral[
                        "hidden_dim"
                    ],
                    errors="raise",
                )
                == 128
            )
            & (
                pd.to_numeric(
                    behavioral[
                        "latent_dim"
                    ],
                    errors="raise",
                )
                == 16
            )
            & behavioral[
                "dropout"
            ].map(
                lambda value: float_close(
                    value,
                    0.1,
                )
            )
            & behavioral[
                "learning_rate"
            ].map(
                lambda value: float_close(
                    value,
                    0.005,
                )
            )
            & behavioral[
                "weight_decay"
            ].map(
                lambda value: float_close(
                    value,
                    0.0001,
                )
            )
            & (
                behavioral[
                    "k"
                ]
                == 54
            )
        ].copy()

    if selected.empty:
        raise ValueError(
            "No selected GAE behavioral balanced runs were found."
        )

    selected[
        "seed"
    ] = pd.to_numeric(
        selected[
            "seed"
        ],
        errors="raise",
    ).astype(
        int
    )

    validate_selected_seeds(
        method_id=(
            "gae_behavioral_balanced"
        ),
        rows=selected,
        expected_seeds=expected_seeds,
    )

    partition_map = discover_partition_map(
        partition_root
    )

    records: list[
        dict[str, Any]
    ] = []

    for row in selected.to_dict(
        orient="records"
    ):
        run_id = required_text(
            row[
                "run_id"
            ],
            "run_id",
        )

        if run_id not in partition_map:
            raise FileNotFoundError(
                f"Missing partition CSV for run_id: {run_id}"
            )

        records.append(
            base_manifest_record(
                method_id=(
                    "gae_behavioral_balanced"
                ),
                run_id=run_id,
                seed=required_int(
                    row[
                        "seed"
                    ],
                    "seed",
                ),
                partition_path=(
                    partition_map[
                        run_id
                    ]
                ),
                assignment_column=(
                    "cluster_id"
                ),
                configuration={
                    "graph": (
                        "behavioral_sequential_k5_decay_support2_min3"
                    ),
                    "feature_mode": (
                        "behavioral"
                    ),
                    "hidden_dim": required_int(
                        row[
                            "hidden_dim"
                        ],
                        "hidden_dim",
                    ),
                    "latent_dim": required_int(
                        row[
                            "latent_dim"
                        ],
                        "latent_dim",
                    ),
                    "dropout": required_float(
                        row[
                            "dropout"
                        ],
                        "dropout",
                    ),
                    "learning_rate": required_float(
                        row[
                            "learning_rate"
                        ],
                        "learning_rate",
                    ),
                    "weight_decay": required_float(
                        row[
                            "weight_decay"
                        ],
                        "weight_decay",
                    ),
                    "k": required_int(
                        row[
                            "k"
                        ],
                        "k",
                    ),
                },
            )
        )

    return records

def build_manifest(
    *,
    behavioral_baseline_runs_csv: Path,
    behavioral_baseline_partition_root: Path,
    node2vec_runs_csv: Path,
    node2vec_partition_root: Path,
    smooth2_runs_csv: Path,
    smooth2_partition_root: Path,
    gae_combined_runs_csv: Path,
    gae_partition_root: Path,
    acoustic_baseline_runs_csv: Path,
    acoustic_baseline_partition_root: Path,
    acoustic_node2vec_runs_csv: Path,
    acoustic_node2vec_partition_root: Path,
    acoustic_smooth2_runs_csv: Path,
    acoustic_smooth2_partition_root: Path,
    acoustic_gae_combined_runs_csv: Path,
    acoustic_gae_partition_root: Path,
    expected_seeds: list[int],
) -> pd.DataFrame:
    behavioral_records = (
        select_behavioral_modularity_runs(
            runs_csv=(
                behavioral_baseline_runs_csv
            ),
            partition_root=(
                behavioral_baseline_partition_root
            ),
            expected_seeds=(
                expected_seeds
            ),
        )
        + select_node2vec_runs(
            runs_csv=(
                node2vec_runs_csv
            ),
            partition_root=(
                node2vec_partition_root
            ),
            expected_seeds=(
                expected_seeds
            ),
        )
        + select_smooth2_runs(
            runs_csv=(
                smooth2_runs_csv
            ),
            partition_root=(
                smooth2_partition_root
            ),
            expected_seeds=(
                expected_seeds
            ),
        )
        + select_gae_runs(
            runs_csv=(
                gae_combined_runs_csv
            ),
            partition_root=(
                gae_partition_root
            ),
            expected_seeds=(
                expected_seeds
            ),
        )
    )

    acoustic_records = (
        select_behavioral_modularity_runs(
            runs_csv=(
                acoustic_baseline_runs_csv
            ),
            partition_root=(
                acoustic_baseline_partition_root
            ),
            expected_seeds=(
                expected_seeds
            ),
        )
        + select_node2vec_runs(
            runs_csv=(
                acoustic_node2vec_runs_csv
            ),
            partition_root=(
                acoustic_node2vec_partition_root
            ),
            expected_seeds=(
                expected_seeds
            ),
        )
        + select_smooth2_runs(
            runs_csv=(
                acoustic_smooth2_runs_csv
            ),
            partition_root=(
                acoustic_smooth2_partition_root
            ),
            expected_seeds=(
                expected_seeds
            ),
        )
        + select_gae_runs(
            runs_csv=(
                acoustic_gae_combined_runs_csv
            ),
            partition_root=(
                acoustic_gae_partition_root
            ),
            expected_seeds=(
                expected_seeds
            ),
        )
    )

    acoustic_records = retag_records(
        acoustic_records,
        method_id_map={
            "leiden_behavioral_balanced": "leiden_acoustic_balanced",
            "louvain_behavioral_balanced": "louvain_acoustic_balanced",
            "node2vec_behavioral_balanced": "node2vec_acoustic_balanced",
            "smooth2_behavioral_control": "smooth2_acoustic_control",
            "gae_behavioral_balanced": "gae_acoustic_balanced",
        },
        run_id_prefix="acoustic::",
        graph_name="acoustic_top10_knn20_min0_40",
    )

    records = behavioral_records + acoustic_records

    manifest = (
        pd.DataFrame.from_records(
            records
        )
        .sort_values(
            [
                "method_order",
                "seed",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    expected_run_count = (
        len(
            METHOD_ORDER
        )
        * len(
            expected_seeds
        )
    )

    if len(
        manifest
    ) != expected_run_count:
        raise ValueError(
            "Unexpected manifest size. "
            f"Expected {expected_run_count}, got {len(manifest)}."
        )

    if manifest[
        "run_id"
    ].duplicated().any():
        raise ValueError(
            "Selected manifest contains duplicate run_id values."
        )

    return manifest


def retag_records(
    records: list[dict[str, Any]],
    *,
    method_id_map: dict[str, str],
    run_id_prefix: str,
    graph_name: str,
) -> list[dict[str, Any]]:
    retagged: list[dict[str, Any]] = []

    for record in records:
        original_method_id = required_text(record["method_id"], "method_id")
        method_id = method_id_map[original_method_id]
        configuration = json.loads(
            required_text(record["configuration_json"], "configuration_json")
        )
        configuration["graph"] = graph_name

        retagged.append(
            {
                **record,
                "method_order": int(METHOD_ORDER.index(method_id)),
                "method_id": method_id,
                "method_display_name": METHOD_DISPLAY_NAMES[method_id],
                "run_id": f"{run_id_prefix}{required_text(record['run_id'], 'run_id')}",
                "configuration_json": json.dumps(
                    configuration,
                    sort_keys=True,
                    ensure_ascii=False,
                ),
            }
        )

    return retagged


def load_labels(
    labels_csv: Path,
) -> pd.DataFrame:
    labels = read_csv_required(
        labels_csv,
        dtype={
            "artist_id": "string",
            "dominant_genre_normalized": "string",
        },
    )

    required_columns = {
        "artist_id",
        "dominant_genre_normalized",
    }

    missing_columns = sorted(
        required_columns - set(
            labels.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "External-label table is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    labels[
        "artist_id"
    ] = (
        labels[
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

    labels[
        "dominant_genre_normalized"
    ] = (
        labels[
            "dominant_genre_normalized"
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
            "External-label table contains missing artist IDs."
        )

    if labels[
        "artist_id"
    ].duplicated().any():
        raise ValueError(
            "External-label table contains duplicate artist IDs."
        )

    return labels[
        [
            "artist_id",
            "dominant_genre_normalized",
        ]
    ].copy()


def load_partition(
    partition_path: Path,
    assignment_column: str,
) -> pd.DataFrame:
    partition = read_csv_required(
        partition_path,
        dtype={
            "artist_id": "string",
        },
    )

    required_columns = {
        "artist_id",
        assignment_column,
    }

    missing_columns = sorted(
        required_columns - set(
            partition.columns
        )
    )

    if missing_columns:
        raise ValueError(
            f"Partition {partition_path} is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    partition[
        "artist_id"
    ] = (
        partition[
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

    if partition[
        "artist_id"
    ].isna().any():
        raise ValueError(
            f"Partition contains missing artist IDs: {partition_path}"
        )

    if partition[
        "artist_id"
    ].duplicated().any():
        raise ValueError(
            f"Partition contains duplicate artist IDs: {partition_path}"
        )

    partition[
        assignment_column
    ] = pd.to_numeric(
        partition[
            assignment_column
        ],
        errors="raise",
    ).astype(
        int
    )

    return (
        partition[
            [
                "artist_id",
                assignment_column,
            ]
        ]
        .rename(
            columns={
                assignment_column: (
                    "assignment_id"
                ),
            }
        )
    )


def weighted_purity(
    truth: list[str],
    predictions: list[int],
) -> float:
    if len(
        truth
    ) != len(
        predictions
    ):
        raise ValueError(
            "Truth and prediction lists have different lengths."
        )

    if not truth:
        raise ValueError(
            "Cannot calculate purity without labeled nodes."
        )

    grouped: dict[
        int,
        list[str],
    ] = {}

    for label, assignment in zip(
        truth,
        predictions,
    ):
        grouped.setdefault(
            int(
                assignment
            ),
            [],
        ).append(
            label
        )

    correct_count = sum(
        int(
            pd.Series(
                labels
            )
            .value_counts()
            .iloc[
                0
            ]
        )
        for labels in (
            grouped.values()
        )
    )

    return (
        correct_count
        / len(
            truth
        )
    )


def calculate_external_metrics(
    partition: pd.DataFrame,
    labels: pd.DataFrame,
    min_genre_support: int,
) -> dict[str, Any]:
    joined = partition.merge(
        labels,
        on="artist_id",
        how="left",
        validate="one_to_one",
    )

    labeled = (
        joined.loc[
            joined[
                "dominant_genre_normalized"
            ].notna()
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    original_counts = (
        labeled[
            "dominant_genre_normalized"
        ]
        .value_counts()
    )

    retained_genres = set(
        original_counts.loc[
            original_counts
            >= min_genre_support
        ]
        .index
        .astype(
            str
        )
    )

    evaluated = (
        labeled.loc[
            labeled[
                "dominant_genre_normalized"
            ].astype(
                str
            )
            .isin(
                retained_genres
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if evaluated.empty:
        raise ValueError(
            "No labeled nodes remain after applying minimum genre "
            f"support {min_genre_support}."
        )

    truth = (
        evaluated[
            "dominant_genre_normalized"
        ]
        .astype(
            str
        )
        .tolist()
    )

    predictions = (
        evaluated[
            "assignment_id"
        ]
        .astype(
            int
        )
        .tolist()
    )

    assignment_sizes = (
        partition[
            "assignment_id"
        ]
        .value_counts()
    )

    return {
        "partition_node_count": int(
            len(
                partition
            )
        ),
        "labeled_node_count_before_support_filter": int(
            len(
                labeled
            )
        ),
        "evaluated_node_count": int(
            len(
                evaluated
            )
        ),
        "retained_genre_count": int(
            len(
                retained_genres
            )
        ),
        "excluded_labeled_node_count": int(
            len(
                labeled
            )
            - len(
                evaluated
            )
        ),
        "nmi": float(
            normalized_mutual_info_score(
                truth,
                predictions,
                average_method=(
                    "arithmetic"
                ),
            )
        ),
        "ami": float(
            adjusted_mutual_info_score(
                truth,
                predictions,
                average_method=(
                    "arithmetic"
                ),
            )
        ),
        "ari": float(
            adjusted_rand_score(
                truth,
                predictions,
            )
        ),
        "purity": float(
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
        "assignment_count": int(
            len(
                assignment_sizes
            )
        ),
        "singleton_assignment_count": int(
            (
                assignment_sizes
                == 1
            ).sum()
        ),
        "minimum_assignment_size": int(
            assignment_sizes.min()
        ),
        "median_assignment_size": float(
            np.median(
                assignment_sizes.values
            )
        ),
        "maximum_assignment_size": int(
            assignment_sizes.max()
        ),
        "maximum_assignment_fraction": float(
            assignment_sizes.max()
            / len(
                partition
            )
        ),
    }


def evaluate_manifest(
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    supports: list[int],
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    metrics_records: list[
        dict[str, Any]
    ] = []

    partition_cache: dict[
        str,
        pd.DataFrame,
    ] = {}

    for row in manifest.to_dict(
        orient="records"
    ):
        run_id = required_text(
            row[
                "run_id"
            ],
            "run_id",
        )

        partition = load_partition(
            partition_path=Path(
                required_text(
                    row[
                        "partition_path"
                    ],
                    "partition_path",
                )
            ),
            assignment_column=required_text(
                row[
                    "assignment_column"
                ],
                "assignment_column",
            ),
        )

        partition_cache[
            run_id
        ] = partition

        for support in supports:
            metrics = calculate_external_metrics(
                partition=partition,
                labels=labels,
                min_genre_support=support,
            )

            metrics_records.append(
                {
                    **row,
                    "min_genre_support": int(
                        support
                    ),
                    **metrics,
                }
            )

    return (
        pd.DataFrame.from_records(
            metrics_records
        ),
        partition_cache,
    )


def summarize_external_metrics(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    return (
        metrics.groupby(
            [
                "method_order",
                "method_id",
                "method_display_name",
                "min_genre_support",
            ],
            as_index=False,
        )
        .agg(
            run_count=(
                "run_id",
                "size",
            ),
            partition_node_count=(
                "partition_node_count",
                "mean",
            ),
            labeled_node_count_before_support_filter=(
                "labeled_node_count_before_support_filter",
                "mean",
            ),
            evaluated_node_count=(
                "evaluated_node_count",
                "mean",
            ),
            retained_genre_count=(
                "retained_genre_count",
                "mean",
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
                "purity",
                "mean",
            ),
            purity_std=(
                "purity",
                "std",
            ),
            homogeneity_mean=(
                "homogeneity",
                "mean",
            ),
            completeness_mean=(
                "completeness",
                "mean",
            ),
            v_measure_mean=(
                "v_measure",
                "mean",
            ),
            assignment_count_mean=(
                "assignment_count",
                "mean",
            ),
            singleton_assignment_count_mean=(
                "singleton_assignment_count",
                "mean",
            ),
            minimum_assignment_size_mean=(
                "minimum_assignment_size",
                "mean",
            ),
            median_assignment_size_mean=(
                "median_assignment_size",
                "mean",
            ),
            maximum_assignment_size_mean=(
                "maximum_assignment_size",
                "mean",
            ),
            maximum_assignment_fraction_mean=(
                "maximum_assignment_fraction",
                "mean",
            ),
        )
        .fillna(
            0.0
        )
        .sort_values(
            [
                "method_order",
                "min_genre_support",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def calculate_pairwise_stability(
    manifest: pd.DataFrame,
    partition_cache: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    records: list[
        dict[str, Any]
    ] = []

    grouped = manifest.groupby(
        [
            "method_order",
            "method_id",
            "method_display_name",
        ],
        sort=True,
    )

    for (
        method_order,
        method_id,
        method_display_name,
    ), group in grouped:
        group_records = group.to_dict(
            orient="records"
        )

        for row_a, row_b in combinations(
            group_records,
            2,
        ):
            run_id_a = required_text(
                row_a[
                    "run_id"
                ],
                "run_id_a",
            )

            run_id_b = required_text(
                row_b[
                    "run_id"
                ],
                "run_id_b",
            )

            left = (
                partition_cache[
                    run_id_a
                ]
                .rename(
                    columns={
                        "assignment_id": (
                            "assignment_id_a"
                        ),
                    }
                )
            )

            right = (
                partition_cache[
                    run_id_b
                ]
                .rename(
                    columns={
                        "assignment_id": (
                            "assignment_id_b"
                        ),
                    }
                )
            )

            joined = left.merge(
                right,
                on="artist_id",
                how="inner",
                validate="one_to_one",
            )

            if joined.empty:
                raise ValueError(
                    f"Partitions {run_id_a} and {run_id_b} "
                    "have no shared nodes."
                )

            labels_a = (
                joined[
                    "assignment_id_a"
                ]
                .astype(
                    int
                )
                .tolist()
            )

            labels_b = (
                joined[
                    "assignment_id_b"
                ]
                .astype(
                    int
                )
                .tolist()
            )

            records.append(
                {
                    "method_order": int(
                        method_order
                    ),
                    "method_id": str(
                        method_id
                    ),
                    "method_display_name": str(
                        method_display_name
                    ),
                    "run_id_a": run_id_a,
                    "run_id_b": run_id_b,
                    "seed_a": required_int(
                        row_a[
                            "seed"
                        ],
                        "seed_a",
                    ),
                    "seed_b": required_int(
                        row_b[
                            "seed"
                        ],
                        "seed_b",
                    ),
                    "shared_node_count": int(
                        len(
                            joined
                        )
                    ),
                    "partition_nmi": float(
                        normalized_mutual_info_score(
                            labels_a,
                            labels_b,
                            average_method=(
                                "arithmetic"
                            ),
                        )
                    ),
                    "partition_ari": float(
                        adjusted_rand_score(
                            labels_a,
                            labels_b,
                        )
                    ),
                }
            )

    return pd.DataFrame.from_records(
        records
    )


def summarize_stability(
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    return (
        pairwise.groupby(
            [
                "method_order",
                "method_id",
                "method_display_name",
            ],
            as_index=False,
        )
        .agg(
            partition_pair_count=(
                "run_id_a",
                "size",
            ),
            shared_node_count=(
                "shared_node_count",
                "mean",
            ),
            partition_nmi_mean=(
                "partition_nmi",
                "mean",
            ),
            partition_nmi_std=(
                "partition_nmi",
                "std",
            ),
            partition_nmi_min=(
                "partition_nmi",
                "min",
            ),
            partition_nmi_max=(
                "partition_nmi",
                "max",
            ),
            partition_ari_mean=(
                "partition_ari",
                "mean",
            ),
            partition_ari_std=(
                "partition_ari",
                "std",
            ),
            partition_ari_min=(
                "partition_ari",
                "min",
            ),
            partition_ari_max=(
                "partition_ari",
                "max",
            ),
        )
        .fillna(
            0.0
        )
        .sort_values(
            [
                "method_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def build_primary_comparison(
    external_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
) -> pd.DataFrame:
    support_one = (
        external_summary.loc[
            external_summary[
                "min_genre_support"
            ]
            == 1
        ]
        .copy()
    )

    comparison = support_one.merge(
        stability_summary[
            [
                "method_order",
                "method_id",
                "partition_nmi_mean",
                "partition_nmi_std",
                "partition_ari_mean",
                "partition_ari_std",
            ]
        ],
        on=[
            "method_order",
            "method_id",
        ],
        how="left",
        validate="one_to_one",
    )

    return (
        comparison[
            [
                "method_order",
                "method_id",
                "method_display_name",
                "run_count",
                "partition_node_count",
                "evaluated_node_count",
                "retained_genre_count",
                "assignment_count_mean",
                "nmi_mean",
                "nmi_std",
                "ami_mean",
                "ami_std",
                "ari_mean",
                "ari_std",
                "purity_mean",
                "maximum_assignment_fraction_mean",
                "partition_nmi_mean",
                "partition_nmi_std",
                "partition_ari_mean",
                "partition_ari_std",
            ]
        ]
        .sort_values(
            "method_order"
        )
        .reset_index(
            drop=True
        )
    )


def run_evaluation(
    *,
    labels_csv: Path,
    output_dir: Path,
    supports: list[int],
    expected_seeds: list[int],
    behavioral_baseline_runs_csv: Path,
    behavioral_baseline_partition_root: Path,
    node2vec_runs_csv: Path,
    node2vec_partition_root: Path,
    smooth2_runs_csv: Path,
    smooth2_partition_root: Path,
    gae_combined_runs_csv: Path,
    gae_partition_root: Path,
    acoustic_baseline_runs_csv: Path,
    acoustic_baseline_partition_root: Path,
    acoustic_node2vec_runs_csv: Path,
    acoustic_node2vec_partition_root: Path,
    acoustic_smooth2_runs_csv: Path,
    acoustic_smooth2_partition_root: Path,
    acoustic_gae_combined_runs_csv: Path,
    acoustic_gae_partition_root: Path,
) -> None:
    if any(
        support < 1
        for support in supports
    ):
        raise ValueError(
            "All minimum genre supports must be at least 1."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Selecting frozen balanced configurations."
    )

    manifest = build_manifest(
        behavioral_baseline_runs_csv=(
            behavioral_baseline_runs_csv
        ),
        behavioral_baseline_partition_root=(
            behavioral_baseline_partition_root
        ),
        node2vec_runs_csv=(
            node2vec_runs_csv
        ),
        node2vec_partition_root=(
            node2vec_partition_root
        ),
        smooth2_runs_csv=(
            smooth2_runs_csv
        ),
        smooth2_partition_root=(
            smooth2_partition_root
        ),
        gae_combined_runs_csv=(
            gae_combined_runs_csv
        ),
        gae_partition_root=(
            gae_partition_root
        ),
        acoustic_baseline_runs_csv=(
            acoustic_baseline_runs_csv
        ),
        acoustic_baseline_partition_root=(
            acoustic_baseline_partition_root
        ),
        acoustic_node2vec_runs_csv=(
            acoustic_node2vec_runs_csv
        ),
        acoustic_node2vec_partition_root=(
            acoustic_node2vec_partition_root
        ),
        acoustic_smooth2_runs_csv=(
            acoustic_smooth2_runs_csv
        ),
        acoustic_smooth2_partition_root=(
            acoustic_smooth2_partition_root
        ),
        acoustic_gae_combined_runs_csv=(
            acoustic_gae_combined_runs_csv
        ),
        acoustic_gae_partition_root=(
            acoustic_gae_partition_root
        ),
        expected_seeds=(
            expected_seeds
        ),
    )

    labels = load_labels(
        labels_csv
    )

    print(
        "Calculating external metrics under common support thresholds."
    )

    (
        metrics,
        partition_cache,
    ) = evaluate_manifest(
        manifest=manifest,
        labels=labels,
        supports=supports,
    )

    external_summary = summarize_external_metrics(
        metrics
    )

    print(
        "Calculating pairwise partition stability across seeds."
    )

    stability_pairwise = calculate_pairwise_stability(
        manifest=manifest,
        partition_cache=partition_cache,
    )

    stability_summary = summarize_stability(
        stability_pairwise
    )

    primary_comparison = build_primary_comparison(
        external_summary=external_summary,
        stability_summary=stability_summary,
    )

    manifest_output = (
        output_dir
        / "selected_balanced_run_manifest.csv"
    )

    metrics_output = (
        output_dir
        / "uniform_external_metrics_by_run_and_support.csv"
    )

    external_summary_output = (
        output_dir
        / "uniform_external_metrics_summary.csv"
    )

    stability_pairwise_output = (
        output_dir
        / "uniform_partition_stability_pairwise.csv"
    )

    stability_summary_output = (
        output_dir
        / "uniform_partition_stability_summary.csv"
    )

    primary_comparison_output = (
        output_dir
        / "primary_balanced_comparison.csv"
    )

    manifest.to_csv(
        manifest_output,
        index=False,
        encoding="utf-8",
    )

    metrics.to_csv(
        metrics_output,
        index=False,
        encoding="utf-8",
    )

    external_summary.to_csv(
        external_summary_output,
        index=False,
        encoding="utf-8",
    )

    stability_pairwise.to_csv(
        stability_pairwise_output,
        index=False,
        encoding="utf-8",
    )

    stability_summary.to_csv(
        stability_summary_output,
        index=False,
        encoding="utf-8",
    )

    primary_comparison.to_csv(
        primary_comparison_output,
        index=False,
        encoding="utf-8",
    )

    metadata = {
        "inputs": {
            "labels_csv": str(
                labels_csv
            ),
            "behavioral_baseline_runs_csv": str(
                behavioral_baseline_runs_csv
            ),
            "behavioral_baseline_partition_root": str(
                behavioral_baseline_partition_root
            ),
            "node2vec_runs_csv": str(
                node2vec_runs_csv
            ),
            "node2vec_partition_root": str(
                node2vec_partition_root
            ),
            "smooth2_runs_csv": str(
                smooth2_runs_csv
            ),
            "smooth2_partition_root": str(
                smooth2_partition_root
            ),
            "gae_combined_runs_csv": str(
                gae_combined_runs_csv
            ),
            "gae_partition_root": str(
                gae_partition_root
            ),
            "acoustic_baseline_runs_csv": str(
                acoustic_baseline_runs_csv
            ),
            "acoustic_baseline_partition_root": str(
                acoustic_baseline_partition_root
            ),
            "acoustic_node2vec_runs_csv": str(
                acoustic_node2vec_runs_csv
            ),
            "acoustic_node2vec_partition_root": str(
                acoustic_node2vec_partition_root
            ),
            "acoustic_smooth2_runs_csv": str(
                acoustic_smooth2_runs_csv
            ),
            "acoustic_smooth2_partition_root": str(
                acoustic_smooth2_partition_root
            ),
            "acoustic_gae_combined_runs_csv": str(
                acoustic_gae_combined_runs_csv
            ),
            "acoustic_gae_partition_root": str(
                acoustic_gae_partition_root
            ),
        },
        "configuration": {
            "minimum_genre_supports": supports,
            "expected_seeds": expected_seeds,
            "selected_methods": (
                METHOD_ORDER
            ),
            "primary_table_support": 1,
            "external_genres_used_during_model_training": False,
            "external_genres_used_during_model_selection": False,
        },
        "outputs": {
            "selected_balanced_run_manifest": str(
                manifest_output
            ),
            "uniform_external_metrics_by_run_and_support": str(
                metrics_output
            ),
            "uniform_external_metrics_summary": str(
                external_summary_output
            ),
            "uniform_partition_stability_pairwise": str(
                stability_pairwise_output
            ),
            "uniform_partition_stability_summary": str(
                stability_summary_output
            ),
            "primary_balanced_comparison": str(
                primary_comparison_output
            ),
        },
    }

    with (
        output_dir
        / "uniform_balanced_evaluation_summary.json"
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
        "Primary balanced comparison"
    )

    print(
        "---------------------------"
    )

    print(
        primary_comparison.to_string(
            index=False
        )
    )

    print()
    print(
        f"Uniform balanced evaluation written to: "
        f"{output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen balanced behavioral community-detection "
            "methods with uniform external genre supports and compare "
            "partition stability across seeds."
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
        "--output-dir",
        type=Path,
        default=Path(
            "results/uniform_balanced_evaluation"
        ),
    )

    parser.add_argument(
        "--supports",
        type=int,
        nargs="+",
        default=DEFAULT_SUPPORTS,
    )

    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_EXPECTED_SEEDS,
    )

    parser.add_argument(
        "--behavioral-baseline-runs-csv",
        type=Path,
        default=Path(
            "results/behavioral_baselines_balanced_grid/"
            "behavioral_baseline_runs.csv"
        ),
    )

    parser.add_argument(
        "--behavioral-baseline-partition-root",
        type=Path,
        default=Path(
            "results/behavioral_baselines_balanced_grid"
        ),
    )

    parser.add_argument(
        "--node2vec-runs-csv",
        type=Path,
        default=Path(
            "results/node2vec_final_candidates/"
            "node2vec_clustering_runs.csv"
        ),
    )

    parser.add_argument(
        "--node2vec-partition-root",
        type=Path,
        default=Path(
            "results/node2vec_final_candidates"
        ),
    )

    parser.add_argument(
        "--smooth2-runs-csv",
        type=Path,
        default=Path(
            "results/behavioral_feature_controls/"
            "behavioral_feature_control_runs.csv"
        ),
    )

    parser.add_argument(
        "--smooth2-partition-root",
        type=Path,
        default=Path(
            "results/behavioral_feature_controls"
        ),
    )

    parser.add_argument(
        "--gae-combined-runs-csv",
        type=Path,
        default=Path(
            "results/graph_autoencoder_final_candidates_no_leakage_summary/"
            "graph_autoencoder_combined_runs.csv"
        ),
    )

    parser.add_argument(
        "--gae-partition-root",
        type=Path,
        default=Path(
            "results/graph_autoencoder_final_candidates_no_leakage"
        ),
    )

    parser.add_argument(
        "--acoustic-baseline-runs-csv",
        type=Path,
        default=Path(
            "results/acoustic_baselines_balanced_grid/"
            "behavioral_baseline_runs.csv"
        ),
    )

    parser.add_argument(
        "--acoustic-baseline-partition-root",
        type=Path,
        default=Path(
            "results/acoustic_baselines_balanced_grid"
        ),
    )

    parser.add_argument(
        "--acoustic-node2vec-runs-csv",
        type=Path,
        default=Path(
            "results/acoustic_node2vec_final_candidates/"
            "node2vec_clustering_runs.csv"
        ),
    )

    parser.add_argument(
        "--acoustic-node2vec-partition-root",
        type=Path,
        default=Path(
            "results/acoustic_node2vec_final_candidates"
        ),
    )

    parser.add_argument(
        "--acoustic-smooth2-runs-csv",
        type=Path,
        default=Path(
            "results/acoustic_feature_controls/"
            "behavioral_feature_control_runs.csv"
        ),
    )

    parser.add_argument(
        "--acoustic-smooth2-partition-root",
        type=Path,
        default=Path(
            "results/acoustic_feature_controls"
        ),
    )

    parser.add_argument(
        "--acoustic-gae-combined-runs-csv",
        type=Path,
        default=Path(
            "results/acoustic_graph_autoencoder_final_candidates_summary/"
            "graph_autoencoder_combined_runs.csv"
        ),
    )

    parser.add_argument(
        "--acoustic-gae-partition-root",
        type=Path,
        default=Path(
            "results/acoustic_graph_autoencoder_final_candidates"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_evaluation(
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        supports=args.supports,
        expected_seeds=args.expected_seeds,
        behavioral_baseline_runs_csv=(
            args.behavioral_baseline_runs_csv
        ),
        behavioral_baseline_partition_root=(
            args.behavioral_baseline_partition_root
        ),
        node2vec_runs_csv=(
            args.node2vec_runs_csv
        ),
        node2vec_partition_root=(
            args.node2vec_partition_root
        ),
        smooth2_runs_csv=(
            args.smooth2_runs_csv
        ),
        smooth2_partition_root=(
            args.smooth2_partition_root
        ),
        gae_combined_runs_csv=(
            args.gae_combined_runs_csv
        ),
        gae_partition_root=(
            args.gae_partition_root
        ),
        acoustic_baseline_runs_csv=(
            args.acoustic_baseline_runs_csv
        ),
        acoustic_baseline_partition_root=(
            args.acoustic_baseline_partition_root
        ),
        acoustic_node2vec_runs_csv=(
            args.acoustic_node2vec_runs_csv
        ),
        acoustic_node2vec_partition_root=(
            args.acoustic_node2vec_partition_root
        ),
        acoustic_smooth2_runs_csv=(
            args.acoustic_smooth2_runs_csv
        ),
        acoustic_smooth2_partition_root=(
            args.acoustic_smooth2_partition_root
        ),
        acoustic_gae_combined_runs_csv=(
            args.acoustic_gae_combined_runs_csv
        ),
        acoustic_gae_partition_root=(
            args.acoustic_gae_partition_root
        ),
    )
