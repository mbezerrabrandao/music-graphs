from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module() -> Any:
    script = (
        Path(__file__).resolve().parent
        / "evaluate_frozen_balanced_methods.py"
    )

    spec = importlib.util.spec_from_file_location(
        "evaluate_frozen_balanced_methods",
        script,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import: {script}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    module = load_module()
    args = module.parse_args()

    module.run_evaluation(
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
        node2vec_runs_csv=args.node2vec_runs_csv,
        node2vec_partition_root=args.node2vec_partition_root,
        smooth2_runs_csv=args.smooth2_runs_csv,
        smooth2_partition_root=args.smooth2_partition_root,
        gae_combined_runs_csv=args.gae_combined_runs_csv,
        gae_partition_root=args.gae_partition_root,
        acoustic_baseline_runs_csv=args.acoustic_baseline_runs_csv,
        acoustic_baseline_partition_root=args.acoustic_baseline_partition_root,
        acoustic_node2vec_runs_csv=args.acoustic_node2vec_runs_csv,
        acoustic_node2vec_partition_root=args.acoustic_node2vec_partition_root,
        acoustic_smooth2_runs_csv=args.acoustic_smooth2_runs_csv,
        acoustic_smooth2_partition_root=args.acoustic_smooth2_partition_root,
        acoustic_gae_combined_runs_csv=args.acoustic_gae_combined_runs_csv,
        acoustic_gae_partition_root=args.acoustic_gae_partition_root,
    )
