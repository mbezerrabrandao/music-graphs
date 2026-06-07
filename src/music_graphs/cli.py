from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import build_plan, execute_plan, render_plan
from .report import write_selection_report
from .stages import ALL_STAGES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reproducible music-artist graph pipeline."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper.yaml"),
    )

    parser.add_argument(
        "--mode",
        choices=["reproduce", "tune"],
        default="reproduce",
    )

    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--list-stages", action="store_true")
    parser.add_argument("--write-selection-report", action="store_true")

    parser.add_argument(
        "--from-stage",
        choices=[stage.stage_id for stage in ALL_STAGES],
    )

    parser.add_argument(
        "--to-stage",
        choices=[stage.stage_id for stage in ALL_STAGES],
    )

    parser.add_argument(
        "--force-stage",
        action="append",
        default=[],
        choices=[stage.stage_id for stage in ALL_STAGES],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_stages:
        for stage in ALL_STAGES:
            optional = (
                ""
                if stage.optional_group is None
                else f" [optional: {stage.optional_group}]"
            )

            print(f"{stage.stage_id}{optional}: {stage.description}")

        return

    config = load_config(args.config)

    if args.write_selection_report:
        output = write_selection_report(config)
        print(f"Selection report written to: {output}")

    plan = build_plan(
        config=config,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        forced_stages=set(args.force_stage),
    )

    print(render_plan(plan))

    if args.plan:
        return

    execute_plan(config=config, plan=plan)


if __name__ == "__main__":
    main()
