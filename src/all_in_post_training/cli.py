from __future__ import annotations

import argparse

from .pipeline.config import DEFAULT_PIPELINE_CONFIG, load_pipeline_config
from .pipeline.runner import PipelineRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="All-In Post-Training pipeline toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pipeline_parser = subparsers.add_parser("pipeline", help="Operate the post-training pipeline")
    pipeline_subparsers = pipeline_parser.add_subparsers(dest="pipeline_command", required=True)

    pipeline_validate = pipeline_subparsers.add_parser("validate", help="Validate a pipeline config")
    pipeline_validate.add_argument(
        "--config", default=str(DEFAULT_PIPELINE_CONFIG), help="Path to pipeline config JSON"
    )

    pipeline_plan = pipeline_subparsers.add_parser("plan", help="Print the stage execution order")
    pipeline_plan.add_argument(
        "--config", default=str(DEFAULT_PIPELINE_CONFIG), help="Path to pipeline config JSON"
    )

    pipeline_run = pipeline_subparsers.add_parser("run", help="Run the pipeline control plane")
    pipeline_run.add_argument(
        "--config", default=str(DEFAULT_PIPELINE_CONFIG), help="Path to pipeline config JSON"
    )
    pipeline_run.add_argument("--run-id", default=None, help="Stable run id for output artifacts")

    args = parser.parse_args(argv)

    if args.command == "pipeline":
        config = load_pipeline_config(args.config)
        runner = PipelineRunner()
        if args.pipeline_command == "validate":
            print(f"ok: {args.config} is a valid pipeline config")
            return 0
        if args.pipeline_command == "plan":
            for index, stage_id in enumerate(runner.plan(config), start=1):
                print(f"{index:02d}. {stage_id}")
            return 0
        if args.pipeline_command == "run":
            result = runner.run(config, run_id=args.run_id)
            print(f"run_dir={result.run_dir}")
            print(f"stages={len(result.stages)} artifacts={len(result.artifacts)}")
            return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
