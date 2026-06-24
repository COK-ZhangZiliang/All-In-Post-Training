from __future__ import annotations

import argparse

from .pipeline.audit import audit_pipeline_readiness
from .pipeline.backends import MissingOptionalDependencyError, create_backend
from .pipeline.config import DEFAULT_PIPELINE_CONFIG, load_pipeline_config
from .pipeline.lineage import DataInspectionError, inspect_pipeline_data
from .pipeline.preflight import write_training_preflight_report
from .pipeline.runner import PipelineRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="All-In Post-Training pipeline toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pipeline_parser = subparsers.add_parser("pipeline", help="Operate the post-training pipeline")
    pipeline_subparsers = pipeline_parser.add_subparsers(dest="pipeline_command", required=True)

    pipeline_validate = pipeline_subparsers.add_parser(
        "validate",
        help="Validate a pipeline config",
    )
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
    pipeline_run.add_argument(
        "--backend",
        choices=("manifest", "torch-smoke", "trl-sft-dry-run", "trl-sft-execute"),
        default="manifest",
        help="Execution backend for pipeline stages",
    )
    pipeline_run.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail torch-backed runs unless CUDA is available",
    )
    pipeline_run.add_argument(
        "--require-trl",
        action="store_true",
        help="Fail trl-sft-dry-run runs unless the optional TRL package is installed",
    )

    pipeline_inspect = pipeline_subparsers.add_parser(
        "inspect-data", help="Inspect dataset lineage and local fixture data"
    )
    pipeline_inspect.add_argument(
        "--config", default=str(DEFAULT_PIPELINE_CONFIG), help="Path to pipeline config JSON"
    )
    pipeline_inspect.add_argument("--run-id", default=None, help="Stable run id for output report")
    pipeline_inspect.add_argument(
        "--fixture-root",
        default=None,
        help="Optional directory with <dataset_id>.jsonl fixture files",
    )
    pipeline_inspect.add_argument(
        "--output-root",
        default=None,
        help="Optional root directory for inspection reports",
    )
    pipeline_inspect.add_argument(
        "--strict",
        action="store_true",
        help="Fail when local inspection records contain schema or path errors",
    )

    pipeline_audit = pipeline_subparsers.add_parser(
        "audit-readiness", help="Audit blockers before real training"
    )
    pipeline_audit.add_argument(
        "--config", default=str(DEFAULT_PIPELINE_CONFIG), help="Path to pipeline config JSON"
    )
    pipeline_audit.add_argument("--run-id", default=None, help="Stable run id for output report")
    pipeline_audit.add_argument(
        "--output-root",
        default=None,
        help="Optional root directory for readiness audit reports",
    )

    pipeline_preflight = pipeline_subparsers.add_parser(
        "preflight", help="Inspect runtime readiness for training backends"
    )
    pipeline_preflight.add_argument(
        "--config", default=str(DEFAULT_PIPELINE_CONFIG), help="Path to pipeline config JSON"
    )
    pipeline_preflight.add_argument(
        "--run-id",
        default=None,
        help="Stable run id for output report",
    )
    pipeline_preflight.add_argument(
        "--output-root",
        default=None,
        help="Optional root directory for preflight reports",
    )
    pipeline_preflight.add_argument(
        "--require-cuda",
        action="store_true",
        help="Mark torch-backed modes blocked unless CUDA is available",
    )
    pipeline_preflight.add_argument(
        "--require-training-extras",
        action="store_true",
        help="Mark preflight blocked unless TRL training extras are installed",
    )

    args = parser.parse_args(argv)

    if args.command == "pipeline":
        config = load_pipeline_config(args.config)
        if args.pipeline_command == "validate":
            print(f"ok: {args.config} is a valid pipeline config")
            return 0
        if args.pipeline_command == "plan":
            runner = PipelineRunner()
            for index, stage_id in enumerate(runner.plan(config), start=1):
                print(f"{index:02d}. {stage_id}")
            return 0
        if args.pipeline_command == "run":
            try:
                backend = create_backend(
                    args.backend,
                    require_cuda=args.require_cuda,
                    require_trl=args.require_trl,
                )
            except MissingOptionalDependencyError as exc:
                print(f"error: {exc}")
                return 1
            runner = PipelineRunner(backend=backend)
            try:
                result = runner.run(config, run_id=args.run_id)
            except RuntimeError as exc:
                print(f"error: {exc}")
                return 1
            print(f"run_dir={result.run_dir}")
            print(f"stages={len(result.stages)} artifacts={len(result.artifacts)}")
            print(f"backend={args.backend}")
            return 0
        if args.pipeline_command == "inspect-data":
            try:
                result = inspect_pipeline_data(
                    config,
                    run_id=args.run_id,
                    fixture_root=args.fixture_root,
                    output_root=args.output_root,
                    strict=args.strict or None,
                )
            except DataInspectionError as exc:
                print(f"error: {exc}")
                return 1
            summary = result.report["summary"]
            print(f"report={result.report_path}")
            print(
                "datasets={datasets} inspected={inspected} "
                "remote_references={remote_references} missing_local={missing_local} "
                "warnings={warnings} rejected_records={rejected_records} "
                "errors={errors}".format(**summary)
            )
            return 0
        if args.pipeline_command == "audit-readiness":
            result = audit_pipeline_readiness(
                config,
                run_id=args.run_id,
                output_root=args.output_root,
            )
            summary = result.report["summary"]
            print(f"report={result.report_path}")
            print(
                "status={status} datasets={datasets} blockers={blockers} "
                "model_blockers={model_blockers} dataset_blockers={dataset_blockers}".format(
                    status=result.report["status"],
                    **summary,
                )
            )
            return 0
        if args.pipeline_command == "preflight":
            result = write_training_preflight_report(
                config,
                run_id=args.run_id,
                output_root=args.output_root,
                require_cuda=args.require_cuda,
                require_training_extras=args.require_training_extras,
            )
            missing = result.report["missing"]
            cuda = result.report["cuda"]
            print(f"report={result.report_path}")
            print(
                "status={status} cuda_available={cuda_available} "
                "missing_training_extras={missing_training_extras} "
                "missing_download_helpers={missing_download_helpers}".format(
                    status=result.report["status"],
                    cuda_available=cuda["available"],
                    missing_training_extras=len(missing["training_extras"]),
                    missing_download_helpers=len(missing["download_helpers"]),
                )
            )
            for mode, status in result.report["modes"].items():
                state = "ready" if status["ready"] else "blocked"
                print(f"mode={mode} status={state} blockers={len(status['blockers'])}")
            return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
