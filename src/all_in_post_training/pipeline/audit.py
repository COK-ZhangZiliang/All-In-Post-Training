from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import utc_now, write_json
from .config import PipelineConfig, validate_pipeline_config


TRAINING_APPROVED_LICENSE_STATUSES = {"approved_for_training", "internal_only", "compatible"}
TRAINING_BLOCKED_LICENSE_STATUSES = {"blocked", "needs_review", "unknown", "verified"}
TRAINING_BLOCKED_CONTAMINATION_STATUSES = {"blocked_until_decontaminated", "unknown"}


@dataclass(frozen=True)
class ReadinessAuditResult:
    run_dir: Path
    report_path: Path
    report: dict[str, Any]


def audit_pipeline_readiness(
    config: PipelineConfig,
    run_id: str | None = None,
    output_root: str | Path | None = None,
) -> ReadinessAuditResult:
    validate_pipeline_config(config)
    actual_run_id = run_id or utc_now().replace(":", "").replace("+00:00", "Z")
    run_dir = Path(output_root) / actual_run_id if output_root else config.output_dir / actual_run_id
    report = build_readiness_report(config)
    report_path = run_dir / "readiness_audit_report.json"
    write_json(report_path, report)
    return ReadinessAuditResult(run_dir=run_dir, report_path=report_path, report=report)


def build_readiness_report(config: PipelineConfig) -> dict[str, Any]:
    validate_pipeline_config(config)
    model_blockers = _model_blockers(config)
    dataset_entries = [_dataset_readiness(dataset) for dataset in config.datasets]
    dataset_blockers = [
        blocker
        for dataset in dataset_entries
        for blocker in dataset["blockers"]
    ]
    blockers = model_blockers + dataset_blockers
    return {
        "created_at": utc_now(),
        "pipeline": config.name,
        "version": config.version,
        "status": "ready" if not blockers else "blocked",
        "model": {
            "name": config.model.name,
            "base_model": config.model.base_model,
            "source_url": config.model.source_url,
            "revision": config.model.revision,
            "revision_status": config.model.revision_status,
            "license": config.model.license,
            "license_status": config.model.license_status,
            "review_checklist": list(config.model.review_checklist),
            "blockers": model_blockers,
        },
        "datasets": dataset_entries,
        "summary": {
            "datasets": len(dataset_entries),
            "blockers": len(blockers),
            "model_blockers": len(model_blockers),
            "dataset_blockers": len(dataset_blockers),
            "approved_datasets": sum(
                1 for dataset in dataset_entries if dataset["status"] == "ready"
            ),
        },
        "blockers": blockers,
    }


def _model_blockers(config: PipelineConfig) -> list[str]:
    blockers: list[str] = []
    if config.model.license_status != "approved_for_training":
        blockers.append(
            "model license_status must be approved_for_training before real training"
        )
    if config.model.revision_status != "pinned":
        blockers.append("model revision_status must be pinned before real training")
    if not config.model.tokenizer_revision or config.model.tokenizer_revision == "main":
        blockers.append("tokenizer_revision must be pinned before real training")
    return blockers


def _dataset_readiness(dataset: Any) -> dict[str, Any]:
    blockers: list[str] = []
    if dataset.license_status in TRAINING_BLOCKED_LICENSE_STATUSES:
        blockers.append(
            f"dataset {dataset.id} license_status is {dataset.license_status}; "
            "training requires approved_for_training, compatible, or internal_only"
        )
    elif dataset.license_status not in TRAINING_APPROVED_LICENSE_STATUSES:
        blockers.append(f"dataset {dataset.id} has unknown training license state")

    if dataset.role == "evaluation":
        if dataset.contamination_status in TRAINING_BLOCKED_CONTAMINATION_STATUSES:
            blockers.append(
                f"dataset {dataset.id} contamination_status is "
                f"{dataset.contamination_status}; evaluation data must be decontaminated"
            )
    elif dataset.contamination_status == "unknown":
        blockers.append(f"dataset {dataset.id} contamination_status is unknown")

    return {
        "id": dataset.id,
        "role": dataset.role,
        "domain": dataset.domain,
        "license": dataset.license,
        "license_status": dataset.license_status,
        "contamination_status": dataset.contamination_status,
        "schema": dataset.schema,
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
    }
