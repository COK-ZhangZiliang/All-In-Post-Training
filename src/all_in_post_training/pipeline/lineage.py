from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import utc_now, write_json
from .config import DatasetConfig, PipelineConfig, validate_pipeline_config


REMOTE_PREFIXES = ("hf://", "modelscope://", "http://", "https://", "s3://", "gs://")


class DataInspectionError(ValueError):
    """Raised when strict data lineage inspection finds invalid local data."""


@dataclass(frozen=True)
class DataInspectionResult:
    run_dir: Path
    report_path: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class DatasetSource:
    files: tuple[Path, ...]
    source_kind: str
    manifest: dict[str, Any] | None = None
    errors: tuple[str, ...] = ()


def inspect_pipeline_data(
    config: PipelineConfig,
    run_id: str | None = None,
    fixture_root: str | Path | None = None,
    output_root: str | Path | None = None,
    strict: bool | None = None,
) -> DataInspectionResult:
    validate_pipeline_config(config)
    actual_run_id = run_id or utc_now().replace(":", "").replace("+00:00", "Z")
    run_dir = Path(output_root) / actual_run_id if output_root else config.output_dir / actual_run_id
    should_be_strict = fixture_root is not None if strict is None else strict
    report = build_data_lineage_report(
        config,
        fixture_root=fixture_root,
        strict=should_be_strict,
    )
    report_path = run_dir / "data_lineage_report.json"
    write_json(report_path, report)
    if should_be_strict and report["summary"]["errors"]:
        raise DataInspectionError(
            f"data inspection failed with {report['summary']['errors']} error(s); "
            f"report={report_path}"
        )
    return DataInspectionResult(run_dir=run_dir, report_path=report_path, report=report)


def build_data_lineage_report(
    config: PipelineConfig,
    dataset_ids: list[str] | tuple[str, ...] | None = None,
    fixture_root: str | Path | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    validate_pipeline_config(config)
    selected = set(dataset_ids or [dataset.id for dataset in config.datasets])
    datasets = [dataset for dataset in config.datasets if dataset.id in selected]
    entries = [
        inspect_dataset(dataset, fixture_root=fixture_root, strict=strict) for dataset in datasets
    ]
    errors = sum(len(entry["inspection"]["errors"]) for entry in entries)
    warnings = sum(len(entry["inspection"].get("warnings", [])) for entry in entries)
    rejected = sum(
        entry["inspection"].get("quality", {}).get("rejected_count", 0) for entry in entries
    )
    inspected = sum(1 for entry in entries if entry["inspection"]["status"] == "ok")
    remote_references = sum(
        1 for entry in entries if entry["inspection"]["status"] == "remote_reference"
    )
    missing_local = sum(1 for entry in entries if entry["inspection"]["status"] == "missing_local")
    return {
        "created_at": utc_now(),
        "pipeline": config.name,
        "version": config.version,
        "model": model_lineage_record(config),
        "datasets": entries,
        "summary": {
            "datasets": len(entries),
            "inspected": inspected,
            "remote_references": remote_references,
            "missing_local": missing_local,
            "warnings": warnings,
            "rejected_records": rejected,
            "errors": errors,
        },
    }


def model_lineage_record(config: PipelineConfig) -> dict[str, Any]:
    return {
        "name": config.model.name,
        "base_model": config.model.base_model,
        "source_url": config.model.source_url,
        "revision": config.model.revision,
        "revision_status": config.model.revision_status,
        "tokenizer": config.model.tokenizer,
        "tokenizer_revision": config.model.tokenizer_revision,
        "license": config.model.license,
        "license_status": config.model.license_status,
        "precision": config.model.precision,
        "max_sequence_length": config.model.max_sequence_length,
        "chat_template": config.model.chat_template,
        "review_checklist": list(config.model.review_checklist),
    }


def inspect_dataset(
    dataset: DatasetConfig,
    fixture_root: str | Path | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    source = _resolve_dataset_source(dataset, fixture_root=fixture_root)
    if source.files:
        inspection = _inspect_jsonl_files(dataset, source)
        if source.errors:
            inspection["errors"].extend(source.errors)
            inspection["status"] = "error"
            inspection["quality"]["status"] = "error"
    elif source.errors:
        inspection = {
            "status": "error",
            "source_kind": source.source_kind,
            "record_count": 0,
            "fingerprint": None,
            "files": [],
            "manifest": source.manifest,
            "quality": _empty_quality_summary(),
            "errors": list(source.errors),
            "warnings": [],
        }
    elif _is_remote_reference(dataset.path):
        inspection = {
            "status": "remote_reference",
            "source_kind": source.source_kind,
            "record_count": None,
            "fingerprint": None,
            "files": [],
            "manifest": source.manifest,
            "quality": _empty_quality_summary(),
            "errors": [],
        }
    else:
        errors = [f"dataset {dataset.id} local path did not resolve: {dataset.path}"]
        inspection = {
            "status": "missing_local",
            "source_kind": source.source_kind,
            "record_count": 0,
            "fingerprint": None,
            "files": [],
            "manifest": source.manifest,
            "quality": _empty_quality_summary(),
            "errors": errors if strict else [],
            "warnings": errors,
        }

    return {
        "id": dataset.id,
        "path": dataset.path,
        "source_url": dataset.source_url,
        "role": dataset.role,
        "domain": dataset.domain,
        "task_role": dataset.task_role,
        "format": dataset.format,
        "schema": dataset.schema,
        "required_columns": list(dataset.required_columns),
        "split": dataset.split,
        "split_policy": dataset.split_policy,
        "license": dataset.license,
        "license_status": dataset.license_status,
        "contamination_status": dataset.contamination_status,
        "quality_filters": list(dataset.quality_filters),
        "inspection": inspection,
    }


def _resolve_dataset_source(
    dataset: DatasetConfig,
    fixture_root: str | Path | None = None,
) -> DatasetSource:
    if fixture_root is not None:
        root = Path(fixture_root)
        manifest = root / f"{dataset.id}.manifest.json"
        if manifest.exists():
            return _source_from_manifest(dataset, manifest, root)
        fixture = root / f"{dataset.id}.jsonl"
        if fixture.exists():
            return DatasetSource(files=(fixture,), source_kind="fixture")

    if _is_remote_reference(dataset.path):
        return DatasetSource(files=(), source_kind="remote")

    matched = [Path(path) for path in glob.glob(dataset.path)]
    if not matched and Path(dataset.path).exists():
        matched = [Path(dataset.path)]

    files: list[Path] = []
    for path in sorted(matched):
        if path.suffix == ".json":
            return _source_from_manifest(dataset, path, path.parent)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
        elif path.suffix == ".jsonl":
            files.append(path)
    return DatasetSource(files=tuple(files), source_kind="local")


def _source_from_manifest(dataset: DatasetConfig, manifest_path: Path, root: Path) -> DatasetSource:
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return DatasetSource(
            files=(),
            source_kind="manifest",
            manifest={"path": str(manifest_path)},
            errors=(f"{manifest_path}: invalid JSON: {exc.msg}",),
        )
    if not isinstance(manifest, dict):
        return DatasetSource(
            files=(),
            source_kind="manifest",
            manifest={"path": str(manifest_path)},
            errors=(f"{manifest_path}: manifest must be a JSON object",),
        )

    if manifest.get("schema") != "dataset_manifest.v1":
        errors.append(f"{manifest_path}: schema must be dataset_manifest.v1")
    if manifest.get("dataset_id") != dataset.id:
        errors.append(f"{manifest_path}: dataset_id must be {dataset.id}")
    if manifest.get("license_status") != dataset.license_status:
        errors.append(f"{manifest_path}: license_status must match pipeline dataset metadata")

    files: list[Path] = []
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        errors.append(f"{manifest_path}: shards must be a non-empty list")
    else:
        for index, shard in enumerate(shards):
            if not isinstance(shard, dict) or not shard.get("path"):
                errors.append(f"{manifest_path}: shard {index} must define path")
                continue
            shard_path = Path(str(shard["path"]))
            if not shard_path.is_absolute():
                shard_path = root / shard_path
            if not shard_path.exists():
                errors.append(f"{manifest_path}: shard path does not exist: {shard_path}")
                continue
            if shard_path.is_dir():
                files.extend(sorted(shard_path.rglob("*.jsonl")))
            else:
                files.append(shard_path)

    manifest_record = {
        "path": str(manifest_path),
        "schema": manifest.get("schema"),
        "dataset_id": manifest.get("dataset_id"),
        "source": manifest.get("source", {}),
        "expected_record_count": manifest.get("expected_record_count"),
        "shards": manifest.get("shards", []),
    }
    return DatasetSource(
        files=tuple(sorted(files)),
        source_kind="manifest",
        manifest=manifest_record,
        errors=tuple(errors),
    )


def _inspect_jsonl_files(dataset: DatasetConfig, source: DatasetSource) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    hasher = hashlib.sha256()
    seen_prompts: dict[str, str] = {}
    record_count = 0
    rejected_count = 0
    warning_count = 0

    for path in sorted(source.files):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{line_number}: invalid JSON: {exc.msg}")
                    continue
                if not isinstance(record, dict):
                    errors.append(f"{path}:{line_number}: record must be a JSON object")
                    continue
                record_count += 1
                record_errors = _validate_required_columns(dataset, record, path, line_number)
                record_errors.extend(
                    _validate_quality_gates(dataset, record, path, line_number, seen_prompts)
                )
                if record_errors:
                    rejected_count += 1
                    errors.extend(record_errors)
                normalized = json.dumps(record, sort_keys=True, separators=(",", ":"))
                hasher.update(normalized.encode("utf-8"))
                hasher.update(b"\n")

    if record_count == 0:
        errors.append(f"dataset {dataset.id} did not contain any JSONL records")
    expected = _expected_record_count(source.manifest)
    if expected is not None and expected != record_count:
        warning_count += 1
        warnings.append(
            f"dataset {dataset.id} expected {expected} record(s) but found {record_count}"
        )

    return {
        "status": "ok" if not errors else "error",
        "source_kind": source.source_kind,
        "record_count": record_count,
        "fingerprint": hasher.hexdigest() if record_count else None,
        "files": [str(path) for path in source.files],
        "manifest": source.manifest,
        "quality": {
            "status": "ok" if not errors else "error",
            "checked_records": record_count,
            "rejected_count": rejected_count,
            "warning_count": warning_count,
            "checks": _quality_checks_for_dataset(dataset),
        },
        "errors": errors,
        "warnings": warnings,
    }


def _validate_required_columns(
    dataset: DatasetConfig,
    record: dict[str, Any],
    path: Path,
    line_number: int,
) -> list[str]:
    errors: list[str] = []
    for column in dataset.required_columns:
        value = record.get(column)
        if value is None:
            errors.append(f"{path}:{line_number}: missing required column {column}")
        elif isinstance(value, str) and not value.strip():
            errors.append(f"{path}:{line_number}: required column {column} is empty")
        elif isinstance(value, list) and not value:
            errors.append(f"{path}:{line_number}: required column {column} is empty")
    return errors


def _validate_quality_gates(
    dataset: DatasetConfig,
    record: dict[str, Any],
    path: Path,
    line_number: int,
    seen_prompts: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    prompt = _record_prompt(record)
    if prompt is None or not prompt.strip():
        errors.append(f"{path}:{line_number}: prompt-like content is empty")
    else:
        prompt_key = " ".join(prompt.split())
        previous = seen_prompts.get(prompt_key)
        if previous is not None:
            errors.append(f"{path}:{line_number}: duplicate prompt also seen at {previous}")
        else:
            seen_prompts[prompt_key] = f"{path}:{line_number}"

    if dataset.schema == "sft_chat" and not _has_non_empty_assistant_message(record):
        errors.append(f"{path}:{line_number}: sft_chat record has no non-empty assistant message")
    if dataset.schema == "math_rl" and not str(record.get("answer", "")).strip():
        errors.append(f"{path}:{line_number}: math_rl record is missing final answer")
    if dataset.schema == "code_rl" and not record.get("tests"):
        errors.append(f"{path}:{line_number}: code_rl record is missing tests")
    if dataset.schema == "tool_agent_rl" and not record.get("tools"):
        errors.append(f"{path}:{line_number}: tool_agent_rl record is missing tools")
    if dataset.schema == "safety_rl" and not str(record.get("policy_label", "")).strip():
        errors.append(f"{path}:{line_number}: safety_rl record is missing policy_label")
    if dataset.schema == "opd_prompt" and not str(record.get("domain", "")).strip():
        errors.append(f"{path}:{line_number}: opd_prompt record is missing domain")
    if dataset.schema == "evaluation":
        if not str(record.get("benchmark", "")).strip():
            errors.append(f"{path}:{line_number}: evaluation record is missing benchmark")
        if not str(record.get("target", "")).strip():
            errors.append(f"{path}:{line_number}: evaluation record is missing target")
    return errors


def _record_prompt(record: dict[str, Any]) -> str | None:
    prompt = record.get("prompt")
    if isinstance(prompt, str):
        return prompt
    messages = record.get("messages")
    if isinstance(messages, list):
        parts = []
        for item in messages:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                parts.append(item["content"])
        return "\n".join(parts)
    return None


def _has_non_empty_assistant_message(record: dict[str, Any]) -> bool:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return False
    for item in messages:
        if (
            isinstance(item, dict)
            and item.get("role") == "assistant"
            and isinstance(item.get("content"), str)
            and item["content"].strip()
        ):
            return True
    return False


def _quality_checks_for_dataset(dataset: DatasetConfig) -> list[str]:
    checks = ["required_columns", "prompt_non_empty", "duplicate_prompt"]
    if dataset.schema == "sft_chat":
        checks.append("assistant_message_non_empty")
    if dataset.schema == "math_rl":
        checks.append("final_answer_present")
    if dataset.schema == "code_rl":
        checks.append("tests_present")
    if dataset.schema == "evaluation":
        checks.extend(["target_present", "benchmark_present"])
    if dataset.schema == "tool_agent_rl":
        checks.extend(["tools_present", "goal_present"])
    if dataset.schema == "safety_rl":
        checks.append("policy_label_present")
    if dataset.schema == "opd_prompt":
        checks.append("domain_present")
    return checks


def _empty_quality_summary() -> dict[str, Any]:
    return {
        "status": "not_inspected",
        "checked_records": 0,
        "rejected_count": 0,
        "warning_count": 0,
        "checks": [],
    }


def _expected_record_count(manifest: dict[str, Any] | None) -> int | None:
    if manifest is None:
        return None
    value = manifest.get("expected_record_count")
    if value is None:
        return None
    return int(value)


def _is_remote_reference(path: str) -> bool:
    return path.startswith(REMOTE_PREFIXES)
