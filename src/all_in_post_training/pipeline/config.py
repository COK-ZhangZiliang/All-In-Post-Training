from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_PIPELINE_CONFIG = Path("examples/post_training_pipeline.json")

ALLOWED_STAGE_TYPES = {
    "data_ingestion",
    "data_mixture",
    "sft",
    "preference_data",
    "reward_model",
    "dpo",
    "environment_rollout",
    "domain_rl",
    "rlvr",
    "opd_distillation",
    "evaluation",
    "release",
}


class PipelineConfigError(ValueError):
    """Raised when a post-training pipeline config is invalid."""


@dataclass(frozen=True)
class ModelConfig:
    name: str
    base_model: str
    tokenizer: str | None = None
    revision: str | None = None


@dataclass(frozen=True)
class DatasetConfig:
    id: str
    path: str
    role: str
    format: str
    split: str | None = None
    license: str | None = None


@dataclass(frozen=True)
class StageConfig:
    id: str
    type: str
    depends_on: tuple[str, ...] = ()
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    version: str
    output_dir: Path
    model: ModelConfig
    datasets: tuple[DatasetConfig, ...]
    stages: tuple[StageConfig, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def load_pipeline_config(path: str | Path = DEFAULT_PIPELINE_CONFIG) -> PipelineConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return parse_pipeline_config(raw)


def parse_pipeline_config(raw: dict[str, Any]) -> PipelineConfig:
    _require(raw, ("name", "version", "output_dir", "model", "datasets", "stages"), "pipeline")

    model_raw = raw["model"]
    _require(model_raw, ("name", "base_model"), "model")
    model = ModelConfig(
        name=str(model_raw["name"]),
        base_model=str(model_raw["base_model"]),
        tokenizer=_optional_str(model_raw.get("tokenizer")),
        revision=_optional_str(model_raw.get("revision")),
    )

    datasets = tuple(_parse_dataset(item) for item in raw["datasets"])
    stages = tuple(_parse_stage(item) for item in raw["stages"])

    config = PipelineConfig(
        name=str(raw["name"]),
        version=str(raw["version"]),
        output_dir=Path(str(raw["output_dir"])),
        model=model,
        datasets=datasets,
        stages=stages,
        metadata=dict(raw.get("metadata", {})),
    )
    validate_pipeline_config(config)
    return config


def validate_pipeline_config(config: PipelineConfig) -> None:
    if not config.name:
        raise PipelineConfigError("pipeline name cannot be empty")
    if not config.datasets:
        raise PipelineConfigError("pipeline must define at least one dataset")
    if not config.stages:
        raise PipelineConfigError("pipeline must define at least one stage")

    dataset_ids = _unique_ids([dataset.id for dataset in config.datasets], "dataset")
    stage_ids = _unique_ids([stage.id for stage in config.stages], "stage")

    for dataset in config.datasets:
        if dataset.role not in {
            "sft",
            "preference",
            "reward",
            "rl",
            "distillation",
            "evaluation",
            "safety",
        }:
            raise PipelineConfigError(f"dataset {dataset.id} has unsupported role {dataset.role}")
        if dataset.format not in {"jsonl", "parquet", "hf_dataset", "folder", "manifest"}:
            raise PipelineConfigError(f"dataset {dataset.id} has unsupported format {dataset.format}")

    for stage in config.stages:
        if stage.type not in ALLOWED_STAGE_TYPES:
            raise PipelineConfigError(f"stage {stage.id} has unsupported type {stage.type}")
        for dependency in stage.depends_on:
            if dependency not in stage_ids:
                raise PipelineConfigError(f"stage {stage.id} depends on unknown stage {dependency}")
        for dataset_id in _stage_dataset_refs(stage):
            if dataset_id not in dataset_ids:
                raise PipelineConfigError(f"stage {stage.id} references unknown dataset {dataset_id}")
        if stage.type == "domain_rl" and not stage.params.get("domain"):
            raise PipelineConfigError(f"stage {stage.id} must set params.domain")
        if stage.type == "opd_distillation" and not stage.inputs.get("teachers"):
            raise PipelineConfigError(f"stage {stage.id} must define teacher checkpoints")

    _assert_acyclic(config.stages)


def topological_stage_order(stages: tuple[StageConfig, ...]) -> list[StageConfig]:
    stage_by_id = {stage.id: stage for stage in stages if stage.enabled}
    visited: set[str] = set()
    visiting: set[str] = set()
    ordered: list[StageConfig] = []

    def visit(stage_id: str) -> None:
        if stage_id in visited:
            return
        if stage_id in visiting:
            raise PipelineConfigError(f"cycle detected at stage {stage_id}")
        visiting.add(stage_id)
        stage = stage_by_id[stage_id]
        for dependency in stage.depends_on:
            if dependency in stage_by_id:
                visit(dependency)
        visiting.remove(stage_id)
        visited.add(stage_id)
        ordered.append(stage)

    for stage_id in stage_by_id:
        visit(stage_id)
    return ordered


def _parse_dataset(raw: dict[str, Any]) -> DatasetConfig:
    _require(raw, ("id", "path", "role", "format"), "dataset")
    return DatasetConfig(
        id=str(raw["id"]),
        path=str(raw["path"]),
        role=str(raw["role"]),
        format=str(raw["format"]),
        split=_optional_str(raw.get("split")),
        license=_optional_str(raw.get("license")),
    )


def _parse_stage(raw: dict[str, Any]) -> StageConfig:
    _require(raw, ("id", "type"), "stage")
    depends_on = raw.get("depends_on", ())
    if not isinstance(depends_on, (list, tuple)):
        raise PipelineConfigError(f"stage {raw['id']} depends_on must be a list")
    return StageConfig(
        id=str(raw["id"]),
        type=str(raw["type"]),
        depends_on=tuple(str(item) for item in depends_on),
        enabled=bool(raw.get("enabled", True)),
        params=dict(raw.get("params", {})),
        inputs=dict(raw.get("inputs", {})),
        outputs=dict(raw.get("outputs", {})),
    )


def _assert_acyclic(stages: tuple[StageConfig, ...]) -> None:
    topological_stage_order(stages)


def _stage_dataset_refs(stage: StageConfig) -> set[str]:
    refs: set[str] = set()
    for source in (stage.inputs, stage.params):
        value = source.get("datasets")
        if isinstance(value, list):
            refs.update(str(item) for item in value)
        single = source.get("dataset")
        if single is not None:
            refs.add(str(single))
    return refs


def _require(raw: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in fields if field not in raw]
    if missing:
        raise PipelineConfigError(f"{label} missing required fields: {', '.join(missing)}")


def _unique_ids(ids: list[str], label: str) -> set[str]:
    seen: set[str] = set()
    for item_id in ids:
        if not item_id:
            raise PipelineConfigError(f"{label} id cannot be empty")
        if item_id in seen:
            raise PipelineConfigError(f"duplicate {label} id: {item_id}")
        seen.add(item_id)
    return seen


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
