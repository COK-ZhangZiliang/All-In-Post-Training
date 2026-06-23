from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .artifacts import Artifact, artifact_to_dict, utc_now, write_json
from .config import PipelineConfig, StageConfig
from .lineage import build_data_lineage_report


class StageBackend(ABC):
    """Execution backend for a pipeline stage."""

    @abstractmethod
    def run_stage(
        self,
        config: PipelineConfig,
        stage: StageConfig,
        run_dir: Path,
        dependency_artifacts: dict[str, list[Artifact]],
    ) -> list[Artifact]:
        """Run a stage and return its produced artifacts."""


class ManifestBackend(StageBackend):
    """A deterministic backend that materializes stage manifests without training."""

    def run_stage(
        self,
        config: PipelineConfig,
        stage: StageConfig,
        run_dir: Path,
        dependency_artifacts: dict[str, list[Artifact]],
    ) -> list[Artifact]:
        artifact_kind = STAGE_OUTPUT_KIND.get(stage.type, "stage_manifest")
        artifact_path = run_dir / "artifacts" / f"{stage.id}.{artifact_kind}.json"
        command = render_command_hint(config, stage)
        payload = {
            "created_at": utc_now(),
            "pipeline": config.name,
            "stage": {
                "id": stage.id,
                "type": stage.type,
                "depends_on": list(stage.depends_on),
                "params": stage.params,
                "inputs": stage.inputs,
                "outputs": stage.outputs,
            },
            "dependency_artifacts": {
                key: [artifact_to_dict(artifact) for artifact in artifacts]
                for key, artifacts in dependency_artifacts.items()
            },
            "command_hint": command,
            "backend": "manifest",
            "note": "This artifact records the control-plane contract for the stage. Replace the backend to execute real training jobs.",
        }
        if stage.type == "data_ingestion":
            payload["lineage_report"] = build_data_lineage_report(
                config,
                dataset_ids=tuple(str(item) for item in stage.inputs.get("datasets", [])),
                strict=False,
            )
        write_json(artifact_path, payload)
        return [
            Artifact(
                id=f"{stage.id}:{artifact_kind}",
                kind=artifact_kind,
                path=str(artifact_path),
                producer_stage=stage.id,
                metadata={"stage_type": stage.type, "command_hint": command},
            )
        ]


STAGE_OUTPUT_KIND = {
    "data_ingestion": "dataset_manifest",
    "data_mixture": "mixture_manifest",
    "sft": "sft_checkpoint",
    "preference_data": "preference_manifest",
    "reward_model": "reward_model",
    "dpo": "preference_checkpoint",
    "environment_rollout": "rollout_traces",
    "domain_rl": "specialist_checkpoint",
    "rlvr": "rl_checkpoint",
    "opd_distillation": "distilled_checkpoint",
    "evaluation": "evaluation_report",
    "release": "release_manifest",
}


def render_command_hint(config: PipelineConfig, stage: StageConfig) -> list[str]:
    backend = str(stage.params.get("backend", "custom"))
    if stage.type == "sft":
        return [
            backend,
            "train-sft",
            "--base-model",
            config.model.base_model,
            "--output",
            stage.outputs.get("checkpoint", f"checkpoints/{stage.id}"),
        ]
    if stage.type == "dpo":
        return [
            backend,
            "train-dpo",
            "--policy",
            stage.inputs.get("policy", "previous_checkpoint"),
            "--preference-data",
            ",".join(stage.inputs.get("datasets", [])),
        ]
    if stage.type == "rlvr":
        return [
            backend,
            "train-rlvr",
            "--policy",
            stage.inputs.get("policy", "previous_checkpoint"),
            "--reward",
            stage.inputs.get("reward", "verified_reward"),
        ]
    if stage.type == "domain_rl":
        return [
            backend,
            "train-domain-rl",
            "--domain",
            str(stage.params["domain"]),
            "--policy",
            stage.inputs.get("policy", "sft_checkpoint"),
            "--reward",
            stage.inputs.get("reward", "domain_reward"),
            "--output",
            stage.outputs.get("checkpoint", f"checkpoints/{stage.id}"),
        ]
    if stage.type == "opd_distillation":
        return [
            backend,
            "distill",
            "--student",
            stage.inputs.get("student", "rl_checkpoint"),
            "--teachers",
            ",".join(stage.inputs.get("teachers", [])),
        ]
    return [backend, stage.type, "--stage-id", stage.id]
