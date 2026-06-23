from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .artifacts import Artifact, artifact_to_dict, utc_now, write_json
from .backends import ManifestBackend, StageBackend
from .config import PipelineConfig, topological_stage_order, validate_pipeline_config


@dataclass(frozen=True)
class PipelineRunResult:
    run_dir: Path
    stages: tuple[str, ...]
    artifacts: tuple[Artifact, ...]


class PipelineRunner:
    def __init__(self, backend: StageBackend | None = None) -> None:
        self.backend = backend or ManifestBackend()

    def plan(self, config: PipelineConfig) -> list[str]:
        validate_pipeline_config(config)
        return [stage.id for stage in topological_stage_order(config.stages)]

    def run(self, config: PipelineConfig, run_id: str | None = None) -> PipelineRunResult:
        validate_pipeline_config(config)
        actual_run_id = run_id or utc_now().replace(":", "").replace("+00:00", "Z")
        run_dir = config.output_dir / actual_run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        stage_artifacts: dict[str, list[Artifact]] = {}
        all_artifacts: list[Artifact] = []
        ordered_stages = topological_stage_order(config.stages)

        write_json(
            run_dir / "pipeline_config.snapshot.json",
            {
                "name": config.name,
                "version": config.version,
                "model": config.model.__dict__,
                "datasets": [dataset.__dict__ for dataset in config.datasets],
                "stages": [
                    {
                        "id": stage.id,
                        "type": stage.type,
                        "depends_on": list(stage.depends_on),
                        "enabled": stage.enabled,
                        "params": stage.params,
                        "inputs": stage.inputs,
                        "outputs": stage.outputs,
                    }
                    for stage in config.stages
                ],
                "metadata": config.metadata,
            },
        )

        for stage in ordered_stages:
            dependency_artifacts = {
                dependency: stage_artifacts.get(dependency, []) for dependency in stage.depends_on
            }
            produced = self.backend.run_stage(config, stage, run_dir, dependency_artifacts)
            stage_artifacts[stage.id] = produced
            all_artifacts.extend(produced)

        write_json(
            run_dir / "run_manifest.json",
            {
                "created_at": utc_now(),
                "pipeline": config.name,
                "version": config.version,
                "run_dir": str(run_dir),
                "stages": [stage.id for stage in ordered_stages],
                "artifacts": [artifact_to_dict(artifact) for artifact in all_artifacts],
            },
        )
        return PipelineRunResult(
            run_dir=run_dir,
            stages=tuple(stage.id for stage in ordered_stages),
            artifacts=tuple(all_artifacts),
        )

