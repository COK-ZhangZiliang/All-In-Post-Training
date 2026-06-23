from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import importlib.metadata
import importlib.util
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


class MissingOptionalDependencyError(RuntimeError):
    """Raised when a requested optional training dependency is unavailable."""


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
            "note": (
                "This artifact records the control-plane contract for the stage. "
                "Replace the backend to execute real training jobs."
            ),
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


class TorchSmokeBackend(StageBackend):
    """A tiny executable backend that proves the full graph can run on torch."""

    def __init__(self, require_cuda: bool = False) -> None:
        self.require_cuda = require_cuda

    def run_stage(
        self,
        config: PipelineConfig,
        stage: StageConfig,
        run_dir: Path,
        dependency_artifacts: dict[str, list[Artifact]],
    ) -> list[Artifact]:
        torch = _import_torch()
        device = _select_torch_device(torch, self.require_cuda, "torch-smoke")
        artifact_kind = STAGE_OUTPUT_KIND.get(stage.type, "stage_manifest")
        artifact_path = run_dir / "artifacts" / f"{stage.id}.{artifact_kind}.json"
        command = render_command_hint(config, stage)
        seed = _stable_seed(config.name, config.version, stage.id, stage.type)
        metric = _run_torch_stage_smoke(torch, device, seed, stage)
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
            "backend": "torch_smoke",
            "device": _torch_device_metadata(torch, device),
            "stage_smoke": {
                "seed": seed,
                "metric": metric,
                "note": (
                    "This is an executable GPU/torch smoke artifact, "
                    "not a real model checkpoint."
                ),
            },
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
                metadata={
                    "stage_type": stage.type,
                    "backend": "torch_smoke",
                    "device": str(device),
                    "metric": metric,
                    "command_hint": command,
                },
            )
        ]


class TrlSftDryRunBackend(StageBackend):
    """Run a tiny SFT training step and use torch smoke for the rest of the graph."""

    def __init__(self, require_cuda: bool = False, require_trl: bool = False) -> None:
        self.require_cuda = require_cuda
        self.require_trl = require_trl
        self.fallback = TorchSmokeBackend(require_cuda=require_cuda)

    def run_stage(
        self,
        config: PipelineConfig,
        stage: StageConfig,
        run_dir: Path,
        dependency_artifacts: dict[str, list[Artifact]],
    ) -> list[Artifact]:
        if stage.type != "sft":
            return self.fallback.run_stage(config, stage, run_dir, dependency_artifacts)

        if self.require_trl:
            require_optional_dependency("trl", "trl-sft-dry-run execute mode")
        torch = _import_torch()
        device = _select_torch_device(torch, self.require_cuda, "trl-sft-dry-run")
        artifact_kind = STAGE_OUTPUT_KIND.get(stage.type, "stage_manifest")
        artifact_path = run_dir / "artifacts" / f"{stage.id}.{artifact_kind}.json"
        command = render_command_hint(config, stage)
        seed = _stable_seed(config.name, config.version, stage.id, stage.type, "trl-sft-dry-run")
        training = _run_sft_dry_run(torch, device, seed, config, stage, run_dir)
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
            "backend": "trl_sft_dry_run",
            "adapter": {
                "target": "trl.SFTTrainer",
                "mode": "synthetic_torch_dry_run",
                "trl": optional_dependency_status("trl"),
                "note": (
                    "This adapter executes a tiny synthetic SFT step. Real TRL execution "
                    "remains gated by model, tokenizer, license, and dataset readiness."
                ),
            },
            "device": _torch_device_metadata(torch, device),
            "training": training,
        }
        write_json(artifact_path, payload)
        return [
            Artifact(
                id=f"{stage.id}:{artifact_kind}",
                kind=artifact_kind,
                path=str(artifact_path),
                producer_stage=stage.id,
                metadata={
                    "stage_type": stage.type,
                    "backend": "trl_sft_dry_run",
                    "device": str(device),
                    "checkpoint_dir": training["checkpoint"]["path"],
                    "final_loss": training["losses"][-1]["loss"],
                    "command_hint": command,
                },
            )
        ]


def create_backend(
    name: str,
    require_cuda: bool = False,
    require_trl: bool = False,
) -> StageBackend:
    if name == "manifest":
        return ManifestBackend()
    if name in {"torch-smoke", "torch_smoke"}:
        return TorchSmokeBackend(require_cuda=require_cuda)
    if name in {"trl-sft-dry-run", "trl_sft_dry_run"}:
        return TrlSftDryRunBackend(require_cuda=require_cuda, require_trl=require_trl)
    raise ValueError(f"unknown backend: {name}")


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


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch-smoke backend requires PyTorch. Install torch or use --backend manifest."
        ) from exc
    return torch


def optional_dependency_status(package_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return {"available": False, "package": package_name, "version": None}
    try:
        version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {"available": True, "package": package_name, "version": version}


def require_optional_dependency(package_name: str, purpose: str) -> dict[str, Any]:
    status = optional_dependency_status(package_name)
    if not status["available"]:
        raise MissingOptionalDependencyError(
            f"{purpose} requires optional package {package_name!r}; "
            "install it or run without the require flag"
        )
    return status


def _select_torch_device(torch: Any, require_cuda: bool, backend_name: str) -> Any:
    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise RuntimeError(f"{backend_name} backend requires CUDA, but torch.cuda is unavailable")
    return torch.device("cuda:0" if cuda_available else "cpu")


def _torch_device_metadata(torch: Any, device: Any) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "type": str(device),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
    }


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _run_torch_stage_smoke(
    torch: Any,
    device: Any,
    seed: int,
    stage: StageConfig,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    base = torch.arange(256, device=device, dtype=torch.float32).reshape(16, 16)
    scale = float((seed % 997) + 1) / 997.0
    matrix = (base / 255.0) * scale

    if stage.type == "data_ingestion":
        result = matrix.mean()
    elif stage.type == "data_mixture":
        weights = torch.softmax(matrix[0], dim=0)
        result = weights.sum()
    elif stage.type == "sft":
        result = torch.nn.functional.mse_loss(matrix, torch.zeros_like(matrix))
    elif stage.type == "domain_rl":
        rewards = torch.tanh(matrix @ matrix.T)
        result = rewards.mean()
    elif stage.type == "opd_distillation":
        teacher = torch.softmax(matrix, dim=-1)
        student = torch.log_softmax(matrix * 0.9, dim=-1)
        result = torch.nn.functional.kl_div(student, teacher, reduction="batchmean")
    elif stage.type == "evaluation":
        result = matrix.diag().mean()
    elif stage.type == "release":
        result = torch.linalg.vector_norm(matrix)
    else:
        result = matrix.sum()

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return {
        "value": float(result.detach().cpu().item()),
        "tensor_shape": list(matrix.shape),
        "stage_type": stage.type,
    }


def _run_sft_dry_run(
    torch: Any,
    device: Any,
    seed: int,
    config: PipelineConfig,
    stage: StageConfig,
    run_dir: Path,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    vocab_size = _bounded_int(
        stage.params.get("dry_run_vocab_size"),
        default=96,
        minimum=16,
        maximum=4096,
    )
    hidden_size = _bounded_int(
        stage.params.get("dry_run_hidden_size"),
        default=48,
        minimum=8,
        maximum=1024,
    )
    batch_size = _bounded_int(
        stage.params.get("dry_run_batch_size"),
        default=2,
        minimum=1,
        maximum=16,
    )
    requested_seq = _bounded_int(
        stage.params.get("dry_run_seq_length"), default=32, minimum=4, maximum=256
    )
    max_model_seq = config.model.max_sequence_length or requested_seq
    sequence_length = min(requested_seq, max_model_seq)
    steps = _bounded_int(stage.params.get("dry_run_steps"), default=2, minimum=1, maximum=16)
    learning_rate = float(stage.params.get("dry_run_learning_rate", 1e-3))

    model = torch.nn.ModuleDict(
        {
            "embedding": torch.nn.Embedding(vocab_size, hidden_size),
            "lm_head": torch.nn.Linear(hidden_size, vocab_size),
        }
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses: list[dict[str, Any]] = []

    for step in range(1, steps + 1):
        input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)
        labels = torch.roll(input_ids, shifts=-1, dims=1)
        logits = model["lm_head"](model["embedding"](input_ids))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "grad_norm": float(grad_norm.detach().cpu().item()),
            }
        )

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()

    checkpoint_dir = run_dir / "checkpoints" / stage.id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_path = checkpoint_dir / "synthetic_sft_state.pt"
    torch.save(model.state_dict(), weights_path)
    trainer_state = {
        "seed": seed,
        "base_model": config.model.base_model,
        "stage_id": stage.id,
        "mode": "synthetic_torch_dry_run",
        "steps": steps,
        "losses": losses,
        "device": str(device),
    }
    adapter_config = {
        "adapter": "trl_sft_dry_run",
        "target": "trl.SFTTrainer",
        "requires_real_model_for_execute": True,
        "synthetic_model": {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
        },
    }
    write_json(checkpoint_dir / "trainer_state.json", trainer_state)
    write_json(checkpoint_dir / "adapter_config.json", adapter_config)

    return {
        "seed": seed,
        "mode": "synthetic_torch_dry_run",
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "losses": losses,
        "synthetic_batch": {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
        },
        "checkpoint": {
            "path": str(checkpoint_dir),
            "weights": str(weights_path),
            "trainer_state": str(checkpoint_dir / "trainer_state.json"),
            "adapter_config": str(checkpoint_dir / "adapter_config.json"),
        },
    }


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        parsed = default
    else:
        parsed = int(value)
    return max(minimum, min(maximum, parsed))
