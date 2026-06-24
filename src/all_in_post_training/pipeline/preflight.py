from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import sys
from typing import Any

from .artifacts import utc_now, write_json
from .audit import build_readiness_report
from .config import PipelineConfig
from .dependencies import optional_dependency_status


TRAINING_EXTRA_PACKAGES = (
    "trl",
    "transformers",
    "accelerate",
    "peft",
    "datasets",
)

DOWNLOAD_HELPER_PACKAGES = (
    "modelscope",
    "huggingface_hub",
)


@dataclass(frozen=True)
class TrainingPreflightResult:
    report_path: Path
    report: dict[str, Any]


def write_training_preflight_report(
    config: PipelineConfig,
    run_id: str | None = None,
    output_root: str | Path | None = None,
    require_cuda: bool = False,
    require_training_extras: bool = False,
) -> TrainingPreflightResult:
    actual_run_id = run_id or utc_now().replace(":", "").replace("+00:00", "Z")
    root = Path(output_root) if output_root is not None else config.output_dir
    report_dir = root / actual_run_id
    report_path = report_dir / "training_preflight_report.json"
    report = build_training_preflight_report(
        config,
        require_cuda=require_cuda,
        require_training_extras=require_training_extras,
    )
    write_json(report_path, report)
    return TrainingPreflightResult(report_path=report_path, report=report)


def build_training_preflight_report(
    config: PipelineConfig,
    require_cuda: bool = False,
    require_training_extras: bool = False,
) -> dict[str, Any]:
    torch_status = optional_dependency_status("torch")
    package_statuses = {
        package: optional_dependency_status(package)
        for package in ("torch", *TRAINING_EXTRA_PACKAGES, *DOWNLOAD_HELPER_PACKAGES)
    }
    cuda = _torch_cuda_report(torch_status["available"])
    readiness = build_readiness_report(config)

    missing_training_extras = [
        package
        for package in TRAINING_EXTRA_PACKAGES
        if not package_statuses[package]["available"]
    ]
    missing_download_helpers = [
        package
        for package in DOWNLOAD_HELPER_PACKAGES
        if not package_statuses[package]["available"]
    ]

    global_blockers: list[str] = []
    if not torch_status["available"]:
        global_blockers.append("PyTorch is not installed.")
    if require_cuda and not cuda["available"]:
        global_blockers.append("CUDA is required but torch.cuda is unavailable.")
    if require_training_extras and missing_training_extras:
        global_blockers.append(
            "Required training extras are missing: " + ", ".join(missing_training_extras)
        )

    execute_blockers = list(global_blockers)
    if missing_training_extras:
        execute_blockers.append(
            "Real TRL SFT execution requires training extras: "
            + ", ".join(missing_training_extras)
        )
    if readiness["status"] != "ready":
        execute_blockers.append(
            "Reference Qwen SFT execution is blocked by readiness audit findings."
        )

    report = {
        "created_at": utc_now(),
        "pipeline": config.name,
        "version": config.version,
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "requirements": {
            "require_cuda": require_cuda,
            "require_training_extras": require_training_extras,
            "training_extra_packages": list(TRAINING_EXTRA_PACKAGES),
            "download_helper_packages": list(DOWNLOAD_HELPER_PACKAGES),
        },
        "packages": package_statuses,
        "cuda": cuda,
        "readiness": {
            "status": readiness["status"],
            "summary": readiness["summary"],
            "blockers": readiness["blockers"],
        },
        "missing": {
            "training_extras": missing_training_extras,
            "download_helpers": missing_download_helpers,
        },
        "modes": {
            "manifest": {"ready": True, "blockers": []},
            "torch_smoke": _mode_status(
                blockers=_torch_mode_blockers(torch_status["available"], cuda, require_cuda)
            ),
            "trl_sft_dry_run": _mode_status(
                blockers=_torch_mode_blockers(torch_status["available"], cuda, require_cuda)
            ),
            "trl_sft_execute": _mode_status(blockers=execute_blockers),
        },
    }
    report["status"] = "ready" if not global_blockers else "blocked"
    return report


def trl_sft_execute_blockers(
    config: PipelineConfig,
    require_cuda: bool = False,
) -> list[str]:
    report = build_training_preflight_report(
        config,
        require_cuda=require_cuda,
        require_training_extras=True,
    )
    return list(report["modes"]["trl_sft_execute"]["blockers"])


def _torch_mode_blockers(
    torch_available: bool,
    cuda: dict[str, Any],
    require_cuda: bool,
) -> list[str]:
    blockers: list[str] = []
    if not torch_available:
        blockers.append("PyTorch is not installed.")
    if require_cuda and not cuda["available"]:
        blockers.append("CUDA is required but torch.cuda is unavailable.")
    return blockers


def _mode_status(blockers: list[str]) -> dict[str, Any]:
    return {"ready": not blockers, "blockers": blockers}


def _torch_cuda_report(torch_available: bool) -> dict[str, Any]:
    if not torch_available:
        return {
            "available": False,
            "device_count": 0,
            "devices": [],
            "torch_version": None,
            "torch_cuda_version": None,
        }

    import torch

    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            device: dict[str, Any] = {
                "index": index,
                "name": torch.cuda.get_device_name(index),
            }
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            except RuntimeError:
                free_bytes, total_bytes = 0, 0
            device["memory"] = {
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
            }
            devices.append(device)

    return {
        "available": cuda_available,
        "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "devices": devices,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
    }
