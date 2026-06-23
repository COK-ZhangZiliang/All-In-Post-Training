<p align="center">
  <img src="assets/icon.svg" alt="All-In Post-Training icon" width="112" height="112">
</p>

<h1 align="center">All-In Post-Training</h1>

<p align="center">
  A backend-first, config-driven control plane for LLM post-training pipelines.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-0f766e"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-2563eb">
  <img alt="Status" src="https://img.shields.io/badge/status-pipeline%20control%20plane-2563eb">
</p>

## Purpose

All-In Post-Training is intended to become a comprehensive post-training pipeline. The repository should orchestrate the practical workflow around modern LLM post-training: dataset ingestion, mixture design, SFT, preference data, reward modeling, DPO, environment rollouts, RLVR, on-policy distillation, evaluation, release packaging, and artifact tracking.

The current implementation is the first backend-oriented control plane. It validates a full pipeline configuration, orders stages by dependency, materializes per-stage artifact manifests, inspects dataset lineage fixtures, runs a tiny torch-based full-flow smoke backend, runs a synthetic SFT dry-run adapter, and records manifests that future training backends can replace with real jobs from TRL, verl, OpenRLHF, custom launchers, or internal systems.

## Current Capabilities

- Pipeline config schema: `examples/post_training_pipeline.json` defines an end-to-end post-training workflow.
- Dependency validation: stage IDs, dataset references, stage types, and dependency cycles are checked before a run starts.
- Dataset lineage inspection: local JSONL fixtures and dataset manifests can be validated for required columns, record counts, license status, quality gates, and deterministic fingerprints.
- Stage planning: the CLI prints the topological execution order for the pipeline.
- Pipeline run manifests: the manifest backend creates deterministic artifacts for each stage, embeds dataset lineage into `ingest_data`, and records `run_manifest.json`.
- Torch smoke execution: the `torch-smoke` backend executes small tensor workloads for every stage, allowing a CUDA container to prove the SFT -> domain RL -> OPD -> evaluation -> release topology runs end to end.
- SFT dry-run adapter: the `trl-sft-dry-run` backend executes a tiny synthetic PyTorch SFT step for `train_sft`, writes checkpoint markers under `runs/<run-id>/checkpoints/`, and uses torch smoke for the remaining stages.
- Extensible execution backend: `StageBackend` is the extension point for real training jobs, schedulers, sandbox rollouts, or cluster launchers.
- Project icon: `assets/icon.svg` is kept as the repository mark.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
all-in-post-training pipeline validate --config examples/post_training_pipeline.json
all-in-post-training pipeline plan --config examples/post_training_pipeline.json
all-in-post-training pipeline inspect-data --config examples/post_training_pipeline.json --fixture-root tests/fixtures/lineage --run-id lineage-smoke
all-in-post-training pipeline audit-readiness --config examples/post_training_pipeline.json --run-id readiness-smoke
all-in-post-training pipeline run --config examples/post_training_pipeline.json --run-id smoke
```

Without installing the package:

```bash
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline validate --config examples/post_training_pipeline.json
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline plan --config examples/post_training_pipeline.json
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline inspect-data --config examples/post_training_pipeline.json --fixture-root tests/fixtures/lineage --run-id lineage-smoke
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline audit-readiness --config examples/post_training_pipeline.json --run-id readiness-smoke
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id smoke
```

For a GPU container with PyTorch and CUDA, run the executable full-flow smoke backend:

```bash
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id gpu-torch-smoke --backend torch-smoke --require-cuda
```

This command intentionally does not train `Qwen/Qwen3.5-2B-Base`. It verifies that the complete pipeline graph can run on torch/CUDA and emit stage artifacts before the real SFT, RL, and OPD launchers are connected.

To run the first SFT adapter dry run on a GPU container:

```bash
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id gpu-trl-sft-dry-run --backend trl-sft-dry-run --require-cuda
```

This dry run trains a tiny synthetic embedding-plus-LM-head model for a few steps, saves ignored checkpoint markers, and records whether the optional `trl` package is installed. Add `--require-trl` when you specifically want the command to fail unless TRL is present.

The lineage command accepts direct fixture files such as `<dataset_id>.jsonl` and manifest files such as `<dataset_id>.manifest.json`. Manifests can reference multiple local JSONL shards without committing real datasets to Git.

The lineage and smoke runs write ignored local artifacts under:

```text
runs/lineage-smoke/
└── data_lineage_report.json
runs/readiness-smoke/
└── readiness_audit_report.json
runs/smoke/
├── artifacts/
├── pipeline_config.snapshot.json
└── run_manifest.json
runs/gpu-torch-smoke/
├── artifacts/
├── pipeline_config.snapshot.json
└── run_manifest.json
runs/gpu-trl-sft-dry-run/
├── artifacts/
├── checkpoints/
├── pipeline_config.snapshot.json
└── run_manifest.json
```

## Reference Pipeline

The initial reference config covers these stages:

1. `ingest_data`: validate and register SFT, RL, distillation, evaluation, and safety datasets.
2. `mix_sft_data`: define the SFT mixture for Qwen3.5-2B-Base.
3. `train_sft`: produce the supervised fine-tuning checkpoint contract.
4. `train_math_rl`: train a math RL specialist from the SFT checkpoint.
5. `train_code_rl`: train a code RL specialist from the SFT checkpoint.
6. `train_tool_agent_rl`: train a tool/agent RL specialist from the SFT checkpoint.
7. `train_safety_instruction_rl`: train a safety/instruction specialist from the SFT checkpoint.
8. `opd_fuse_specialists`: fuse specialist knowledge into one student through OPD.
9. `evaluate_policy`: run capability, regression, specialist-loss, and safety gates.
10. `package_release`: emit release, model-card, dataset-card, and gate artifacts.

## Local Data Layout

Real datasets and model artifacts should stay outside Git. The repository already ignores `datasets/`, `runs/`, `checkpoints/`, and `models/`.

Use this local layout for small auditable samples:

```text
datasets/
└── samples/
    ├── sft_chat/train-00000.jsonl
    ├── math_rl/train-00000.jsonl
    ├── code_rl/train-00000.jsonl
    ├── opd/train-00000.jsonl
    └── eval/final-00000.jsonl
```

Tracked manifest examples live under `examples/dataset_manifests/`. Copy one next to a local sample root, update `dataset_id`, `license_status`, `source`, `shards`, and `expected_record_count`, then run `pipeline inspect-data` against that manifest root.

Before real training, `pipeline audit-readiness` should report no blockers. Current reference config is intentionally blocked until the exact Qwen model revision, tokenizer revision, and remaining dataset licenses are approved.

## Repository Structure

```text
.
├── AGENTS.md                         # Project collaboration, validation, and Git rules
├── PLAN.md                           # Pipeline roadmap and milestone tracker
├── README.md                         # Project overview
├── examples/post_training_pipeline.json
├── examples/dataset_manifests/       # Local sample manifest examples
├── src/all_in_post_training/
│   ├── pipeline/                     # Pipeline config, lineage, backends, artifacts, and runner
│   ├── cli.py                        # Command-line entry point
├── assets/icon.svg                   # Project icon
└── tests/                            # Offline unit tests and lineage fixtures
```

## Common Commands

```bash
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline validate --config examples/post_training_pipeline.json
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline plan --config examples/post_training_pipeline.json
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline inspect-data --config examples/post_training_pipeline.json --fixture-root tests/fixtures/lineage --run-id lineage-smoke
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline audit-readiness --config examples/post_training_pipeline.json --run-id readiness-smoke
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id smoke
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id gpu-torch-smoke --backend torch-smoke --require-cuda
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id gpu-trl-sft-dry-run --backend trl-sft-dry-run --require-cuda
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## License

This project is released under the [MIT License](LICENSE).
