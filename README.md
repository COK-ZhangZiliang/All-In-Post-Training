<p align="center">
  <img src="assets/icon.svg" alt="All-In Post-Training icon" width="112" height="112">
</p>

<h1 align="center">All-In Post-Training</h1>

<p align="center">
  A full-stack, config-driven control plane for LLM post-training pipelines.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-0f766e"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-2563eb">
  <img alt="Status" src="https://img.shields.io/badge/status-pipeline%20control%20plane-2563eb">
</p>

## Purpose

All-In Post-Training is intended to become a comprehensive post-training pipeline. The repository should orchestrate the practical workflow around modern LLM post-training: dataset ingestion, mixture design, SFT, preference data, reward modeling, DPO, environment rollouts, RLVR, on-policy distillation, evaluation, release packaging, and artifact tracking.

The current implementation is the first backend-oriented control plane. It validates a full pipeline configuration, orders stages by dependency, materializes per-stage artifact manifests, and records a run manifest that future training backends can replace with real jobs from TRL, verl, OpenRLHF, custom launchers, or internal systems.

## Current Capabilities

- Pipeline config schema: `examples/post_training_pipeline.json` defines an end-to-end post-training workflow.
- Dependency validation: stage IDs, dataset references, stage types, and dependency cycles are checked before a run starts.
- Stage planning: the CLI prints the topological execution order for the pipeline.
- Pipeline run manifests: the manifest backend creates deterministic artifacts for each stage and records `run_manifest.json`.
- Extensible execution backend: `StageBackend` is the extension point for real training jobs, schedulers, sandbox rollouts, or cluster launchers.
- Project icon: `assets/icon.svg` is kept as the repository mark.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
all-in-post-training pipeline validate --config examples/post_training_pipeline.json
all-in-post-training pipeline plan --config examples/post_training_pipeline.json
all-in-post-training pipeline run --config examples/post_training_pipeline.json --run-id smoke
```

Without installing the package:

```bash
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline validate --config examples/post_training_pipeline.json
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline plan --config examples/post_training_pipeline.json
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id smoke
```

The smoke run writes ignored local artifacts under:

```text
runs/smoke/
├── artifacts/
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

## Repository Structure

```text
.
├── AGENTS.md                         # Project collaboration, validation, and Git rules
├── PLAN.md                           # Pipeline roadmap and milestone tracker
├── README.md                         # Project overview
├── examples/post_training_pipeline.json
├── src/all_in_post_training/
│   ├── pipeline/                     # Pipeline config, backends, artifacts, and runner
│   ├── cli.py                        # Command-line entry point
├── assets/icon.svg                   # Project icon
└── tests/                            # Offline unit tests
```

## Common Commands

```bash
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline validate --config examples/post_training_pipeline.json
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline plan --config examples/post_training_pipeline.json
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id smoke
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## License

This project is released under the [MIT License](LICENSE).
