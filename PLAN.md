# All-In Post-Training Plan

Last updated: 2026-06-24

## Mission

Build a comprehensive, backend-first LLM post-training pipeline. The target training recipe is now: supervised fine-tuning first, then independent reinforcement-learning specialists across multiple capability domains, then on-policy distillation to fuse those specialists into one deployable policy. The project should move from research taxonomy into executable infrastructure for data preparation, SFT, domain RL, OPD fusion, evaluation, and release governance.

The earlier static visualization path has been removed. The main product is the pipeline control plane and the training/evaluation backends behind it.

## Corrected Product Direction

The intended system is a practical post-training stack with these layers:

| Layer | Responsibility | Current Status |
|-|-|-|
| Pipeline config | Describe datasets, model lineage, stages, dependencies, parameters, and outputs | Initial JSON config implemented |
| Validation | Fail fast on broken dataset references, unsupported stage types, duplicate IDs, and dependency cycles | Implemented |
| Orchestration | Order stages, run the control plane, and record run manifests | Implemented with manifest backend |
| Artifact tracking | Emit per-stage artifact contracts and a top-level run manifest | Implemented |
| Training backends | Connect stages to torch smoke execution now, then TRL SFT, verl, OpenRLHF, custom launchers, or internal schedulers | Torch smoke and SFT dry run complete; real trainer preflight in progress |
| Data processing | Ingest, deduplicate, filter, license-check, mix, and version datasets | Planned |
| Rollout systems | Run agentic environments, sandbox tools, collect traces, and attach rewards | Planned |
| Evaluation gates | Run capability, regression, safety, and release gates | Planned |
| Release governance | Emit model cards, dataset cards, metrics, and reproducibility metadata | Planned |

## Target Training Recipe

### Base Model

Use `Qwen/Qwen3.5-2B-Base` from ModelScope as the initial base model:

- Model URL: https://modelscope.cn/models/Qwen/Qwen3.5-2B-Base
- First implementation requirement: pin the exact ModelScope revision, tokenizer revision, model license, max context length, chat template assumptions, and supported precision before the first real training run.
- Rationale: a 2B dense base is small enough for fast iteration, LoRA/QLoRA experiments, and single-node smoke tests, while still large enough to expose meaningful SFT/RL/OPD interactions.

### High-Level Flow

```text
Qwen/Qwen3.5-2B-Base
  -> SFT general policy
      -> math RL specialist
      -> code RL specialist
      -> tool/agent RL specialist
      -> safety/instruction RL specialist
  -> OPD fusion student
  -> evaluation gates
  -> release package
```

Key design choice: every RL specialist starts from the same SFT checkpoint. This avoids early multi-objective reward interference and makes each specialist's deltas easier to audit before OPD fusion.

### Stage 1 - SFT General Policy

Goal: teach the base model the chat format, instruction-following behavior, basic reasoning style, coding answer format, and tool-call syntax before RL.

Recommended SFT mixture for the first run:

| Slice | Candidate Sources | Target Share | Notes |
|-|-|-|-|
| General instruction/chat | `allenai/tulu-3-sft-mixture`, `BAAI/Infinity-Instruct`, `HuggingFaceH4/ultrachat_200k`, `OpenAssistant/oasst1` | 45% | Use aggressive quality filters; avoid overfitting the 2B model on overly verbose synthetic conversations. |
| Math reasoning demos | `AI-MO/NuminaMath-CoT`, `nvidia/OpenMathReasoning`, MATH train-style derivations | 20% | Keep final-answer extraction fields for later RL rewards. |
| Code instruction demos | `OpenCodeInstruct`, APPS training split, MBPP train split | 20% | Prefer samples with tests, execution feedback, or canonical solutions. |
| Tool/agent demonstrations | ToolBench, WebShop demonstrations, tau-bench style tool trajectories if license-compatible | 10% | Normalize into function-call messages plus observations. |
| Safety/instruction constraints | safety slices from open post-training mixtures plus internal policy data if available | 5% | Keep this small in SFT and enforce stronger gates during evaluation. |

Initial SFT policy:

- Start with 100k-300k curated examples for the first 2B run instead of ingesting every available row.
- Train one full-rank baseline if resources permit; otherwise start with LoRA/QLoRA to validate data and loss plumbing.
- Keep a held-out validation split per slice to catch catastrophic style or domain drift.
- Output artifact: `qwen3.5-2b-sft`.

### Stage 2 - Domain RL Specialists

Train each specialist independently from `qwen3.5-2b-sft`.

| Specialist | Candidate Data | Reward | First Algorithm | Evaluation Gate |
|-|-|-|-|-|
| Math RL | MATH train, NuminaMath prompts, OpenR1-Math/OpenMathReasoning prompts after decontamination | Exact final-answer match, symbolic equivalence, numeric tolerance, format validity | GRPO via verl-style backend | GSM8K, MATH/MATH500, AIME-style heldouts |
| Code RL | APPS, CodeContests, MBPP, OpenCodeInstruct tasks with executable tests | Compile success, unit-test pass rate, timeout/memory safety, no forbidden IO | GRPO or PPO-lite with sandboxed execution | HumanEval, MBPP, LiveCodeBench-style heldouts |
| Tool/Agent RL | tau-bench, WebShop, ToolBench/StableToolBench-style tool tasks | Goal-state match, tool-call validity, environment success, step budget penalty | GRPO with trajectory rewards; later step-level rewards | tau-bench pass^k, WebShop success, tool-call format regression |
| Safety/Instruction RL | safety prompts, refusal/comply pairs, instruction hierarchy tasks, adversarial format-following tasks | Policy-rule compliance, refusal correctness, helpfulness retention, no unsafe completion | Reward-model or rule-plus-judge RL; keep low KL | Safety benchmark, instruction-following eval, over-refusal checks |

Operational constraints:

- Use a separate adapter/checkpoint namespace per specialist.
- Keep KL to `qwen3.5-2b-sft` for every specialist to prevent style collapse.
- Store rollout traces, reward components, raw verifier outputs, and final normalized rewards.
- Prefer verifiable rewards for math/code/tool domains; learned or judge rewards should be explicitly marked lower-confidence.
- Do not mix specialists during this phase. Cross-domain interference should be handled by OPD, not by a single mixed RL job.

### Stage 3 - OPD Fusion

Goal: merge specialist improvements into one student policy without destructive multi-domain RL interference.

Recommended OPD setup:

- Student initialization: `qwen3.5-2b-sft` for the first fusion run; compare against initializing from the strongest average specialist later.
- Teachers: math RL, code RL, tool/agent RL, and safety/instruction RL specialist checkpoints.
- Routing: classify each on-policy prompt into a domain and query the matching teacher; for mixed prompts, use weighted teacher logits or teacher priority order.
- Objective: combine teacher KL guidance with task rewards where rewards are cheap and reliable.
- KL direction: start with reverse-KL or sampled-token policy-gradient distillation for decisive domains such as math/code; compare forward-KL for general instruction retention.
- Fallback when full logits are expensive: use teacher sampled outputs plus token logprobs, or a TGPO-style top-token guidance objective.
- Mixture for first OPD run: math 30%, code 25%, tool/agent 20%, safety/instruction 15%, general SFT retention 10%.
- Output artifact: `qwen3.5-2b-opd`.

OPD validation must compare:

- Base model vs SFT.
- SFT vs each specialist in its own domain.
- SFT vs OPD on every domain.
- Best specialist vs OPD on its domain to measure fusion loss.
- OPD vs SFT on general chat and safety to catch regressions.

### Source Seeds For Dataset Research

Use these as the first source list; every actual dataset import must still run license, contamination, schema, and quality checks.

| Area | Source | URL | Why It Matters |
|-|-|-|-|
| Open post-training recipe | Tulu 3 | https://arxiv.org/abs/2411.15124 | Open SFT/DPO/RLVR recipe and dataset philosophy for modern post-training. |
| SFT mixture | Tulu 3 SFT mixture | https://huggingface.co/datasets/allenai/tulu-3-sft-mixture | Broad multilingual instruction mixture; useful as a high-level SFT baseline. |
| SFT scale | Infinity-Instruct | https://huggingface.co/datasets/BAAI/Infinity-Instruct | Large instruction dataset with reported Qwen-family fine-tuning experiments. |
| SFT chat | UltraChat 200k | https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k | Compact chat SFT source for format and conversational behavior. |
| Human instruction data | OpenAssistant OASST1 | https://huggingface.co/datasets/OpenAssistant/oasst1 | Human-generated multilingual instruction corpus with quality annotations. |
| Math SFT/RL | NuminaMath-CoT | https://huggingface.co/datasets/AI-MO/NuminaMath-CoT | 860k-row math CoT dataset with Apache-2.0 license metadata on Hugging Face. |
| Math RL/eval | MATH | https://arxiv.org/abs/2103.03874 | Competition math dataset with final-answer verification potential. |
| Math reasoning | OpenMathReasoning | https://huggingface.co/datasets/nvidia/OpenMathReasoning | Large math reasoning source from the AIMO-2 winning recipe. |
| Code RL | APPS | https://arxiv.org/abs/2105.09938 | Code generation tasks evaluated by test cases. |
| Code RL/eval | MBPP | https://arxiv.org/abs/2108.07732 | Small Python program synthesis tasks with unit-test-style evaluation. |
| Code RL | CodeContests | https://arxiv.org/abs/2203.07814 | Competitive programming data and evaluation setting behind AlphaCode. |
| Code SFT | OpenCodeInstruct | https://arxiv.org/abs/2504.04030 | Large code instruction dataset with tests and execution feedback. |
| Tool use | ToolBench / ToolLLM | https://arxiv.org/abs/2307.16789 | Tool-use dataset spanning many real-world APIs. |
| Agent RL | tau-bench | https://arxiv.org/abs/2406.12045 | Tool-agent-user benchmark with goal-state evaluation. |
| Web agent RL | WebShop | https://arxiv.org/abs/2207.01206 | Simulated e-commerce environment with demonstrations and RL-style success rewards. |

## Target Pipeline Shape

The reference pipeline config now follows this shape:

```text
ingest_data
  -> mix_sft_data
  -> train_sft
      -> train_math_rl
      -> train_code_rl
      -> train_tool_agent_rl
      -> train_safety_instruction_rl
  -> opd_fuse_specialists
  -> evaluate_policy
  -> package_release
```

The current implementation executes this shape through a manifest backend. DPO and standalone reward-model training can remain optional adapter stages, but they should not define the main path.

## Architecture

```text
examples/post_training_pipeline.json
        |
        v
src/all_in_post_training/pipeline/config.py
        |  validates schema, datasets, stage types, dependencies
        v
src/all_in_post_training/pipeline/lineage.py
        |  inspects model and dataset lineage fixtures
        v
src/all_in_post_training/pipeline/runner.py
        |  topologically orders stages and drives execution
        v
src/all_in_post_training/pipeline/backends.py
        |  backend interface; manifest backend emits contracts; torch-smoke and SFT dry-run backends execute tiny workloads
        v
src/all_in_post_training/pipeline/preflight.py
        |  runtime dependency, CUDA, and readiness preflight for training backends
        v
runs/<run-id>/
        |-- artifacts/<stage>.<kind>.json
        |-- pipeline_config.snapshot.json
        `-- run_manifest.json
```

## Research Principles

- Treat post-training as an end-to-end system, not a single algorithm.
- Preserve data lineage and model lineage across every stage.
- Prefer verifiable rewards where possible, especially for math, code, tool use, and agentic environments.
- Keep reward models and process rewards auditable because reward hacking is a primary failure mode.
- Treat agentic RL as environment/reward/policy co-design with trace replay and sandboxing.
- Treat OPD and specialist distillation as first-class capability-fusion stages.

## Milestones

### P0 - Pipeline Control Plane

Status: complete

- Add `pipeline` package with config parsing, validation, stage ordering, artifact records, and runner.
- Add a reference post-training pipeline config.
- Add CLI commands for `pipeline validate`, `pipeline plan`, and `pipeline run`.
- Add tests for valid configs, dependency errors, and artifact materialization.

Exit evidence:

- `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline validate --config examples/post_training_pipeline.json`
- `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline plan --config examples/post_training_pipeline.json`
- `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id smoke-sft-rl-opd`
- `PYTHONPATH=src python3 -m unittest discover -s tests -v`

### P1.1 - Dataset Lineage Foundation

Status: complete

Objective: make the reference SFT -> domain RL -> OPD recipe auditable before connecting real training backends. The next implementation should turn the current dataset IDs and model name into validated lineage records with schema, license, source, split, fingerprint, and local inspection results.

Why this comes next:

- Real SFT is blocked until the base model revision, tokenizer assumptions, dataset schemas, and license status are explicit.
- Multi-domain RL will be fragile if math, code, tool-agent, safety, OPD, and evaluation data do not have separate roles and quality gates.
- OPD fusion needs teacher/student lineage and domain routing metadata, so lineage should be solved before the OPD backend.
- This can be tested offline with small fixture datasets before downloading large public corpora or launching GPU jobs.

Implementation slice:

- Add model lineage metadata for `Qwen/Qwen3.5-2B-Base`: source URL, revision placeholder, tokenizer revision, license status, precision assumptions, max sequence length, and chat-template notes.
- Extend dataset records with domain, task role, license status, schema name, expected columns, split policy, contamination status, and quality filters.
- Add a small manifest format for local JSONL/folder datasets and Hugging Face or ModelScope dataset references.
- Implement an offline data inspection command that validates local fixtures, counts records, checks required fields, computes content fingerprints, and writes a lineage report under `runs/<run-id>/`.
- Add fixture datasets for SFT chat, math RL, code RL, tool-agent RL, safety RL, OPD prompts, and evaluation gates.
- Update the manifest backend so `ingest_data` emits a richer dataset lineage artifact instead of only a generic stage manifest.
- Add unit tests for valid lineage records, missing license metadata, missing required columns, duplicate dataset IDs, and deterministic fingerprints.

Acceptance criteria:

- `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline validate --config examples/post_training_pipeline.json` still works offline.
- A new data inspection CLI path runs against fixture datasets without network access.
- The inspection report lists every reference dataset, its role, domain, license status, schema status, and fingerprint or remote-reference marker.
- Invalid fixture records fail with actionable validation errors.
- `PYTHONPATH=src python3 -m unittest discover -s tests -v` covers the new data lineage path.

Non-goals for this slice:

- Do not download full public datasets.
- Do not run SFT or RL training.
- Do not add heavyweight dependencies unless they are required for robust local validation.
- Do not commit generated `runs/` reports or real dataset files.

Completed scope:

- Extended model metadata with source URL, revision status, tokenizer revision, license status, precision assumptions, max sequence length, and chat-template notes.
- Extended dataset metadata with domain, task role, schema, required columns, split policy, license status, contamination status, quality filters, and source URL.
- Added offline JSONL fixture inspection with deterministic content fingerprints and actionable schema errors.
- Added fixture datasets for the reference SFT, math RL, code RL, tool-agent RL, safety RL, OPD, and evaluation datasets.
- Embedded a lineage report into the manifest backend's `ingest_data` artifact.
- Added unit tests for lineage metadata, fixture inspection, missing license metadata, missing required columns, and deterministic fingerprints.

Exit evidence:

- `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline validate --config examples/post_training_pipeline.json`
- `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline inspect-data --config examples/post_training_pipeline.json --fixture-root tests/fixtures/lineage --run-id lineage-smoke`
- `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id lineage-pipeline-smoke`
- `PYTHONPATH=src python3 -m unittest discover -s tests -v`

### Immediate Next Step - P1.2 Local Dataset Manifests And Quality Gates

Status: complete

Objective: move from tiny fixtures to local dataset manifests that can describe sampled public data, internal data, or synthetic data without committing real datasets to Git.

Implementation slice:

- Add a `dataset_manifest` JSON schema that points to local JSONL shards, declares source provenance, license status, split, domain, schema, and expected record counts.
- Extend `inspect-data` to read manifest files and inspect all referenced shards.
- Add quality checks for duplicate prompts, empty prompts, empty assistant responses, missing final answers, missing tests, and missing benchmark labels.
- Add per-dataset quality summaries with record counts, rejected counts, and warning counts.
- Add a small fixture manifest that references multiple JSONL shards to verify folder and manifest handling.
- Keep generated reports under `runs/` and keep real datasets out of Git.

Acceptance criteria:

- `inspect-data` validates both direct fixture JSONL files and fixture manifest files.
- Duplicate and empty-record fixtures fail in strict mode with actionable errors.
- The lineage report includes per-dataset record count, fingerprint, schema status, quality status, and rejection summary.
- Existing `pipeline validate`, `pipeline run`, and unit tests stay green.

Completed scope:

- Added `dataset_manifest.v1` support for local manifests that reference multiple JSONL shards.
- Added fixture manifest coverage for `code_rl_tasks` with two local shards and an expected record count.
- Added quality gates for duplicate prompt-like content, empty prompt-like content, empty assistant responses, missing final answers, missing code tests, missing tool definitions, missing safety labels, missing OPD domains, and missing evaluation targets or benchmark labels.
- Added per-dataset quality summaries with checked record count, rejected count, warning count, and active checks.
- Added strict-mode tests for duplicate prompts and missing quality fields.

Exit evidence:

- `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline validate --config examples/post_training_pipeline.json`
- `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline inspect-data --config examples/post_training_pipeline.json --fixture-root tests/fixtures/lineage --run-id lineage-manifest-smoke`
- `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id quality-pipeline-smoke`
- `PYTHONPATH=src python3 -m unittest discover -s tests -v`

### P1.3 - Source Sampling And License Audit

Status: complete

Objective: prepare the first real, tiny, auditable dataset samples for SFT and one RL domain without committing dataset rows or credentials to Git.

Implementation slice:

- Add a local-only `datasets/` layout convention documented in the README and ignored by Git.
- Add sample manifest examples for SFT chat, math RL, code RL, OPD prompts, and eval gates.
- Add a command or helper mode that checks whether referenced local shards exist and whether sample sizes match manifest expectations.
- Add a model metadata review checklist for the exact `Qwen/Qwen3.5-2B-Base` revision, tokenizer revision, license, precision, max sequence length, and chat-template assumptions.
- Add license audit states that distinguish `needs_review`, `approved_for_training`, `internal_only`, and `blocked`.
- Run one tiny local sample inspection path on the GPU container after the changes are committed and pulled from GitHub.

Acceptance criteria:

- The repository documents where local datasets should live without tracking them.
- Manifests can describe real local sample shards while keeping rows out of Git.
- `inspect-data` can validate those sample manifests offline.
- The plan names exactly what remains blocked before the first real SFT job.

Completed local scope:

- Added `examples/dataset_manifests/` with sample manifests for SFT chat, math RL, code RL, OPD prompts, and final evaluation.
- Documented the local-only `datasets/samples/` layout in the README while keeping `datasets/` ignored by Git.
- Added explicit license states for `approved_for_training` and `blocked`; retained `needs_review`, `internal_only`, `compatible`, and legacy `verified` for migration.
- Added model review checklist metadata to the reference Qwen config.
- Added `pipeline audit-readiness` to emit `runs/<run-id>/readiness_audit_report.json`.
- Added readiness blockers for unpinned model revisions, unapproved model licenses, unpinned tokenizer revisions, dataset license states that still need review, and held-out evaluation data that still requires decontamination.

Current blockers before real SFT:

- `Qwen/Qwen3.5-2B-Base` model revision must be pinned to an exact ModelScope revision.
- Tokenizer revision must be pinned.
- Model license must be explicitly approved for the intended training and release use.
- SFT sources still marked `needs_review` must be approved or removed from the first run.
- Evaluation sources marked `blocked_until_decontaminated` must be decontaminated or replaced.

Exit evidence so far:

- `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline audit-readiness --config examples/post_training_pipeline.json --run-id readiness-smoke`
- `PYTHONPATH=src python3 -m unittest discover -s tests -v`

Completed remote scope:

- Committed and pushed the P1.1-P1.3 changes to GitHub.
- Ran `validate`, `inspect-data`, `audit-readiness`, `pipeline run`, `unittest`, and `compileall` from the GPU container using GitHub-managed code.
- Confirmed the readiness audit intentionally blocks real training until model revision, tokenizer revision, model license, dataset licenses, and evaluation decontamination are approved.

Remote exit evidence:

- GPU environment: NVIDIA GeForce RTX 5090, CUDA available through PyTorch.
- `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline validate --config examples/post_training_pipeline.json`
- `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline inspect-data --config examples/post_training_pipeline.json --fixture-root tests/fixtures/lineage --run-id lineage-manifest-smoke`
- `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline audit-readiness --config examples/post_training_pipeline.json --run-id readiness-smoke`
- `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id gpu-manifest-smoke`
- `PYTHONPATH=src python3 -m unittest discover -s tests -v`

Remaining optional follow-up:

- Add a tiny local sample manifest root on the GPU container once real sample rows are available outside Git.

### P1.4 - GPU Full-Flow Smoke Backend

Status: complete

Objective: prove that the complete SFT -> multi-domain RL -> OPD -> evaluation -> release topology can execute inside a CUDA container before the real Qwen SFT/RL/OPD launchers exist.

Implementation slice:

- Add a `torch-smoke` stage backend that lazily imports PyTorch and runs a tiny deterministic tensor workload for each stage type.
- Add a `--backend torch-smoke` CLI option for `pipeline run`.
- Add `--require-cuda` so GPU validation fails clearly when CUDA is unavailable.
- Emit the same artifact kinds as the manifest backend, with torch version, CUDA version, device name, tensor shape, and smoke metric metadata.
- Keep the smoke backend honest: artifacts must state that they are executable torch smoke artifacts, not real model checkpoints.
- Add tests for backend selection and torch-smoke artifact materialization when PyTorch is installed.

Acceptance criteria:

- Local manifest backend validation remains green.
- A CUDA container can run `pipeline run --backend torch-smoke --require-cuda` against the full reference config.
- The smoke run writes 10 stage artifacts and a run manifest.
- Unit tests pass on the GPU container with PyTorch installed.

Completed scope:

- Added the `torch-smoke` backend and CLI selection path.
- Added CUDA-required failure mode through `--require-cuda`.
- Added torch device metadata and smoke metrics to per-stage artifacts.
- Added tests for backend selection and torch-smoke materialization when PyTorch is installed.
- Pushed the implementation to GitHub as commit `e6a170e`.

Exit evidence:

- `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id gpu-torch-smoke --backend torch-smoke --require-cuda`
- `PYTHONPATH=src python3 -m unittest discover -s tests -v`
- `PYTHONPYCACHEPREFIX=/tmp/aitp-pycache python3 -m compileall -q src tests`

Remote GPU result:

- GPU: NVIDIA GeForce RTX 5090, 16 GB VRAM, driver 580.82.07.
- PyTorch: `2.7.0a0+7c8ec84dab.nv25.03`, CUDA `12.8`, `torch.cuda.is_available() == True`.
- Full torch smoke run: `stages=10 artifacts=10 backend=torch-smoke`.
- The SFT artifact records `device.type == "cuda:0"` and `backend == "torch_smoke"`.
- GPU unit tests: 11 passed, 0 skipped.

### P1 - Data and Dataset Lineage

Status: in progress

- Encode the SFT -> domain RL specialists -> OPD fusion recipe in the reference pipeline config. Completed for the manifest-backed reference config.
- Add initial `Qwen/Qwen3.5-2B-Base` model lineage metadata. Exact model revision pin and final license approval remain required before real training.
- Add dataset manifest schemas for SFT, RL, distillation, evaluation, and safety datasets. Preference and reward-model schemas remain optional until those paths return to the main recipe.
- Implement local JSONL fixture inspection with fingerprints and local manifest inspection with multi-shard support. Folder manifests remain planned.
- Add initial data quality checks: duplicates, empty prompts, empty assistant responses, missing final answers, missing tests, missing benchmark labels, and missing license metadata. Preference-pair checks remain optional until preference data returns to the main recipe.
- Add mixture recipes with capability weights and sampling policies.
- Record dataset fingerprints and content hashes.

### P2 - Training Backend Adapters

Status: in progress

- Add backend adapters for TRL SFT first, starting with a tiny LoRA or synthetic-batch dry run.
- Add backend adapter contracts for verl/OpenRLHF GRPO-style RL jobs.
- Add specialist checkpoint namespaces for math, code, tool/agent, and safety/instruction RL.
- Add command rendering, environment variable handling, and dry-run vs execute modes.
- Add resource specs: GPU count, memory hints, distributed strategy, checkpoint cadence.
- Add failure and retry states to run manifests.

### P2.1 - TRL SFT Dry-Run Adapter

Status: complete

Objective: replace the SFT stage's torch-only smoke with the first real trainer integration while keeping the run small enough for a single GPU validation pass.

Implementation slice:

- Add a TRL SFT adapter contract with dry-run and execute modes.
- Keep the first execute mode tiny: a synthetic or fixture-backed batch, LoRA enabled, one or a few optimizer steps, and output under ignored `checkpoints/`.
- Resolve the first real dependency boundary: either optional TRL installation instructions or a runtime check that reports missing packages clearly.
- Materialize a real SFT checkpoint marker, trainer config snapshot, loss sample, and tokenizer/model lineage reference.
- Keep `audit-readiness` authoritative: real public-data training remains blocked until model revision, tokenizer revision, license approval, and dataset license/decontamination issues are resolved.

Acceptance criteria:

- `pipeline run --backend manifest` remains dependency-light and offline.
- `pipeline run --backend torch-smoke --require-cuda` remains the fast CUDA topology check.
- A new SFT adapter can execute a tiny GPU job and emit a checkpoint marker without downloading or committing large datasets.
- Tests cover dry-run command rendering and missing-dependency errors.

Completed scope:

- Added a `trl-sft-dry-run` backend that executes a synthetic PyTorch SFT micro-run for `sft` stages.
- The backend writes ignored checkpoint artifacts under `runs/<run-id>/checkpoints/<stage-id>/`.
- Non-SFT stages use torch smoke so the full reference graph still executes.
- The adapter records optional TRL package availability and supports `--require-trl` for explicit dependency checks.
- Missing optional dependency failures now report clean CLI errors instead of tracebacks.
- Pushed the implementation as commit `d09174a` and the CLI error handling fix as commit `c4e5998`.

Exit evidence:

- Local: `PYTHONPATH=src python3 -m unittest discover -s tests -v` passed with PyTorch-dependent tests skipped because PyTorch is not installed locally.
- Local: `PYTHONPYCACHEPREFIX=/private/tmp/aitp-pycache python3 -m compileall -q src tests`.
- Local: `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id local-require-trl-expected-fail --backend trl-sft-dry-run --require-trl` failed cleanly with `error: trl-sft-dry-run execute mode requires optional package 'trl'`.
- GPU: `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id gpu-trl-sft-dry-run-final --backend trl-sft-dry-run --require-cuda`.
- GPU: `PYTHONPATH=src python3 -m unittest discover -s tests -v` passed with 13 tests and no skips.
- GPU: `PYTHONPYCACHEPREFIX=/tmp/aitp-pycache-final python3 -m compileall -q src tests`.
- GPU: `--require-trl` failed cleanly with status code 1 because TRL is not installed in the current container.

Remote GPU result:

- GPU: NVIDIA GeForce RTX 5090.
- PyTorch: `2.7.0a0+7c8ec84dab.nv25.03`, CUDA available.
- Full SFT dry-run pipeline: `stages=10 artifacts=10 backend=trl-sft-dry-run`.
- SFT artifact: `backend == "trl_sft_dry_run"`, `device.type == "cuda:0"`, `loss_steps == 2`.
- Checkpoint files: `adapter_config.json`, `synthetic_sft_state.pt`, and `trainer_state.json`.
- TRL package availability recorded as `false`; this is expected for the current container and keeps real TRL execution as the next dependency boundary.

### P2.2 - Real TRL SFT Execution Boundary

Status: in progress

Objective: move from a synthetic SFT dry run to an optional real TRL-backed tiny run while still avoiding large model downloads or unapproved public training data.

Implementation slice:

- Add a dependency profile for optional training extras: TRL, Transformers, Accelerate, PEFT, Datasets, and ModelScope or Hugging Face download helpers.
- Add a preflight command that reports installed versions, CUDA availability, free VRAM, and whether a real SFT execute mode can run.
- Add an execute mode that can train a tiny local fixture-backed model or a deliberately tiny test model before attempting Qwen.
- Keep Qwen execution blocked until model revision, tokenizer revision, model license, dataset license, and decontamination blockers are resolved.
- Emit trainer logs, package versions, effective config, checkpoint marker, and failure reason artifacts.

Acceptance criteria:

- The default `manifest` backend remains dependency-light and offline.
- `trl-sft-dry-run` remains available without TRL installed.
- A new real execute path fails fast with actionable dependency or readiness errors.
- If all optional dependencies are present, the real execute path runs a tiny SFT job on GPU and emits trainer artifacts.

Implementation status:

- Added `pipeline preflight` to emit `training_preflight_report.json`.
- Added a dependency profile for `trl`, `transformers`, `accelerate`, `peft`, `datasets`, `modelscope`, and `huggingface_hub`.
- Added CUDA reporting through PyTorch when PyTorch is installed.
- Added mode readiness for `manifest`, `torch_smoke`, `trl_sft_dry_run`, and `trl_sft_execute`.
- Added `trl-sft-execute` backend selection with fail-fast missing dependency errors.
- Kept real Qwen execution blocked by readiness audit findings until model, tokenizer, license, dataset, and decontamination gates pass.

Exit evidence:

- Local: `PYTHONPATH=src python3 -m unittest discover -s tests -v` passed with PyTorch-dependent tests skipped because PyTorch is not installed locally.
- Local: `PYTHONPYCACHEPREFIX=/private/tmp/aitp-pycache python3 -m compileall -q src tests`.
- Local: `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline preflight --config examples/post_training_pipeline.json --run-id local-preflight --require-training-extras`.
- Local: `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id local-trl-execute-expected-fail --backend trl-sft-execute` failed cleanly on missing optional training packages.
- GPU: `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline preflight --config examples/post_training_pipeline.json --run-id gpu-preflight --require-cuda --require-training-extras`.
- GPU: `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id gpu-trl-sft-execute --backend trl-sft-execute --require-cuda` failed cleanly on missing optional training packages.
- GPU: `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id gpu-trl-sft-dry-run-preflight --backend trl-sft-dry-run --require-cuda`.
- GPU: `PYTHONPATH=src python3 -m unittest discover -s tests -v` passed with 15 tests and no skips.
- GPU: `PYTHONPYCACHEPREFIX=/tmp/aitp-pycache-preflight python3 -m compileall -q src tests`.

Remote GPU result:

- CUDA available: `true`.
- Missing training extras: `trl`, `transformers`, `accelerate`, `peft`, and `datasets`.
- `trl_sft_dry_run` mode: ready.
- `trl_sft_execute` mode: blocked with 3 blocker groups.
- `trl-sft-execute` CLI status: 1, with a clean `error:` message and no traceback.

Remaining scope:

- Add an optional tiny real SFT runner that uses installed TRL/Transformers/Datasets when they are present.
- Add artifact emission for real trainer logs, package versions, effective config, checkpoint marker, and failure reasons.

### P2.3 - Distributed Fixture SFT

Status: in progress

Objective: validate that two GPU containers can run one coordinated SFT training job before attempting large-model or unapproved-data training.

Implementation slice:

- Add a pure PyTorch distributed SFT fixture trainer that works with `torchrun`.
- Use one GPU per container with DDP/NCCL when the runtime supports it.
- Provide a Gloo CPU-allreduce fallback that keeps model compute on CUDA and synchronizes gradients through TCP when NCCL collectives are unavailable.
- Keep the model tiny and local so no external model or dataset download is required.
- Save rank0 checkpoint, trainer state, and the fixture SFT data under ignored `runs/`.
- Use this as the distributed systems smoke before real Qwen or TRL training.

Acceptance criteria:

- Two containers can reach the rank0 rendezvous IP and port.
- `torchrun --nnodes=2 --nproc-per-node=1` completes with world size 2.
- Rank0 writes `model_state.pt`, `trainer_state.json`, and `sft_fixture.json`.
- The trainer state records `distributed == true`, `world_size == 2`, the gradient sync mode, and a finite final loss.

### P3 - Reward and Agentic Rollout Layer

Status: planned

- Add reward definitions for exact match, code tests, tool outcomes, safety policies, and learned reward models.
- Add environment rollout specs for sandboxed tools and multi-turn agent traces.
- Add trace schema references, replay hooks, and deterministic reward attachment.
- Add step-level and episode-level reward aggregation contracts.
- Add reward normalizers so math/code/tool/safety rewards can be compared during reporting without being mixed during specialist RL.

### P3.5 - OPD Fusion Layer

Status: planned

- Add teacher registry and domain router for specialist checkpoints.
- Add OPD data sampler that generates on-policy student rollouts across the domain mixture.
- Add distillation objectives: reverse KL, forward KL, sampled-token logprob distillation, and TGPO-style guidance fallback.
- Add fusion-loss reporting: OPD vs each specialist on its own domain.

### P4 - Evaluation Gates

Status: planned

- Add evaluation suite schemas for reasoning, coding, tool use, long-horizon agent tasks, regression, and safety.
- Add gate policies: required metrics, allowed regressions, and release blockers.
- Add model comparison reports across SFT, DPO, RLVR, and distilled checkpoints.

### P5 - Release and Governance

Status: planned

- Emit model cards, dataset cards, config snapshots, metrics, and reproducibility bundles.
- Add CI checks for config validation and unit tests.
- Add optional artifact upload hooks while keeping generated runs out of Git.
- Add changelog generation for pipeline config changes.

## Open Questions

- Which training backend should be connected first: TRL, verl, OpenRLHF, or a custom local launcher?
- Should the first real execution target be SFT/DPO on local small models or an RLVR smoke path with mocked rewards?
- What artifact store should be used beyond local `runs/`: local filesystem only, S3-compatible storage, or a database?
- What trace schema should agentic rollouts use for deterministic replay?

## Update Protocol

- Update this file when a milestone changes status or scope.
- Mark a milestone complete only after the relevant CLI path or test path has been executed.
- Keep repository documentation and code comments in English.
