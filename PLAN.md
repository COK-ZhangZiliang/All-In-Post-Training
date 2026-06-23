# All-In Post-Training Plan

Last updated: 2026-06-23

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
| Training backends | Connect stages to TRL, verl, OpenRLHF, custom launchers, or internal schedulers | Planned |
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
src/all_in_post_training/pipeline/runner.py
        |  topologically orders stages and drives execution
        v
src/all_in_post_training/pipeline/backends.py
        |  backend interface; current backend emits deterministic manifests
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

### Immediate Next Step - P1.1 Dataset Lineage Foundation

Status: next

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

### P1 - Data and Dataset Lineage

Status: in progress

- Encode the SFT -> domain RL specialists -> OPD fusion recipe in the reference pipeline config. Completed for the manifest-backed reference config.
- Pin `Qwen/Qwen3.5-2B-Base` revision and license metadata in the model manifest.
- Add dataset manifest schemas for SFT, preference, reward, RL, evaluation, and safety datasets.
- Implement local dataset inspection for JSONL and folder manifests.
- Add data quality checks: duplicates, empty prompts, missing fields, invalid preference pairs, and license metadata.
- Add mixture recipes with capability weights and sampling policies.
- Record dataset fingerprints and content hashes.

### P2 - Training Backend Adapters

Status: planned

- Add backend adapters for TRL SFT first.
- Add backend adapter contracts for verl/OpenRLHF GRPO-style RL jobs.
- Add specialist checkpoint namespaces for math, code, tool/agent, and safety/instruction RL.
- Add command rendering, environment variable handling, and dry-run vs execute modes.
- Add resource specs: GPU count, memory hints, distributed strategy, checkpoint cadence.
- Add failure and retry states to run manifests.

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
