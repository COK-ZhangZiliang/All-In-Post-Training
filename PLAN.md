# All-In Post-Training Plan

Last updated: 2026-06-22

## Mission

Build a maintainable panorama of LLM post-training: methods, research lineage, systems, agentic environments, evaluation, and safety boundaries. The first product shape is a data-driven static atlas that can later grow into an interactive knowledge graph and research operations tool.

## Knowledge Baseline

This plan is grounded in the provided Lark document, "迈向万亿参数之后：大模型后训练技术梳理", and a fresh pass over primary or near-primary sources. The working thesis is:

- Post-training has shifted from instruction alignment toward capability elicitation.
- RLVR and critic-free policy optimization are now central for reasoning.
- Specialist training plus on-policy distillation is the main answer to multi-capability interference.
- Agentic RL introduces a different systems problem: long-horizon environments, tool actions, dense/verified rewards, replay, sandboxing, and step-level credit assignment.

## Research Map

| Track | Key Ideas | Representative Sources | Product Treatment |
|-|-|-|-|
| Alignment foundations | SFT, RLHF, preference optimization, DPO | InstructGPT, DPO | Establish the historical baseline and vocabulary. |
| Reasoning RL | RLVR, GRPO, DAPO, CISPO, verifiable rewards | DeepSeekMath, DeepSeek-R1, DAPO, MiniMax-M1 | Explain why outcome-verifiable domains changed the training signal. |
| Multi-capability fusion | GKD/OPD, specialist RL, MOPD, teacher-guided distillation, self-distillation | GKD, TGPO, SDPO, MiMo reports | Model capability merging as a graph of teacher, student, and reward signals. |
| Agentic RL | Multi-turn environments, tool actions, dense reward, step/turn credit | Agentic RL guide, MT reward design, GiGPO, StepPO | Treat agent training as environment/reward/policy co-design. |
| Systems infrastructure | Rollout engines, sandbox execution, replay, async scheduling, prefix reuse | MiniMax Forge, long-context model reports | Track production bottlenecks separately from algorithm ideas. |
| Evaluation and safety | SWE-style tasks, tool-use tasks, reward hacking, unsafe tool behavior | SWE-bench family, Tau-bench family, safety papers to be expanded | Keep evaluation and safety visible instead of appendix-only. |

## Confirmed Source Seeds

- InstructGPT: https://arxiv.org/abs/2203.02155
- DPO: https://arxiv.org/abs/2305.18290
- GKD / on-policy distillation: https://arxiv.org/abs/2306.13649
- DeepSeekMath / GRPO: https://arxiv.org/abs/2402.03300
- VinePPO credit assignment: https://arxiv.org/abs/2410.01679
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Kimi k1.5: https://arxiv.org/abs/2501.12599
- DAPO: https://arxiv.org/abs/2503.14476
- Multi-turn reward design: https://arxiv.org/abs/2505.11821
- GiGPO: https://arxiv.org/abs/2505.10978
- MiniMax-M1 / CISPO: https://arxiv.org/abs/2506.13585
- Agentic RL practitioner guide: https://arxiv.org/abs/2510.01132
- GDPO: https://arxiv.org/abs/2601.05242
- SDPO: https://arxiv.org/abs/2601.20802
- StepPO: https://arxiv.org/abs/2604.18401
- TGPO: https://arxiv.org/abs/2605.13230
- MiniMax Forge: https://huggingface.co/blog/MiniMax-AI/forge-scalable-agent-rl-framework-and-algorithm

## Architecture

```text
data/panorama.json
        |
        v
src/all_in_post_training/catalog.py  -> validate schema and graph integrity
        |
        v
src/all_in_post_training/site.py     -> render static HTML/CSS/JS bundle
        |
        v
site/index.html                      -> local interactive panorama
```

Core design choices:

- Keep the knowledge graph as JSON so future agents can patch it safely.
- Keep rendering deterministic and offline.
- Validate references, node IDs, track IDs, and edge endpoints before building.
- Separate research data from presentation logic.

## Milestones

### P0 - Repository Foundation

Status: complete

- Create project documentation: `README.md`, `PLAN.md`, `AGENTS.md`, `LICENSE`.
- Add a Git-Rules-based contribution and commit policy.
- Seed the post-training panorama dataset.
- Implement an offline CLI with `validate`, `stats`, and `build` commands.
- Implement the first static panorama page.
- Add unit tests for catalog validation and site build.

Exit evidence:

- `PYTHONPATH=src python3 -m all_in_post_training.cli validate`
- `PYTHONPATH=src python3 -m all_in_post_training.cli build --out site`
- `PYTHONPATH=src python3 -m unittest discover -s tests -v`

### P1 - Research Curation Depth

Status: planned

- Add paper metadata fields: venue, date, authors, confidence, tags, and BibTeX key.
- Add source confidence levels: confirmed, likely, speculative, deprecated.
- Expand evaluation nodes: SWE-bench, SWE-Gym, Tau-bench, BFCL, ToolBench, long-horizon browser tasks.
- Add safety/alignment nodes: reward hacking, tool abuse, sandboxes, policy filters, red-team evals.
- Create a repeatable source review checklist.

### P2 - Better Panorama UX

Status: planned

- Add track-by-track comparison pages.
- Add clickable edges with relation details.
- Add search facets for algorithm, reward signal, environment, and system component.
- Add a compact export mode for screenshots and presentations.
- Add responsive layout QA for desktop and mobile.

### P3 - Research Operations

Status: planned

- Add a paper ingestion command that normalizes arXiv URLs into reference records.
- Add duplicate detection for titles and arXiv IDs.
- Add changelog generation from data diffs.
- Add optional BibTeX export.
- Add source freshness checks for frontier papers.

### P4 - Deployment

Status: planned

- Add GitHub Pages or static artifact deployment.
- Add CI for validation and tests.
- Add release tags for stable snapshots.
- Add a generated `site/` preview artifact in CI, not in the repository.

### P5 - Advanced Graph Intelligence

Status: planned

- Add graph queries: "what changed after RLVR", "which methods solve credit assignment", "which systems need sandbox replay".
- Add matrix views for algorithms vs reward signals and systems vs bottlenecks.
- Add optional local LLM summarization hooks while keeping secrets out of tracked files.

## Open Questions

- Should the primary UI language be Chinese, English, or bilingual?
- Should the atlas prioritize papers, model releases, or reusable engineering patterns?
- Should generated static artifacts be deployed through GitHub Pages or kept local until the dataset matures?
- How strict should evidence confidence be for fast-moving 2026 reports?

## Update Protocol

- Update this file when a milestone changes status or scope.
- Keep completed items tied to executable evidence.
- Keep new research claims traceable to a URL in `data/panorama.json`.

