<p align="center">
  <img src="assets/icon.svg" alt="All-In Post-Training icon" width="112" height="112">
</p>

<h1 align="center">All-In Post-Training</h1>

<p align="center">
  A maintainable panorama for LLM post-training research, engineering, and agentic RL.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-0f766e"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-2563eb">
  <img alt="Status" src="https://img.shields.io/badge/status-initial%20framework-f59e0b">
</p>

## Purpose

All-In Post-Training organizes post-training knowledge into an evolving research map: from SFT, RLHF, and DPO, to RLVR, GRPO/DAPO/CISPO, OPD/MOPD, multi-capability fusion, long-horizon agentic RL, sandboxed environments, and evaluation systems. The first version focuses on the data model, research seeds, static panorama page, and future roadmap.

## Current Capabilities

- Data-driven knowledge base: `data/panorama.json` stores papers, methods, systems, relationship edges, and roadmap metadata.
- Offline CLI: validate data and generate the static page without third-party runtime dependencies.
- Static panorama: generate `site/index.html` and open it directly in a browser, with search, track filtering, node details, and an SVG relationship graph.
- Project governance: `PLAN.md` records the research and engineering roadmap, while `AGENTS.md` captures collaboration, commit, and validation rules.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
all-in-post-training validate
all-in-post-training build --out site
python -m http.server 8000 --directory site
```

You can also run the project without installing it:

```bash
PYTHONPATH=src python3 -m all_in_post_training.cli validate
PYTHONPATH=src python3 -m all_in_post_training.cli build --out site
```

Then open `http://localhost:8000`.

## Repository Structure

```text
.
├── AGENTS.md                         # Project collaboration, validation, and Git rules
├── PLAN.md                           # Post-training panorama roadmap
├── README.md                         # Project overview
├── data/panorama.json                # Research panorama data source
├── src/all_in_post_training/         # CLI, data validation, and static site generator
├── assets/icon.svg                   # Project icon
└── tests/                            # Offline unit tests
```

## Research Scope

The first version organizes the provided document knowledge and external research into six tracks:

1. Alignment foundations: SFT, RLHF, DPO, and preference data.
2. Reasoning RL: RLVR, GRPO, DAPO, CISPO, and verifiable rewards.
3. Multi-capability fusion: GKD/OPD, specialist RL, MOPD, TGPO, and SDPO.
4. Agentic RL: multi-turn environments, turn/step-level credit assignment, and dense rewards.
5. Systems infrastructure: sandboxes, rollout, replay, asynchronous scheduling, and prefix trees.
6. Evaluation and safety: SWE-bench, tool use, long-horizon evaluation, and safety boundaries.

## Common Commands

```bash
PYTHONPATH=src python3 -m all_in_post_training.cli validate
PYTHONPATH=src python3 -m all_in_post_training.cli stats
PYTHONPATH=src python3 -m all_in_post_training.cli build --out site
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## License

This project is released under the [MIT License](LICENSE).
