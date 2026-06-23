# Agent Guidelines

This repository builds a backend-first LLM post-training pipeline. Treat the repository as a living training-systems product: every code change should keep the pipeline config, execution contracts, artifacts, tests, and roadmap coherent.

## Source of Truth

- `PLAN.md` tracks milestones, open questions, and research-to-implementation decisions.
- `examples/post_training_pipeline.json` is the first reference pipeline config.
- `src/all_in_post_training/pipeline/` contains pipeline config parsing, validation, execution backends, artifact tracking, and runner logic.
- `README.md` is the user-facing entry point and should stay accurate after feature changes.

## Research Rules

- Prefer primary sources: arXiv papers, official technical reports, official project pages, or organization blogs.
- Record source URLs in `PLAN.md`, backend design documents, or stage metadata when a pipeline decision depends on a paper or technical report.
- Distinguish confirmed research from speculative or roadmap ideas.
- When adding frontier claims, include dates and avoid implying that unverified model specs are confirmed.

## Engineering Rules

- Keep the first version dependency-light. Add external packages only when they remove real pipeline complexity.
- Preserve offline usability: `PYTHONPATH=src python3 -m all_in_post_training.cli pipeline validate --config examples/post_training_pipeline.json` should work without network access.
- Add or update tests for pipeline schema, validation, stage ordering, artifact tracking, and backend behavior when the data model changes.
- Do not commit generated `site/`, cache directories, model weights, datasets, secrets, or one-off outputs.
- Write project comments, docstrings, and repository documentation in English unless a user-facing localization file explicitly requires another language.
- Keep code comments short and useful; explain non-obvious validation or rendering choices.

## Validation

Before proposing or committing code changes, run the relevant checks:

```bash
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline validate --config examples/post_training_pipeline.json
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline plan --config examples/post_training_pipeline.json
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline inspect-data --config examples/post_training_pipeline.json --fixture-root tests/fixtures/lineage --run-id lineage-smoke
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline audit-readiness --config examples/post_training_pipeline.json --run-id readiness-smoke
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id smoke
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

When a CUDA container is available, also run:

```bash
PYTHONPATH=src python3 -m all_in_post_training.cli pipeline run --config examples/post_training_pipeline.json --run-id gpu-torch-smoke --backend torch-smoke --require-cuda
```

If a check cannot run, state exactly why and what was verified instead.

## Plan Maintenance

Update `PLAN.md` whenever milestones are added, completed, blocked, or materially re-scoped. Do not mark an item complete just because helper code exists; verify the actual command or user-facing path.

## Git Rules

This project uses the repository convention from [COK-ZhangZiliang/Git-Rules](https://github.com/COK-ZhangZiliang/Git-Rules).

Key local rules:

- Commit only when the user or maintainer explicitly asks for a commit.
- Keep each commit to one logical, testable change.
- Use explicit path staging; do not use `git add -A` or `git add .`.
- Use Conventional Commits:

```text
<type>(<scope>): <subject>
```

Recommended types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, and `perf`.

Examples:

```text
feat(pipeline): add post-training stage runner
docs(plan): define rlvr backend roadmap
test(pipeline): cover dependency validation
```

- Write subjects in English, imperative mood, lowercase, and without a trailing period.
- Run relevant validation before committing.
- Confirm branch and remote target before pushing.
- Do not rewrite shared history unless explicitly requested.
