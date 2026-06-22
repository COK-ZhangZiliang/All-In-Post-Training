# Agent Guidelines

This repository builds a data-driven panorama of LLM post-training research and systems. Treat the repository as a living research product: every code change should keep the knowledge graph, static site, and roadmap coherent.

## Source of Truth

- `PLAN.md` tracks milestones, open questions, and research-to-implementation decisions.
- `data/panorama.json` is the canonical panorama dataset. Do not hard-code knowledge in the frontend when it belongs in data.
- `src/all_in_post_training/` contains the offline CLI, validation logic, and site generator.
- `README.md` is the user-facing entry point and should stay accurate after feature changes.

## Research Rules

- Prefer primary sources: arXiv papers, official technical reports, official project pages, or organization blogs.
- Record source URLs in `data/panorama.json` when a node depends on a paper or technical report.
- Distinguish confirmed research from speculative or roadmap ideas.
- When adding frontier claims, include dates and avoid implying that unverified model specs are confirmed.

## Engineering Rules

- Keep the first version dependency-light. Add external packages only when they remove real complexity.
- Preserve offline usability: `PYTHONPATH=src python3 -m all_in_post_training.cli validate` should work without network access.
- Add or update tests for schema, validation, and site generation when the data model changes.
- Do not commit generated `site/`, cache directories, model weights, datasets, secrets, or one-off outputs.
- Keep code comments short and useful; explain non-obvious validation or rendering choices.

## Validation

Before proposing or committing code changes, run the relevant checks:

```bash
PYTHONPATH=src python3 -m all_in_post_training.cli validate
PYTHONPATH=src python3 -m all_in_post_training.cli build --out site
PYTHONPATH=src python3 -m unittest discover -s tests -v
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
feat(panorama): add post-training knowledge graph seed
docs(plan): define agentic rl roadmap
test(catalog): cover edge endpoint validation
```

- Write subjects in English, imperative mood, lowercase, and without a trailing period.
- Run relevant validation before committing.
- Confirm branch and remote target before pushing.
- Do not rewrite shared history unless explicitly requested.

