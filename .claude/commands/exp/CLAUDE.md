# Experiment Commands

Slash commands for ML experiment workflow.

## Philosophy

These commands support the user in experimentation — they don't replace the user's thinking. The pattern:
1. **Gather context** — pull configs, results, prior experiments (the tedious part)
2. **Present clearly** — so the user can see what matters
3. **Engage as thought partner** — follow the user's lead, probe their reasoning

## The Commands

| Command | Phase | Purpose |
|---------|-------|---------|
| `plan` | Before | Clarify hypothesis before running |
| `iterate` | During | Set up next iteration from checkpoint |
| `learn` | After | Capture results while fresh |
| `compare` | Anytime | Side-by-side run comparison |
| `status` | Anytime | Quick dashboard |

## Data Sources

Commands pull from:
- `experiments/LOG.md` — experiment entries, goals, results
- `experiments/IDEAS.md` — background thinking
- `experiments/manifest.yaml` — artifact lineage
- `experiments/runs/{name}/` — config.yaml, notes.txt
- `experiments/benchmarks/{name}/` — results.json

## Adding Commands

New commands should:
- Have a clear phase (before/during/after/anytime)
- Do the tedious gathering, let user drive interpretation
- Pull from the same data sources for consistency
