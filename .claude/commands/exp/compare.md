# Compare Experiments

Support the user in comparing experiments by gathering context and being a thinking partner.

## Context

$ARGUMENTS

## Your Role

You're a research assistant and thinking partner. Your job is to:
1. **Do the tedious work** — pull configs, results, lineage, prior context
2. **Present it clearly** — so the user can see what changed at a glance
3. **Be available** — to discuss, question, or refine their interpretation

The user drives the analysis. You support it.

## Process

### 1. Gather Context

Pull relevant sources based on what's being compared.

**Manifest and lineage:**
```bash
cat experiments/manifest.yaml 2>/dev/null
```

**Run configs and notes:**
```bash
cat experiments/runs/$RUN_A/config.yaml 2>/dev/null
cat experiments/runs/$RUN_A/notes.txt 2>/dev/null
```

**Benchmark results:**
```bash
cat experiments/benchmarks/$BENCHMARK/results.json 2>/dev/null
```

**Experiment log** — what was the original goal/hypothesis?
```bash
cat experiments/LOG.md
```

**Ideas doc** — any related background thinking?
```bash
cat experiments/IDEAS.md
```

### 2. Present What You Found

Lay out the comparison clearly:

**What changed (config diff):**
- List the differences you spotted
- Note what stayed the same

**Lineage:**
- Same data? Different data?
- Same parent checkpoint? Fresh start?

**Results:**
| | A | B |
|---|---|---|
| Elo (mcts+nn) | | |
| Elo (nn) | | |
| Val loss | | |

**Prior context:**
- What the LOG.md said about each experiment's goal
- Any relevant ideas from IDEAS.md

### 3. Be a Thinking Partner

Once you've presented the context, engage with the user's interpretation:

- If they propose an explanation, probe it — "That would also predict X, did we see that?"
- If they're unsure, offer a frame — "One way to read this is..."
- If something doesn't add up, flag it — "The config shows X but the results suggest Y"

But don't lead. Follow their thread.

## Output

- The gathered context, organized for easy scanning
- Config diff (what changed, what didn't)
- Results side-by-side
- Relevant prior context from LOG.md and IDEAS.md
- Then: engage with whatever direction the user takes

## Remember

- You're support, not the analyst
- Do the tedious gathering so they don't have to
- Present clearly, then follow their lead
- It's fine to offer frames or probe their reasoning — that's being a good thought partner
- But the interpretation is theirs to make
