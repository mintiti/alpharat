# Iterate on an Experiment

Set up the next iteration — continue from a checkpoint with fresh data.

## Context

$ARGUMENTS

## Your Role

Help set up the next round of the experiment loop. This is continuation, not a new hypothesis — we're building on what worked.

## Process

### 1. Identify What We're Iterating From

Figure out the source:
- Which run/checkpoint are we continuing from?
- What iteration number is this? (v1 → v2, or iteration 2 → iteration 3)

```bash
# Check existing runs
cat experiments/manifest.yaml 2>/dev/null
```

```bash
# Check the source run's config and results
cat experiments/runs/$RUN_NAME/config.yaml 2>/dev/null
cat experiments/runs/$RUN_NAME/notes.txt 2>/dev/null
```

### 2. Determine Iteration Type

**NN-guided sampling** (most common):
- Use the trained checkpoint as MCTS prior for sampling
- Generate new data that's "smarter" than uniform-prior data
- Train on the new data, resuming from the checkpoint

**Same config, more data**:
- Sample more games with same setup
- Add to existing shards or create new ones
- Continue training

**Parameter tweak**:
- Small adjustment based on what we learned
- Still an iteration, not a new experiment

### 3. Set Up the Configs

**For NN-guided sampling**, you need:

1. **Sampling config** pointing to the checkpoint:
```yaml
# configs/sample_iter2.yaml (or similar)
group: iter2_from_mlp_v1  # or whatever naming makes sense

checkpoint: experiments/runs/mlp_v1/checkpoints/best_model.pt

mcts:
  simulations: 554
  c_puct: 8.34
  # ... same as before, or adjusted

game:
  # ... same game params
```

2. **Training config** with resume_from:
```yaml
name: mlp_v2  # next version

resume_from: experiments/runs/mlp_v1/checkpoints/best_model.pt

data:
  train_dir: experiments/shards/<NEW_SHARD>/train
  val_dir: experiments/shards/<NEW_SHARD>/val

# ... rest of config
```

### 4. Suggest the Workflow

Typical iteration workflow:
```bash
# 1. Sample with NN-guided MCTS
uv run python scripts/sample.py configs/sample_iter2.yaml --workers 8

# 2. Create shards from new data
uv run python scripts/prepare_shards.py --group iter2 --batches "iter2_from_mlp_v1/*"

# 3. Train, resuming from checkpoint
uv run python scripts/train.py configs/train_iter2.yaml --epochs 100

# 4. Benchmark
uv run python scripts/benchmark.py configs/benchmark.yaml
```

### 5. Draft the Log Entry

Since this is an iteration, the log entry is simpler:

```markdown
### YYYY-MM-DD: [Model name] Iteration N

**Goal:** Continue self-play loop from [parent run].

**Parent:** [parent run name] (Elo: X, val_loss: Y)

**Setup:**
- NN-guided sampling with [parent] checkpoint
- [N] games sampled
- Training resumed from [parent] checkpoint

**Type:** iteration

---
*Results to be added after running*
```

### 6. Note What Carries Forward

Remind what context carries over:
- The hypothesis/goal from the original experiment
- What we learned from previous iterations
- What we're watching for (diminishing returns? plateau?)

## Output

- Sampling config (or modifications to existing)
- Training config (or modifications)
- The workflow commands to run
- Log entry draft
- Suggested run name (e.g., `mlp_v2`, `symmetric_iter3`)

## Remember

- Iterations don't need elaborate justification
- The goal is continuity — build on what's working
- Watch for diminishing returns across iterations
- Keep naming consistent (v1 → v2, or iter1 → iter2)
- Link back to parent in the notes
