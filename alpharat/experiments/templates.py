"""Templates for notes.txt and CLAUDE.md files.

These templates are written to the experiments folder on initialization.
The CLAUDE.md files document the folder structure for AI assistants.
"""

NOTES_TEMPLATE = """\
## Goal

What are you testing? What hypothesis?

## Observations

What did you notice during/after the run?

## Results

Summary of outcomes, conclusions.
"""

EXPERIMENTS_CLAUDE_MD = """\
# Experiments Folder

Experiment artifacts and context. NOT version controlled.

## Structure

```
experiments/
├── manifest.yaml      # Artifact index with lineage
├── LOG.md             # Experiment log: roadmap, entries, results
├── IDEAS.md           # Parking lot for unstructured thinking
├── batches/           # Raw game recordings
├── shards/            # Processed train/val splits
├── runs/              # Training runs with checkpoints
└── benchmarks/        # Tournament results
```

## Context Files

**LOG.md** — the official record:
- Roadmap with phases
- Decisions and rationale
- Individual experiment entries (goal, hypothesis, setup, results)

**IDEAS.md** — the scratchpad:
- Unstructured thoughts
- Ideas not ready to test
- Architecture sketches

## Manifest & Lineage

`manifest.yaml` tracks all artifacts and how they connect:
- Which checkpoint produced a batch? → `batches.{id}.parent_checkpoint`
- Which batches made shards? → `shards.{id}.source_batches`
- Which shards trained a model? → `runs.{name}.source_shards`

## Working Here

The `/exp:*` commands are designed to help:
- `/exp:plan` — before running, to clarify hypothesis
- `/exp:iterate` — to set up next iteration
- `/exp:learn` — after running, to capture results
- `/exp:compare` — to compare runs side-by-side
- `/exp:status` — quick overview

These commands read LOG.md, IDEAS.md, and manifest automatically.
"""

BATCHES_CLAUDE_MD = """\
# Batches

Raw game recordings from self-play sampling.

## Structure

```
batches/
└── {group}/              # Human-readable grouping (e.g., "uniform_5x5")
    └── {uuid}/           # Unique batch ID
        ├── metadata.json # Batch configuration and provenance
        └── games/        # Individual game recordings (.npz)
```

## metadata.json

Contains:
- `batch_id`: UUID of this batch
- `created_at`: Timestamp
- `checkpoint_path`: Parent checkpoint (null for uniform prior)
- `mcts_config`: MCTS algorithm configuration
- `game`: Game dimensions, cheese count, etc.

## Creating Batches

```python
exp = ExperimentManager()
batch_dir = exp.create_batch(
    group="uniform_5x5",
    mcts_config=mcts_config,
    game=game_config,
    checkpoint_path=None,  # or path to parent checkpoint
)
```
"""

SHARDS_CLAUDE_MD = """\
# Shards

Processed training data derived from one or more batches.

## Structure

```
shards/
└── {uuid}/
    ├── manifest.json    # Shard metadata and source batches
    ├── train/           # Training split
    │   ├── manifest.json
    │   └── shard_*.npz
    └── val/             # Validation split
        ├── manifest.json
        └── shard_*.npz
```

## manifest.json

Contains:
- `training_set_id`: Unique identifier
- `created_at`: Timestamp
- `source_batches`: List of batch IDs used
- `total_positions`: Total training examples
- `shard_count`: Number of shard files

## Creating Shards

Use `scripts/prepare_shards.py` or:

```python
exp = ExperimentManager()
shard_dir = exp.create_shards(
    source_batches=["uniform_5x5/abc123"],
    train_val_split=0.9,
)
```
"""

RUNS_CLAUDE_MD = """\
# Runs

Training runs with checkpoints, configs, and notes.

## Structure

```
runs/
└── {run_name}/           # Human-readable name (e.g., "mlp_v1")
    ├── config.yaml       # Frozen training config
    ├── notes.txt         # Experiment notes (Goal/Observations/Results)
    └── checkpoints/
        ├── best_model.pt
        ├── checkpoint_epoch_10.pt
        └── ...
```

## config.yaml

Frozen copy of the training configuration used. Allows exact reproduction.

## notes.txt

Freeform notes with prompts:
- **Goal**: What hypothesis are you testing?
- **Observations**: What did you notice?
- **Results**: Summary and conclusions

## Creating Runs

```python
exp = ExperimentManager()
run_dir = exp.create_run(
    name="mlp_v1",
    config=train_config,
    source_shards="xyz789",
)
```
"""

BENCHMARKS_CLAUDE_MD = """\
# Benchmarks

Tournament and benchmark results.

## Structure

```
benchmarks/
└── {benchmark_name}/     # Human-readable name (e.g., "tournament_20260107")
    ├── config.yaml       # Tournament/benchmark config
    ├── notes.txt         # Notes
    └── results.json      # Structured results (Elo, W/D/L, etc.)
```

## results.json

Contains structured results:
- Elo ratings
- Win/Draw/Loss records
- Cheese statistics
- Per-matchup breakdowns

## Creating Benchmarks

```python
exp = ExperimentManager()
bench_dir = exp.create_benchmark(
    name="tournament_001",
    config=tournament_config,
    checkpoints=["mlp_v1"],
)
exp.save_benchmark_results(name="tournament_001", results=elo_result)
```
"""
