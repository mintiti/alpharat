# configs/

Reusable configuration templates for sampling, training, and evaluation.

## Gitignore Note

This folder is in `.gitignore` to prevent accidental commits of experiment-specific configs. To commit changes here, use `git add -f`:

```bash
git add -f configs/path/to/file.yaml
```

## Structure

```
configs/
├── iterate.yaml         # Auto-iteration loop entry point (recommended)
├── sample.yaml          # Main sampling entry point
├── train.yaml           # Main training entry point
├── tournament.yaml      # Main benchmark entry point
├── game/                # Game presets (grid size, topology, cheese)
├── mcts/                # MCTS parameter sets
├── model/               # Model architecture + optimizer configs
└── sample/              # Composed sampling configs
```

## Naming Convention

### Game configs: `{size}_{topology}[_asymmetric]`

**Topology:**
- `open` — no walls, no mud (`wall_density=0, mud_density=0`)
- `walls` — pyrat defaults (walls + mud)

**Cheese placement:**
- (default) — symmetric cheese (fair game, theoretical draw under perfect play)
- `_asymmetric` — random cheese placement (may favor one player)

Examples:
- `5x5_open.yaml` — 5×5, no walls/mud, symmetric cheese
- `5x5_open_asymmetric.yaml` — 5×5, no walls/mud, random cheese
- `5x5_walls.yaml` — 5×5, walls+mud, symmetric cheese
- `7x7_open.yaml` — 7×7, no walls/mud, symmetric cheese

### Sample configs

Named after the game config they use. All use the same MCTS params (`7x7_scalar_tuned`).

### Model configs

Named after architecture: `mlp.yaml`, `symmetric.yaml`

## What Belongs Here

- **Reusable templates** — configs you'll use repeatedly across experiments
- **Standard presets** — common grid sizes, well-tuned parameters
- **Composable building blocks** — sub-configs that combine via Hydra `defaults:`

## What Does NOT Belong Here

- **Experiment-specific configs** — one-off variations with hardcoded paths
- **Hardcoded data paths** — use CLI overrides (`--shards GROUP/UUID`)
- **Checkpoint paths** — pass via CLI, don't bake into committed configs
- **Configs for a single experiment** — keep those in `experiments/` or don't commit them

## Usage

Main entry points compose from sub-configs:

```bash
# Auto-iteration (recommended) — uses /game, /mcts, /model defaults
alpharat-iterate configs/iterate.yaml --prefix sym_5x5
alpharat-iterate configs/iterate.yaml --prefix sym_5x5 --iterations 3

# Override defaults for iteration
alpharat-iterate configs/iterate.yaml --prefix mlp_7x7 game=7x7_open model=mlp

# Sampling — uses /game and /mcts defaults
alpharat-sample configs/sample.yaml --group my_batch

# Override game config
alpharat-sample configs/sample.yaml game=7x7_open --group my_7x7_batch

# Training — uses /model defaults
alpharat-train configs/train.yaml --name my_run --shards group/uuid

# Override model
alpharat-train configs/train.yaml model=symmetric --name my_run --shards group/uuid

# Tournament
alpharat-benchmark configs/tournament.yaml
```

Or use the pre-composed sample configs directly:

```bash
alpharat-sample configs/sample/7x7_open.yaml --group 7x7_batch
```
