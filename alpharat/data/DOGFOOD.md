
## Train/val split happens at file level, not game level

**Friction:** When testing the workflow with small datasets, the train/val split produced 0 validation samples. With 8 bundle files and 10% val ratio, `int(8 * 0.1) = 0`. Training then crashed with `FileNotFoundError` for `val/manifest.json`.

**Resolution:** Worked around by using `--val-ratio 0.3` to force at least 2 files into validation.

**Suggestion:** Two fixes needed:
1. Split at game level within bundles, not file level — this is the root cause
2. Training loop should gracefully handle empty validation (skip val metrics, don't crash)
