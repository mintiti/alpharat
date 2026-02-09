# Experiment Status

Quick overview of where we are — roadmap progress, recent activity, open threads.

## Context

$ARGUMENTS

## Your Role

Give a fast situational snapshot. The user wants to know "where are we" without digging through files.

**Keep it brief.** This is a dashboard, not a deep dive.

## Process

### 1. Read the Sources

```bash
# Experiment logs — roadmap, phases, recent entries
cat experiments/LOG*.md
```

```bash
# Manifest — artifact counts, recent runs
cat experiments/manifest.yaml 2>/dev/null
```

```bash
# Ideas — pending items
cat experiments/IDEAS.md
```

### 2. Summarize

**Roadmap position:**
- Current phase and what's done
- What's next on the roadmap

**Recent activity:**
- Last few experiments (from log)
- Recent runs/benchmarks (from manifest)
- Key results in one line each

**Open threads:**
- Open questions from the log
- Pending ideas that seem ready to test
- Experiments started but not concluded (missing results/conclusions)

**Artifacts:**
- Count of batches, shards, runs, benchmarks
- Latest of each with key stats

### 3. Surface Anything Stuck

Flag if:
- An experiment was planned but never run
- Results exist but conclusions weren't captured
- An open question has been sitting for a while
- Multiple ideas point to the same unexplored area

## Output Format

Keep it scannable:

```
## Current Phase
[Phase N]: [description] — [status]

## Recent Experiments
- [date]: [name] — [one-line result]
- [date]: [name] — [one-line result]

## Open Threads
- [question or pending item]
- [question or pending item]

## Artifacts
- Batches: N (latest: [group], [games] games)
- Shards: N (latest: [group], [positions] positions)
- Runs: N (latest: [name], val_loss [X])
- Benchmarks: N (latest: [name])

## Suggested Next
[If there's an obvious next step, mention it]
```

## Remember

- This is a glance, not a report
- Surface what's actionable
- If nothing's stuck, say so — "all caught up" is useful info
