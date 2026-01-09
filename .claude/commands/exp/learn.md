# Capture Experiment Learnings

Help the user articulate what they learned from an experiment and update the log.

## Context

$ARGUMENTS

## Your Role

You're helping close the loop on an experiment. The goal is to capture what happened, what it means, and what comes next — while it's still fresh.

**This is reflection, not just reporting.** The value is in extracting insight, not just recording numbers.

## Process

### 1. Gather Context

Identify which experiment we're discussing. Read relevant sources:

**Experiment log** — find the entry for this experiment:
```bash
cat experiments/LOG.md
```

**Manifest / lineage** — understand what this experiment was built on:
```bash
cat experiments/manifest.yaml 2>/dev/null
```
This shows which batches → shards → runs → benchmarks. Useful for understanding the data and checkpoints behind a run.

**Run artifacts** — if there's a run directory:
```bash
ls experiments/runs/$RUN_NAME/ 2>/dev/null
cat experiments/runs/$RUN_NAME/notes.txt 2>/dev/null
cat experiments/runs/$RUN_NAME/config.yaml 2>/dev/null
```

**Benchmark results** — if there's a benchmark:
```bash
cat experiments/benchmarks/$BENCHMARK_NAME/results.json 2>/dev/null
```

**Ideas doc** — for context on what we were trying to learn:
```bash
cat experiments/IDEAS.md
```

### 2. Understand What Happened

Get the facts first:
- What were the actual results? (Elo, val_loss, head-to-head, etc.)
- How does this compare to the baseline?
- Any surprises or unexpected observations?

If the user hasn't shared results yet, ask for them. Numbers matter for the log.

### 3. Interpret the Results

Help the user think through what the results mean.

**For clear outcomes:**
- Did we confirm or deny the hypothesis?
- Was the effect size what we expected?
- Any caveats or confounds?

**For ambiguous outcomes:**
- What direction does this point, even if not conclusive?
- What did we rule out?
- Did this narrow the space of possibilities?

**For iterations:**
- Did we see improvement? Plateauing?
- Signs of diminishing returns?

**Propose interpretations** rather than asking open-ended questions. "The +34 Elo for raw NN suggests constant-sum regularization helps standalone inference more than MCTS+NN — does that match your read?"

### 4. Capture Different Kinds of Learning

Not all learning is "hypothesis confirmed/denied." Help capture:

**Concrete findings** — things we can state with evidence:
- "Constant-sum regularization improved raw NN by +34 Elo"
- "Tree reuse didn't help and sometimes hurt"

**Intuitions developed** — patterns you're starting to feel, even without hard proof:
- "I'm getting the sense that structural constraints help more for raw NN than MCTS+NN"
- "Something feels off about how we're handling sparse matrices"

**Questions sharpened** — the experiment clarified what to ask next:
- "The real question isn't 'does X help' but 'when does X help'"

**Dead ends identified** — knowing what not to pursue is valuable:
- "Sigmoid × remaining_cheese looked right but hurt learning — not worth revisiting"

Intuitions are worth capturing. You're building a feel for the problem space through experience — that's real, even when you can't point to a p-value.

### 5. Identify What's Next

Based on what we learned:
- What's the natural follow-up?
- Any new hypotheses forming?
- Should we update the ideas doc?
- Is there an open question this touches?

### 6. Draft the Log Update

Complete the experiment log entry:

```markdown
**Results:**
[Key numbers — Elo, val_loss, head-to-head records]

| Agent | Elo |
|-------|-----|
| ... | ... |

**Observations:**
- [What we noticed beyond the numbers]
- [Surprises, unexpected patterns]
- [Intuitions forming, even if not proven]

**Conclusion:** [What we learned — concrete findings, intuitions, or "inconclusive but..."]

**Next:** [What this suggests we should try, or why we're moving on]
```

### 7. Surface Connections

Check if this learning:
- Resolves or informs an open question from the log
- Validates, invalidates, or refines an idea from ideas.md
- Echoes patterns from other experiments

Convergent findings across experiments are worth noting — if three different approaches point the same direction, that's signal even if no single experiment was conclusive.

## Output

- The completed log entry (ready to paste into `experiments/LOG.md`)
- Any suggested updates to ideas.md
- Suggested next experiment if there's a clear thread to pull

## Remember

- Capture while it's fresh — details and intuitions fade quickly
- "It didn't work" is still learning — document what you ruled out
- Intuitions are real — you're building a feel for the space through experience
- Not everything needs a verdict — "inconclusive, moving on" is fine
- Negative results narrow the search space
- If you're starting to sense a pattern across experiments, write it down
