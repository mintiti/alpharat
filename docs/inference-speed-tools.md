# Inference speed tools

`alpharat-infer` runs saved workloads through AlphaRat's production inference and
self-play paths. It records clean throughput, reruns a selected attempt with
diagnostics, and compares compatible results. The v1 supervisor runs on Linux with [pidfd support](https://man7.org/linux/man-pages/man2/pidfd_open.2.html) (kernel 5.3 or newer).

The useful loop is: save a workload, establish clean performance, observe where
its time goes, change one declared factor, and compare clean runs. A trace helps
explain a result; its instrumented throughput is not a substitute for that result.

## Build and try it without a GPU

From the repository root:

```sh
cargo build -p alpharat-bench --features inference --bin alpharat-infer
target/debug/alpharat-infer check --plan crates/alpharat-bench/examples/inference/cpu.json
target/debug/alpharat-infer run --plan crates/alpharat-bench/examples/inference/cpu.json --out target/inference-runs/cpu-1
target/debug/alpharat-infer report --run target/inference-runs/cpu-1
target/debug/alpharat-infer diagnose --run target/inference-runs/cpu-1 --attempt a00001 --mode latency --out target/inference-runs/cpu-1-latency
```

The CPU example uses SmartUniform, the real encoder for corpus identity, the
production eager mux, and the native self-play sampler. Its rates exercise the
apparatus; they say nothing about TensorRT performance. The example executes the
same backend twice in AB then BA order. Comparable builds need a clean Git tree
or a reconstructable tracked patch. New untracked source files make the build
non-reproducible; stage intentional additions or commit them before building.

The TensorRT example is
[examples/inference/tensorrt.json](../crates/alpharat-bench/examples/inference/tensorrt.json).
Set its model path to an existing ONNX model with the encoder's observation
shape; the included 7x7 corpus has 349 input elements. Set executable and engine
cache paths explicitly. Paths in a plan resolve against the plan's directory.

## Build with TensorRT and optional timelines

Use the SDK and CUDA libraries selected for your workload. This does not install
or select a project-wide runtime version.

```sh
export TENSORRT_RTX_ROOT=/path/to/TensorRT-RTX-SDK
export VIRTUAL_ENV=/path/to/venv-with-nvidia-cuda-libraries
export NVTX_INCLUDE_DIR=/path/containing/nvtx3
export LD_LIBRARY_PATH="$TENSORRT_RTX_ROOT/lib:/path/to/cuda/lib:${LD_LIBRARY_PATH:-}"
cargo build --release -p alpharat-bench --features inference,tensorrt,inference-trace --bin alpharat-infer
```

The existing build also supports CUDA_HOME. NVTX is only needed for the
`inference-trace` feature; omit that feature for a build without timeline hooks.
Use an optimized build for actual throughput work. Build flags, Cargo features,
compiler/target, source revision and patch, Cargo.lock, executable, model,
encoded corpus, serialized engine, mapped TensorRT/CUDA libraries, GPU/driver,
CPU and selected runtime environment variables are retained.

A one-context production TensorRT backend remains one context when several
callers supply it. Eager mux combines callers' requests into one worker's actual
batches. The tool records callers, requested sizes, merged batches and physical
contexts separately.

## Workload contract

Plans use `alpharat.inference.plan`, schema version 1. Unknown fields and
unsupported settings are errors. The examples are the complete configuration
reference for direct capacity, eager mux, self-play, warmup, time limits, and
paired ordering. `check` validates corpus replay and each executable's protocol
and compiled capabilities, without loading a model or creating a CUDA context.
Actual model/SDK/engine compatibility is validated during child setup and warmup.

A corpus contains explicit boards: dimensions, player starts, cheese,
max_turns, creation_seed, optional wall edges, optional mud edges
(`{"from":[x,y],"to":[x,y],"cost":3}`), and simultaneous action prefixes.
Actions use the engine's direction numbers: up 0, right 1, down 2, left 3, stay 4.
Walls and mud must join adjacent cells. Duplicate/conflicting edges and terminal
positions are rejected. An omitted maze edge is an ordinary open passage.
This reconstructs actual GameStates, including scores and mud timers after
the prefix; it does not infer hidden state from a training observation.

For capacity, each caller executes exactly calls_per_caller requests.
A constant workload uses one size; a sequence cycles through its batches.
Request i, caller c, element j uses corpus position
`(corpus_offset + c + i + j) % corpus_len`.
The complete sequence is prepared before the measured interval; no artificial
interarrival delay is imposed. Eager mux's scheduling-dependent batch histogram
is measured, not prescribed.

Warmup evaluates every requested shape, including reachable merged shapes for
the configured caller count and mux bound. It requires both the pass count and
minimum duration, with a hard maximum. Warmup is outside the measurement.
Self-play warms sizes 1 through its search batch bound. Search uses the current
production SearchConfig defaults; its exact source identity is part of the run.

Native self-play completes the declared independent games, search simulations
and output bundles. Its wall time includes search, inference and bundle output.
Games/s is its primary comparison rate; NN evaluations/s and native simulation,
terminal and collision counters remain visible. Self-play trajectories can
change with scheduling or an implementation change. Capacity and self-play
answer different performance questions.

## Observation modes

| Mode | Measured boundary | Added observation |
|---|---|---|
| clean | Caller interval around production Backend calls; or native self-play wall | Local capacity counters and existing production counters |
| latency | Same work, separate attempt | Caller timestamps in bounded per-thread buffers |
| timeline | Same work, separate attempt | Latency samples, NVTX ranges, request/batch links, Nsight CUDA activity |
| stages | Same work, separate attempt | Existing opt-in TensorRT CUDA-event and host stage timings |

Clean capacity allocates no per-request recorder, starts no timing events, and
emits no active AlphaRat NVTX ranges. Normal backend counters and finite-output
checks remain. Builds with timeline support retain disabled hook checks.
Latency/timeline self-play initializes its thread-local buffers on first use;
that initialization is part of diagnostic self-play wall time.

Request p50/p95/max are computed from retained individual calls, never by
averaging per-worker percentiles or dividing aggregate elapsed time by calls.
A per-thread cap limits memory; dropped samples are explicit. Once truncated,
quantiles describe the retained prefix and cannot establish the full tail.
Full samples remain in requests.csv and reports validate/recompute the summary.

Mux worker_wait_drain_ns includes the worker's wait, lock acquisition and drain.
It is not per-request queue residence. Timeline links retain the actual enqueue
and dequeue timestamps under the queue lock, positions, request ID and batch ID.
Request latency includes waiting for the worker, backend execution and delivery.
It is not pure GPU service time.

Stage measurements retain the existing production timing meanings. H2D/infer/D2H
CUDA-event durations and host encode/parse/allocation work are diagnostic
observations. Their sum is not a critical path or GPU active-time percentage.
This tool does not reinterpret the old capacity harness's interval union as
device activity.

## Diagnose one saved attempt

```sh
target/release/alpharat-infer diagnose --run target/inference-runs/trt-1 --attempt a00001 --mode stages --out target/inference-runs/trt-1-stages
target/release/alpharat-infer diagnose --run target/inference-runs/trt-1 --attempt a00001 --mode timeline --nsys /path/to/nsys --out target/inference-runs/trt-1-trace
```

The selected case, caller supply, corpus, model, backend configuration and
executable are reused. The child must match the parent's complete build/runtime/
input identity before measurement. A changed executable or engine is rejected.
The diagnostic stores the original run ID location, attempt, and manifest hash.

Timeline collection is limited to the `inference.measure` NVTX range. Nsight
records CUDA and NVTX, with CPU sampling/context-switch collectors disabled.
The launcher passes `--capture-range-end=stop --kill=none` so normal output
finalization follows the measured range. Its command, profiler binary identity,
logs, .nsys-rep and links remain with the attempt. No privileges are changed.

Ranges label request, merged batch, queue enqueue, session lock wait, encoding,
H2D/D2H submission, enqueue, completion wait, parse and scatter. Push/pop ranges
remain on their originating threads; numeric IDs connect caller and worker.
Submission ranges are host API time. Inspect CUDA kernel/copy activity in
Nsight to distinguish device work from host waits.

Nsight launch behavior follows the
[NVIDIA Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
Profiler availability, trace contents and observer overhead must be qualified
on the selected host. A nonempty trace file alone is not proof that expected
kernels and ranges were captured. V1 preserves the native report for inspection;
it does not depend on an unstable Nsight SQLite schema for routine reporting.

## Compare and regenerate reports

A paired plan specifies exactly two variants, baseline/candidate IDs, allowed
axes, and an AB/BA order for every repetition. Its report compares matching case
keys and repetition indices. It reports the median and range of paired
candidate/baseline rates, plus the individual ratios and median arm rates.
These are not pooled-work/pooled-time estimates or a confidence interval.

For two independent one-variant runs:

```sh
target/release/alpharat-infer compare --left target/inference-runs/a --right target/inference-runs/b --vary host_io
target/release/alpharat-infer report --run target/inference-runs/a --format json
```

Separate runs pair by repetition index and are explicitly labeled as
non-interleaved. Allowed axes are host_io, profile, source, build, runtime,
hardware, topology and requests. Model and corpus changes are always rejected.
An undeclared difference, diagnostic mode, incomplete/failed attempt, missing
repetition, within-arm identity drift, or non-reconstructable source rejects
the comparison. Changing self-play's game/search configuration is not a
compatible comparison. No speedup threshold automatically selects a winner.

report/compare print by default; `--out FILE` exclusively creates a new report
file. Cached report.json/report.md are conveniences: regenerating a report
validates the frozen plan, attempt requests/results, exact completed work,
raw latency summaries, mux histogram totals and artifact hashes.

## Failure and artifact ownership

A fresh output directory is reserved before preflight or GPU setup. Each attempt
has its own request, stdout/stderr, command, progress snapshot, append-only
progress events, result/error and source patch. run.json is atomically replaced
and synced as attempts advance. Completed records are never silently dropped.

Every attempt launches in a separate owned process group. Setup/warmup,
measurement/finalization and whole-run time limits are enforced by the parent.
Before GPU setup, the parent verifies the worker's ancestry and obtains a stable
Linux process handle. SIGINT/SIGTERM stops that exact worker even if a profiler
gave it another process group, then stops the owned launcher group. Unrelated
PIDs are rejected; there is no system-wide descendant cleanup.
The first failure, timeout or interruption stops the run and marks remaining
attempts not_run. There are no automatic retries. A hard kill of the supervisor
or host failure can leave a running manifest; it remains incomplete evidence,
not a successful run or a resumable queue.

Setup failures retain failure.json or a structured unsupported run. Failed
children retain logs and available partial artifacts; partial work has no
success rate. To try again, choose a fresh output directory. Model and
executable identities are verified before every attempt. Their large binaries
and engine caches stay at their recorded locations; keep those files if you
need to rerun diagnostics. The corpus and source patch are copied into the run.

GPU state snapshots are taken outside timing. They are context, not proof of
exclusive GPU ownership. Arrange an idle machine for performance conclusions.
Use repeated clean runs to assess throughput and keep diagnostic runs separate.
