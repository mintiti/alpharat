#!/usr/bin/env python3
"""Compare pageable and reusable pinned TensorRT host I/O under one eager lane.

The record is intentionally descriptive: validity is a hard gate, but the script
does not encode a numerical performance threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODES = ("pageable", "pinned")
CAPACITY_ROW = re.compile(
    r"^\s*(?P<batch>\d+)\s+(?P<callers>\d+)\s+"
    r"(?P<stage>[\d.]+)\s+(?P<h2d>[\d.]+)\s+(?P<infer>[\d.]+)\s+"
    r"(?P<alloc>[\d.]+)\s+(?P<d2h>[\d.]+)\s+(?P<parse>[\d.]+)\s+"
    r"(?P<residual>[\d.]+)\s+(?P<call>[\d.]+)\s+(?P<effective>[\d.]+)\s+"
    r"(?P<throughput>[\d.]+[kM]?)\s*$"
)
PARITY = re.compile(
    r"Parity: one context, batch=(?P<batch>\d+), "
    r"max_abs_diff=(?P<value>[\d.eE+-]+)"
)
ROOT_BEHAVIOR = re.compile(
    r"Root behavior: policy_l1_p1=(?P<policy_p1>[\d.eE+-]+), "
    r"policy_l1_p2=(?P<policy_p2>[\d.eE+-]+), "
    r"value_abs_p1=(?P<value_p1>[\d.eE+-]+), "
    r"value_abs_p2=(?P<value_p2>[\d.eE+-]+)"
)


def _parse_ints(value: str) -> list[int]:
    parsed = [int(item) for item in value.split(",") if item]
    if not parsed or len(set(parsed)) != len(parsed) or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("expected distinct comma-separated positive integers")
    return parsed


def _throughput(value: str) -> float:
    if value.endswith("M"):
        return float(value[:-1]) * 1_000_000.0
    if value.endswith("k"):
        return float(value[:-1]) * 1_000.0
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _percentile(histogram: list[list[int]], percentile: float) -> int:
    total = sum(count for _, count in histogram)
    if total == 0:
        return 0
    rank = max(1, int(total * percentile + 0.999999))
    cumulative = 0
    for batch, count in histogram:
        cumulative += count
        if cumulative >= rank:
            return batch
    return histogram[-1][0]


def _mode_order(repetition: int) -> tuple[str, str]:
    return MODES if repetition % 2 == 0 else tuple(reversed(MODES))


def _batch_order(batches: list[int], repetition: int) -> list[int]:
    if repetition % 2 == 0:
        return batches
    return list(reversed(batches))


def _run_capacity(
    *,
    binary: Path,
    model: Path,
    cache_dir: Path,
    raw_path: Path,
    mode: str,
    batches: list[int],
    iterations: int,
    max_batch: int,
) -> dict[str, Any]:
    command = [
        str(binary),
        str(model),
        "--device",
        "tensorrt",
        "--max-batch",
        str(max_batch),
        "--opt-batch",
        str(max_batch),
        "--callers",
        "1",
        "--host-io",
        mode,
        "--cache-dir",
        str(cache_dir),
        "--batches",
        ",".join(str(batch) for batch in batches),
        "--iters",
        str(iterations),
        "--verify-parity",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    raw = completed.stdout + completed.stderr
    raw_path.write_text(raw, encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        match = CAPACITY_ROW.match(line)
        if match is None:
            continue
        rows.append(
            {
                "batch": int(match["batch"]),
                "callers": int(match["callers"]),
                "input_stage_us": float(match["stage"]),
                "h2d_us": float(match["h2d"]),
                "infer_us": float(match["infer"]),
                "output_alloc_us": float(match["alloc"]),
                "d2h_us": float(match["d2h"]),
                "parse_us": float(match["parse"]),
                "residual_us": float(match["residual"]),
                "call_us": float(match["call"]),
                "effective_us": float(match["effective"]),
                "positions_per_second": _throughput(match["throughput"]),
            }
        )

    parity = {
        int(match["batch"]): float(match["value"]) for match in PARITY.finditer(completed.stdout)
    }
    root_match = ROOT_BEHAVIOR.search(completed.stdout)
    measured = {row["batch"] for row in rows}
    expected = set(batches)
    if measured != expected or set(parity) != expected or root_match is None:
        raise RuntimeError(f"incomplete capacity record; inspect {raw_path}")
    return {
        "command": command,
        "batch_order": batches,
        "rows": rows,
        "max_abs_output_diff_by_batch": {str(batch): parity[batch] for batch in batches},
        "root_behavior": {
            "policy_l1_p1": float(root_match["policy_p1"]),
            "policy_l1_p2": float(root_match["policy_p2"]),
            "value_abs_p1": float(root_match["value_p1"]),
            "value_abs_p2": float(root_match["value_p2"]),
        },
    }


def _run_selfplay(
    *,
    model: Path,
    games_dir: Path,
    mode: str,
    games: int,
    simulations: int,
    threads: int,
    search_batch: int,
    max_batch: int,
) -> dict[str, Any]:
    from alpharat_sampling import rust_self_play

    stats = rust_self_play(
        width=7,
        height=7,
        cheese_count=10,
        max_turns=50,
        num_games=games,
        maze_type="open",
        positions="corners",
        cheese_symmetric=True,
        simulations=simulations,
        batch_size=search_batch,
        c_puct=0.512,
        fpu_reduction=0.459,
        force_k=0.103,
        noise_epsilon=0.25,
        noise_concentration=10.83,
        collision_limit_min=1,
        collision_limit_max=256,
        collision_scaling_start=800,
        collision_scaling_end=50_000,
        collision_scaling_power=1.0,
        num_threads=threads,
        output_dir=str(games_dir),
        max_games_per_bundle=32,
        onnx_model_path=str(model),
        device="tensorrt",
        mux_max_batch_size=max_batch,
        tensorrt_opt_batch=max_batch,
        tensorrt_pinned_host_io=mode == "pinned",
        tensorrt_profile_stages=True,
        use_inference_mux=True,
        cache_size=0,
    )
    elapsed = float(stats.elapsed_secs)
    histogram = [[int(batch), int(count)] for batch, count in stats.inference_batch_histogram]
    result = {
        "host_io": str(stats.tensorrt_host_io),
        "pinned_bytes": int(stats.tensorrt_pinned_bytes),
        "profiled_calls": int(stats.tensorrt_profiled_calls),
        "profiled_positions": int(stats.tensorrt_profiled_positions),
        "total_games": int(stats.total_games),
        "total_positions": int(stats.total_positions),
        "total_simulations": int(stats.total_simulations),
        "total_nn_evals": int(stats.total_nn_evals),
        "total_terminals": int(stats.total_terminals),
        "total_collisions": int(stats.total_collisions),
        "elapsed_seconds": elapsed,
        "simulations_per_second": float(stats.total_simulations) / elapsed,
        "nn_evals_per_second": float(stats.total_nn_evals) / elapsed,
        "inference_batches": int(stats.inference_batches),
        "inference_positions": int(stats.inference_positions),
        "inference_avg_batch_size": float(stats.inference_avg_batch_size),
        "inference_nn_seconds": float(stats.inference_nn_seconds),
        "inference_wait_seconds": float(stats.inference_wait_seconds),
        "inference_batch_histogram": histogram,
        "inference_batch_p50": _percentile(histogram, 0.5),
        "inference_batch_p90": _percentile(histogram, 0.9),
        "encode_seconds": float(stats.tensorrt_encode_seconds),
        "input_stage_seconds": float(stats.tensorrt_input_stage_seconds),
        "h2d_seconds": float(stats.tensorrt_h2d_seconds),
        "infer_seconds": float(stats.tensorrt_infer_seconds),
        "output_alloc_seconds": float(stats.tensorrt_output_alloc_seconds),
        "d2h_seconds": float(stats.tensorrt_d2h_seconds),
        "parse_seconds": float(stats.tensorrt_parse_seconds),
        "backend_profiled_seconds": float(stats.tensorrt_total_seconds),
    }
    explicit_seconds = sum(
        float(result[field])
        for field in (
            "encode_seconds",
            "input_stage_seconds",
            "h2d_seconds",
            "infer_seconds",
            "output_alloc_seconds",
            "d2h_seconds",
            "parse_seconds",
        )
    )
    result["residual_wall_seconds"] = max(
        0.0,
        float(result["backend_profiled_seconds"]) - explicit_seconds,
    )
    if result["host_io"] != mode:
        raise RuntimeError(f"requested {mode} host I/O but backend reported {result['host_io']}")
    if result["profiled_calls"] != result["inference_batches"]:
        raise RuntimeError("TensorRT and mux call counts disagree")
    if result["profiled_positions"] != result["inference_positions"]:
        raise RuntimeError("TensorRT and mux position counts disagree")
    return result


def _median(records: list[dict[str, Any]], field: str) -> float:
    return statistics.median(float(record[field]) for record in records)


def _render_comparison(record: dict[str, Any]) -> str:
    cases = record["cases"]
    pageable = cases["pageable"]
    pinned = cases["pinned"]
    lines = [
        "# TensorRT reusable pinned-host I/O experiment",
        "",
        (
            "Validity is a hard gate. Performance below is descriptive; this experiment "
            "does not encode a numerical pass/fail speed threshold."
        ),
        "",
        "## Long production-shaped self-play",
        "",
        (
            "| Host I/O | median wall | median sims/s | median NN evals/s | "
            "device batch avg | p50 | p90 | backend time | queue wait |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        trials = cases[mode]["selfplay_trials"]
        lines.append(
            "| {mode} | {wall:.3f}s | {sims:,.0f} | {nn:,.0f} | {avg:.2f} | "
            "{p50:.0f} | {p90:.0f} | {backend:.3f}s | {wait:.3f}s |".format(
                mode=mode,
                wall=_median(trials, "elapsed_seconds"),
                sims=_median(trials, "simulations_per_second"),
                nn=_median(trials, "nn_evals_per_second"),
                avg=_median(trials, "inference_avg_batch_size"),
                p50=_median(trials, "inference_batch_p50"),
                p90=_median(trials, "inference_batch_p90"),
                backend=_median(trials, "inference_nn_seconds"),
                wait=_median(trials, "inference_wait_seconds"),
            )
        )
    page_rate = _median(pageable["selfplay_trials"], "simulations_per_second")
    pinned_rate = _median(pinned["selfplay_trials"], "simulations_per_second")
    lines.extend(
        [
            "",
            f"Pinned/pageable median self-play sims/s ratio: **{pinned_rate / page_rate:.4f}×**.",
            "",
            "## Profiled TensorRT stages (median aggregate seconds per self-play trial)",
            "",
            (
                "| Host I/O | encode | input stage | H2D | inference | output alloc | "
                "D2H | parse | residual wall | profiled backend | pinned bytes |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        trials = cases[mode]["selfplay_trials"]
        lines.append(
            "| {mode} | {encode:.4f} | {stage:.4f} | {h2d:.4f} | {infer:.4f} | "
            "{alloc:.4f} | {d2h:.4f} | {parse:.4f} | {residual:.4f} | "
            "{total:.4f} | {pinned:.0f} |".format(
                mode=mode,
                encode=_median(trials, "encode_seconds"),
                stage=_median(trials, "input_stage_seconds"),
                h2d=_median(trials, "h2d_seconds"),
                infer=_median(trials, "infer_seconds"),
                alloc=_median(trials, "output_alloc_seconds"),
                d2h=_median(trials, "d2h_seconds"),
                parse=_median(trials, "parse_seconds"),
                residual=_median(trials, "residual_wall_seconds"),
                total=_median(trials, "backend_profiled_seconds"),
                pinned=_median(trials, "pinned_bytes"),
            )
        )
    lines.extend(
        [
            "",
            (
                "Residual wall time is total profiled backend wall time minus the explicit "
                "host stages and non-overlapping CUDA-event intervals. It includes shape/lane "
                "setup, CUDA/TensorRT API overhead, and completion-wait overhead not otherwise "
                "attributed."
            ),
            "",
            "## Capacity, parity, and lifecycle",
            "",
            (
                f"Each mode ran in {record['config']['capacity_repetitions']} fresh processes. "
                "Every process created and dropped its backend, alternated batch order, checked "
                "all requested batches against a pageable baseline, and repeated deterministic "
                "root behavior. Raw process output remains under `capacity-raw/`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--capacity-binary",
        type=Path,
        default=Path("target/release/bench_nn_throughput"),
    )
    parser.add_argument("--batches", type=_parse_ints, default=_parse_ints("1,16,32,64,128"))
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--capacity-iters", type=int, default=200)
    parser.add_argument("--capacity-repetitions", type=int, default=3)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--selfplay-trials", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=1_897)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--search-batch", type=int, default=16)
    args = parser.parse_args()

    model = args.model.resolve()
    binary = args.capacity_binary.resolve()
    output_dir = args.output_dir.resolve()
    if not model.is_file():
        parser.error(f"model does not exist: {model}")
    if not binary.is_file():
        parser.error(f"capacity binary does not exist: {binary}")
    if max(args.batches) > args.max_batch:
        parser.error("every requested batch must be <= max batch")
    if args.capacity_repetitions < 1 or args.selfplay_trials < 1:
        parser.error("repetition counts must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    capacity_raw = output_dir / "capacity-raw"
    capacity_raw.mkdir()
    selfplay_root = output_dir / "selfplay-artifacts"
    selfplay_root.mkdir()
    cache_dir = output_dir / "engine-cache"
    cache_dir.mkdir()

    from alpharat_sampling import preload_tensorrt_libs

    preload_tensorrt_libs()
    git_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        check=True,
        capture_output=True,
    ).stdout
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
        nvidia_smi = "/usr/lib/wsl/lib/nvidia-smi"
    record: dict[str, Any] = {
        "protocol_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "scope": {
            "execution_contexts": 1,
            "cuda_graphs": False,
            "inference_mux": "eager",
            "profile": f"MIN=1,OPT={args.max_batch},MAX={args.max_batch}",
        },
        "source": {
            "git_head": _command_output(["git", "rev-parse", "HEAD"]),
            "git_status": _command_output(["git", "status", "--short"]),
            "git_diff_sha256": hashlib.sha256(git_diff).hexdigest(),
            "binary": str(binary),
            "binary_sha256": _sha256(binary),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "nvidia_smi": (
                _command_output(
                    [
                        nvidia_smi,
                        (
                            "--query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,"
                            "temperature.gpu,power.draw"
                        ),
                        "--format=csv,noheader",
                    ]
                )
                if nvidia_smi is not None
                else None
            ),
        },
        "model": {"path": str(model), "sha256": _sha256(model)},
        "config": {
            "batches": args.batches,
            "max_batch": args.max_batch,
            "capacity_iterations": args.capacity_iters,
            "capacity_repetitions": args.capacity_repetitions,
            "games_per_trial": args.games,
            "selfplay_trials": args.selfplay_trials,
            "simulations": args.simulations,
            "threads": args.threads,
            "search_batch": args.search_batch,
        },
        "execution_log": [],
        "cases": {mode: {"capacity_runs": [], "selfplay_trials": []} for mode in MODES},
    }
    _write_json(output_dir / "record.json", record)

    execution_order = 0
    for repetition in range(args.capacity_repetitions):
        batches = _batch_order(args.batches, repetition)
        for mode in _mode_order(repetition):
            execution_order += 1
            raw_path = capacity_raw / f"{execution_order:02d}-{mode}-rep{repetition + 1}.txt"
            run = _run_capacity(
                binary=binary,
                model=model,
                cache_dir=cache_dir,
                raw_path=raw_path,
                mode=mode,
                batches=batches,
                iterations=args.capacity_iters,
                max_batch=args.max_batch,
            )
            run.update(
                {
                    "repetition": repetition + 1,
                    "execution_order": execution_order,
                    "raw_output": str(raw_path),
                }
            )
            record["cases"][mode]["capacity_runs"].append(run)
            record["execution_log"].append(
                {"order": execution_order, "phase": "capacity", "mode": mode}
            )
            _write_json(output_dir / "record.json", record)

    for trial in range(args.selfplay_trials):
        for mode in _mode_order(trial):
            execution_order += 1
            games_dir = selfplay_root / f"{execution_order:02d}-{mode}-trial{trial + 1}-games"
            games_dir.mkdir()
            result = _run_selfplay(
                model=model,
                games_dir=games_dir,
                mode=mode,
                games=args.games,
                simulations=args.simulations,
                threads=args.threads,
                search_batch=args.search_batch,
                max_batch=args.max_batch,
            )
            result.update(
                {
                    "trial": trial + 1,
                    "execution_order": execution_order,
                    "artifacts": str(games_dir),
                }
            )
            record["cases"][mode]["selfplay_trials"].append(result)
            record["execution_log"].append(
                {"order": execution_order, "phase": "selfplay", "mode": mode}
            )
            _write_json(output_dir / "record.json", record)

    record["completed_at"] = datetime.now(UTC).isoformat()
    _write_json(output_dir / "record.json", record)
    (output_dir / "comparison.md").write_text(_render_comparison(record), encoding="utf-8")
    print(f"Wrote {output_dir / 'record.json'}")
    print(f"Wrote {output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
