#!/usr/bin/env python3
"""Measure TensorRT profile points against real independent-game self-play.

This is the rerunnable Chunk 2 experiment surface. It preserves raw capacity output,
exact achieved device-batch histograms, repeated whole-self-play trials, numerical
parity, and deterministic root-behavior drift in one folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CAPACITY_ROW = re.compile(
    r"^\s*(?P<batch>\d+)\s+(?P<callers>\d+)\s+"
    r"(?P<h2d>[\d.]+)\s+(?P<infer>[\d.]+)\s+(?P<d2h>[\d.]+)\s+"
    r"(?P<effective>[\d.]+)\s+(?P<throughput>[\d.]+[kM]?)\s*$"
)
PARITY = re.compile(r"max_abs_diff=(?P<value>[\d.eE+-]+)")
ROOT_BEHAVIOR = re.compile(
    r"Root behavior: policy_l1_p1=(?P<policy_p1>[\d.eE+-]+), "
    r"policy_l1_p2=(?P<policy_p2>[\d.eE+-]+), "
    r"value_abs_p1=(?P<value_p1>[\d.eE+-]+), "
    r"value_abs_p2=(?P<value_p2>[\d.eE+-]+)"
)


def _parse_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _parse_graph_modes(value: str) -> tuple[bool, ...]:
    names = [item.strip().lower() for item in value.split(",")]
    if not names or any(name not in {"off", "on"} for name in names):
        raise argparse.ArgumentTypeError("graph modes must be a comma-separated subset of off,on")
    return tuple(name == "on" for name in dict.fromkeys(names))


def _throughput(value: str) -> float:
    if value.endswith("M"):
        return float(value[:-1]) * 1_000_000.0
    if value.endswith("k"):
        return float(value[:-1]) * 1_000.0
    return float(value)


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_capacity(
    binary: Path,
    model: Path,
    cache_dir: Path,
    case_dir: Path,
    opt_batch: int,
    max_batch: int,
    batches: list[int],
    iterations: int,
    graphs: bool,
) -> dict[str, Any]:
    command = [
        str(binary),
        str(model),
        "--device",
        "tensorrt",
        "--max-batch",
        str(max_batch),
        "--opt-batch",
        str(opt_batch),
        "--contexts",
        "1",
        "--callers",
        "1",
        "--cache-dir",
        str(cache_dir),
        "--batches",
        ",".join(str(batch) for batch in batches),
        "--iters",
        str(iterations),
        "--verify-parity",
    ]
    if graphs:
        command.append("--cuda-graphs")

    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    raw = completed.stdout + completed.stderr
    (case_dir / "capacity.txt").write_text(raw, encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        match = CAPACITY_ROW.match(line)
        if match is None:
            continue
        rows.append(
            {
                "batch": int(match["batch"]),
                "callers": int(match["callers"]),
                "h2d_us": float(match["h2d"]),
                "infer_us": float(match["infer"]),
                "d2h_us": float(match["d2h"]),
                "effective_us": float(match["effective"]),
                "positions_per_second": _throughput(match["throughput"]),
            }
        )

    parity_match = PARITY.search(completed.stdout)
    root_match = ROOT_BEHAVIOR.search(completed.stdout)
    measured_batches = {row["batch"] for row in rows}
    missing_batches = set(batches) - measured_batches
    if parity_match is None or root_match is None or missing_batches:
        raise RuntimeError(
            f"could not parse complete capacity record from {case_dir / 'capacity.txt'}"
        )

    return {
        "command": command,
        "rows": rows,
        "max_abs_output_diff": float(parity_match["value"]),
        "root_behavior": {
            "policy_l1_p1": float(root_match["policy_p1"]),
            "policy_l1_p2": float(root_match["policy_p2"]),
            "value_abs_p1": float(root_match["value_p1"]),
            "value_abs_p2": float(root_match["value_p2"]),
        },
    }


def _run_selfplay_trial(
    model: Path,
    output_dir: Path,
    games: int,
    simulations: int,
    threads: int,
    search_batch: int,
    max_batch: int,
    opt_batch: int,
    graphs: bool,
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
        output_dir=str(output_dir),
        max_games_per_bundle=32,
        onnx_model_path=str(model),
        device="tensorrt",
        mux_max_batch_size=max_batch,
        tensorrt_opt_batch=opt_batch,
        tensorrt_execution_contexts=1,
        tensorrt_cuda_graphs=graphs,
        use_inference_mux=True,
        cache_size=0,
    )
    histogram = [[int(batch), int(count)] for batch, count in stats.inference_batch_histogram]
    elapsed = float(stats.elapsed_secs)
    return {
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
    }


def _median(trials: list[dict[str, Any]], field: str) -> float:
    return statistics.median(float(trial[field]) for trial in trials)


def _render_comparison(record: dict[str, Any]) -> str:
    cases = record["cases"]
    baseline = next(
        case
        for case in cases
        if case["opt_batch"] == record["config"]["max_batch"] and not case["graphs"]
    )
    baseline_rate = _median(baseline["selfplay_trials"], "simulations_per_second")
    lines = [
        "# NN sampling workload/profile comparison",
        "",
        (
            "This record compares runtime profile points only. It does not select a model or "
            "search default."
        ),
        "",
        "| TensorRT profile | Graphs | self-play sims/s | vs OPT=MAX graph-off | "
        "NN evals/s | device batch avg | p50 | p90 | capacity @32 | output max Δ | "
        "root policy L1 (P1/P2) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        trials = case["selfplay_trials"]
        rate = _median(trials, "simulations_per_second")
        capacity_32 = next(
            row["positions_per_second"] for row in case["capacity"]["rows"] if row["batch"] == 32
        )
        root = case["capacity"]["root_behavior"]
        lines.append(
            "| OPT={opt}, MAX={max} | {graphs} | {rate:,.0f} | {delta:+.1%} | "
            "{nn:,.0f} | {avg:.1f} | {p50:.0f} | {p90:.0f} | {capacity:,.0f} | "
            "{output:.2e} | {p1:.3f}/{p2:.3f} |".format(
                opt=case["opt_batch"],
                max=record["config"]["max_batch"],
                graphs="on" if case["graphs"] else "off",
                rate=rate,
                delta=rate / baseline_rate - 1.0,
                nn=_median(trials, "nn_evals_per_second"),
                avg=_median(trials, "inference_avg_batch_size"),
                p50=_median(trials, "inference_batch_p50"),
                p90=_median(trials, "inference_batch_p90"),
                capacity=capacity_32,
                output=case["capacity"]["max_abs_output_diff"],
                p1=root["policy_l1_p1"],
                p2=root["policy_l1_p2"],
            )
        )
    lines.extend(
        [
            "",
            (
                "Self-play rows are medians of repeated complete-game trials. Capacity, output "
                "parity, and deterministic root behavior are measured once per profile point. "
                "Raw outputs and game bundles remain beside this file."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="7x7 scalar-head ONNX model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--capacity-binary",
        type=Path,
        default=Path("target/release/bench_nn_throughput"),
    )
    parser.add_argument("--opt-batches", type=_parse_ints, default=_parse_ints("16,32,64,128"))
    parser.add_argument("--capacity-batches", type=_parse_ints, default=_parse_ints("16,32,64,128"))
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=1_897)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--search-batch", type=int, default=16)
    parser.add_argument("--capacity-iters", type=int, default=200)
    parser.add_argument(
        "--graph-modes",
        type=_parse_graph_modes,
        default=_parse_graph_modes("off,on"),
        help="Comma-separated TensorRT CUDA graph modes: off,on (default), off, or on",
    )
    args = parser.parse_args()

    model = args.model.resolve()
    output_dir = args.output_dir.resolve()
    binary = args.capacity_binary.resolve()
    if not model.is_file():
        parser.error(f"model does not exist: {model}")
    if not binary.is_file():
        parser.error(f"capacity binary does not exist: {binary}")
    if any(opt > args.max_batch for opt in args.opt_batches):
        parser.error("every optimization point must be <= max batch")
    if args.max_batch not in args.opt_batches or False not in args.graph_modes:
        parser.error("the comparison baseline requires OPT=MAX with graph mode off")

    output_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = output_dir / "engine-cache"
    cache_dir.mkdir()
    selfplay_root = output_dir / "selfplay-artifacts"
    selfplay_root.mkdir()

    from alpharat_sampling import preload_tensorrt_libs

    preload_tensorrt_libs()
    record: dict[str, Any] = {
        "protocol_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model": {
            "path": str(model),
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        },
        "config": {
            "max_batch": args.max_batch,
            "opt_batches": args.opt_batches,
            "capacity_batches": args.capacity_batches,
            "games_per_trial": args.games,
            "trials": args.trials,
            "simulations": args.simulations,
            "threads": args.threads,
            "search_batch": args.search_batch,
            "capacity_iterations": args.capacity_iters,
            "graph_modes": ["on" if graphs else "off" for graphs in args.graph_modes],
        },
        "cases": [],
    }
    _write_json(output_dir / "record.json", record)

    cases: list[dict[str, Any]] = record["cases"]
    for opt_batch in args.opt_batches:
        for graphs in args.graph_modes:
            label = f"opt{opt_batch}-graphs-{'on' if graphs else 'off'}"
            case_dir = output_dir / label
            case_dir.mkdir()
            case: dict[str, Any] = {
                "label": label,
                "opt_batch": opt_batch,
                "graphs": graphs,
                "capacity": _run_capacity(
                    binary,
                    model,
                    cache_dir,
                    case_dir,
                    opt_batch,
                    args.max_batch,
                    args.capacity_batches,
                    args.capacity_iters,
                    graphs,
                ),
                "selfplay_trials": [],
            }
            cases.append(case)
            _write_json(output_dir / "record.json", record)

    execution_order = 0
    for trial_index in range(args.trials):
        trial_cases = cases if trial_index % 2 == 0 else list(reversed(cases))
        for case in trial_cases:
            label = str(case["label"])
            opt_batch = int(case["opt_batch"])
            graphs = bool(case["graphs"])
            execution_order += 1
            games_dir = selfplay_root / f"{label}-trial{trial_index + 1}-games"
            games_dir.mkdir()
            trial = _run_selfplay_trial(
                model,
                games_dir,
                args.games,
                args.simulations,
                args.threads,
                args.search_batch,
                args.max_batch,
                opt_batch,
                graphs,
            )
            trial["trial"] = trial_index + 1
            trial["execution_order"] = execution_order
            case["selfplay_trials"].append(trial)
            _write_json(output_dir / "record.json", record)

    (output_dir / "comparison.md").write_text(_render_comparison(record), encoding="utf-8")
    print(f"Wrote {output_dir / 'record.json'}")
    print(f"Wrote {output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
