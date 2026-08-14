#!/usr/bin/env python3
"""Run paired pageable/pinned 512-game sample->shard->train loops."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODES = ("pageable", "pinned")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _mode_order(trial: int) -> tuple[str, str]:
    return MODES if trial % 2 == 0 else tuple(reversed(MODES))


def _median(records: list[dict[str, Any]], field: str) -> float:
    return statistics.median(float(record[field]) for record in records)


def _summarize_timing(timing: dict[str, Any]) -> dict[str, Any]:
    iteration = timing["iterations"][0]
    phases = iteration["phases"]
    sampling = phases["sampling"]
    metrics = sampling["metrics"]
    return {
        "wall_seconds": float(iteration["wall_seconds"]),
        "sampling_wall_seconds": float(sampling["wall_seconds"]),
        "sharding_wall_seconds": float(phases["sharding"]["wall_seconds"]),
        "training_wall_seconds": float(phases["training"]["wall_seconds"]),
        "sampling_elapsed_seconds": float(metrics["elapsed_seconds"]),
        "simulations_per_second": float(metrics["total_simulations"])
        / float(metrics["elapsed_seconds"]),
        "nn_evals_per_second": float(metrics["total_nn_evals"]) / float(metrics["elapsed_seconds"]),
        "total_games": int(metrics["total_games"]),
        "total_positions": int(metrics["total_positions"]),
        "total_simulations": int(metrics["total_simulations"]),
        "total_nn_evals": int(metrics["total_nn_evals"]),
        "inference_batches": int(metrics["inference_batches"]),
        "inference_positions": int(metrics["inference_positions"]),
        "inference_avg_batch_size": float(metrics["inference_positions"])
        / float(metrics["inference_batches"]),
        "inference_nn_seconds": float(metrics["inference_nn_seconds"]),
        "inference_wait_seconds": float(metrics["inference_wait_seconds"]),
        "host_io": str(metrics["tensorrt_host_io"]),
        "pinned_bytes": int(metrics["tensorrt_pinned_bytes"]),
        "profiled_calls": int(metrics["tensorrt_profiled_calls"]),
        "profiled_positions": int(metrics["tensorrt_profiled_positions"]),
        "encode_seconds": float(metrics["tensorrt_encode_seconds"]),
        "input_stage_seconds": float(metrics["tensorrt_input_stage_seconds"]),
        "h2d_seconds": float(metrics["tensorrt_h2d_seconds"]),
        "infer_seconds": float(metrics["tensorrt_infer_seconds"]),
        "output_alloc_seconds": float(metrics["tensorrt_output_alloc_seconds"]),
        "d2h_seconds": float(metrics["tensorrt_d2h_seconds"]),
        "parse_seconds": float(metrics["tensorrt_parse_seconds"]),
        "backend_profiled_seconds": float(metrics["tensorrt_total_seconds"]),
        "sampling_artifact": str(sampling["artifact"]),
        "sharding_artifact": str(phases["sharding"]["artifact"]),
        "training_artifact": str(phases["training"]["artifact"]),
    }


def _render_comparison(record: dict[str, Any]) -> str:
    cases = record["cases"]
    lines = [
        "# Paired TensorRT host-I/O full-loop comparison",
        "",
        (
            "Each row is the median of complete 512-game, 300-epoch "
            "sample → shard → train loops from the same checkpoint."
        ),
        "",
        "| Host I/O | total wall | sampling | sharding | training | sims/s | NN evals/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        trials = cases[mode]
        lines.append(
            "| {mode} | {total:.3f}s | {sampling:.3f}s | {sharding:.3f}s | "
            "{training:.3f}s | {sims:,.0f} | {nn:,.0f} |".format(
                mode=mode,
                total=_median(trials, "wall_seconds"),
                sampling=_median(trials, "sampling_wall_seconds"),
                sharding=_median(trials, "sharding_wall_seconds"),
                training=_median(trials, "training_wall_seconds"),
                sims=_median(trials, "simulations_per_second"),
                nn=_median(trials, "nn_evals_per_second"),
            )
        )
    pageable_total = _median(cases["pageable"], "wall_seconds")
    pinned_total = _median(cases["pinned"], "wall_seconds")
    pageable_sampling = _median(cases["pageable"], "sampling_wall_seconds")
    pinned_sampling = _median(cases["pinned"], "sampling_wall_seconds")
    lines.extend(
        [
            "",
            f"Pinned/pageable median total-wall ratio: **{pinned_total / pageable_total:.4f}×**.",
            (
                "Pinned/pageable median sampling-wall ratio: "
                f"**{pinned_sampling / pageable_sampling:.4f}×**."
            ),
            "",
            (
                "Training was observed without changing its recipe. Sampling remains stochastic; "
                "order alternates by pair, while sharding and training retain their fixed seed 42."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pageable-config", type=Path, required=True)
    parser.add_argument("--pinned-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    configs = {
        "pageable": args.pageable_config.resolve(),
        "pinned": args.pinned_config.resolve(),
    }
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    for path in (*configs.values(), checkpoint):
        if not path.is_file():
            parser.error(f"input does not exist: {path}")
    if args.trials < 1:
        parser.error("trials must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)

    git_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        check=True,
        capture_output=True,
    ).stdout
    record: dict[str, Any] = {
        "protocol_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "scope": {
            "games": 512,
            "training_epochs": 300,
            "execution_contexts": 1,
            "cuda_graphs": False,
            "inference_mux": "eager",
            "profile": "MIN=1,OPT=128,MAX=128",
        },
        "source": {
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "git_diff_sha256": hashlib.sha256(git_diff).hexdigest(),
        },
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "configs": {
            mode: {"path": str(path), "sha256": _sha256(path)} for mode, path in configs.items()
        },
        "trials": args.trials,
        "execution_log": [],
        "cases": {mode: [] for mode in MODES},
    }
    _write_json(output_dir / "record.json", record)

    execution_order = 0
    iterate = Path("scripts/iterate.py").resolve()
    for trial in range(args.trials):
        for mode in _mode_order(trial):
            execution_order += 1
            case_dir = output_dir / f"{execution_order:02d}-{mode}-trial{trial + 1}"
            artifacts = case_dir / "artifacts"
            case_dir.mkdir()
            command = [
                sys.executable,
                str(iterate),
                str(configs[mode]),
                "--prefix",
                f"host_io_{mode}_trial{trial + 1}",
                "--iterations",
                "1",
                "--start-checkpoint",
                str(checkpoint),
                "--no-benchmark",
                "--experiments-dir",
                str(artifacts),
                "--device",
                "tensorrt",
                "--timing-output",
                str(case_dir / "timing.json"),
            ]
            log_path = case_dir / "iterate.log"
            started_at = datetime.now(UTC).isoformat()
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode != 0:
                record["failure"] = {
                    "mode": mode,
                    "trial": trial + 1,
                    "execution_order": execution_order,
                    "returncode": completed.returncode,
                    "log": str(log_path),
                }
                _write_json(output_dir / "record.json", record)
                raise SystemExit(completed.returncode)

            timing_path = case_dir / "timing.json"
            result = _summarize_timing(json.loads(timing_path.read_text(encoding="utf-8")))
            if result["host_io"] != mode:
                raise RuntimeError(f"requested {mode}, backend reported {result['host_io']}")
            if result["profiled_calls"] != result["inference_batches"]:
                raise RuntimeError("TensorRT and mux call counts disagree")
            if result["profiled_positions"] != result["inference_positions"]:
                raise RuntimeError("TensorRT and mux position counts disagree")
            result.update(
                {
                    "trial": trial + 1,
                    "execution_order": execution_order,
                    "started_at": started_at,
                    "completed_at": datetime.now(UTC).isoformat(),
                    "command": command,
                    "timing_record": str(timing_path),
                    "log": str(log_path),
                }
            )
            record["cases"][mode].append(result)
            record["execution_log"].append(
                {"order": execution_order, "mode": mode, "trial": trial + 1}
            )
            _write_json(output_dir / "record.json", record)

    record["completed_at"] = datetime.now(UTC).isoformat()
    _write_json(output_dir / "record.json", record)
    (output_dir / "comparison.md").write_text(_render_comparison(record), encoding="utf-8")
    print(f"Wrote {output_dir / 'record.json'}")
    print(f"Wrote {output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
