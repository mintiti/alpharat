"""Profile Rust self-play pipeline with nsys.

Usage:
    # CPU (creates a random model if no checkpoint given)
    uv run python scripts/profile_rust_selfplay.py --device cpu

    # CUDA
    uv run python scripts/profile_rust_selfplay.py --device cuda

    # With a real checkpoint
    uv run python scripts/profile_rust_selfplay.py --device cuda --checkpoint path/to/best_model.pt

    # Wrap with nsys for GPU profiling
    nsys profile -o selfplay_cuda --force-overwrite \
        uv run python scripts/profile_rust_selfplay.py --device cuda
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch


def _export_random_model(width: int, height: int, onnx_path: Path) -> None:
    """Create a random SymmetricMLP and export directly to ONNX."""
    from alpharat.nn.architectures.symmetric.config import SymmetricModelConfig
    from scripts.export_onnx import OnnxWrapper

    config = SymmetricModelConfig(hidden_dim=256)
    config.set_data_dimensions(width, height)
    model = config.build_model()
    builder = config.build_observation_builder(width, height)

    wrapper = OnnxWrapper(model)  # type: ignore[arg-type]
    wrapper.eval()

    obs_dim: int = builder.obs_shape[0]  # type: ignore[attr-defined]
    dummy_input = torch.randn(1, obs_dim)

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(onnx_path),
        input_names=["observation"],
        output_names=["policy_p1", "policy_p2", "pred_value_p1", "pred_value_p2"],
        dynamic_axes={
            "observation": {0: "batch"},
            "policy_p1": {0: "batch"},
            "policy_p2": {0: "batch"},
            "pred_value_p1": {0: "batch"},
            "pred_value_p2": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported random model to {onnx_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Rust self-play with NN")
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument(
        "--checkpoint",
        default=None,
        help="Path to .pt checkpoint (omit to use a random model)",
    )
    model_source.add_argument(
        "--onnx",
        default=None,
        help="Path to an already-exported ONNX model (avoids Python model/config imports)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "coreml", "tensorrt"],
        help="Execution provider (cpu/cuda/coreml use ORT, tensorrt uses TensorRT-RTX)",
    )
    parser.add_argument("--games", type=int, default=50, help="Number of games")
    parser.add_argument("--sims", type=int, default=600, help="MCTS simulations per move")
    parser.add_argument("--threads", type=int, default=4, help="Worker threads")
    parser.add_argument("--batch-size", type=int, default=16, help="Within-tree NN batch size")
    parser.add_argument("--mux-batch", type=int, default=256, help="Mux max batch size")
    parser.add_argument(
        "--tensorrt-opt-batch",
        type=int,
        default=None,
        help="TensorRT dynamic-profile optimization point (defaults to mux batch)",
    )
    parser.add_argument(
        "--tensorrt-contexts",
        type=int,
        default=1,
        help="Independent TensorRT context/stream/buffer lanes",
    )
    parser.add_argument(
        "--tensorrt-cuda-graphs",
        action="store_true",
        help="Request whole-model CUDA Graph capture for TensorRT lanes",
    )
    parser.add_argument(
        "--no-inference-mux",
        action="store_true",
        help="Send worker requests directly to the backend",
    )
    args = parser.parse_args()

    width, height = 7, 7

    # Get or create ONNX model
    if args.onnx is not None:
        onnx_path = args.onnx
    elif args.checkpoint is not None:
        from alpharat.data.rust_sampling import _ensure_onnx

        onnx_path = _ensure_onnx(args.checkpoint)
    else:
        onnx_path = str(Path(tempfile.gettempdir()) / "alpharat_profile_model.onnx")
        _export_random_model(width, height, Path(onnx_path))

    print(f"ONNX model: {onnx_path}")

    from alpharat_sampling import rust_self_play

    if args.device == "tensorrt":
        from alpharat_sampling import preload_tensorrt_libs

        preload_tensorrt_libs()
    elif args.device != "cpu":
        from alpharat_sampling import preload_cuda_libs

        preload_cuda_libs()

    profile_root = Path(tempfile.gettempdir()) / "alpharat_profile_selfplay"
    output_dir = profile_root / "games"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)

    print(
        f"Running {args.games} games, {args.sims} sims, "
        f"{args.threads} threads, batch={args.batch_size}, "
        f"mux_batch={args.mux_batch}, trt_contexts={args.tensorrt_contexts}, "
        f"trt_opt_batch={args.tensorrt_opt_batch or args.mux_batch}, "
        f"trt_cuda_graphs={args.tensorrt_cuda_graphs}, "
        f"inference_mux={not args.no_inference_mux}, device={args.device}"
    )

    stats = rust_self_play(
        width=width,
        height=height,
        cheese_count=10,
        max_turns=50,
        num_games=args.games,
        maze_type="open",
        positions="corners",
        cheese_symmetric=True,
        simulations=args.sims,
        batch_size=args.batch_size,
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
        num_threads=args.threads,
        output_dir=str(output_dir),
        max_games_per_bundle=32,
        onnx_model_path=onnx_path,
        mux_max_batch_size=args.mux_batch,
        tensorrt_opt_batch=args.tensorrt_opt_batch,
        tensorrt_execution_contexts=args.tensorrt_contexts,
        tensorrt_cuda_graphs=args.tensorrt_cuda_graphs,
        use_inference_mux=not args.no_inference_mux,
        device=args.device,
    )

    elapsed = stats.elapsed_secs
    print("\nResults:")
    print(f"  Games: {stats.total_games}")
    print(f"  Positions: {stats.total_positions}")
    print(f"  Simulations: {stats.total_simulations}")
    print(f"  NN evals: {stats.total_nn_evals}")
    print(f"  Terminals: {stats.total_terminals}")
    print(f"  Collisions: {stats.total_collisions}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Sims/s: {stats.total_simulations / elapsed:,.0f}")
    print(f"  NN evals/s: {stats.total_nn_evals / elapsed:,.0f}")
    print(f"  Collision%: {stats.collision_fraction * 100:.1f}%")
    if stats.inference_batches:
        histogram = ",".join(f"{batch}:{count}" for batch, count in stats.inference_batch_histogram)
        print(f"  Device batches: {stats.inference_batches}")
        print(f"  Average device batch: {stats.inference_avg_batch_size:.2f}")
        print(f"  Device batch histogram: {histogram}")
        print(f"  Inference backend time: {stats.inference_nn_seconds:.3f}s")
        print(f"  Inference queue wait: {stats.inference_wait_seconds:.3f}s")


if __name__ == "__main__":
    main()
