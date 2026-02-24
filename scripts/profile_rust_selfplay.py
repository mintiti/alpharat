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
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to .pt checkpoint (omit to use a random model)",
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
    args = parser.parse_args()

    width, height = 7, 7

    from alpharat.mcts.config import RustMCTSConfig

    mcts = RustMCTSConfig(
        simulations=args.sims,
        c_puct=0.512,
        force_k=0.103,
        fpu_reduction=0.459,
        batch_size=args.batch_size,
        noise_epsilon=0.25,
        noise_concentration=10.83,
    )

    # Get or create ONNX model
    if args.checkpoint is not None:
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

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "games"
        output_dir.mkdir()

        print(
            f"Running {args.games} games, {args.sims} sims, "
            f"{args.threads} threads, batch={args.batch_size}, "
            f"mux_batch={args.mux_batch}, device={args.device}"
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
            simulations=mcts.simulations,
            batch_size=mcts.batch_size,
            c_puct=mcts.c_puct,
            fpu_reduction=mcts.fpu_reduction,
            force_k=mcts.force_k,
            noise_epsilon=mcts.noise_epsilon,
            noise_concentration=mcts.noise_concentration,
            max_collisions=mcts.max_collisions,
            num_threads=args.threads,
            output_dir=str(output_dir),
            max_games_per_bundle=32,
            onnx_model_path=onnx_path,
            mux_max_batch_size=args.mux_batch,
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


if __name__ == "__main__":
    main()
