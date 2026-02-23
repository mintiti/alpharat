#!/usr/bin/env python3
"""Export a PyTorch checkpoint to ONNX format for Rust inference.

Usage:
    alpharat-export-onnx experiments/runs/scalar_baseline_7x7_iter0/checkpoints/best_model.pt
    alpharat-export-onnx path/to/model.pt --output /tmp/model.onnx
    alpharat-export-onnx path/to/model.pt --opset 17 --verify

The ONNX model includes softmax on policy outputs (applied in predict(),
not forward()), so the Rust backend gets probabilities directly.

Output tensor names:
    policy_p1     [N, 5]  — P1 action probabilities
    policy_p2     [N, 5]  — P2 action probabilities
    pred_value_p1 [N]     — P1 scalar value estimate
    pred_value_p2 [N]     — P2 scalar value estimate
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn

from alpharat.config.checkpoint import load_model_from_checkpoint


class OnnxWrapper(nn.Module):
    """Wraps a model to route through predict() for ONNX export.

    torch.onnx.export traces forward(), but we need predict() because
    that's where softmax is applied. This wrapper makes forward() call
    predict() and return a flat tuple of the 4 tensors we need.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        # TrainableModel protocol has predict(), but nn.Module doesn't
        self._predict: Any = model.predict

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        result: dict[str, Any] = self._predict(x)
        return (
            result["policy_p1"],
            result["policy_p2"],
            result["pred_value_p1"],
            result["pred_value_p2"],
        )


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    opset_version: int = 17,
    verify: bool = False,
) -> Path:
    """Export checkpoint to ONNX.

    Args:
        checkpoint_path: Path to .pt checkpoint.
        output_path: Where to save .onnx file. Defaults to same dir as checkpoint.
        opset_version: ONNX opset version.
        verify: If True, run a forward pass through both PyTorch and ONNX and compare.

    Returns:
        Path to the exported .onnx file.
    """
    checkpoint_path = Path(checkpoint_path)

    model, builder, width, height = load_model_from_checkpoint(
        checkpoint_path, device="cpu", compile_model=False
    )

    output_path = checkpoint_path.with_suffix(".onnx") if output_path is None else Path(output_path)

    wrapper = OnnxWrapper(model)
    wrapper.eval()

    obs_dim: int = builder.obs_shape[0]  # type: ignore[attr-defined]
    dummy_input = torch.randn(1, obs_dim)

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(output_path),
        input_names=["observation"],
        output_names=["policy_p1", "policy_p2", "pred_value_p1", "pred_value_p2"],
        dynamic_axes={
            "observation": {0: "batch"},
            "policy_p1": {0: "batch"},
            "policy_p2": {0: "batch"},
            "pred_value_p1": {0: "batch"},
            "pred_value_p2": {0: "batch"},
        },
        opset_version=opset_version,
        dynamo=False,
    )

    print(f"Exported to {output_path}")
    print(f"  Input: observation [{obs_dim}] (width={width}, height={height})")
    print(f"  Opset: {opset_version}")

    if verify:
        _verify_onnx(wrapper, output_path, dummy_input)

    return output_path


def _verify_onnx(wrapper: nn.Module, onnx_path: Path, dummy_input: torch.Tensor) -> None:
    """Compare PyTorch and ONNX outputs on the same input."""
    import numpy as np

    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError:
        print("  Skipping verification: onnxruntime not installed (pip install onnxruntime)")
        return

    # PyTorch reference
    with torch.inference_mode():
        pt_outputs = wrapper(dummy_input)

    # ONNX inference
    session = ort.InferenceSession(str(onnx_path))
    ort_outputs = session.run(None, {"observation": dummy_input.numpy()})

    names = ["policy_p1", "policy_p2", "pred_value_p1", "pred_value_p2"]
    all_close = True
    for name, pt_out, ort_out in zip(names, pt_outputs, ort_outputs, strict=True):
        diff = float(np.max(np.abs(pt_out.numpy() - ort_out)))
        status = "ok" if diff < 1e-5 else "MISMATCH"
        if diff >= 1e-5:
            all_close = False
        print(f"  Verify {name}: max_diff={diff:.2e} [{status}]")

    if all_close:
        print("  Verification passed.")
    else:
        print("  WARNING: Outputs differ beyond tolerance.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch checkpoint to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX output matches PyTorch")
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.output, args.opset, args.verify)


if __name__ == "__main__":
    main()
