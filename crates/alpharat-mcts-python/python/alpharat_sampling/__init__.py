"""Python bindings for the Rust self-play sampling pipeline."""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat_engine._core.sampling import (  # type: ignore[import-not-found]
        SelfPlayProgress as SelfPlayProgress,
    )
    from pyrat_engine._core.sampling import (
        SelfPlayStats as SelfPlayStats,
    )
    from pyrat_engine._core.sampling import (
        rust_self_play as rust_self_play,
    )
else:
    import pyrat_engine._core as _impl

    SelfPlayStats = _impl.sampling.SelfPlayStats
    SelfPlayProgress = _impl.sampling.SelfPlayProgress
    rust_self_play = _impl.sampling.rust_self_play

_logger = logging.getLogger(__name__)

_cuda_preloaded = False


def preload_cuda_libs() -> None:
    """Preload ORT CUDA provider libs so ``dlopen`` finds them at runtime.

    Maturin only copies the Python extension ``.so`` into the package directory,
    leaving the ORT provider dylibs in ``target/release/``.  When ORT later
    calls ``dlopen("libonnxruntime_providers_shared.so")``, it won't find them.

    This function manually loads the provider libs (with ``RTLD_GLOBAL``) before
    ORT needs them.  It also ensures the CUDA runtime libs from the ``nvidia-*``
    pip packages are loaded first (PyTorch does this on import, but we handle it
    explicitly in case torch hasn't been imported yet).

    Call this *before* creating an ORT CUDA session (i.e. before ``rust_self_play``
    with ``device="cuda"``).
    """
    global _cuda_preloaded  # noqa: PLW0603
    if _cuda_preloaded:
        return
    _cuda_preloaded = True

    # --- 1. CUDA runtime libs (from nvidia pip packages) -----------------
    # PyTorch preloads these on import, but if torch isn't imported yet we
    # need to find them ourselves.
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        import torch  # noqa: F401 — triggers PyTorch's own CUDA preloading

    # --- 2. ORT provider dylibs (from Cargo build output) ----------------
    # Walk up from this file to find the Cargo workspace root, then look in
    # target/release/.
    _this_dir = Path(__file__).resolve().parent
    _workspace_root = _this_dir.parent.parent.parent.parent  # → project root
    _target_dir = _workspace_root / "target" / "release"

    _provider_libs = [
        "libonnxruntime_providers_shared.so",
        "libonnxruntime_providers_cuda.so",
    ]

    for lib_name in _provider_libs:
        lib_path = _target_dir / lib_name
        if lib_path.exists():
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                _logger.debug("Preloaded %s", lib_path)
            except OSError as e:
                _logger.warning("Failed to preload %s: %s", lib_path, e)
        else:
            _logger.warning(
                "ORT CUDA provider not found at %s — "
                "rebuild with: uv pip install -e crates/alpharat-mcts-python",
                lib_path,
            )


_tensorrt_preloaded = False


def preload_tensorrt_libs() -> None:
    """Preload TensorRT-RTX shared libs so ``dlopen`` finds them at runtime.

    Reads ``TENSORRT_RTX_ROOT`` and loads ``libtensorrt_rtx.so`` and
    ``libtensorrt_onnxparser_rtx.so`` from ``$TENSORRT_RTX_ROOT/lib`` with
    ``RTLD_GLOBAL``.  Also preloads ``libcudart`` from the ``nvidia-cuda-runtime``
    pip package if available.  The Rust ``trtx`` crate then finds these via its
    own ``dlopen`` call.

    Call this *before* creating a TensorRT backend (i.e. before
    ``rust_self_play`` with ``device="tensorrt"``).
    """
    global _tensorrt_preloaded  # noqa: PLW0603
    if _tensorrt_preloaded:
        return
    _tensorrt_preloaded = True

    import os

    # --- 1. CUDA runtime from nvidia pip package ---
    try:
        import nvidia.cuda_runtime as _cr

        cuda_lib_dir = Path(_cr.__path__[0]) / "lib"
        for name in sorted(cuda_lib_dir.glob("libcudart.so*")):
            try:
                ctypes.CDLL(str(name), mode=ctypes.RTLD_GLOBAL)
                _logger.debug("Preloaded %s", name)
                break
            except OSError:
                pass
    except ImportError:
        _logger.debug("nvidia-cuda-runtime not installed, skipping cudart preload")

    # --- 2. TensorRT-RTX libs ---
    trt_root = os.environ.get("TENSORRT_RTX_ROOT")
    if not trt_root:
        _logger.warning("TENSORRT_RTX_ROOT not set — TensorRT-RTX libs may not be found at runtime")
        return

    lib_dir = Path(trt_root) / "lib"
    for lib_name in ["libtensorrt_rtx.so", "libtensorrt_onnxparser_rtx.so"]:
        lib_path = lib_dir / lib_name
        if lib_path.exists():
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                _logger.debug("Preloaded %s", lib_path)
            except OSError as e:
                _logger.warning("Failed to preload %s: %s", lib_path, e)
        else:
            _logger.warning(
                "TensorRT-RTX lib not found at %s — check TENSORRT_RTX_ROOT points to the SDK root",
                lib_path,
            )


__all__ = [
    "preload_cuda_libs",
    "preload_tensorrt_libs",
    "rust_self_play",
    "SelfPlayStats",
    "SelfPlayProgress",
]
