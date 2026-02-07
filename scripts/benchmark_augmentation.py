#!/usr/bin/env python
"""Benchmark per-sample vs batch-level augmentation."""

from __future__ import annotations

import statistics
import time

import numpy as np
import torch

from alpharat.nn.augmentation import BatchAugmentation, swap_player_perspective


def create_synthetic_numpy_batch(
    batch_size: int, width: int = 5, height: int = 5
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Create synthetic batch as numpy arrays (simulating shard data)."""
    obs_dim = width * height * 7 + 6

    observations = np.random.randn(batch_size, obs_dim).astype(np.float32)
    policy_p1 = np.random.dirichlet(np.ones(5), batch_size).astype(np.float32)
    policy_p2 = np.random.dirichlet(np.ones(5), batch_size).astype(np.float32)
    p1_value = np.random.randn(batch_size, 1).astype(np.float32)
    p2_value = np.random.randn(batch_size, 1).astype(np.float32)
    action_p1 = np.random.randint(0, 5, (batch_size, 1), dtype=np.int8)
    action_p2 = np.random.randint(0, 5, (batch_size, 1), dtype=np.int8)

    return (
        observations,
        policy_p1,
        policy_p2,
        p1_value,
        p2_value,
        action_p1,
        action_p2,
    )


def create_synthetic_torch_batch(
    batch_size: int, width: int = 5, height: int = 5, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    """Create synthetic batch as torch tensors."""
    obs_dim = width * height * 7 + 6

    return {
        "observation": torch.randn(batch_size, obs_dim, device=device),
        "policy_p1": torch.softmax(torch.randn(batch_size, 5, device=device), dim=-1),
        "policy_p2": torch.softmax(torch.randn(batch_size, 5, device=device), dim=-1),
        "p1_value": torch.randn(batch_size, 1, device=device),
        "p2_value": torch.randn(batch_size, 1, device=device),
        "action_p1": torch.randint(0, 5, (batch_size, 1), device=device, dtype=torch.int8),
        "action_p2": torch.randint(0, 5, (batch_size, 1), device=device, dtype=torch.int8),
    }


def benchmark_per_sample(
    batch_size: int,
    width: int,
    height: int,
    n_warmup: int = 3,
    n_trials: int = 10,
) -> tuple[float, float]:
    """Benchmark per-sample numpy augmentation.

    Returns:
        Tuple of (mean_time_ms, std_time_ms).
    """
    obs, p1, p2, p1_val, p2_val, a1, a2 = create_synthetic_numpy_batch(batch_size, width, height)
    rng = np.random.default_rng(42)

    # Warm-up
    for _ in range(n_warmup):
        for j in range(batch_size):
            if rng.random() < 0.5:
                swap_player_perspective(
                    obs[j],
                    p1[j],
                    p2[j],
                    p1_val[j : j + 1],
                    p2_val[j : j + 1],
                    a1[j : j + 1],
                    a2[j : j + 1],
                    width,
                    height,
                )

    # Timed trials
    times = []
    for _ in range(n_trials):
        # Fresh data each trial
        obs, p1, p2, p1_val, p2_val, a1, a2 = create_synthetic_numpy_batch(
            batch_size, width, height
        )
        rng = np.random.default_rng(42)

        start = time.perf_counter()
        for j in range(batch_size):
            if rng.random() < 0.5:
                swap_player_perspective(
                    obs[j],
                    p1[j],
                    p2[j],
                    p1_val[j : j + 1],
                    p2_val[j : j + 1],
                    a1[j : j + 1],
                    a2[j : j + 1],
                    width,
                    height,
                )
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def benchmark_batch_cpu(
    batch_size: int,
    width: int,
    height: int,
    n_warmup: int = 3,
    n_trials: int = 10,
) -> tuple[float, float]:
    """Benchmark batch-level CPU augmentation.

    Returns:
        Tuple of (mean_time_ms, std_time_ms).
    """
    augment = BatchAugmentation(width, height, p_swap=0.5)

    # Warm-up
    for _ in range(n_warmup):
        batch = create_synthetic_torch_batch(batch_size, width, height, device="cpu")
        augment(batch)

    # Timed trials
    times = []
    for _ in range(n_trials):
        batch = create_synthetic_torch_batch(batch_size, width, height, device="cpu")

        start = time.perf_counter()
        augment(batch)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def get_gpu_device() -> str | None:
    """Get available GPU device name, or None if no GPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def sync_device(device: str) -> None:
    """Synchronize GPU device for accurate timing."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def benchmark_batch_gpu(
    batch_size: int,
    width: int,
    height: int,
    n_warmup: int = 3,
    n_trials: int = 10,
) -> tuple[float, float, str] | None:
    """Benchmark batch-level GPU augmentation.

    Returns:
        Tuple of (mean_time_ms, std_time_ms, device_name), or None if no GPU available.
    """
    device = get_gpu_device()
    if device is None:
        return None

    augment = BatchAugmentation(width, height, p_swap=0.5)

    # Warm-up
    for _ in range(n_warmup):
        batch = create_synthetic_torch_batch(batch_size, width, height, device=device)
        augment(batch)
        sync_device(device)

    # Timed trials
    times = []
    for _ in range(n_trials):
        batch = create_synthetic_torch_batch(batch_size, width, height, device=device)
        sync_device(device)

        start = time.perf_counter()
        augment(batch)
        sync_device(device)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0, device


def main() -> None:
    """Run augmentation benchmarks."""
    width, height = 5, 5
    batch_sizes = [64, 256, 1024]
    n_trials = 20

    gpu_device = get_gpu_device()

    print("=" * 70)
    print("Augmentation Benchmark: Per-Sample (numpy) vs Batch-Level (PyTorch)")
    print("=" * 70)
    print(f"Maze size: {width}x{height}")
    print(f"Trials per configuration: {n_trials}")
    print(f"GPU device: {gpu_device or 'None'}")
    print()

    for batch_size in batch_sizes:
        print(f"Batch size: {batch_size}")
        print("-" * 50)

        # Per-sample numpy
        mean_ps, std_ps = benchmark_per_sample(batch_size, width, height, n_trials=n_trials)
        print(f"  Per-sample (numpy):  {mean_ps:8.3f} ± {std_ps:6.3f} ms")

        # Batch CPU
        mean_cpu, std_cpu = benchmark_batch_cpu(batch_size, width, height, n_trials=n_trials)
        speedup_cpu = mean_ps / mean_cpu if mean_cpu > 0 else float("inf")
        print(f"  Batch (CPU):         {mean_cpu:8.3f} ± {std_cpu:6.3f} ms  ({speedup_cpu:.1f}x)")

        # Batch GPU
        gpu_result = benchmark_batch_gpu(batch_size, width, height, n_trials=n_trials)
        if gpu_result:
            mean_gpu, std_gpu, device = gpu_result
            speedup_gpu = mean_ps / mean_gpu if mean_gpu > 0 else float("inf")
            dev = device.upper()
            print(f"  Batch ({dev}):  {mean_gpu:8.3f} ± {std_gpu:6.3f} ms  ({speedup_gpu:.1f}x)")
        else:
            print("  Batch (GPU):         N/A (no GPU)")

        print()

    print("=" * 70)
    print("Note: Times include only augmentation, not data loading or I/O.")
    print("=" * 70)


if __name__ == "__main__":
    main()
