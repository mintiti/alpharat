from __future__ import annotations

from inspect import signature

from alpharat.data.rust_sampling import RustSamplingMetrics, run_rust_sampling


def _metrics(histogram: tuple[tuple[int, int], ...]) -> RustSamplingMetrics:
    return RustSamplingMetrics(
        total_games=1,
        total_positions=1,
        total_simulations=1,
        elapsed_seconds=1.0,
        p1_wins=0,
        p2_wins=0,
        draws=1,
        total_cheese_collected=0.0,
        total_cheese_available=1,
        min_turns=1,
        max_turns=1,
        total_nn_evals=1,
        total_terminals=0,
        total_collisions=0,
        cache_hits=0,
        cache_misses=0,
        inference_batches=sum(count for _, count in histogram),
        inference_positions=sum(batch * count for batch, count in histogram),
        inference_nn_seconds=0.5,
        inference_wait_seconds=0.5,
        inference_batch_histogram=histogram,
    )


def test_inference_batch_summary_uses_exact_histogram() -> None:
    metrics = _metrics(((8, 2), (16, 5), (32, 3)))

    assert metrics.inference_avg_batch_size == 19.2
    assert metrics.inference_batch_percentile(0.5) == 16
    assert metrics.inference_batch_percentile(0.9) == 32


def test_inference_batch_summary_handles_uninstrumented_backend() -> None:
    metrics = _metrics(())

    assert metrics.inference_avg_batch_size == 0.0
    assert metrics.inference_batch_percentile(0.5) == 0


def test_tensor_rt_pinned_host_io_is_the_python_default() -> None:
    parameter = signature(run_rust_sampling).parameters["tensorrt_pinned_host_io"]

    assert parameter.default is True


def test_sampling_seed_is_optional_by_default() -> None:
    parameter = signature(run_rust_sampling).parameters["seed"]

    assert parameter.default is None
