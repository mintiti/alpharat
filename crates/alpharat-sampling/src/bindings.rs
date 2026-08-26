//! PyO3 bindings exposing the Rust self-play pipeline to Python.

use pyo3::prelude::*;
use std::path::Path;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

use alpharat_mcts::{Backend, SearchConfig, SmartUniformBackend};
use pyrat::{GameBuilder, GameState, MazeParams};

#[cfg(any(feature = "onnx", feature = "tensorrt"))]
use crate::backends::mux::MuxStats;
use crate::backends::mux::MuxStatsSnapshot;
use crate::selfplay::{self, SelfPlayConfig, SelfPlayError, SelfPlayStats};

#[cfg(any(feature = "onnx", feature = "tensorrt"))]
use crate::backends::mux::{MuxBackend, MuxConfig};
use crate::CachedBackend;
#[cfg(any(feature = "onnx", feature = "tensorrt"))]
use crate::FlatEncoder;
#[cfg(feature = "onnx")]
use crate::{ExecutionProvider, OnnxBackend};
#[cfg(feature = "tensorrt")]
use crate::{TensorrtBackend, TensorrtConfig};

// ---------------------------------------------------------------------------
// PySelfPlayStats
// ---------------------------------------------------------------------------

/// Python-visible wrapper around SelfPlayStats.
#[pyclass(name = "SelfPlayStats")]
#[derive(Clone)]
pub struct PySelfPlayStats {
    inner: SelfPlayStats,
    inference: Option<MuxStatsSnapshot>,
}

#[pymethods]
impl PySelfPlayStats {
    #[getter]
    fn total_games(&self) -> u32 {
        self.inner.total_games
    }
    #[getter]
    fn total_positions(&self) -> u64 {
        self.inner.total_positions
    }
    #[getter]
    fn total_simulations(&self) -> u64 {
        self.inner.total_simulations
    }
    #[getter]
    fn elapsed_secs(&self) -> f64 {
        self.inner.elapsed_secs
    }
    #[getter]
    fn p1_wins(&self) -> u32 {
        self.inner.p1_wins
    }
    #[getter]
    fn p2_wins(&self) -> u32 {
        self.inner.p2_wins
    }
    #[getter]
    fn draws(&self) -> u32 {
        self.inner.draws
    }
    #[getter]
    fn total_cheese_collected(&self) -> f32 {
        self.inner.total_cheese_collected
    }
    #[getter]
    fn total_cheese_available(&self) -> u32 {
        self.inner.total_cheese_available
    }
    #[getter]
    fn min_turns(&self) -> u32 {
        self.inner.min_turns
    }
    #[getter]
    fn max_turns(&self) -> u32 {
        self.inner.max_turns
    }

    #[getter]
    fn total_nn_evals(&self) -> u64 {
        self.inner.total_nn_evals
    }
    #[getter]
    fn total_terminals(&self) -> u64 {
        self.inner.total_terminals
    }
    #[getter]
    fn total_collisions(&self) -> u64 {
        self.inner.total_collisions
    }

    // Derived metrics
    #[getter]
    fn games_per_second(&self) -> f64 {
        self.inner.games_per_second()
    }
    #[getter]
    fn positions_per_second(&self) -> f64 {
        self.inner.positions_per_second()
    }
    #[getter]
    fn simulations_per_second(&self) -> f64 {
        self.inner.simulations_per_second()
    }
    #[getter]
    fn cheese_utilization(&self) -> f64 {
        self.inner.cheese_utilization()
    }
    #[getter]
    fn avg_turns(&self) -> f64 {
        self.inner.avg_turns()
    }
    #[getter]
    fn draw_rate(&self) -> f64 {
        self.inner.draw_rate()
    }
    #[getter]
    fn nn_evals_per_second(&self) -> f64 {
        self.inner.nn_evals_per_second()
    }
    #[getter]
    fn nn_eval_fraction(&self) -> f64 {
        self.inner.nn_eval_fraction()
    }
    #[getter]
    fn terminal_fraction(&self) -> f64 {
        self.inner.terminal_fraction()
    }
    #[getter]
    fn collision_fraction(&self) -> f64 {
        self.inner.collision_fraction()
    }
    #[getter]
    fn cache_hits(&self) -> u64 {
        self.inner.cache_hits
    }
    #[getter]
    fn cache_misses(&self) -> u64 {
        self.inner.cache_misses
    }
    #[getter]
    fn cache_hit_rate(&self) -> f64 {
        self.inner.cache_hit_rate()
    }

    #[getter]
    fn inference_batches(&self) -> u64 {
        self.inference
            .as_ref()
            .map_or(0, |stats| stats.total_batches)
    }

    #[getter]
    fn inference_positions(&self) -> u64 {
        self.inference
            .as_ref()
            .map_or(0, |stats| stats.total_positions)
    }

    #[getter]
    fn inference_avg_batch_size(&self) -> f64 {
        self.inference.as_ref().map_or(0.0, |stats| {
            if stats.total_batches == 0 {
                0.0
            } else {
                stats.total_positions as f64 / stats.total_batches as f64
            }
        })
    }

    #[getter]
    fn inference_nn_seconds(&self) -> f64 {
        self.inference
            .as_ref()
            .map_or(0.0, |stats| stats.nn_time_ns as f64 / 1e9)
    }

    #[getter]
    fn inference_wait_seconds(&self) -> f64 {
        self.inference
            .as_ref()
            .map_or(0.0, |stats| stats.wait_time_ns as f64 / 1e9)
    }

    /// Exact `(device_batch_size, call_count)` distribution from the eager mux.
    #[getter]
    fn inference_batch_histogram(&self) -> Vec<(usize, u64)> {
        self.inference
            .as_ref()
            .map_or_else(Vec::new, |stats| stats.batch_histogram.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "SelfPlayStats(games={}, positions={}, sims={}, elapsed={:.2}s, sims/s={:.0})",
            self.inner.total_games,
            self.inner.total_positions,
            self.inner.total_simulations,
            self.inner.elapsed_secs,
            self.inner.simulations_per_second(),
        )
    }
}

// ---------------------------------------------------------------------------
// PySelfPlayProgress
// ---------------------------------------------------------------------------

/// Atomic progress counters, shareable between threads.
///
/// Create in Python, pass to `rust_self_play()`, poll from another thread.
#[pyclass(name = "SelfPlayProgress")]
#[derive(Clone)]
pub struct PySelfPlayProgress {
    inner: Arc<selfplay::SelfPlayProgress>,
}

#[pymethods]
impl PySelfPlayProgress {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(selfplay::SelfPlayProgress::new()),
        }
    }

    #[getter]
    fn games_completed(&self) -> u32 {
        self.inner.games_completed.load(Relaxed)
    }

    #[getter]
    fn positions_completed(&self) -> u64 {
        self.inner.positions_completed.load(Relaxed)
    }

    #[getter]
    fn simulations_completed(&self) -> u64 {
        self.inner.simulations_completed.load(Relaxed)
    }

    #[getter]
    fn nn_evals_completed(&self) -> u64 {
        self.inner.nn_evals_completed.load(Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Backend construction helpers
// ---------------------------------------------------------------------------

/// Wrap a backend in MuxBackend if multi-threaded, otherwise return as-is.
#[cfg(any(feature = "onnx", feature = "tensorrt"))]
fn maybe_mux<B: Backend + 'static>(
    backend: B,
    max_batch_size: usize,
    num_threads: u32,
    enabled: bool,
) -> (Box<dyn Backend>, Option<Arc<MuxStats>>) {
    if enabled && num_threads > 1 {
        let mux = MuxBackend::new(backend, MuxConfig { max_batch_size });
        let stats = Arc::clone(mux.stats());
        (Box::new(mux), Some(stats))
    } else {
        (Box::new(backend), None)
    }
}

#[cfg(feature = "tensorrt")]
fn create_tensorrt_backend(
    model_path: &str,
    width: u8,
    height: u8,
    max_batch_size: usize,
    opt_batch_size: Option<usize>,
    output_dir: &str,
    num_threads: u32,
    execution_contexts: usize,
    cuda_graphs: bool,
    use_inference_mux: bool,
) -> Result<(Box<dyn Backend>, Option<Arc<MuxStats>>), SelfPlayError> {
    if cuda_graphs && execution_contexts > 1 && !use_inference_mux {
        return Err(SelfPlayError::Backend(alpharat_mcts::BackendError::msg(
            concat!(
                "TensorRT CUDA graphs with multiple direct self-play contexts are unsupported: ",
                "concurrent capture across variable batch shapes exits inside TensorRT-RTX 1.3. ",
                "Use one context, keep the inference mux enabled, or disable CUDA graphs."
            ),
        )));
    }
    let encoder = FlatEncoder::new(width, height);
    let cache_dir = Path::new(output_dir).parent().map(|p| p.join(".trt_cache"));
    let config = TensorrtConfig {
        opt_batch: opt_batch_size.unwrap_or(max_batch_size),
        max_batch: max_batch_size,
        cache_dir,
        execution_contexts,
        cuda_graphs,
    };
    let trt = TensorrtBackend::new(model_path, encoder, config).map_err(SelfPlayError::Backend)?;
    Ok(maybe_mux(
        trt,
        max_batch_size,
        num_threads,
        use_inference_mux,
    ))
}

#[cfg(feature = "onnx")]
fn create_onnx_backend(
    model_path: &str,
    device: &str,
    width: u8,
    height: u8,
    max_batch_size: usize,
    num_threads: u32,
    use_inference_mux: bool,
) -> Result<(Box<dyn Backend>, Option<Arc<MuxStats>>), SelfPlayError> {
    let encoder = FlatEncoder::new(width, height);
    let provider = ExecutionProvider::try_from(device)
        .map_err(|e| SelfPlayError::Backend(alpharat_mcts::BackendError::msg(e)))?;
    let onnx = OnnxBackend::with_provider(model_path, encoder, provider)
        .map_err(SelfPlayError::Backend)?;
    Ok(maybe_mux(
        onnx,
        max_batch_size,
        num_threads,
        use_inference_mux,
    ))
}

// ---------------------------------------------------------------------------
// rust_self_play — main entry point
// ---------------------------------------------------------------------------

/// Run Rust self-play pipeline end-to-end.
///
/// Creates games, optionally loads an ONNX model, runs multi-threaded
/// self-play, and writes NPZ bundles to `output_dir`. Returns stats.
///
/// GIL is released during the entire self-play computation.
#[pyfunction]
#[pyo3(signature = (
    *,
    width,
    height,
    cheese_count,
    max_turns,
    num_games,
    cheese_symmetric = true,
    maze_type = "open",
    positions = "corners",
    wall_density = 0.7,
    mud_density = 0.1,
    maze_symmetric = true,
    simulations,
    batch_size = 8,
    c_puct = 1.5,
    fpu_reduction = 0.2,
    force_k = 2.0,
    noise_epsilon = 0.0,
    noise_concentration = 10.83,
    collision_limit_min = 1,
    collision_limit_max = 256,
    collision_scaling_start = 800,
    collision_scaling_end = 50000,
    collision_scaling_power = 1.0,
    num_threads = 4,
    output_dir,
    max_games_per_bundle = 32,
    onnx_model_path = None,
    device = "auto",
    mux_max_batch_size = 256,
    tensorrt_opt_batch = None,
    tensorrt_execution_contexts = 1,
    tensorrt_cuda_graphs = false,
    use_inference_mux = true,
    cache_size = 0,
    progress = None,
))]
#[allow(clippy::too_many_arguments)]
fn rust_self_play(
    py: Python<'_>,
    // Game
    width: u8,
    height: u8,
    cheese_count: u16,
    max_turns: u16,
    num_games: usize,
    cheese_symmetric: bool,
    maze_type: &str,
    positions: &str,
    wall_density: f32,
    mud_density: f32,
    maze_symmetric: bool,
    // Search
    simulations: u32,
    batch_size: u32,
    c_puct: f32,
    fpu_reduction: f32,
    force_k: f32,
    noise_epsilon: f32,
    noise_concentration: f32,
    collision_limit_min: u32,
    collision_limit_max: u32,
    collision_scaling_start: u32,
    collision_scaling_end: u32,
    collision_scaling_power: f32,
    // Sampling
    num_threads: u32,
    output_dir: &str,
    max_games_per_bundle: usize,
    // NN (optional)
    onnx_model_path: Option<&str>,
    device: &str,
    mux_max_batch_size: usize,
    tensorrt_opt_batch: Option<usize>,
    tensorrt_execution_contexts: usize,
    tensorrt_cuda_graphs: bool,
    use_inference_mux: bool,
    // Cache (optional, 0 = disabled)
    cache_size: usize,
    // Progress (optional)
    progress: Option<PySelfPlayProgress>,
) -> PyResult<PySelfPlayStats> {
    // Build game states
    let games = make_games(
        num_games,
        width,
        height,
        cheese_count,
        max_turns,
        cheese_symmetric,
        maze_type,
        positions,
        wall_density,
        mud_density,
        maze_symmetric,
    );

    let search_config = SearchConfig {
        c_puct,
        fpu_reduction,
        force_k,
        noise_epsilon,
        noise_concentration,
        collision_limit_min,
        collision_limit_max,
        collision_scaling_start,
        collision_scaling_end,
        collision_scaling_power,
    };

    let selfplay_config = SelfPlayConfig {
        n_sims: simulations,
        batch_size,
        num_threads,
    };

    let output_path = Path::new(output_dir);

    // Get a reference to the inner progress if provided
    let progress_arc = progress.as_ref().map(|p| Arc::clone(&p.inner));

    // Build backend and run (GIL released)
    let result = py.allow_threads(move || {
        let progress_ref = progress_arc.as_deref();

        // Choose backend based on model path + device, then run self-play.
        // Cache only wraps NN backends — SmartUniform is trivial arithmetic
        // (cheaper than hashing), so caching it is pure overhead.
        match onnx_model_path {
            #[cfg(any(feature = "onnx", feature = "tensorrt"))]
            Some(model_path) => {
                let (backend, mux_stats): (Box<dyn Backend>, Option<Arc<MuxStats>>) = match device {
                    #[cfg(feature = "tensorrt")]
                    "tensorrt" => create_tensorrt_backend(
                        model_path,
                        width,
                        height,
                        mux_max_batch_size,
                        tensorrt_opt_batch,
                        output_dir,
                        num_threads,
                        tensorrt_execution_contexts,
                        tensorrt_cuda_graphs,
                        use_inference_mux,
                    )?,
                    #[cfg(not(feature = "tensorrt"))]
                    "tensorrt" => {
                        return Err(SelfPlayError::Backend(alpharat_mcts::BackendError::msg(
                            "TensorRT support not compiled (build with --features tensorrt)",
                        )));
                    }
                    #[cfg(feature = "onnx")]
                    _ => create_onnx_backend(
                        model_path,
                        device,
                        width,
                        height,
                        mux_max_batch_size,
                        num_threads,
                        use_inference_mux,
                    )?,
                    #[cfg(not(feature = "onnx"))]
                    _ => {
                        return Err(SelfPlayError::Backend(alpharat_mcts::BackendError::msg(
                            "ONNX support not compiled (build with --features onnx)",
                        )));
                    }
                };

                let disk_result = if cache_size > 0 {
                    let cached = CachedBackend::new(backend, cache_size);
                    let mut disk_result = selfplay::run_self_play_to_disk(
                        &games,
                        &cached,
                        &search_config,
                        &selfplay_config,
                        output_path,
                        max_games_per_bundle,
                        progress_ref,
                    )?;
                    disk_result.stats.cache_hits = cached.stats.hits.load(Relaxed);
                    disk_result.stats.cache_misses = cached.stats.misses.load(Relaxed);
                    disk_result
                } else {
                    selfplay::run_self_play_to_disk(
                        &games,
                        backend.as_ref(),
                        &search_config,
                        &selfplay_config,
                        output_path,
                        max_games_per_bundle,
                        progress_ref,
                    )?
                };
                let inference = mux_stats.as_ref().map(|stats| stats.snapshot());
                Ok((disk_result, inference))
            }
            #[cfg(not(any(feature = "onnx", feature = "tensorrt")))]
            Some(_) => Err(SelfPlayError::Backend(alpharat_mcts::BackendError::msg(
                "No NN backend compiled (build with --features onnx or tensorrt)",
            ))),
            None => selfplay::run_self_play_to_disk(
                &games,
                &SmartUniformBackend,
                &search_config,
                &selfplay_config,
                output_path,
                max_games_per_bundle,
                progress_ref,
            )
            .map(|disk_result| (disk_result, None)),
        }
    });

    match result {
        Ok((disk_result, inference)) => Ok(PySelfPlayStats {
            inner: disk_result.stats,
            inference,
        }),
        Err(SelfPlayError::Backend(e)) => {
            Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }
        Err(SelfPlayError::Io(e)) => Err(pyo3::exceptions::PyIOError::new_err(e.to_string())),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_games(
    n: usize,
    width: u8,
    height: u8,
    cheese_count: u16,
    max_turns: u16,
    cheese_symmetric: bool,
    maze_type: &str,
    positions: &str,
    wall_density: f32,
    mud_density: f32,
    maze_symmetric: bool,
) -> Vec<GameState> {
    let base = GameBuilder::new(width, height).with_max_turns(max_turns);

    // Maze axis (NeedsMaze → NeedsPlayers)
    let with_maze = match maze_type {
        "open" => base.with_open_maze(),
        "classic" => base.with_classic_maze(),
        "random" => {
            let mud_range: u8 = if mud_density > 0.0 { 3 } else { 2 };
            base.with_random_maze(MazeParams {
                wall_density,
                mud_density,
                mud_range,
                connected: true,
                symmetric: maze_symmetric,
            })
        }
        _ => panic!("unknown maze_type: {maze_type}"),
    };

    // Positions axis (NeedsPlayers → NeedsCheese)
    let with_positions = match positions {
        "corners" => with_maze.with_corner_positions(),
        "random" => with_maze.with_random_positions(),
        _ => panic!("unknown positions: {positions}"),
    };

    let config = with_positions
        .with_random_cheese(cheese_count, cheese_symmetric)
        .build();

    (0..n)
        .map(|_| config.create(None).expect("game creation failed"))
        .collect()
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register_sampling_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySelfPlayStats>()?;
    m.add_class::<PySelfPlayProgress>()?;
    m.add_function(wrap_pyfunction!(rust_self_play, m)?)?;
    Ok(())
}
