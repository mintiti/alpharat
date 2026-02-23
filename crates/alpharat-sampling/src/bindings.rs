//! PyO3 bindings exposing the Rust self-play pipeline to Python.

use pyo3::prelude::*;
use std::path::Path;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

use alpharat_mcts::{Backend, SearchConfig, SmartUniformBackend};
use pyrat::{CheeseConfig, GameState, MazeConfig};

use crate::selfplay::{self, SelfPlayConfig, SelfPlayError, SelfPlayStats};

#[cfg(any(feature = "onnx", feature = "tensorrt"))]
use crate::backends::mux::{MuxBackend, MuxConfig};
#[cfg(any(feature = "onnx", feature = "tensorrt"))]
use crate::FlatEncoder;
#[cfg(feature = "onnx")]
use crate::OnnxBackend;
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
) -> Box<dyn Backend> {
    if num_threads > 1 {
        Box::new(MuxBackend::new(backend, MuxConfig { max_batch_size }))
    } else {
        Box::new(backend)
    }
}

#[cfg(feature = "tensorrt")]
fn create_tensorrt_backend(
    model_path: &str,
    width: u8,
    height: u8,
    max_batch_size: usize,
    output_dir: &str,
    num_threads: u32,
) -> Result<Box<dyn Backend>, SelfPlayError> {
    let encoder = FlatEncoder::new(width, height);
    let cache_dir = Path::new(output_dir).parent().map(|p| p.join(".trt_cache"));
    let config = TensorrtConfig {
        max_batch: max_batch_size,
        cache_dir,
    };
    let trt =
        TensorrtBackend::new(model_path, encoder, config).map_err(SelfPlayError::Backend)?;
    Ok(maybe_mux(trt, max_batch_size, num_threads))
}

#[cfg(feature = "onnx")]
fn create_onnx_backend(
    model_path: &str,
    device: &str,
    width: u8,
    height: u8,
    max_batch_size: usize,
    num_threads: u32,
) -> Result<Box<dyn Backend>, SelfPlayError> {
    let encoder = FlatEncoder::new(width, height);
    let onnx = match device {
        "cpu" => OnnxBackend::new(model_path, encoder),
        #[cfg(feature = "onnx-coreml")]
        "coreml" | "mps" => OnnxBackend::with_coreml(model_path, encoder),
        #[cfg(feature = "onnx-cuda")]
        "cuda" => OnnxBackend::with_cuda(model_path, encoder),
        "auto" => {
            #[cfg(feature = "onnx-coreml")]
            {
                OnnxBackend::with_coreml(model_path, encoder)
            }
            #[cfg(all(feature = "onnx-cuda", not(feature = "onnx-coreml")))]
            {
                OnnxBackend::with_cuda(model_path, encoder)
            }
            #[cfg(all(not(feature = "onnx-coreml"), not(feature = "onnx-cuda")))]
            {
                OnnxBackend::new(model_path, encoder)
            }
        }
        other => {
            return Err(SelfPlayError::Backend(
                alpharat_mcts::BackendError::msg(format!("unknown device: {other}")),
            ))
        }
    }
    .map_err(SelfPlayError::Backend)?;
    Ok(maybe_mux(onnx, max_batch_size, num_threads))
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
    symmetric = true,
    wall_density = None,
    mud_density = None,
    simulations,
    batch_size = 8,
    c_puct = 1.5,
    fpu_reduction = 0.2,
    force_k = 2.0,
    noise_epsilon = 0.0,
    noise_concentration = 10.83,
    max_collisions = 0,
    num_threads = 4,
    output_dir,
    max_games_per_bundle = 32,
    onnx_model_path = None,
    device = "auto",
    mux_max_batch_size = 256,
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
    symmetric: bool,
    wall_density: Option<f32>,
    mud_density: Option<f32>,
    // Search
    simulations: u32,
    batch_size: u32,
    c_puct: f32,
    fpu_reduction: f32,
    force_k: f32,
    noise_epsilon: f32,
    noise_concentration: f32,
    max_collisions: u32,
    // Sampling
    num_threads: u32,
    output_dir: &str,
    max_games_per_bundle: usize,
    // NN (optional)
    onnx_model_path: Option<&str>,
    device: &str,
    mux_max_batch_size: usize,
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
        symmetric,
        wall_density.unwrap_or(0.0),
        mud_density.unwrap_or(0.0),
    );

    let search_config = SearchConfig {
        c_puct,
        fpu_reduction,
        force_k,
        noise_epsilon,
        noise_concentration,
        max_collisions,
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

        // Choose backend based on model path + device
        let backend: Box<dyn Backend> = match onnx_model_path {
            #[cfg(any(feature = "onnx", feature = "tensorrt"))]
            Some(model_path) => match device {
                #[cfg(feature = "tensorrt")]
                "tensorrt" => create_tensorrt_backend(
                    model_path,
                    width,
                    height,
                    mux_max_batch_size,
                    output_dir,
                    num_threads,
                )?,
                #[cfg(not(feature = "tensorrt"))]
                "tensorrt" => {
                    return Err(SelfPlayError::Backend(
                        alpharat_mcts::BackendError::msg(
                            "TensorRT support not compiled (build with --features tensorrt)",
                        ),
                    ));
                }
                #[cfg(feature = "onnx")]
                _ => create_onnx_backend(
                    model_path,
                    device,
                    width,
                    height,
                    mux_max_batch_size,
                    num_threads,
                )?,
                #[cfg(not(feature = "onnx"))]
                _ => {
                    return Err(SelfPlayError::Backend(
                        alpharat_mcts::BackendError::msg(
                            "ONNX support not compiled (build with --features onnx)",
                        ),
                    ));
                }
            },
            #[cfg(not(any(feature = "onnx", feature = "tensorrt")))]
            Some(_) => {
                return Err(SelfPlayError::Backend(
                    alpharat_mcts::BackendError::msg(
                        "No NN backend compiled (build with --features onnx or tensorrt)",
                    ),
                ));
            }
            None => Box::new(SmartUniformBackend),
        };

        selfplay::run_self_play_to_disk(
            &games,
            backend.as_ref(),
            &search_config,
            &selfplay_config,
            output_path,
            max_games_per_bundle,
            progress_ref,
        )
    });

    match result {
        Ok(disk_result) => Ok(PySelfPlayStats {
            inner: disk_result.stats,
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
    symmetric: bool,
    wall_density: f32,
    mud_density: f32,
) -> Vec<GameState> {
    let mud_range: u8 = if mud_density > 0.0 { 3 } else { 0 };
    (0..n)
        .map(|_| {
            let maze_config = MazeConfig {
                width,
                height,
                target_density: wall_density,
                connected: true,
                symmetry: symmetric,
                mud_density,
                mud_range,
                seed: None,
            };
            let cheese_config = CheeseConfig {
                count: cheese_count,
                symmetry: symmetric,
            };
            let mut game = GameState::new_random(width, height, maze_config, cheese_config);
            game.max_turns = max_turns;
            game
        })
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
