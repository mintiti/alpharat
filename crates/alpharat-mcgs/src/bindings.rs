//! Python bindings for the Rust MCGS search.
//!
//! Gated behind the `python` feature. Exposes `register_mcgs_module()` for
//! the combined extension crate to call — no `#[pymodule]` here.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::search::{run_search, SearchConfig, SearchResult};
use crate::tree::MCGSTree;
use crate::{Backend, BackendError, EvalResult, SmartUniformBackend};

use pyrat::{GameBuilder, GameState, PyRat};

// ---------------------------------------------------------------------------
// PyMCGSSearchResult
// ---------------------------------------------------------------------------

/// MCGS search result exposed to Python.
///
/// Policies are numpy arrays in 5-action space (UP, RIGHT, DOWN, LEFT, STAY).
/// Blocked actions have probability 0.
#[pyclass(name = "SearchResult")]
pub struct PyMCGSSearchResult {
    inner: SearchResult,
}

#[pymethods]
impl PyMCGSSearchResult {
    #[getter]
    fn policy_p1<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.policy_p1)
    }

    #[getter]
    fn policy_p2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.policy_p2)
    }

    #[getter]
    fn value_p1(&self) -> f32 {
        self.inner.value_p1
    }

    #[getter]
    fn value_p2(&self) -> f32 {
        self.inner.value_p2
    }

    #[getter]
    fn visit_counts_p1<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.visit_counts_p1)
    }

    #[getter]
    fn visit_counts_p2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.visit_counts_p2)
    }

    #[getter]
    fn prior_p1<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.prior_p1)
    }

    #[getter]
    fn prior_p2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.prior_p2)
    }

    #[getter]
    fn q_values_p1<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.q_values_p1)
    }

    #[getter]
    fn q_values_p2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.q_values_p2)
    }

    #[getter]
    fn total_visits(&self) -> u32 {
        self.inner.total_visits
    }

    #[getter]
    fn nn_evals(&self) -> u32 {
        self.inner.nn_evals
    }

    #[getter]
    fn terminals(&self) -> u32 {
        self.inner.terminals
    }

    #[getter]
    fn collisions(&self) -> u32 {
        self.inner.collisions
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchResult(value_p1={:.4}, value_p2={:.4}, total_visits={})",
            self.inner.value_p1, self.inner.value_p2, self.inner.total_visits
        )
    }
}

// ---------------------------------------------------------------------------
// PyCallbackBackend — calls Python predict_fn for NN evaluation
// ---------------------------------------------------------------------------

/// Backend that delegates evaluation to a Python callable.
///
/// Same interface as the MCTS PyCallbackBackend — the Backend trait is shared.
struct PyCallbackBackend {
    predict_fn: PyObject,
}

impl Backend for PyCallbackBackend {
    fn evaluate(&self, game: &GameState) -> Result<EvalResult, BackendError> {
        Ok(self.evaluate_batch(&[game])?.into_iter().next().unwrap())
    }

    fn evaluate_batch(&self, games: &[&GameState]) -> Result<Vec<EvalResult>, BackendError> {
        Python::with_gil(|py| {
            let py_games: Vec<Py<PyRat>> = games
                .iter()
                .map(|gs| {
                    let game = (*gs).clone();
                    let config = GameBuilder::new(game.width, game.height)
                        .with_max_turns(game.max_turns)
                        .with_open_maze()
                        .with_corner_positions()
                        .with_custom_cheese(game.cheese.get_all_cheese_positions())
                        .build();
                    let pyrat = PyRat::from_game_state(game, config);
                    Py::new(py, pyrat).expect("failed to wrap GameState as PyRat")
                })
                .collect();

            let result = self
                .predict_fn
                .call1(py, (py_games,))
                .map_err(|e| BackendError::msg(format!("predict_fn raised an exception: {e}")))?;

            Ok(parse_eval_results(result.bind(py), games.len()))
        })
    }
}

/// Parse the 4-tuple of numpy arrays returned by predict_fn into EvalResults.
fn parse_eval_results(result: &Bound<'_, PyAny>, n: usize) -> Vec<EvalResult> {
    let tuple = result
        .downcast::<PyTuple>()
        .expect("predict_fn must return a tuple");
    assert!(
        tuple.len() == 4,
        "predict_fn must return (policy_p1, policy_p2, value_p1, value_p2)"
    );

    let pp1: PyReadonlyArray2<f32> = tuple
        .get_item(0)
        .unwrap()
        .extract()
        .expect("policy_p1: expected float32 array [N, 5]");
    let pp2: PyReadonlyArray2<f32> = tuple
        .get_item(1)
        .unwrap()
        .extract()
        .expect("policy_p2: expected float32 array [N, 5]");
    let vp1: PyReadonlyArray1<f32> = tuple
        .get_item(2)
        .unwrap()
        .extract()
        .expect("value_p1: expected float32 array [N]");
    let vp2: PyReadonlyArray1<f32> = tuple
        .get_item(3)
        .unwrap()
        .extract()
        .expect("value_p2: expected float32 array [N]");

    let pp1 = pp1.as_array();
    let pp2 = pp2.as_array();
    let vp1 = vp1.as_array();
    let vp2 = vp2.as_array();

    (0..n)
        .map(|i| {
            let mut policy_p1 = [0.0f32; 5];
            let mut policy_p2 = [0.0f32; 5];
            for j in 0..5 {
                policy_p1[j] = pp1[[i, j]];
                policy_p2[j] = pp2[[i, j]];
            }
            EvalResult {
                policy_p1,
                policy_p2,
                value_p1: vp1[i],
                value_p2: vp2[i],
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// rust_mcgs_search
// ---------------------------------------------------------------------------

/// Run MCGS search on a PyRat game state.
///
/// Returns a `SearchResult` with policies and values for both players.
/// When `predict_fn` is None, uses smart uniform priors (no neural network).
///
/// With `predict_fn`, the GIL is held during search (Python callbacks need it).
/// Without `predict_fn`, the GIL is released for pure Rust computation.
#[pyfunction]
#[pyo3(signature = (game, *, predict_fn=None, simulations=100, batch_size=8, c_puct=1.5, fpu_reduction=0.2, force_k=2.0, noise_epsilon=0.0, noise_concentration=10.83, collision_limit_min=1, collision_limit_max=256, collision_scaling_start=800, collision_scaling_end=50000, collision_scaling_power=1.0, seed=None))]
fn rust_mcgs_search(
    py: Python<'_>,
    game: PyRef<'_, PyRat>,
    predict_fn: Option<PyObject>,
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
    seed: Option<u64>,
) -> PyResult<PyMCGSSearchResult> {
    let game_state = game.game_state().clone();

    let config = SearchConfig {
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
    let mut rng = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_entropy(),
    };

    let result = match predict_fn {
        Some(pf) => {
            let backend = PyCallbackBackend { predict_fn: pf };
            let mut tree = MCGSTree::new(&game_state);
            run_search(
                &mut tree,
                &game_state,
                &backend,
                &config,
                simulations,
                batch_size,
                &mut rng,
            )
        }
        None => {
            py.allow_threads(|| {
                let mut tree = MCGSTree::new(&game_state);
                run_search(
                    &mut tree,
                    &game_state,
                    &SmartUniformBackend,
                    &config,
                    simulations,
                    batch_size,
                    &mut rng,
                )
            })
        }
    };

    match result {
        Ok(r) => Ok(PyMCGSSearchResult { inner: r }),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register MCGS types and functions on the given module.
///
/// Called by the combined extension crate — not a standalone pymodule.
pub fn register_mcgs_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_mcgs_search, m)?)?;
    m.add_class::<PyMCGSSearchResult>()?;
    Ok(())
}
