//! Python bindings for the Rust MCTS search.
//!
//! Gated behind the `python` feature. Exposes `register_mcts_module()` for
//! the combined extension crate to call — no `#[pymodule]` here.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::backend::{Backend, EvalResult, SmartUniformBackend};
use crate::search::{run_search, SearchConfig, SearchResult};
use crate::tree::MCTSTree;

use pyrat::{GameState, PyRat};

// ---------------------------------------------------------------------------
// PySearchResult
// ---------------------------------------------------------------------------

/// MCTS search result exposed to Python.
///
/// Policies are numpy arrays in 5-action space (UP, RIGHT, DOWN, LEFT, STAY).
/// Blocked actions have probability 0.
#[pyclass(name = "SearchResult")]
pub struct PySearchResult {
    inner: SearchResult,
}

#[pymethods]
impl PySearchResult {
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
    fn total_visits(&self) -> u32 {
        self.inner.total_visits
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
/// The callable signature:
/// ```python
/// def predict_fn(games: list[PyRat]) -> tuple[
///     np.ndarray,  # policy_p1 [N, 5] float32
///     np.ndarray,  # policy_p2 [N, 5] float32
///     np.ndarray,  # value_p1  [N]    float32
///     np.ndarray,  # value_p2  [N]    float32
/// ]: ...
/// ```
///
/// One GIL acquisition per batch. Within-tree batching amortizes the
/// Python round-trip cost.
struct PyCallbackBackend {
    predict_fn: PyObject,
}

// Py<PyAny> is Send + Sync, so this is auto-derived.
// The GIL is acquired inside evaluate_batch when needed.

impl Backend for PyCallbackBackend {
    fn evaluate(&self, game: &GameState) -> EvalResult {
        self.evaluate_batch(&[game]).into_iter().next().unwrap()
    }

    fn evaluate_batch(&self, games: &[&GameState]) -> Vec<EvalResult> {
        Python::with_gil(|py| {
            // Wrap each GameState as a PyRat Python object.
            let py_games: Vec<Py<PyRat>> = games
                .iter()
                .map(|gs| {
                    let pyrat = PyRat::from_game_state((*gs).clone(), true);
                    Py::new(py, pyrat).expect("failed to wrap GameState as PyRat")
                })
                .collect();

            // Call predict_fn(list[PyRat]) -> 4-tuple of numpy arrays
            let result = self
                .predict_fn
                .call1(py, (py_games,))
                .expect("predict_fn raised an exception");

            parse_eval_results(result.bind(py), games.len())
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
// rust_mcts_search
// ---------------------------------------------------------------------------

/// Run MCTS search on a PyRat game state.
///
/// Returns a `SearchResult` with policies and values for both players.
/// When `predict_fn` is None, uses smart uniform priors (no neural network).
///
/// With `predict_fn`, the GIL is held during search (Python callbacks need it).
/// Without `predict_fn`, the GIL is released for pure Rust computation.
#[pyfunction]
#[pyo3(signature = (game, *, predict_fn=None, simulations=100, batch_size=8, c_puct=1.5, fpu_reduction=0.2, force_k=2.0, seed=None))]
fn rust_mcts_search(
    py: Python<'_>,
    game: PyRef<'_, PyRat>,
    predict_fn: Option<PyObject>,
    simulations: u32,
    batch_size: u32,
    c_puct: f32,
    fpu_reduction: f32,
    force_k: f32,
    seed: Option<u64>,
) -> PyResult<PySearchResult> {
    let game_state = game.game_state().clone();

    let config = SearchConfig {
        c_puct,
        fpu_reduction,
        force_k,
    };
    let mut rng = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_entropy(),
    };

    let result = match predict_fn {
        Some(pf) => {
            // GIL held — PyCallbackBackend re-acquires it internally via
            // Python::with_gil, which is a no-op when already held.
            let backend = PyCallbackBackend { predict_fn: pf };
            let mut tree = MCTSTree::new(&game_state);
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
            // GIL released — pure Rust computation.
            py.allow_threads(|| {
                let mut tree = MCTSTree::new(&game_state);
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

    Ok(PySearchResult { inner: result })
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register MCTS types and functions on the given module.
///
/// Called by the combined extension crate — not a standalone pymodule.
pub fn register_mcts_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_mcts_search, m)?)?;
    m.add_class::<PySearchResult>()?;
    Ok(())
}
