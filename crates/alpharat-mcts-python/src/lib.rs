use pyo3::prelude::*;

/// Combined Python extension: pyrat-engine types + MCTS bindings.
///
/// module-name = "pyrat_engine._core" so this overwrites pyrat-engine's .so,
/// giving a superset that includes the mcts submodule. All existing
/// `from pyrat_engine.core.game import PyRat` imports still work because
/// the pure Python wrappers import from `_core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register pyrat types (same submodules as pyrat-extension)
    let types_module = PyModule::new(m.py(), "types")?;
    pyrat::bindings::register_types_module(&types_module)?;
    m.add_submodule(&types_module)?;

    let game_module = PyModule::new(m.py(), "game")?;
    pyrat::bindings::register_game_module(&game_module)?;
    m.add_submodule(&game_module)?;

    let observation_module = PyModule::new(m.py(), "observation")?;
    pyrat::bindings::register_observation_module(&observation_module)?;
    m.add_submodule(&observation_module)?;

    let builder_module = PyModule::new(m.py(), "builder")?;
    pyrat::bindings::register_builder_module(&builder_module)?;
    m.add_submodule(&builder_module)?;

    // Register mcts types
    let mcts_module = PyModule::new(m.py(), "mcts")?;
    alpharat_mcts::bindings::register_mcts_module(&mcts_module)?;
    m.add_submodule(&mcts_module)?;

    // Register sampling types
    let sampling_module = PyModule::new(m.py(), "sampling")?;
    alpharat_sampling::bindings::register_sampling_module(&sampling_module)?;
    m.add_submodule(&sampling_module)?;

    Ok(())
}
