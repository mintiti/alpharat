use pyrat::GameState;

/// Encodes a `GameState` into a flat f32 buffer for neural network inference.
///
/// Implementations must produce output identical to the corresponding Python
/// observation builder (e.g. `FlatObservationBuilder`).
pub trait ObservationEncoder: Send + Sync {
    /// Total number of f32 values produced per game state.
    fn obs_dim(&self) -> usize;

    /// Write the encoded observation into `buf` starting at `offset`.
    ///
    /// The caller guarantees `buf[offset..offset + obs_dim()]` is valid.
    fn encode_into(&self, game: &GameState, buf: &mut [f32], offset: usize);
}
