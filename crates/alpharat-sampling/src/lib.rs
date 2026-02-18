pub mod encoder;
pub mod flat_encoder;
pub mod mux_backend;
pub mod onnx_backend;
pub mod selfplay;

pub use encoder::ObservationEncoder;
pub use flat_encoder::FlatEncoder;
pub use mux_backend::{MuxBackend, MuxConfig};
pub use selfplay::{
    play_game, run_self_play, CheeseOutcome, GameRecord, PositionRecord, SelfPlayProgress,
    SelfPlayResult, SelfPlayStats,
};

#[cfg(feature = "onnx")]
pub use onnx_backend::OnnxBackend;
