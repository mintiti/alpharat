pub mod encoder;
pub mod flat_encoder;
pub mod mux_backend;
pub mod npz_writer;
pub mod onnx_backend;
pub mod recording;
pub mod selfplay;

pub use encoder::ObservationEncoder;
pub use flat_encoder::FlatEncoder;
pub use mux_backend::{MuxBackend, MuxConfig};
pub use recording::{write_bundle, BundleWriter};
pub use selfplay::{
    play_game, run_self_play, run_self_play_to_disk, CheeseOutcome, GameRecord, PositionRecord,
    SelfPlayConfig, SelfPlayProgress, SelfPlayResult, SelfPlayStats, SelfPlayToDiskResult,
};

#[cfg(feature = "onnx")]
pub use onnx_backend::OnnxBackend;
