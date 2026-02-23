#[cfg(feature = "python")]
pub mod bindings;
pub mod encoder;
pub mod flat_encoder;
pub mod mux_backend;
pub mod npz_writer;
pub mod onnx_backend;
pub mod recording;
pub mod selfplay;
pub mod tensorrt_backend;

/// ONNX tensor names shared between backends.
pub const TENSOR_INPUT: &str = "observation";
pub const TENSOR_POLICY_P1: &str = "policy_p1";
pub const TENSOR_POLICY_P2: &str = "policy_p2";
pub const TENSOR_VALUE_P1: &str = "pred_value_p1";
pub const TENSOR_VALUE_P2: &str = "pred_value_p2";

pub use encoder::ObservationEncoder;
pub use flat_encoder::FlatEncoder;
pub use mux_backend::{MuxBackend, MuxConfig, MuxStats};
pub use recording::{write_bundle, BundleWriter};
pub use selfplay::{
    play_game, run_self_play, run_self_play_to_disk, CheeseOutcome, GameRecord, PositionRecord,
    SelfPlayConfig, SelfPlayError, SelfPlayProgress, SelfPlayResult, SelfPlayStats,
    SelfPlayToDiskResult,
};

#[cfg(feature = "onnx")]
pub use onnx_backend::OnnxBackend;

#[cfg(feature = "tensorrt")]
pub use tensorrt_backend::{load_trt_libs, TensorrtBackend, TensorrtConfig, TrtTimingInfo};
