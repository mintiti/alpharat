#[cfg(feature = "python")]
pub mod bindings;
pub mod cached_backend;
pub mod encoder;
pub mod flat_encoder;
pub mod backends;
pub mod nn_cache;
pub mod npz_writer;
pub mod recording;
pub mod selfplay;

/// ONNX tensor names shared between backends.
pub const TENSOR_INPUT: &str = "observation";
pub const TENSOR_POLICY_P1: &str = "policy_p1";
pub const TENSOR_POLICY_P2: &str = "policy_p2";
pub const TENSOR_VALUE_P1: &str = "pred_value_p1";
pub const TENSOR_VALUE_P2: &str = "pred_value_p2";

pub use cached_backend::{CacheStats, CachedBackend};
pub use encoder::ObservationEncoder;
pub use flat_encoder::FlatEncoder;
pub use backends::mux::{MuxBackend, MuxConfig, MuxStats};
pub use recording::{write_bundle, BundleWriter};
pub use selfplay::{
    play_game, run_self_play, run_self_play_to_disk, CheeseOutcome, GameRecord, PositionRecord,
    SelfPlayConfig, SelfPlayError, SelfPlayProgress, SelfPlayResult, SelfPlayStats,
    SelfPlayToDiskResult,
};

#[cfg(feature = "onnx")]
pub use backends::onnx::OnnxBackend;

#[cfg(feature = "tensorrt")]
pub use backends::tensorrt::{load_trt_libs, TensorrtBackend, TensorrtConfig, TrtTimingInfo};
