#[cfg(feature = "onnx")]
mod inner {
    use crate::encoder::ObservationEncoder;
    use alpharat_mcts::{Backend, BackendError, EvalResult};
    use ort::session::Session;
    use ort::value::Tensor;
    use pyrat::GameState;
    use std::sync::Mutex;

    /// Execution provider for ONNX Runtime.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ExecutionProvider {
        Cpu,
        #[cfg(feature = "onnx-cuda")]
        Cuda,
        #[cfg(feature = "onnx-coreml")]
        CoreMl,
    }

    impl TryFrom<&str> for ExecutionProvider {
        type Error = String;

        fn try_from(s: &str) -> Result<Self, Self::Error> {
            match s {
                "cpu" => Ok(Self::Cpu),
                #[cfg(feature = "onnx-cuda")]
                "cuda" => Ok(Self::Cuda),
                #[cfg(feature = "onnx-coreml")]
                "coreml" => Ok(Self::CoreMl),
                other => {
                    #[allow(unused_mut)]
                    let mut supported = vec!["cpu"];
                    #[cfg(feature = "onnx-cuda")]
                    supported.push("cuda");
                    #[cfg(feature = "onnx-coreml")]
                    supported.push("coreml");
                    Err(format!(
                        "unknown execution provider '{other}', supported: {supported:?}"
                    ))
                }
            }
        }
    }

    /// ONNX Runtime backend for neural network inference.
    ///
    /// Wraps an `ort::Session` and an `ObservationEncoder` to evaluate game
    /// states without touching Python. Implements the `Backend` trait from
    /// `alpharat-mcts`.
    ///
    /// Thread safety: `ort::Session` is `Send + Sync`, but `run()` takes
    /// `&mut self`, so we wrap in a `Mutex`. This serializes inference calls,
    /// which is fine — the real parallelism comes from batching.
    pub struct OnnxBackend<E: ObservationEncoder> {
        session: Mutex<Session>,
        encoder: E,
    }

    impl<E: ObservationEncoder> OnnxBackend<E> {
        /// Create an ONNX backend with CPU execution provider.
        pub fn new(
            model_path: impl AsRef<std::path::Path>,
            encoder: E,
        ) -> Result<Self, BackendError> {
            let session = Session::builder()
                .map_err(|e| BackendError::msg(format!("failed to create ONNX session builder: {e}")))?
                .with_intra_threads(1)
                .map_err(|e| BackendError::msg(format!("failed to set intra-op thread count: {e}")))?
                .commit_from_file(model_path)
                .map_err(|e| BackendError::msg(format!("failed to load ONNX model: {e}")))?;
            Self::build(session, encoder)
        }

        /// Create an ONNX backend with CoreML execution provider (macOS).
        #[cfg(feature = "onnx-coreml")]
        pub fn with_coreml(
            model_path: impl AsRef<std::path::Path>,
            encoder: E,
        ) -> Result<Self, BackendError> {
            let session = Session::builder()
                .map_err(|e| BackendError::msg(format!("failed to create ONNX session builder: {e}")))?
                .with_intra_threads(1)
                .map_err(|e| BackendError::msg(format!("failed to set intra-op thread count: {e}")))?
                .with_execution_providers([
                    ort::execution_providers::CoreMLExecutionProvider::default()
                        .with_profile_compute_plan(true)
                        .build()
                        .error_on_failure(),
                ])
                .map_err(|e| BackendError::msg(format!("failed to register CoreML EP: {e}")))?
                .commit_from_file(model_path)
                .map_err(|e| BackendError::msg(format!("failed to load ONNX model: {e}")))?;
            Self::build(session, encoder)
        }

        /// Create an ONNX backend with CUDA execution provider (Linux).
        #[cfg(feature = "onnx-cuda")]
        pub fn with_cuda(
            model_path: impl AsRef<std::path::Path>,
            encoder: E,
        ) -> Result<Self, BackendError> {
            let session = Session::builder()
                .map_err(|e| BackendError::msg(format!("failed to create ONNX session builder: {e}")))?
                .with_intra_threads(1)
                .map_err(|e| BackendError::msg(format!("failed to set intra-op thread count: {e}")))?
                .with_execution_providers([
                    ort::execution_providers::CUDAExecutionProvider::default()
                        .build()
                        .error_on_failure(),
                ])
                .map_err(|e| BackendError::msg(format!("failed to register CUDA EP: {e}")))?
                .commit_from_file(model_path)
                .map_err(|e| BackendError::msg(format!("failed to load ONNX model: {e}")))?;
            Self::build(session, encoder)
        }

        /// Create an ONNX backend with the given execution provider.
        pub fn with_provider(
            model_path: impl AsRef<std::path::Path>,
            encoder: E,
            provider: ExecutionProvider,
        ) -> Result<Self, BackendError> {
            match provider {
                ExecutionProvider::Cpu => Self::new(model_path, encoder),
                #[cfg(feature = "onnx-cuda")]
                ExecutionProvider::Cuda => Self::with_cuda(model_path, encoder),
                #[cfg(feature = "onnx-coreml")]
                ExecutionProvider::CoreMl => Self::with_coreml(model_path, encoder),
            }
        }

        /// Validate encoder/model compatibility, then construct.
        fn build(session: Session, encoder: E) -> Result<Self, BackendError> {
            let inputs = session.inputs();
            if inputs.is_empty() {
                return Err(BackendError::msg("ONNX model has no inputs"));
            }
            if let Some(shape) = inputs[0].dtype().tensor_shape() {
                // shape[1] is obs_dim in [batch, obs_dim]
                if shape.len() >= 2 && shape[1] >= 0 {
                    // -1 means dynamic, skip check
                    if shape[1] as usize != encoder.obs_dim() {
                        return Err(BackendError::msg(format!(
                            "encoder obs_dim ({}) doesn't match ONNX model input dim ({})",
                            encoder.obs_dim(),
                            shape[1]
                        )));
                    }
                }
            }
            Ok(Self {
                session: Mutex::new(session),
                encoder,
            })
        }
    }

    impl<E: ObservationEncoder> Backend for OnnxBackend<E> {
        fn evaluate(&self, game: &GameState) -> Result<EvalResult, BackendError> {
            Ok(self.evaluate_batch(&[game])?[0])
        }

        fn evaluate_batch(&self, games: &[&GameState]) -> Result<Vec<EvalResult>, BackendError> {
            let n = games.len();
            let obs_dim = self.encoder.obs_dim();

            // Encode all game states into a flat buffer
            let mut buf = vec![0.0f32; n * obs_dim];
            for (i, game) in games.iter().enumerate() {
                self.encoder.encode_into(game, &mut buf, i * obs_dim);
            }

            // Create input tensor from (shape, data) — no ndarray needed
            let input = Tensor::from_array(([n, obs_dim], buf))
                .map_err(|e| BackendError::msg(format!("failed to create ONNX input tensor: {e}")))?;

            // Run inference (needs &mut session)
            let mut session = self.session.lock().expect("session lock poisoned");
            let outputs = session
                .run(ort::inputs!["observation" => input])
                .map_err(|e| BackendError::msg(format!("ONNX inference failed: {e}")))?;

            // Extract outputs by name as flat slices:
            //   policy_p1[N,5], policy_p2[N,5], pred_value_p1[N], pred_value_p2[N]
            let (_shape, pp1_data) = outputs["policy_p1"]
                .try_extract_tensor::<f32>()
                .map_err(|e| BackendError::msg(format!("failed to extract policy_p1: {e}")))?;
            let (_shape, pp2_data) = outputs["policy_p2"]
                .try_extract_tensor::<f32>()
                .map_err(|e| BackendError::msg(format!("failed to extract policy_p2: {e}")))?;
            let (_shape, v1_data) = outputs["pred_value_p1"]
                .try_extract_tensor::<f32>()
                .map_err(|e| BackendError::msg(format!("failed to extract pred_value_p1: {e}")))?;
            let (_shape, v2_data) = outputs["pred_value_p2"]
                .try_extract_tensor::<f32>()
                .map_err(|e| BackendError::msg(format!("failed to extract pred_value_p2: {e}")))?;

            (0..n)
                .map(|i| {
                    let p1_off = i * 5;
                    let p2_off = i * 5;
                    let result = EvalResult {
                        policy_p1: [
                            pp1_data[p1_off],
                            pp1_data[p1_off + 1],
                            pp1_data[p1_off + 2],
                            pp1_data[p1_off + 3],
                            pp1_data[p1_off + 4],
                        ],
                        policy_p2: [
                            pp2_data[p2_off],
                            pp2_data[p2_off + 1],
                            pp2_data[p2_off + 2],
                            pp2_data[p2_off + 3],
                            pp2_data[p2_off + 4],
                        ],
                        value_p1: v1_data[i],
                        value_p2: v2_data[i],
                    };
                    if !result.policy_p1.iter().all(|v| v.is_finite())
                        || !result.policy_p2.iter().all(|v| v.is_finite())
                        || !result.value_p1.is_finite()
                        || !result.value_p2.is_finite()
                    {
                        return Err(BackendError::msg(format!(
                            "ONNX output contains non-finite values (NaN/Inf) for game {i}"
                        )));
                    }
                    Ok(result)
                })
                .collect()
        }
    }
}

#[cfg(feature = "onnx")]
pub use inner::{ExecutionProvider, OnnxBackend};
