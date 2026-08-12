//! Construction of the production backends shared by benchmark commands and calibration runs.

use std::path::PathBuf;
#[cfg(all(feature = "mcgs-profile", any(feature = "onnx", feature = "tensorrt")))]
use std::sync::Mutex;
#[cfg(all(feature = "mcgs-profile", any(feature = "onnx", feature = "tensorrt")))]
use std::time::Duration;

use alpharat_eval_core::{Backend, SmartUniformBackend};
#[cfg(all(feature = "mcgs-profile", any(feature = "onnx", feature = "tensorrt")))]
use alpharat_eval_core::{BackendError, EvalResult};
#[cfg(all(feature = "mcgs-profile", any(feature = "onnx", feature = "tensorrt")))]
use pyrat::GameState;
use thiserror::Error;

use crate::calibration::{BackendRequest, OnnxProvider, ResolvedBackend};
use crate::capacity::direct_serialization;
#[cfg(all(feature = "mcgs-profile", any(feature = "onnx", feature = "tensorrt")))]
use crate::search::{DeviceSnapshot, DeviceStats};
#[cfg(feature = "mcgs-profile")]
use crate::search::{SearchBackendProbe, SearchWorkload};

#[derive(Clone, Debug, Default)]
pub struct ProductionBackendConfig {
    pub model: Option<PathBuf>,
    pub tensorrt_cache: Option<PathBuf>,
}

#[derive(Debug, Error)]
pub enum BackendBuildError {
    #[error("unsupported backend setup: {0}")]
    Unsupported(String),
    #[error("failed to construct backend: {0}")]
    Failed(String),
}

/// Construct one direct production backend for a capacity case.
pub fn build_capacity_backend(
    request: &BackendRequest,
    config: &ProductionBackendConfig,
    width: u8,
    height: u8,
    max_batch: u32,
) -> Result<(Box<dyn Backend>, ResolvedBackend), BackendBuildError> {
    let backend: Box<dyn Backend> = match request {
        BackendRequest::SmartUniform => Box::new(SmartUniformBackend),
        BackendRequest::Onnx { provider } => build_onnx(config, *provider, width, height)?,
        BackendRequest::TensorRt => build_tensorrt(config, width, height, max_batch)?,
    };
    Ok((
        backend,
        ResolvedBackend {
            backend: request.clone(),
            serialization: direct_serialization(request),
        },
    ))
}

#[cfg(feature = "mcgs-profile")]
pub fn build_search_backend(
    request: &BackendRequest,
    config: &ProductionBackendConfig,
    mux_max_batch: Option<u32>,
    max_device_batch: u32,
    workload: &SearchWorkload,
) -> Result<(SearchBackendProbe, ResolvedBackend), BackendBuildError> {
    let backend = match request {
        BackendRequest::SmartUniform => instrument_backend(SmartUniformBackend, mux_max_batch)?,
        BackendRequest::Onnx { provider } => {
            let backend = build_onnx(config, *provider, workload.game.width, workload.game.height)?;
            instrument_boxed_backend(backend, mux_max_batch)?
        }
        BackendRequest::TensorRt => {
            let backend = build_tensorrt(
                config,
                workload.game.width,
                workload.game.height,
                max_device_batch,
            )?;
            instrument_boxed_backend(backend, mux_max_batch)?
        }
    };
    Ok((
        backend,
        ResolvedBackend {
            backend: request.clone(),
            serialization: direct_serialization(request),
        },
    ))
}

#[cfg(feature = "onnx")]
fn build_onnx(
    config: &ProductionBackendConfig,
    provider: OnnxProvider,
    width: u8,
    height: u8,
) -> Result<Box<dyn Backend>, BackendBuildError> {
    use alpharat_sampling::{ExecutionProvider, FlatEncoder, OnnxBackend};

    let provider = match provider {
        OnnxProvider::Cpu => ExecutionProvider::Cpu,
        OnnxProvider::Coreml => {
            #[cfg(feature = "onnx-coreml")]
            {
                ExecutionProvider::CoreMl
            }
            #[cfg(not(feature = "onnx-coreml"))]
            {
                return Err(BackendBuildError::Unsupported(
                    "CoreML support is not compiled; rebuild with --features onnx-coreml"
                        .to_owned(),
                ));
            }
        }
        OnnxProvider::Cuda => {
            #[cfg(feature = "onnx-cuda")]
            {
                ExecutionProvider::Cuda
            }
            #[cfg(not(feature = "onnx-cuda"))]
            {
                return Err(BackendBuildError::Unsupported(
                    "CUDA support is not compiled; rebuild with --features onnx-cuda".to_owned(),
                ));
            }
        }
    };
    let model = config
        .model
        .as_ref()
        .ok_or_else(|| BackendBuildError::Failed("ONNX cases require a model path".to_owned()))?;
    OnnxBackend::with_provider(model, FlatEncoder::new(width, height), provider)
        .map(|backend| Box::new(backend) as Box<dyn Backend>)
        .map_err(|error| BackendBuildError::Failed(format!("ONNX: {error}")))
}

#[cfg(not(feature = "onnx"))]
fn build_onnx(
    _config: &ProductionBackendConfig,
    _provider: OnnxProvider,
    _width: u8,
    _height: u8,
) -> Result<Box<dyn Backend>, BackendBuildError> {
    Err(BackendBuildError::Unsupported(
        "ONNX support is not compiled; rebuild with --features onnx (or a provider feature)"
            .to_owned(),
    ))
}

#[cfg(feature = "tensorrt")]
fn build_tensorrt(
    config: &ProductionBackendConfig,
    width: u8,
    height: u8,
    max_batch: u32,
) -> Result<Box<dyn Backend>, BackendBuildError> {
    use alpharat_sampling::{FlatEncoder, TensorrtBackend, TensorrtConfig};

    let model = config.model.as_ref().ok_or_else(|| {
        BackendBuildError::Failed("TensorRT cases require a model path".to_owned())
    })?;
    let max_batch = usize::try_from(max_batch).map_err(|_| {
        BackendBuildError::Failed(format!(
            "TensorRT max batch {max_batch} does not fit this platform"
        ))
    })?;
    let tensorrt = TensorrtConfig {
        max_batch,
        cache_dir: config.tensorrt_cache.clone(),
    };
    TensorrtBackend::new(model, FlatEncoder::new(width, height), tensorrt)
        .map(|backend| Box::new(backend) as Box<dyn Backend>)
        .map_err(|error| BackendBuildError::Failed(format!("TensorRT: {error}")))
}

#[cfg(not(feature = "tensorrt"))]
fn build_tensorrt(
    _config: &ProductionBackendConfig,
    _width: u8,
    _height: u8,
    _max_batch: u32,
) -> Result<Box<dyn Backend>, BackendBuildError> {
    Err(BackendBuildError::Unsupported(
        "TensorRT support is not compiled; rebuild with --features tensorrt".to_owned(),
    ))
}

#[cfg(feature = "mcgs-profile")]
fn instrument_boxed_backend(
    backend: Box<dyn Backend>,
    mux_max_batch: Option<u32>,
) -> Result<SearchBackendProbe, BackendBuildError> {
    match mux_max_batch {
        Some(max_batch_size) => instrument_boxed_with_mux(backend, max_batch_size),
        None => Ok(SearchBackendProbe::direct(backend)),
    }
}

#[cfg(all(feature = "mcgs-profile", any(feature = "onnx", feature = "tensorrt")))]
fn instrument_boxed_with_mux(
    backend: Box<dyn Backend>,
    max_batch_size: u32,
) -> Result<SearchBackendProbe, BackendBuildError> {
    use alpharat_sampling::{MuxBackend, MuxConfig};

    let max_batch_size = usize::try_from(max_batch_size).map_err(|_| {
        BackendBuildError::Failed(format!(
            "mux max batch {max_batch_size} does not fit this platform"
        ))
    })?;
    let mux = MuxBackend::new(DynBackend(backend), MuxConfig { max_batch_size });
    let stats = MuxDeviceStats::new(mux.stats().clone());
    Ok(SearchBackendProbe::with_device_stats(
        Box::new(mux),
        Box::new(stats),
    ))
}

#[cfg(all(feature = "mcgs-profile", any(feature = "onnx", feature = "tensorrt")))]
struct DynBackend(Box<dyn Backend>);

#[cfg(all(feature = "mcgs-profile", any(feature = "onnx", feature = "tensorrt")))]
impl Backend for DynBackend {
    fn evaluate(&self, game: &GameState) -> Result<EvalResult, BackendError> {
        self.0.evaluate(game)
    }

    fn evaluate_batch(&self, games: &[&GameState]) -> Result<Vec<EvalResult>, BackendError> {
        self.0.evaluate_batch(games)
    }
}

#[cfg(all(
    feature = "mcgs-profile",
    not(any(feature = "onnx", feature = "tensorrt"))
))]
fn instrument_boxed_with_mux(
    _backend: Box<dyn Backend>,
    _max_batch_size: u32,
) -> Result<SearchBackendProbe, BackendBuildError> {
    Err(BackendBuildError::Unsupported(
        "mux measurement requires an alpharat-sampling backend feature such as onnx or tensorrt"
            .to_owned(),
    ))
}

#[cfg(feature = "mcgs-profile")]
fn instrument_backend(
    backend: impl Backend + 'static,
    mux_max_batch: Option<u32>,
) -> Result<SearchBackendProbe, BackendBuildError> {
    instrument_boxed_backend(Box::new(backend), mux_max_batch)
}

#[cfg(all(feature = "mcgs-profile", any(feature = "onnx", feature = "tensorrt")))]
struct MuxDeviceStats {
    stats: std::sync::Arc<alpharat_sampling::MuxStats>,
    baseline: Mutex<(u64, u64, u64)>,
}

#[cfg(all(feature = "mcgs-profile", any(feature = "onnx", feature = "tensorrt")))]
impl MuxDeviceStats {
    fn new(stats: std::sync::Arc<alpharat_sampling::MuxStats>) -> Self {
        Self {
            stats,
            baseline: Mutex::new((0, 0, 0)),
        }
    }

    fn current(&self) -> (u64, u64, u64) {
        use std::sync::atomic::Ordering;

        (
            self.stats.total_batches.load(Ordering::Relaxed),
            self.stats.total_positions.load(Ordering::Relaxed),
            self.stats.nn_time_ns.load(Ordering::Relaxed),
        )
    }
}

#[cfg(all(feature = "mcgs-profile", any(feature = "onnx", feature = "tensorrt")))]
impl DeviceStats for MuxDeviceStats {
    fn reset(&self) {
        *self.baseline.lock().expect("mux baseline mutex poisoned") = self.current();
    }

    fn snapshot(&self) -> DeviceSnapshot {
        let current = self.current();
        let baseline = *self.baseline.lock().expect("mux baseline mutex poisoned");
        DeviceSnapshot {
            calls: current.0.saturating_sub(baseline.0),
            positions: current.1.saturating_sub(baseline.1),
            inference: Duration::from_nanos(current.2.saturating_sub(baseline.2)),
        }
    }
}
