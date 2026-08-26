#[cfg(feature = "tensorrt")]
mod inner {
    use crate::encoder::ObservationEncoder;
    use alpharat_mcts::{Backend, BackendError, EvalResult};
    use pyrat::GameState;
    use sha2::{Digest, Sha256};
    use std::ffi::{c_void, CString};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Condvar, Mutex, MutexGuard};

    // -----------------------------------------------------------------------
    // CUDA FFI — thin declarations for GPU memory management
    // -----------------------------------------------------------------------

    extern "C" {
        fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
        fn cudaFree(ptr: *mut c_void) -> i32;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
        fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: i32,
            stream: *mut c_void,
        ) -> i32;
        fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
        fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
        fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
        fn cudaStreamDestroy(stream: *mut c_void) -> i32;
        fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
        fn cudaGetErrorString(error: i32) -> *const std::ffi::c_char;
        fn cudaGetErrorName(error: i32) -> *const std::ffi::c_char;
    }

    const CUDA_MEMCPY_H2D: i32 = 1;
    const CUDA_MEMCPY_D2H: i32 = 2;
    const CUDA_STREAM_NON_BLOCKING: u32 = 1;

    fn cuda_check(code: i32, op: &str) -> Result<(), BackendError> {
        if code != 0 {
            let name = unsafe {
                let p = cudaGetErrorName(code);
                if p.is_null() {
                    "unknown"
                } else {
                    std::ffi::CStr::from_ptr(p).to_str().unwrap_or("unknown")
                }
            };
            let msg = unsafe {
                let p = cudaGetErrorString(code);
                if p.is_null() {
                    "no description"
                } else {
                    std::ffi::CStr::from_ptr(p)
                        .to_str()
                        .unwrap_or("no description")
                }
            };
            return Err(BackendError::msg(format!(
                "CUDA {op} failed: {name} ({code}): {msg}"
            )));
        }
        Ok(())
    }

    fn compute_capability() -> Result<(i32, i32), BackendError> {
        let (mut major, mut minor) = (0i32, 0i32);
        // cudaDevAttrComputeCapabilityMajor = 75, Minor = 76
        cuda_check(
            unsafe { cudaDeviceGetAttribute(&mut major, 75, 0) },
            "cudaDeviceGetAttribute(compute_major)",
        )?;
        cuda_check(
            unsafe { cudaDeviceGetAttribute(&mut minor, 76, 0) },
            "cudaDeviceGetAttribute(compute_minor)",
        )?;
        Ok((major, minor))
    }

    // -----------------------------------------------------------------------
    // C++ TRT shim FFI (compiled from cpp/trt_shim.cpp)
    // -----------------------------------------------------------------------

    extern "C" {
        fn trt_build_engine(
            onnx_data: *const c_void,
            onnx_len: usize,
            min_batch: i32,
            opt_batch: i32,
            max_batch: i32,
            workspace_mb: usize,
            out_data: *mut *mut c_void,
            out_len: *mut usize,
        ) -> i32;

        fn trt_free_buffer(data: *mut c_void);

        fn trt_create_engine(engine_data: *const c_void, engine_len: usize) -> *mut c_void;
        fn trt_destroy_engine(handle: *mut c_void);

        fn trt_create_session(engine_handle: *mut c_void, enable_cuda_graphs: i32) -> *mut c_void;
        fn trt_destroy_session(handle: *mut c_void);
        fn trt_session_cuda_graphs_requested(handle: *mut c_void) -> i32;

        fn trt_set_tensor_address(handle: *mut c_void, name: *const i8, ptr: *mut c_void) -> i32;
        fn trt_set_input_shape(
            handle: *mut c_void,
            name: *const i8,
            ndims: i32,
            shape: *const i64,
        ) -> i32;
        fn trt_enqueue_v3(handle: *mut c_void, stream: *mut c_void) -> i32;

        fn trt_get_nb_io_tensors(handle: *mut c_void) -> i32;
        fn trt_get_tensor_name(handle: *mut c_void, index: i32) -> *const i8;

        fn trt_load_libs(trt_lib_path: *const i8, parser_lib_path: *const i8) -> i32;

        fn trt_get_version() -> i32;
    }

    /// Load TRT-RTX shared libraries via dlopen (idempotent).
    /// Called automatically by `TensorrtBackend::new`. Also available
    /// for standalone Rust binaries that don't have Python preloading.
    pub fn load_trt_libs() -> Result<(), BackendError> {
        let rc = unsafe { trt_load_libs(std::ptr::null(), std::ptr::null()) };
        if rc != 0 {
            return Err(BackendError::msg(
                "Failed to load TensorRT-RTX libraries — \
                 is $TENSORRT_RTX_ROOT/lib in LD_LIBRARY_PATH?",
            ));
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    /// Configuration for the TensorRT-RTX backend.
    pub struct TensorrtConfig {
        /// Batch size TensorRT should optimize the dynamic profile around.
        pub opt_batch: usize,
        /// Maximum batch size. GPU buffers are pre-allocated for this size.
        pub max_batch: usize,
        /// Directory for cached serialized engines. `None` disables caching.
        pub cache_dir: Option<PathBuf>,
        /// Number of independent execution context/stream/buffer lanes.
        pub execution_contexts: usize,
        /// Request TensorRT-RTX whole-model CUDA Graph capture for each lane.
        ///
        /// TensorRT may silently fall back when a model or stream cannot be captured,
        /// so graph-off/on measurement remains the behavioral check.
        pub cuda_graphs: bool,
    }

    impl Default for TensorrtConfig {
        fn default() -> Self {
            Self {
                opt_batch: 256,
                max_batch: 256,
                cache_dir: None,
                execution_contexts: 1,
                cuda_graphs: false,
            }
        }
    }

    // -----------------------------------------------------------------------
    // GPU buffer management
    // -----------------------------------------------------------------------

    struct GpuBuffers {
        d_input: *mut c_void,
        d_policy_p1: *mut c_void,
        d_policy_p2: *mut c_void,
        d_value_p1: *mut c_void,
        d_value_p2: *mut c_void,
    }

    // SAFETY: GPU pointers are thread-safe when access is serialized via Mutex.
    unsafe impl Send for GpuBuffers {}

    impl GpuBuffers {
        /// Allocate GPU buffers for the given batch/obs dimensions.
        ///
        /// On `?`-return, `Drop` runs on the partially-initialized struct.
        /// All pointers start as `null_mut()` and `cudaFree(NULL)` is a
        /// documented no-op, so cleanup is safe even on partial allocation.
        fn alloc(max_batch: usize, obs_dim: usize) -> Result<Self, BackendError> {
            let f = std::mem::size_of::<f32>();
            let mut b = Self {
                d_input: std::ptr::null_mut(),
                d_policy_p1: std::ptr::null_mut(),
                d_policy_p2: std::ptr::null_mut(),
                d_value_p1: std::ptr::null_mut(),
                d_value_p2: std::ptr::null_mut(),
            };
            unsafe {
                cuda_check(
                    cudaMalloc(&mut b.d_input, max_batch * obs_dim * f),
                    "cudaMalloc(input)",
                )?;
                cuda_check(
                    cudaMalloc(&mut b.d_policy_p1, max_batch * 5 * f),
                    "cudaMalloc(policy_p1)",
                )?;
                cuda_check(
                    cudaMalloc(&mut b.d_policy_p2, max_batch * 5 * f),
                    "cudaMalloc(policy_p2)",
                )?;
                cuda_check(
                    cudaMalloc(&mut b.d_value_p1, max_batch * f),
                    "cudaMalloc(value_p1)",
                )?;
                cuda_check(
                    cudaMalloc(&mut b.d_value_p2, max_batch * f),
                    "cudaMalloc(value_p2)",
                )?;
            }
            Ok(b)
        }
    }

    impl Drop for GpuBuffers {
        fn drop(&mut self) {
            unsafe {
                cudaFree(self.d_input);
                cudaFree(self.d_policy_p1);
                cudaFree(self.d_policy_p2);
                cudaFree(self.d_value_p1);
                cudaFree(self.d_value_p2);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Engine caching
    // -----------------------------------------------------------------------

    /// Magic bytes at the start of cached engine files for format detection.
    const CACHE_MAGIC: [u8; 4] = *b"TRTE";
    /// Header: 4 bytes magic + 32 bytes ONNX SHA-256 hash.
    const CACHE_HEADER_SIZE: usize = 4 + 32;

    fn cache_key(
        onnx_hash: &[u8; 32],
        opt_batch: usize,
        max_batch: usize,
    ) -> Result<String, BackendError> {
        let (major, minor) = compute_capability()?;
        let trt_version = unsafe { trt_get_version() };
        let hash_hex: String = onnx_hash
            .iter()
            .take(8)
            .map(|b| format!("{b:02x}"))
            .collect();
        Ok(format!(
            "trt{trt_version}_sm{major}{minor}_{hash_hex}_opt{opt_batch}_max{max_batch}.engine"
        ))
    }

    fn try_load_cache(cache_dir: &Path, key: &str, onnx_hash: &[u8; 32]) -> Option<Vec<u8>> {
        let data = fs::read(cache_dir.join(key)).ok()?;
        if data.len() < CACHE_HEADER_SIZE {
            eprintln!("[TensorRT] Cache file too short, rebuilding");
            return None;
        }
        if data[..4] != CACHE_MAGIC {
            eprintln!("[TensorRT] Cache file missing magic bytes, rebuilding");
            return None;
        }
        if data[4..CACHE_HEADER_SIZE] != onnx_hash[..] {
            eprintln!("[TensorRT] Cache file ONNX hash mismatch, rebuilding");
            return None;
        }
        Some(data[CACHE_HEADER_SIZE..].to_vec())
    }

    fn save_cache(cache_dir: &Path, key: &str, engine_data: &[u8], onnx_hash: &[u8; 32]) {
        let _ = fs::create_dir_all(cache_dir);
        let path = cache_dir.join(key);
        let mut buf = Vec::with_capacity(CACHE_HEADER_SIZE + engine_data.len());
        buf.extend_from_slice(&CACHE_MAGIC);
        buf.extend_from_slice(onnx_hash);
        buf.extend_from_slice(engine_data);
        if let Err(e) = fs::write(&path, &buf) {
            eprintln!(
                "[TensorRT] Warning: failed to cache engine to {}: {e}",
                path.display()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Engine building (via C++ shim)
    // -----------------------------------------------------------------------

    fn trt_build_error_message(rc: i32) -> &'static str {
        match rc {
            -1 => "failed to resolve TRT factory functions (libs not loaded?)",
            -2 => "failed to create network definition",
            -3 => "failed to create ONNX parser",
            -4 => "ONNX parse failed (check stderr for details)",
            -5 => "failed to create optimization profile",
            -6 => "failed to set optimization profile dimensions",
            -7 => "failed to create builder config",
            -8 => "failed to add optimization profile to config",
            -9 => "engine serialization failed",
            _ => "unknown error",
        }
    }

    /// Build a serialized TensorRT engine from ONNX bytes, with optimization
    /// profiles for dynamic batch sizes.
    fn build_engine(
        onnx_bytes: &[u8],
        opt_batch: usize,
        max_batch: usize,
    ) -> Result<Vec<u8>, BackendError> {
        let mut out_data: *mut c_void = std::ptr::null_mut();
        let mut out_len: usize = 0;

        let rc = unsafe {
            trt_build_engine(
                onnx_bytes.as_ptr() as *const c_void,
                onnx_bytes.len(),
                1, // min_batch
                opt_batch as i32,
                max_batch as i32, // max_batch
                256,              // workspace MB
                &mut out_data,
                &mut out_len,
            )
        };

        if rc != 0 || out_data.is_null() {
            return Err(BackendError::msg(format!(
                "TRT engine build failed: {} (rc={rc})",
                trt_build_error_message(rc)
            )));
        }

        let data = unsafe { std::slice::from_raw_parts(out_data as *const u8, out_len) }.to_vec();
        unsafe { trt_free_buffer(out_data) };

        eprintln!("[TensorRT] Engine built ({} bytes)", data.len());
        Ok(data)
    }

    // -----------------------------------------------------------------------
    // TrtSession — owns TRT context + CUDA resources
    // -----------------------------------------------------------------------

    struct TrtEngine {
        handle: *mut c_void,
    }

    // SAFETY: TensorRT engines are immutable during inference. Context creation
    // happens serially in `TensorrtBackend::new`, and every mutable execution
    // object lives in its own leased `TrtSession`.
    unsafe impl Send for TrtEngine {}
    unsafe impl Sync for TrtEngine {}

    impl TrtEngine {
        fn new(engine_data: &[u8]) -> Result<Self, BackendError> {
            let handle = unsafe {
                trt_create_engine(engine_data.as_ptr() as *const c_void, engine_data.len())
            };
            if handle.is_null() {
                return Err(BackendError::msg("Failed to deserialize TRT engine"));
            }
            Ok(Self { handle })
        }

        fn io_tensor_names(&self) -> Vec<String> {
            let n_io = unsafe { trt_get_nb_io_tensors(self.handle) };
            (0..n_io)
                .map(|i| {
                    let ptr = unsafe { trt_get_tensor_name(self.handle, i) };
                    if ptr.is_null() {
                        "<null>".to_string()
                    } else {
                        unsafe { std::ffi::CStr::from_ptr(ptr) }
                            .to_string_lossy()
                            .into_owned()
                    }
                })
                .collect()
        }
    }

    impl Drop for TrtEngine {
        fn drop(&mut self) {
            unsafe { trt_destroy_engine(self.handle) };
        }
    }

    struct LanePool<T> {
        lanes: Vec<Mutex<T>>,
        available: Mutex<Vec<usize>>,
        ready: Condvar,
    }

    impl<T> LanePool<T> {
        fn new(lanes: Vec<T>) -> Result<Self, BackendError> {
            if lanes.is_empty() {
                return Err(BackendError::msg(
                    "TensorRT execution_contexts must be at least 1",
                ));
            }
            let available = (0..lanes.len()).collect();
            Ok(Self {
                lanes: lanes.into_iter().map(Mutex::new).collect(),
                available: Mutex::new(available),
                ready: Condvar::new(),
            })
        }

        fn lease(&self) -> Result<LaneLease<'_, T>, BackendError> {
            let mut available = self
                .available
                .lock()
                .map_err(|_| BackendError::msg("TensorRT lane queue lock poisoned"))?;
            while available.is_empty() {
                available = self
                    .ready
                    .wait(available)
                    .map_err(|_| BackendError::msg("TensorRT lane queue lock poisoned"))?;
            }
            let index = available.pop().expect("non-empty lane queue");
            drop(available);

            let guard = match self.lanes[index].lock() {
                Ok(guard) => guard,
                Err(_) => {
                    self.release(index);
                    return Err(BackendError::msg("TensorRT execution lane lock poisoned"));
                }
            };
            Ok(LaneLease {
                pool: self,
                index,
                guard: Some(guard),
            })
        }

        fn release(&self, index: usize) {
            let mut available = self
                .available
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            available.push(index);
            self.ready.notify_one();
        }
    }

    struct LaneLease<'a, T> {
        pool: &'a LanePool<T>,
        index: usize,
        guard: Option<MutexGuard<'a, T>>,
    }

    impl<T> LaneLease<'_, T> {
        fn get_mut(&mut self) -> &mut T {
            self.guard.as_deref_mut().expect("lane lease guard present")
        }
    }

    impl<T> Drop for LaneLease<'_, T> {
        fn drop(&mut self) {
            drop(self.guard.take());
            self.pool.release(self.index);
        }
    }

    struct TrtSession {
        handle: *mut c_void, // opaque TrtSession from C++ shim
        stream: *mut c_void,
        buffers: GpuBuffers,
        max_batch: usize,
        obs_dim: usize,
        stream_scoped_copies: bool,
        _engine: Arc<TrtEngine>,
    }

    // SAFETY: A session is used by at most one `LaneLease` at a time.
    unsafe impl Send for TrtSession {}

    impl Drop for TrtSession {
        fn drop(&mut self) {
            unsafe {
                cudaStreamDestroy(self.stream);
                trt_destroy_session(self.handle);
            }
        }
    }

    impl TrtSession {
        fn new(
            engine: Arc<TrtEngine>,
            obs_dim: usize,
            max_batch: usize,
            cuda_graphs: bool,
            log_configuration: bool,
        ) -> Result<Self, BackendError> {
            let enable_cuda_graphs = if cuda_graphs { 1 } else { 0 };
            let handle = unsafe { trt_create_session(engine.handle, enable_cuda_graphs) };
            if handle.is_null() {
                return Err(BackendError::msg(format!(
                    "Failed to create TRT execution context (cuda_graphs={cuda_graphs})"
                )));
            }

            match Self::init_session(
                handle,
                engine,
                obs_dim,
                max_batch,
                cuda_graphs,
                log_configuration,
            ) {
                Ok(session) => Ok(session),
                Err(e) => {
                    // Clean up the C++ session on init failure
                    unsafe { trt_destroy_session(handle) };
                    Err(e)
                }
            }
        }

        /// Initialize the session after handle creation. Separated so that
        /// `new()` can destroy the handle on error.
        fn init_session(
            handle: *mut c_void,
            engine: Arc<TrtEngine>,
            obs_dim: usize,
            max_batch: usize,
            cuda_graphs: bool,
            log_configuration: bool,
        ) -> Result<Self, BackendError> {
            if log_configuration {
                let names = engine.io_tensor_names();
                eprintln!("[TensorRT] IO tensors: {names:?}");
                let accepted = unsafe { trt_session_cuda_graphs_requested(handle) } != 0;
                eprintln!(
                    "[TensorRT] RTX CUDA graphs: requested={cuda_graphs}, config_accepted={accepted}"
                );
            }

            // Allocate GPU buffers
            let buffers = GpuBuffers::alloc(max_batch, obs_dim)?;

            // Bind tensor addresses
            Self::bind_tensors(handle, &buffers)?;

            let mut stream: *mut c_void = std::ptr::null_mut();
            if cuda_graphs {
                // Graph capture cannot interact with CUDA's legacy stream, so
                // graph lanes use non-blocking streams plus stream-scoped copies.
                cuda_check(
                    unsafe { cudaStreamCreateWithFlags(&mut stream, CUDA_STREAM_NON_BLOCKING) },
                    "cudaStreamCreateWithFlags(non-blocking)",
                )?;
            } else {
                // Preserve the pre-pool default path exactly when graphs are off.
                cuda_check(unsafe { cudaStreamCreate(&mut stream) }, "cudaStreamCreate")?;
            }

            Ok(Self {
                handle,
                stream,
                buffers,
                max_batch,
                obs_dim,
                stream_scoped_copies: cuda_graphs,
                _engine: engine,
            })
        }

        /// Bind the 5 named tensors (1 input + 4 outputs) to GPU buffer addresses.
        fn bind_tensors(handle: *mut c_void, buffers: &GpuBuffers) -> Result<(), BackendError> {
            let bindings: [(&str, *mut c_void); 5] = [
                (crate::TENSOR_INPUT, buffers.d_input),
                (crate::TENSOR_POLICY_P1, buffers.d_policy_p1),
                (crate::TENSOR_POLICY_P2, buffers.d_policy_p2),
                (crate::TENSOR_VALUE_P1, buffers.d_value_p1),
                (crate::TENSOR_VALUE_P2, buffers.d_value_p2),
            ];
            for (name, ptr) in bindings {
                let cname = CString::new(name).unwrap();
                let rc = unsafe { trt_set_tensor_address(handle, cname.as_ptr(), ptr) };
                if rc != 0 {
                    return Err(BackendError::msg(format!(
                        "Failed to bind '{name}' tensor (rc={rc})"
                    )));
                }
            }
            Ok(())
        }

        fn copy(
            &self,
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: i32,
            op: &str,
        ) -> Result<(), BackendError> {
            let code = if self.stream_scoped_copies {
                unsafe { cudaMemcpyAsync(dst, src, count, kind, self.stream) }
            } else {
                unsafe { cudaMemcpy(dst, src, count, kind) }
            };
            cuda_check(code, op)
        }

        fn finish_output_copies(&self) -> Result<(), BackendError> {
            if self.stream_scoped_copies {
                cuda_check(
                    unsafe { cudaStreamSynchronize(self.stream) },
                    "cudaStreamSynchronize(outputs)",
                )?;
            }
            Ok(())
        }

        /// Run inference on encoded observations. Returns (pp1, pp2, v1, v2).
        fn infer(
            &mut self,
            input: &[f32],
            n: usize,
        ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), BackendError> {
            if n > self.max_batch {
                return Err(BackendError::msg(format!(
                    "batch size {n} exceeds max_batch {}",
                    self.max_batch
                )));
            }
            if input.len() != n * self.obs_dim {
                return Err(BackendError::msg(format!(
                    "input length {} != expected {}",
                    input.len(),
                    n * self.obs_dim
                )));
            }

            let f = std::mem::size_of::<f32>();

            // Set dynamic input shape for this batch
            let obs_name = CString::new(crate::TENSOR_INPUT).unwrap();
            let shape = [n as i64, self.obs_dim as i64];
            let rc =
                unsafe { trt_set_input_shape(self.handle, obs_name.as_ptr(), 2, shape.as_ptr()) };
            if rc != 0 {
                return Err(BackendError::msg(format!(
                    "Failed to set input shape for batch size {n} (rc={rc})"
                )));
            }

            // H2D: copy observation buffer to GPU
            self.copy(
                self.buffers.d_input,
                input.as_ptr() as *const c_void,
                n * self.obs_dim * f,
                CUDA_MEMCPY_H2D,
                "input H2D",
            )?;

            // Run inference
            let rc = unsafe { trt_enqueue_v3(self.handle, self.stream) };
            if rc != 0 {
                return Err(BackendError::msg(format!(
                    "TRT enqueue_v3 failed (rc={rc})"
                )));
            }
            cuda_check(
                unsafe { cudaStreamSynchronize(self.stream) },
                "cudaStreamSynchronize",
            )?;

            // D2H: copy outputs back
            let mut pp1 = vec![0.0f32; n * 5];
            let mut pp2 = vec![0.0f32; n * 5];
            let mut v1 = vec![0.0f32; n];
            let mut v2 = vec![0.0f32; n];

            self.copy(
                pp1.as_mut_ptr() as *mut c_void,
                self.buffers.d_policy_p1,
                n * 5 * f,
                CUDA_MEMCPY_D2H,
                "policy_p1 D2H",
            )?;
            self.copy(
                pp2.as_mut_ptr() as *mut c_void,
                self.buffers.d_policy_p2,
                n * 5 * f,
                CUDA_MEMCPY_D2H,
                "policy_p2 D2H",
            )?;
            self.copy(
                v1.as_mut_ptr() as *mut c_void,
                self.buffers.d_value_p1,
                n * f,
                CUDA_MEMCPY_D2H,
                "value_p1 D2H",
            )?;
            self.copy(
                v2.as_mut_ptr() as *mut c_void,
                self.buffers.d_value_p2,
                n * f,
                CUDA_MEMCPY_D2H,
                "value_p2 D2H",
            )?;
            self.finish_output_copies()?;

            Ok((pp1, pp2, v1, v2))
        }
    }

    // -----------------------------------------------------------------------
    // TensorrtBackend — public API, implements Backend
    // -----------------------------------------------------------------------

    /// TensorRT-RTX backend for neural network inference.
    ///
    /// Loads an ONNX model via TensorRT's ONNX parser, JIT-compiles an
    /// optimized engine for the current GPU, and runs inference directly on
    /// the GPU. Engines are cached to disk for fast subsequent startups.
    ///
    /// Thread safety: callers lease one independent execution context, CUDA
    /// stream, and buffer set from a bounded pool. A single caller retains the
    /// old one-lane behavior when `execution_contexts` is 1.
    pub struct TensorrtBackend<E: ObservationEncoder> {
        sessions: LanePool<TrtSession>,
        encoder: E,
    }

    impl<E: ObservationEncoder> TensorrtBackend<E> {
        /// Create a TensorRT backend from an ONNX model file.
        ///
        /// On first run for a given model + GPU combination, this builds
        /// the TRT engine (~10-30s). Subsequent runs load from cache (~100ms).
        pub fn new(
            model_path: impl AsRef<Path>,
            encoder: E,
            config: TensorrtConfig,
        ) -> Result<Self, BackendError> {
            if config.max_batch == 0 {
                return Err(BackendError::msg("TensorRT max_batch must be at least 1"));
            }
            if config.opt_batch == 0 || config.opt_batch > config.max_batch {
                return Err(BackendError::msg(format!(
                    "TensorRT opt_batch must be in 1..={} (got {})",
                    config.max_batch, config.opt_batch
                )));
            }
            if config.execution_contexts == 0 {
                return Err(BackendError::msg(
                    "TensorRT execution_contexts must be at least 1",
                ));
            }
            load_trt_libs()?;

            let obs_dim = encoder.obs_dim();
            let onnx_path = model_path.as_ref();

            let onnx_bytes = fs::read(onnx_path).map_err(|e| {
                BackendError::msg(format!(
                    "Failed to read ONNX model at {}: {e}",
                    onnx_path.display()
                ))
            })?;

            let onnx_hash: [u8; 32] = Sha256::digest(&onnx_bytes).into();
            let key = cache_key(&onnx_hash, config.opt_batch, config.max_batch)?;

            // Load cached engine or build from scratch
            let engine_data = match &config.cache_dir {
                Some(dir) => match try_load_cache(dir, &key, &onnx_hash) {
                    Some(data) => {
                        eprintln!("[TensorRT] Loaded cached engine: {key}");
                        data
                    }
                    None => {
                        eprintln!("[TensorRT] Building engine from ONNX (this may take 10-30s)...");
                        let data = build_engine(&onnx_bytes, config.opt_batch, config.max_batch)?;
                        save_cache(dir, &key, &data, &onnx_hash);
                        eprintln!("[TensorRT] Engine cached as {key}");
                        data
                    }
                },
                None => {
                    eprintln!("[TensorRT] Building engine (no cache dir configured)...");
                    build_engine(&onnx_bytes, config.opt_batch, config.max_batch)?
                }
            };

            let engine = Arc::new(TrtEngine::new(&engine_data)?);
            let mut sessions = Vec::with_capacity(config.execution_contexts);
            for index in 0..config.execution_contexts {
                sessions.push(TrtSession::new(
                    Arc::clone(&engine),
                    obs_dim,
                    config.max_batch,
                    config.cuda_graphs,
                    index == 0,
                )?);
            }
            let sessions = LanePool::new(sessions)?;
            eprintln!(
                "[TensorRT] Profile: MIN=1, OPT={}, MAX={}; execution contexts: {}; CUDA graphs requested: {}",
                config.opt_batch,
                config.max_batch,
                config.execution_contexts,
                config.cuda_graphs
            );

            Ok(Self { sessions, encoder })
        }
    }

    /// Parse flat output buffers from TRT inference into `EvalResult` vec.
    /// Returns an error if any values are non-finite (NaN/Inf from GPU inference).
    pub fn parse_eval_results(
        pp1: &[f32],
        pp2: &[f32],
        v1: &[f32],
        v2: &[f32],
        n: usize,
    ) -> Result<Vec<EvalResult>, BackendError> {
        (0..n)
            .map(|i| {
                let p1_off = i * 5;
                let p2_off = i * 5;
                let result = EvalResult {
                    policy_p1: [
                        pp1[p1_off],
                        pp1[p1_off + 1],
                        pp1[p1_off + 2],
                        pp1[p1_off + 3],
                        pp1[p1_off + 4],
                    ],
                    policy_p2: [
                        pp2[p2_off],
                        pp2[p2_off + 1],
                        pp2[p2_off + 2],
                        pp2[p2_off + 3],
                        pp2[p2_off + 4],
                    ],
                    value_p1: v1[i],
                    value_p2: v2[i],
                };
                if !result.policy_p1.iter().all(|v| v.is_finite())
                    || !result.policy_p2.iter().all(|v| v.is_finite())
                    || !result.value_p1.is_finite()
                    || !result.value_p2.is_finite()
                {
                    return Err(BackendError::msg(format!(
                        "TensorRT output contains non-finite values (NaN/Inf) for position {i}"
                    )));
                }
                Ok(result)
            })
            .collect()
    }

    /// Timing breakdown for a single TRT inference call.
    pub struct TrtTimingInfo {
        /// Host-to-device transfer (microseconds).
        pub h2d_us: f64,
        /// GPU inference + synchronization (microseconds).
        pub infer_us: f64,
        /// Device-to-host transfer (microseconds).
        pub d2h_us: f64,
    }

    impl<E: ObservationEncoder> TensorrtBackend<E> {
        /// Run inference on pre-encoded observations with per-phase timing.
        ///
        /// Unlike `evaluate_batch` (which encodes game states), this takes
        /// a flat f32 buffer that's already encoded. Returns results + timing.
        ///
        /// This method is for benchmarking — it duplicates some of
        /// `TrtSession::infer` to insert timing around individual phases.
        pub fn evaluate_encoded_timed(
            &self,
            encoded: &[f32],
            n: usize,
        ) -> Result<(Vec<EvalResult>, TrtTimingInfo), BackendError> {
            let mut lease = self.sessions.lease()?;
            let session = lease.get_mut();
            if n > session.max_batch {
                return Err(BackendError::msg(format!(
                    "batch size {n} exceeds max_batch {}",
                    session.max_batch
                )));
            }
            if encoded.len() != n * session.obs_dim {
                return Err(BackendError::msg(format!(
                    "encoded length {} != expected {}",
                    encoded.len(),
                    n * session.obs_dim
                )));
            }

            let f = std::mem::size_of::<f32>();

            // Set dynamic input shape
            let obs_name = CString::new(crate::TENSOR_INPUT).unwrap();
            let shape = [n as i64, session.obs_dim as i64];
            let rc = unsafe {
                trt_set_input_shape(session.handle, obs_name.as_ptr(), 2, shape.as_ptr())
            };
            if rc != 0 {
                return Err(BackendError::msg(format!(
                    "Failed to set input shape for batch size {n} (rc={rc})"
                )));
            }

            // Phase 1: H2D
            let t0 = std::time::Instant::now();
            session.copy(
                session.buffers.d_input,
                encoded.as_ptr() as *const c_void,
                n * session.obs_dim * f,
                CUDA_MEMCPY_H2D,
                "input H2D",
            )?;
            let h2d_us = t0.elapsed().as_micros() as f64;

            // Phase 2: Inference + sync
            let t1 = std::time::Instant::now();
            let rc = unsafe { trt_enqueue_v3(session.handle, session.stream) };
            if rc != 0 {
                return Err(BackendError::msg(format!(
                    "TRT enqueue_v3 failed (rc={rc})"
                )));
            }
            cuda_check(
                unsafe { cudaStreamSynchronize(session.stream) },
                "cudaStreamSynchronize",
            )?;
            let infer_us = t1.elapsed().as_micros() as f64;

            // Phase 3: D2H
            let t2 = std::time::Instant::now();
            let mut pp1 = vec![0.0f32; n * 5];
            let mut pp2 = vec![0.0f32; n * 5];
            let mut v1 = vec![0.0f32; n];
            let mut v2 = vec![0.0f32; n];
            session.copy(
                pp1.as_mut_ptr() as *mut c_void,
                session.buffers.d_policy_p1,
                n * 5 * f,
                CUDA_MEMCPY_D2H,
                "policy_p1 D2H",
            )?;
            session.copy(
                pp2.as_mut_ptr() as *mut c_void,
                session.buffers.d_policy_p2,
                n * 5 * f,
                CUDA_MEMCPY_D2H,
                "policy_p2 D2H",
            )?;
            session.copy(
                v1.as_mut_ptr() as *mut c_void,
                session.buffers.d_value_p1,
                n * f,
                CUDA_MEMCPY_D2H,
                "value_p1 D2H",
            )?;
            session.copy(
                v2.as_mut_ptr() as *mut c_void,
                session.buffers.d_value_p2,
                n * f,
                CUDA_MEMCPY_D2H,
                "value_p2 D2H",
            )?;
            session.finish_output_copies()?;
            let d2h_us = t2.elapsed().as_micros() as f64;

            let results = parse_eval_results(&pp1, &pp2, &v1, &v2, n)?;
            let timing = TrtTimingInfo {
                h2d_us,
                infer_us,
                d2h_us,
            };
            Ok((results, timing))
        }
    }

    impl<E: ObservationEncoder> Backend for TensorrtBackend<E> {
        fn evaluate(&self, game: &GameState) -> Result<EvalResult, BackendError> {
            Ok(self.evaluate_batch(&[game])?[0])
        }

        fn evaluate_batch(&self, games: &[&GameState]) -> Result<Vec<EvalResult>, BackendError> {
            let n = games.len();
            if n == 0 {
                return Ok(Vec::new());
            }
            let obs_dim = self.encoder.obs_dim();

            // Encode all game states into a flat buffer
            let mut buf = vec![0.0f32; n * obs_dim];
            for (i, game) in games.iter().enumerate() {
                self.encoder.encode_into(game, &mut buf, i * obs_dim);
            }

            // Run TRT inference
            let mut lease = self.sessions.lease()?;
            let (pp1, pp2, v1, v2) = lease.get_mut().infer(&buf, n)?;

            parse_eval_results(&pp1, &pp2, &v1, &v2, n)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{LanePool, TensorrtBackend, TensorrtConfig};
        use crate::FlatEncoder;
        use std::sync::{mpsc, Arc};
        use std::time::Duration;

        #[test]
        fn lane_pool_reuses_the_recently_released_lane() {
            let pool = LanePool::new(vec![10_u8, 20_u8]).unwrap();

            let mut first = pool.lease().unwrap();
            assert_eq!(*first.get_mut(), 20);
            drop(first);

            let mut second = pool.lease().unwrap();
            assert_eq!(*second.get_mut(), 20);
        }

        #[test]
        fn lane_pool_waits_until_capacity_is_released() {
            let pool = Arc::new(LanePool::new(vec![()]).unwrap());
            let held = pool.lease().unwrap();
            let (started_tx, started_rx) = mpsc::channel();
            let (acquired_tx, acquired_rx) = mpsc::channel();
            let waiter_pool = Arc::clone(&pool);

            let waiter = std::thread::spawn(move || {
                started_tx.send(()).unwrap();
                let _lease = waiter_pool.lease().unwrap();
                acquired_tx.send(()).unwrap();
            });

            started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
            assert!(acquired_rx.recv_timeout(Duration::from_millis(25)).is_err());
            drop(held);
            acquired_rx.recv_timeout(Duration::from_secs(1)).unwrap();
            waiter.join().unwrap();
        }

        #[test]
        fn lane_pool_rejects_zero_capacity() {
            let Err(error) = LanePool::<()>::new(Vec::new()) else {
                panic!("zero-capacity lane pool should be rejected");
            };
            assert!(error.to_string().contains("at least 1"));
        }

        #[test]
        fn invalid_optimization_batch_is_rejected_before_runtime_load() {
            let error = TensorrtBackend::new(
                "not-read.onnx",
                FlatEncoder::new(7, 7),
                TensorrtConfig {
                    opt_batch: 129,
                    max_batch: 128,
                    ..TensorrtConfig::default()
                },
            )
            .err()
            .expect("invalid optimization point should fail");

            assert!(error.to_string().contains("opt_batch must be in 1..=128"));
        }
    }
}

#[cfg(feature = "tensorrt")]
pub use inner::{
    load_trt_libs, parse_eval_results, TensorrtBackend, TensorrtConfig, TrtTimingInfo,
};
