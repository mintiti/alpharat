#[cfg(feature = "tensorrt")]
mod inner {
    use crate::encoder::ObservationEncoder;
    use alpharat_mcts::{Backend, EvalResult};
    use pyrat::GameState;
    use sha2::{Digest, Sha256};
    use std::ffi::{c_void, CString};
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use std::{fs, io};

    // -----------------------------------------------------------------------
    // CUDA FFI — thin declarations for GPU memory management
    // -----------------------------------------------------------------------

    extern "C" {
        fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
        fn cudaFree(ptr: *mut c_void) -> i32;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
        fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
        fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
        fn cudaStreamDestroy(stream: *mut c_void) -> i32;
        fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    }

    const CUDA_MEMCPY_H2D: i32 = 1;
    const CUDA_MEMCPY_D2H: i32 = 2;

    fn cuda_check(code: i32, op: &str) {
        assert_eq!(code, 0, "CUDA {op} failed with error code {code}");
    }

    fn compute_capability() -> (i32, i32) {
        let (mut major, mut minor) = (0i32, 0i32);
        // cudaDevAttrComputeCapabilityMajor = 75, Minor = 76
        cuda_check(
            unsafe { cudaDeviceGetAttribute(&mut major, 75, 0) },
            "cudaDeviceGetAttribute(compute_major)",
        );
        cuda_check(
            unsafe { cudaDeviceGetAttribute(&mut minor, 76, 0) },
            "cudaDeviceGetAttribute(compute_minor)",
        );
        (major, minor)
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

        fn trt_create_session(engine_data: *const c_void, engine_len: usize) -> *mut c_void;
        fn trt_destroy_session(handle: *mut c_void);

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

        fn trt_load_libs(
            trt_lib_path: *const i8,
            parser_lib_path: *const i8,
        ) -> i32;
    }

    /// Load TRT-RTX shared libraries via dlopen (idempotent).
    /// Called automatically by `TensorrtBackend::new`. Also available
    /// for standalone Rust binaries that don't have Python preloading.
    pub fn load_trt_libs() {
        let rc = unsafe { trt_load_libs(std::ptr::null(), std::ptr::null()) };
        assert_eq!(
            rc, 0,
            "Failed to load TensorRT-RTX libraries — \
             is $TENSORRT_RTX_ROOT/lib in LD_LIBRARY_PATH?"
        );
    }

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    /// Configuration for the TensorRT-RTX backend.
    pub struct TensorrtConfig {
        /// Maximum batch size. GPU buffers are pre-allocated for this size.
        pub max_batch: usize,
        /// Directory for cached serialized engines. `None` disables caching.
        pub cache_dir: Option<PathBuf>,
    }

    impl Default for TensorrtConfig {
        fn default() -> Self {
            Self {
                max_batch: 256,
                cache_dir: None,
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
        fn alloc(max_batch: usize, obs_dim: usize) -> Self {
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
                );
                cuda_check(
                    cudaMalloc(&mut b.d_policy_p1, max_batch * 5 * f),
                    "cudaMalloc(policy_p1)",
                );
                cuda_check(
                    cudaMalloc(&mut b.d_policy_p2, max_batch * 5 * f),
                    "cudaMalloc(policy_p2)",
                );
                cuda_check(
                    cudaMalloc(&mut b.d_value_p1, max_batch * f),
                    "cudaMalloc(value_p1)",
                );
                cuda_check(
                    cudaMalloc(&mut b.d_value_p2, max_batch * f),
                    "cudaMalloc(value_p2)",
                );
            }
            b
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

    fn cache_key(onnx_bytes: &[u8], max_batch: usize) -> String {
        let (major, minor) = compute_capability();
        let hash = Sha256::digest(onnx_bytes);
        let hash_hex: String = hash.iter().take(8).map(|b| format!("{b:02x}")).collect();
        format!("sm{major}{minor}_{hash_hex}_{max_batch}.engine")
    }

    fn try_load_cache(cache_dir: &Path, key: &str) -> Option<Vec<u8>> {
        fs::read(cache_dir.join(key)).ok()
    }

    fn save_cache(cache_dir: &Path, key: &str, data: &[u8]) {
        let _ = fs::create_dir_all(cache_dir);
        let path = cache_dir.join(key);
        if let Err(e) = fs::write(&path, data) {
            eprintln!(
                "[TensorRT] Warning: failed to cache engine to {}: {e}",
                path.display()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Engine building (via C++ shim)
    // -----------------------------------------------------------------------

    /// Build a serialized TensorRT engine from ONNX bytes, with optimization
    /// profiles for dynamic batch sizes.
    fn build_engine(onnx_bytes: &[u8], max_batch: usize) -> Vec<u8> {
        let mut out_data: *mut c_void = std::ptr::null_mut();
        let mut out_len: usize = 0;

        let rc = unsafe {
            trt_build_engine(
                onnx_bytes.as_ptr() as *const c_void,
                onnx_bytes.len(),
                1,                 // min_batch
                max_batch as i32,  // opt_batch
                max_batch as i32,  // max_batch
                256, // workspace MB
                &mut out_data,
                &mut out_len,
            )
        };

        assert!(
            rc == 0 && !out_data.is_null(),
            "TRT engine build failed (error code {rc})"
        );

        let data = unsafe { std::slice::from_raw_parts(out_data as *const u8, out_len) }.to_vec();
        unsafe { trt_free_buffer(out_data) };

        eprintln!("[TensorRT] Engine built ({} bytes)", data.len());
        data
    }

    // -----------------------------------------------------------------------
    // TrtSession — owns TRT context + CUDA resources
    // -----------------------------------------------------------------------

    struct TrtSession {
        handle: *mut c_void, // opaque TrtSession from C++ shim
        stream: *mut c_void,
        buffers: GpuBuffers,
        max_batch: usize,
        obs_dim: usize,
    }

    // SAFETY: All mutable access serialized via Mutex in TensorrtBackend.
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
        fn new(engine_data: &[u8], obs_dim: usize, max_batch: usize) -> Self {
            let handle = unsafe {
                trt_create_session(engine_data.as_ptr() as *const c_void, engine_data.len())
            };
            assert!(!handle.is_null(), "Failed to create TRT session");

            // Log IO tensors
            let n_io = unsafe { trt_get_nb_io_tensors(handle) };
            let names: Vec<String> = (0..n_io)
                .map(|i| {
                    let ptr = unsafe { trt_get_tensor_name(handle, i) };
                    if ptr.is_null() {
                        "<null>".to_string()
                    } else {
                        unsafe { std::ffi::CStr::from_ptr(ptr) }
                            .to_string_lossy()
                            .into_owned()
                    }
                })
                .collect();
            eprintln!("[TensorRT] IO tensors: {names:?}");

            // Allocate GPU buffers
            let buffers = GpuBuffers::alloc(max_batch, obs_dim);

            // Bind tensor addresses
            let obs_name = CString::new("observation").unwrap();
            let pp1_name = CString::new("policy_p1").unwrap();
            let pp2_name = CString::new("policy_p2").unwrap();
            let vp1_name = CString::new("pred_value_p1").unwrap();
            let vp2_name = CString::new("pred_value_p2").unwrap();

            unsafe {
                assert_eq!(
                    trt_set_tensor_address(handle, obs_name.as_ptr(), buffers.d_input),
                    0,
                    "Failed to bind 'observation' tensor"
                );
                assert_eq!(
                    trt_set_tensor_address(handle, pp1_name.as_ptr(), buffers.d_policy_p1),
                    0,
                    "Failed to bind 'policy_p1' tensor"
                );
                assert_eq!(
                    trt_set_tensor_address(handle, pp2_name.as_ptr(), buffers.d_policy_p2),
                    0,
                    "Failed to bind 'policy_p2' tensor"
                );
                assert_eq!(
                    trt_set_tensor_address(handle, vp1_name.as_ptr(), buffers.d_value_p1),
                    0,
                    "Failed to bind 'pred_value_p1' tensor"
                );
                assert_eq!(
                    trt_set_tensor_address(handle, vp2_name.as_ptr(), buffers.d_value_p2),
                    0,
                    "Failed to bind 'pred_value_p2' tensor"
                );
            }

            // Create dedicated CUDA stream
            let mut stream: *mut c_void = std::ptr::null_mut();
            cuda_check(unsafe { cudaStreamCreate(&mut stream) }, "cudaStreamCreate");

            Self {
                handle,
                stream,
                buffers,
                max_batch,
                obs_dim,
            }
        }

        /// Run inference on encoded observations. Returns (pp1, pp2, v1, v2).
        fn infer(
            &mut self,
            input: &[f32],
            n: usize,
        ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
            assert!(
                n <= self.max_batch,
                "batch size {n} exceeds max_batch {}",
                self.max_batch
            );
            assert_eq!(input.len(), n * self.obs_dim);

            let f = std::mem::size_of::<f32>();

            // Set dynamic input shape for this batch
            let obs_name = CString::new("observation").unwrap();
            let shape = [n as i64, self.obs_dim as i64];
            assert_eq!(
                unsafe {
                    trt_set_input_shape(self.handle, obs_name.as_ptr(), 2, shape.as_ptr())
                },
                0,
                "Failed to set input shape for batch size {n}"
            );

            // H2D: copy observation buffer to GPU
            unsafe {
                cuda_check(
                    cudaMemcpy(
                        self.buffers.d_input,
                        input.as_ptr() as *const c_void,
                        n * self.obs_dim * f,
                        CUDA_MEMCPY_H2D,
                    ),
                    "cudaMemcpy(input H2D)",
                );
            }

            // Run inference
            assert_eq!(
                unsafe { trt_enqueue_v3(self.handle, self.stream) },
                0,
                "TRT enqueue_v3 failed"
            );
            cuda_check(
                unsafe { cudaStreamSynchronize(self.stream) },
                "cudaStreamSynchronize",
            );

            // D2H: copy outputs back
            let mut pp1 = vec![0.0f32; n * 5];
            let mut pp2 = vec![0.0f32; n * 5];
            let mut v1 = vec![0.0f32; n];
            let mut v2 = vec![0.0f32; n];

            unsafe {
                cuda_check(
                    cudaMemcpy(
                        pp1.as_mut_ptr() as *mut c_void,
                        self.buffers.d_policy_p1,
                        n * 5 * f,
                        CUDA_MEMCPY_D2H,
                    ),
                    "cudaMemcpy(policy_p1 D2H)",
                );
                cuda_check(
                    cudaMemcpy(
                        pp2.as_mut_ptr() as *mut c_void,
                        self.buffers.d_policy_p2,
                        n * 5 * f,
                        CUDA_MEMCPY_D2H,
                    ),
                    "cudaMemcpy(policy_p2 D2H)",
                );
                cuda_check(
                    cudaMemcpy(
                        v1.as_mut_ptr() as *mut c_void,
                        self.buffers.d_value_p1,
                        n * f,
                        CUDA_MEMCPY_D2H,
                    ),
                    "cudaMemcpy(value_p1 D2H)",
                );
                cuda_check(
                    cudaMemcpy(
                        v2.as_mut_ptr() as *mut c_void,
                        self.buffers.d_value_p2,
                        n * f,
                        CUDA_MEMCPY_D2H,
                    ),
                    "cudaMemcpy(value_p2 D2H)",
                );
            }

            (pp1, pp2, v1, v2)
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
    /// Thread safety: access to the TRT execution context and GPU buffers
    /// is serialized via `Mutex` (same pattern as `OnnxBackend`).
    pub struct TensorrtBackend<E: ObservationEncoder> {
        session: Mutex<TrtSession>,
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
        ) -> io::Result<Self> {
            load_trt_libs();

            let obs_dim = encoder.obs_dim();
            let onnx_path = model_path.as_ref();

            let onnx_bytes = fs::read(onnx_path).map_err(|e| {
                io::Error::new(
                    e.kind(),
                    format!("Failed to read ONNX model at {}: {e}", onnx_path.display()),
                )
            })?;

            let key = cache_key(&onnx_bytes, config.max_batch);

            // Load cached engine or build from scratch
            let engine_data = match &config.cache_dir {
                Some(dir) => match try_load_cache(dir, &key) {
                    Some(data) => {
                        eprintln!("[TensorRT] Loaded cached engine: {key}");
                        data
                    }
                    None => {
                        eprintln!(
                            "[TensorRT] Building engine from ONNX (this may take 10-30s)..."
                        );
                        let data = build_engine(&onnx_bytes, config.max_batch);
                        save_cache(dir, &key, &data);
                        eprintln!("[TensorRT] Engine cached as {key}");
                        data
                    }
                },
                None => {
                    eprintln!("[TensorRT] Building engine (no cache dir configured)...");
                    build_engine(&onnx_bytes, config.max_batch)
                }
            };

            let session = TrtSession::new(&engine_data, obs_dim, config.max_batch);

            Ok(Self {
                session: Mutex::new(session),
                encoder,
            })
        }
    }

    impl<E: ObservationEncoder> Backend for TensorrtBackend<E> {
        fn evaluate(&self, game: &GameState) -> EvalResult {
            self.evaluate_batch(&[game])[0]
        }

        fn evaluate_batch(&self, games: &[&GameState]) -> Vec<EvalResult> {
            let n = games.len();
            let obs_dim = self.encoder.obs_dim();

            // Encode all game states into a flat buffer
            let mut buf = vec![0.0f32; n * obs_dim];
            for (i, game) in games.iter().enumerate() {
                self.encoder.encode_into(game, &mut buf, i * obs_dim);
            }

            // Run TRT inference
            let mut session = self.session.lock().expect("TRT session lock poisoned");
            let (pp1, pp2, v1, v2) = session.infer(&buf, n);

            // Parse flat outputs into EvalResults
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
                    debug_assert!(
                        result.policy_p1.iter().all(|v| v.is_finite())
                            && result.policy_p2.iter().all(|v| v.is_finite())
                            && result.value_p1.is_finite()
                            && result.value_p2.is_finite(),
                        "TensorRT output contains non-finite values (NaN/Inf) for game {i}"
                    );
                    result
                })
                .collect()
        }
    }
}

#[cfg(feature = "tensorrt")]
pub use inner::{load_trt_libs, TensorrtBackend, TensorrtConfig};
