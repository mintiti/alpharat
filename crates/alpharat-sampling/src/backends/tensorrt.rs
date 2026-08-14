#[cfg(feature = "tensorrt")]
mod inner {
    use crate::encoder::ObservationEncoder;
    use alpharat_mcts::{Backend, BackendError, EvalResult};
    use pyrat::GameState;
    use sha2::{Digest, Sha256};
    use std::ffi::{c_void, CString};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Condvar, Mutex, MutexGuard};
    use std::time::Instant;

    // -----------------------------------------------------------------------
    // CUDA FFI — thin declarations for GPU memory management
    // -----------------------------------------------------------------------

    extern "C" {
        fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
        fn cudaFree(ptr: *mut c_void) -> i32;
        fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> i32;
        fn cudaFreeHost(ptr: *mut c_void) -> i32;
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
        fn cudaEventCreate(event: *mut *mut c_void) -> i32;
        fn cudaEventDestroy(event: *mut c_void) -> i32;
        fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
        fn cudaEventSynchronize(event: *mut c_void) -> i32;
        fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
        fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
        fn cudaGetErrorString(error: i32) -> *const std::ffi::c_char;
        fn cudaGetErrorName(error: i32) -> *const std::ffi::c_char;
    }

    const CUDA_MEMCPY_H2D: i32 = 1;
    const CUDA_MEMCPY_D2H: i32 = 2;
    const CUDA_STREAM_NON_BLOCKING: u32 = 1;

    type OwnedTrtOutputs = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    type TrtOutputSlices<'a> = (&'a [f32], &'a [f32], &'a [f32], &'a [f32]);

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

    /// Host-memory lifetime used by one TensorRT execution session.
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub enum TrtHostIoMode {
        /// Preserve the existing per-call pageable vectors and stage barriers.
        #[default]
        Pageable,
        /// Reuse one page-locked input/output slot and wait once after D2H.
        Pinned,
    }

    impl std::fmt::Display for TrtHostIoMode {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Pageable => f.write_str("pageable"),
                Self::Pinned => f.write_str("pinned"),
            }
        }
    }

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
        /// Experiment-only host I/O lifetime. Defaults to the existing pageable path.
        pub host_io: TrtHostIoMode,
        /// Record CUDA-event and host-stage timing for production calls.
        pub profile_stages: bool,
    }

    impl Default for TensorrtConfig {
        fn default() -> Self {
            Self {
                opt_batch: 256,
                max_batch: 256,
                cache_dir: None,
                execution_contexts: 1,
                cuda_graphs: false,
                host_io: TrtHostIoMode::Pageable,
                profile_stages: false,
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

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct OutputLayout {
        policy_p1: usize,
        policy_p2: usize,
        value_p1: usize,
        value_p2: usize,
        total: usize,
    }

    impl OutputLayout {
        fn new(max_batch: usize) -> Self {
            Self {
                policy_p1: 0,
                policy_p2: max_batch * 5,
                value_p1: max_batch * 10,
                value_p2: max_batch * 11,
                total: max_batch * 12,
            }
        }
    }

    struct PinnedAllocation {
        ptr: *mut c_void,
        len_f32: usize,
    }

    // SAFETY: the allocation is exclusively owned by one leased TrtSession.
    unsafe impl Send for PinnedAllocation {}

    impl PinnedAllocation {
        fn alloc(len_f32: usize, label: &str) -> Result<Self, BackendError> {
            let mut ptr = std::ptr::null_mut();
            cuda_check(
                unsafe {
                    cudaMallocHost(
                        &mut ptr,
                        len_f32
                            .checked_mul(std::mem::size_of::<f32>())
                            .ok_or_else(|| BackendError::msg("pinned allocation size overflow"))?,
                    )
                },
                label,
            )?;
            Ok(Self { ptr, len_f32 })
        }

        fn as_mut_slice(&mut self, len: usize) -> &mut [f32] {
            assert!(len <= self.len_f32);
            unsafe { std::slice::from_raw_parts_mut(self.ptr.cast::<f32>(), len) }
        }

        fn ptr_at(&self, offset: usize) -> *mut c_void {
            assert!(offset <= self.len_f32);
            unsafe { self.ptr.cast::<f32>().add(offset).cast::<c_void>() }
        }

        fn slice_at(&self, offset: usize, len: usize) -> &[f32] {
            assert!(offset + len <= self.len_f32);
            unsafe { std::slice::from_raw_parts(self.ptr.cast::<f32>().add(offset), len) }
        }

        fn bytes(&self) -> usize {
            self.len_f32 * std::mem::size_of::<f32>()
        }
    }

    impl Drop for PinnedAllocation {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe {
                    cudaFreeHost(self.ptr);
                }
            }
        }
    }

    struct PinnedHostBuffers {
        input: PinnedAllocation,
        output: PinnedAllocation,
        layout: OutputLayout,
        max_batch: usize,
        obs_dim: usize,
    }

    impl PinnedHostBuffers {
        fn alloc(max_batch: usize, obs_dim: usize) -> Result<Self, BackendError> {
            let layout = OutputLayout::new(max_batch);
            Ok(Self {
                input: PinnedAllocation::alloc(max_batch * obs_dim, "cudaMallocHost(input)")?,
                output: PinnedAllocation::alloc(layout.total, "cudaMallocHost(output)")?,
                layout,
                max_batch,
                obs_dim,
            })
        }

        fn input_mut(&mut self, n: usize) -> &mut [f32] {
            assert!(n <= self.max_batch);
            self.input.as_mut_slice(n * self.obs_dim)
        }

        fn input_ptr(&self) -> *const c_void {
            self.input.ptr.cast_const()
        }

        fn output_ptrs(&self) -> [*mut c_void; 4] {
            [
                self.output.ptr_at(self.layout.policy_p1),
                self.output.ptr_at(self.layout.policy_p2),
                self.output.ptr_at(self.layout.value_p1),
                self.output.ptr_at(self.layout.value_p2),
            ]
        }

        fn output_slices(&self, n: usize) -> TrtOutputSlices<'_> {
            assert!(n <= self.max_batch);
            (
                self.output.slice_at(self.layout.policy_p1, n * 5),
                self.output.slice_at(self.layout.policy_p2, n * 5),
                self.output.slice_at(self.layout.value_p1, n),
                self.output.slice_at(self.layout.value_p2, n),
            )
        }

        fn bytes(&self) -> usize {
            self.input.bytes() + self.output.bytes()
        }
    }

    struct CudaEvent {
        handle: *mut c_void,
    }

    // SAFETY: events are only recorded and queried by their leased session.
    unsafe impl Send for CudaEvent {}

    impl CudaEvent {
        fn new() -> Result<Self, BackendError> {
            let mut handle = std::ptr::null_mut();
            cuda_check(unsafe { cudaEventCreate(&mut handle) }, "cudaEventCreate")?;
            Ok(Self { handle })
        }

        fn record(&self, stream: *mut c_void, label: &str) -> Result<(), BackendError> {
            cuda_check(unsafe { cudaEventRecord(self.handle, stream) }, label)
        }

        fn synchronize(&self, label: &str) -> Result<(), BackendError> {
            cuda_check(unsafe { cudaEventSynchronize(self.handle) }, label)
        }

        fn elapsed_us(&self, end: &Self, label: &str) -> Result<f64, BackendError> {
            let mut milliseconds = 0.0_f32;
            cuda_check(
                unsafe { cudaEventElapsedTime(&mut milliseconds, self.handle, end.handle) },
                label,
            )?;
            Ok(f64::from(milliseconds) * 1_000.0)
        }
    }

    impl Drop for CudaEvent {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe {
                    cudaEventDestroy(self.handle);
                }
            }
        }
    }

    struct StageEvents {
        h2d_start: CudaEvent,
        h2d_end: CudaEvent,
        infer_start: CudaEvent,
        infer_end: CudaEvent,
        d2h_start: CudaEvent,
        d2h_end: CudaEvent,
    }

    impl StageEvents {
        fn new() -> Result<Self, BackendError> {
            Ok(Self {
                h2d_start: CudaEvent::new()?,
                h2d_end: CudaEvent::new()?,
                infer_start: CudaEvent::new()?,
                infer_end: CudaEvent::new()?,
                d2h_start: CudaEvent::new()?,
                d2h_end: CudaEvent::new()?,
            })
        }

        fn timing(&self) -> Result<TrtTimingInfo, BackendError> {
            Ok(TrtTimingInfo {
                h2d_us: self
                    .h2d_start
                    .elapsed_us(&self.h2d_end, "cudaEventElapsedTime(H2D)")?,
                infer_us: self
                    .infer_start
                    .elapsed_us(&self.infer_end, "cudaEventElapsedTime(inference)")?,
                d2h_us: self
                    .d2h_start
                    .elapsed_us(&self.d2h_end, "cudaEventElapsedTime(D2H)")?,
                ..TrtTimingInfo::default()
            })
        }
    }

    struct StreamFlight {
        stream: *mut c_void,
        armed: bool,
    }

    impl StreamFlight {
        fn new(stream: *mut c_void) -> Self {
            Self {
                stream,
                armed: true,
            }
        }

        fn finish(&mut self) {
            self.armed = false;
        }
    }

    impl Drop for StreamFlight {
        fn drop(&mut self) {
            if self.armed {
                unsafe {
                    cudaStreamSynchronize(self.stream);
                }
            }
        }
    }

    /// Host and CUDA-device timing for one TensorRT call.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct TrtTimingInfo {
        pub encode_us: f64,
        pub input_stage_us: f64,
        pub h2d_us: f64,
        pub infer_us: f64,
        pub output_alloc_us: f64,
        pub d2h_us: f64,
        pub parse_us: f64,
        pub total_us: f64,
    }

    /// Cumulative stage telemetry for production TensorRT calls.
    pub struct TrtStats {
        host_io: TrtHostIoMode,
        pinned_bytes: usize,
        calls: AtomicU64,
        positions: AtomicU64,
        encode_ns: AtomicU64,
        input_stage_ns: AtomicU64,
        h2d_ns: AtomicU64,
        infer_ns: AtomicU64,
        output_alloc_ns: AtomicU64,
        d2h_ns: AtomicU64,
        parse_ns: AtomicU64,
        total_ns: AtomicU64,
    }

    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct TrtStatsSnapshot {
        pub host_io: TrtHostIoMode,
        pub pinned_bytes: usize,
        pub calls: u64,
        pub positions: u64,
        pub encode_ns: u64,
        pub input_stage_ns: u64,
        pub h2d_ns: u64,
        pub infer_ns: u64,
        pub output_alloc_ns: u64,
        pub d2h_ns: u64,
        pub parse_ns: u64,
        pub total_ns: u64,
    }

    impl TrtStats {
        fn new(host_io: TrtHostIoMode, pinned_bytes: usize) -> Self {
            Self {
                host_io,
                pinned_bytes,
                calls: AtomicU64::new(0),
                positions: AtomicU64::new(0),
                encode_ns: AtomicU64::new(0),
                input_stage_ns: AtomicU64::new(0),
                h2d_ns: AtomicU64::new(0),
                infer_ns: AtomicU64::new(0),
                output_alloc_ns: AtomicU64::new(0),
                d2h_ns: AtomicU64::new(0),
                parse_ns: AtomicU64::new(0),
                total_ns: AtomicU64::new(0),
            }
        }

        fn record(&self, positions: usize, timing: &TrtTimingInfo) {
            let ns = |microseconds: f64| (microseconds.max(0.0) * 1_000.0).round() as u64;
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.positions
                .fetch_add(positions as u64, Ordering::Relaxed);
            self.encode_ns
                .fetch_add(ns(timing.encode_us), Ordering::Relaxed);
            self.input_stage_ns
                .fetch_add(ns(timing.input_stage_us), Ordering::Relaxed);
            self.h2d_ns.fetch_add(ns(timing.h2d_us), Ordering::Relaxed);
            self.infer_ns
                .fetch_add(ns(timing.infer_us), Ordering::Relaxed);
            self.output_alloc_ns
                .fetch_add(ns(timing.output_alloc_us), Ordering::Relaxed);
            self.d2h_ns.fetch_add(ns(timing.d2h_us), Ordering::Relaxed);
            self.parse_ns
                .fetch_add(ns(timing.parse_us), Ordering::Relaxed);
            self.total_ns
                .fetch_add(ns(timing.total_us), Ordering::Relaxed);
        }

        pub fn snapshot(&self) -> TrtStatsSnapshot {
            TrtStatsSnapshot {
                host_io: self.host_io,
                pinned_bytes: self.pinned_bytes,
                calls: self.calls.load(Ordering::Relaxed),
                positions: self.positions.load(Ordering::Relaxed),
                encode_ns: self.encode_ns.load(Ordering::Relaxed),
                input_stage_ns: self.input_stage_ns.load(Ordering::Relaxed),
                h2d_ns: self.h2d_ns.load(Ordering::Relaxed),
                infer_ns: self.infer_ns.load(Ordering::Relaxed),
                output_alloc_ns: self.output_alloc_ns.load(Ordering::Relaxed),
                d2h_ns: self.d2h_ns.load(Ordering::Relaxed),
                parse_ns: self.parse_ns.load(Ordering::Relaxed),
                total_ns: self.total_ns.load(Ordering::Relaxed),
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
        pinned: Option<PinnedHostBuffers>,
        events: StageEvents,
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
                // Host and device buffers are fields dropped after this method.
                // Quiesce first so no asynchronous work can outlive them.
                cudaStreamSynchronize(self.stream);
                trt_destroy_session(self.handle);
                cudaStreamDestroy(self.stream);
            }
        }
    }

    impl TrtSession {
        fn new(
            engine: Arc<TrtEngine>,
            obs_dim: usize,
            max_batch: usize,
            cuda_graphs: bool,
            host_io: TrtHostIoMode,
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
                host_io,
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
            host_io: TrtHostIoMode,
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

            // Lifecycle resources are created once, outside every timed call.
            let pinned = match host_io {
                TrtHostIoMode::Pageable => None,
                TrtHostIoMode::Pinned => Some(PinnedHostBuffers::alloc(max_batch, obs_dim)?),
            };
            let events = StageEvents::new()?;

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
                pinned,
                events,
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

        fn validate_input(&self, input_len: usize, n: usize) -> Result<(), BackendError> {
            if n > self.max_batch {
                return Err(BackendError::msg(format!(
                    "batch size {n} exceeds max_batch {}",
                    self.max_batch
                )));
            }
            if input_len != n * self.obs_dim {
                return Err(BackendError::msg(format!(
                    "input length {input_len} != expected {}",
                    n * self.obs_dim
                )));
            }
            Ok(())
        }

        fn set_input_shape(&self, n: usize) -> Result<(), BackendError> {
            let obs_name = CString::new(crate::TENSOR_INPUT).unwrap();
            let shape = [n as i64, self.obs_dim as i64];
            let rc =
                unsafe { trt_set_input_shape(self.handle, obs_name.as_ptr(), 2, shape.as_ptr()) };
            if rc != 0 {
                return Err(BackendError::msg(format!(
                    "Failed to set input shape for batch size {n} (rc={rc})"
                )));
            }
            Ok(())
        }

        fn copy_control(
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

        fn copy_async(
            &self,
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: i32,
            op: &str,
        ) -> Result<(), BackendError> {
            cuda_check(
                unsafe { cudaMemcpyAsync(dst, src, count, kind, self.stream) },
                op,
            )
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

        /// Preserve the existing pageable allocation and synchronization path.
        fn infer_pageable(
            &mut self,
            input: &[f32],
            n: usize,
            profile_stages: bool,
        ) -> Result<(OwnedTrtOutputs, TrtTimingInfo), BackendError> {
            self.validate_input(input.len(), n)?;
            self.set_input_shape(n)?;
            let f = std::mem::size_of::<f32>();
            let total_start = Instant::now();
            let mut timing = TrtTimingInfo::default();

            if profile_stages {
                self.events
                    .h2d_start
                    .record(self.stream, "cudaEventRecord(H2D start)")?;
                self.copy_async(
                    self.buffers.d_input,
                    input.as_ptr().cast::<c_void>(),
                    n * self.obs_dim * f,
                    CUDA_MEMCPY_H2D,
                    "input H2D async",
                )?;
                self.events
                    .h2d_end
                    .record(self.stream, "cudaEventRecord(H2D end)")?;
                // Match the control's existing H2D-before-enqueue barrier.
                self.events
                    .h2d_end
                    .synchronize("cudaEventSynchronize(H2D)")?;
                self.events
                    .infer_start
                    .record(self.stream, "cudaEventRecord(inference start)")?;
            } else {
                self.copy_control(
                    self.buffers.d_input,
                    input.as_ptr().cast::<c_void>(),
                    n * self.obs_dim * f,
                    CUDA_MEMCPY_H2D,
                    "input H2D",
                )?;
            }

            let rc = unsafe { trt_enqueue_v3(self.handle, self.stream) };
            if rc != 0 {
                return Err(BackendError::msg(format!(
                    "TRT enqueue_v3 failed (rc={rc})"
                )));
            }
            if profile_stages {
                self.events
                    .infer_end
                    .record(self.stream, "cudaEventRecord(inference end)")?;
                self.events
                    .infer_end
                    .synchronize("cudaEventSynchronize(inference)")?;
            } else {
                cuda_check(
                    unsafe { cudaStreamSynchronize(self.stream) },
                    "cudaStreamSynchronize",
                )?;
            }

            let alloc_start = Instant::now();
            let mut pp1 = vec![0.0f32; n * 5];
            let mut pp2 = vec![0.0f32; n * 5];
            let mut v1 = vec![0.0f32; n];
            let mut v2 = vec![0.0f32; n];
            timing.output_alloc_us = alloc_start.elapsed().as_secs_f64() * 1_000_000.0;

            if profile_stages {
                self.events
                    .d2h_start
                    .record(self.stream, "cudaEventRecord(D2H start)")?;
                self.copy_async(
                    pp1.as_mut_ptr().cast::<c_void>(),
                    self.buffers.d_policy_p1,
                    n * 5 * f,
                    CUDA_MEMCPY_D2H,
                    "policy_p1 D2H async",
                )?;
                self.copy_async(
                    pp2.as_mut_ptr().cast::<c_void>(),
                    self.buffers.d_policy_p2,
                    n * 5 * f,
                    CUDA_MEMCPY_D2H,
                    "policy_p2 D2H async",
                )?;
                self.copy_async(
                    v1.as_mut_ptr().cast::<c_void>(),
                    self.buffers.d_value_p1,
                    n * f,
                    CUDA_MEMCPY_D2H,
                    "value_p1 D2H async",
                )?;
                self.copy_async(
                    v2.as_mut_ptr().cast::<c_void>(),
                    self.buffers.d_value_p2,
                    n * f,
                    CUDA_MEMCPY_D2H,
                    "value_p2 D2H async",
                )?;
                self.events
                    .d2h_end
                    .record(self.stream, "cudaEventRecord(D2H end)")?;
                self.events
                    .d2h_end
                    .synchronize("cudaEventSynchronize(D2H)")?;
                let device = self.events.timing()?;
                timing.h2d_us = device.h2d_us;
                timing.infer_us = device.infer_us;
                timing.d2h_us = device.d2h_us;
            } else {
                self.copy_control(
                    pp1.as_mut_ptr().cast::<c_void>(),
                    self.buffers.d_policy_p1,
                    n * 5 * f,
                    CUDA_MEMCPY_D2H,
                    "policy_p1 D2H",
                )?;
                self.copy_control(
                    pp2.as_mut_ptr().cast::<c_void>(),
                    self.buffers.d_policy_p2,
                    n * 5 * f,
                    CUDA_MEMCPY_D2H,
                    "policy_p2 D2H",
                )?;
                self.copy_control(
                    v1.as_mut_ptr().cast::<c_void>(),
                    self.buffers.d_value_p1,
                    n * f,
                    CUDA_MEMCPY_D2H,
                    "value_p1 D2H",
                )?;
                self.copy_control(
                    v2.as_mut_ptr().cast::<c_void>(),
                    self.buffers.d_value_p2,
                    n * f,
                    CUDA_MEMCPY_D2H,
                    "value_p2 D2H",
                )?;
                self.finish_output_copies()?;
            }

            timing.total_us = total_start.elapsed().as_secs_f64() * 1_000_000.0;
            Ok(((pp1, pp2, v1, v2), timing))
        }

        fn pinned_input_mut(&mut self, n: usize) -> Result<&mut [f32], BackendError> {
            self.validate_input(n * self.obs_dim, n)?;
            self.pinned
                .as_mut()
                .map(|buffers| buffers.input_mut(n))
                .ok_or_else(|| BackendError::msg("pinned input requested on pageable session"))
        }

        fn infer_pinned_prepared(
            &mut self,
            n: usize,
            profile_stages: bool,
        ) -> Result<TrtTimingInfo, BackendError> {
            self.validate_input(n * self.obs_dim, n)?;
            self.set_input_shape(n)?;
            let f = std::mem::size_of::<f32>();
            let total_start = Instant::now();
            let pinned = self.pinned.as_ref().ok_or_else(|| {
                BackendError::msg("pinned inference requested on pageable session")
            })?;
            let input_ptr = pinned.input_ptr();
            let [pp1_ptr, pp2_ptr, v1_ptr, v2_ptr] = pinned.output_ptrs();
            let mut flight = StreamFlight::new(self.stream);

            if profile_stages {
                self.events
                    .h2d_start
                    .record(self.stream, "cudaEventRecord(H2D start)")?;
            }
            self.copy_async(
                self.buffers.d_input,
                input_ptr,
                n * self.obs_dim * f,
                CUDA_MEMCPY_H2D,
                "pinned input H2D async",
            )?;
            if profile_stages {
                self.events
                    .h2d_end
                    .record(self.stream, "cudaEventRecord(H2D end)")?;
                self.events
                    .infer_start
                    .record(self.stream, "cudaEventRecord(inference start)")?;
            }

            let rc = unsafe { trt_enqueue_v3(self.handle, self.stream) };
            if rc != 0 {
                return Err(BackendError::msg(format!(
                    "TRT enqueue_v3 failed (rc={rc})"
                )));
            }
            if profile_stages {
                self.events
                    .infer_end
                    .record(self.stream, "cudaEventRecord(inference end)")?;
                self.events
                    .d2h_start
                    .record(self.stream, "cudaEventRecord(D2H start)")?;
            }

            self.copy_async(
                pp1_ptr,
                self.buffers.d_policy_p1,
                n * 5 * f,
                CUDA_MEMCPY_D2H,
                "pinned policy_p1 D2H async",
            )?;
            self.copy_async(
                pp2_ptr,
                self.buffers.d_policy_p2,
                n * 5 * f,
                CUDA_MEMCPY_D2H,
                "pinned policy_p2 D2H async",
            )?;
            self.copy_async(
                v1_ptr,
                self.buffers.d_value_p1,
                n * f,
                CUDA_MEMCPY_D2H,
                "pinned value_p1 D2H async",
            )?;
            self.copy_async(
                v2_ptr,
                self.buffers.d_value_p2,
                n * f,
                CUDA_MEMCPY_D2H,
                "pinned value_p2 D2H async",
            )?;
            self.events
                .d2h_end
                .record(self.stream, "cudaEventRecord(completion)")?;
            self.events
                .d2h_end
                .synchronize("cudaEventSynchronize(completion)")?;

            let mut timing = if profile_stages {
                self.events.timing()?
            } else {
                TrtTimingInfo::default()
            };
            timing.total_us = total_start.elapsed().as_secs_f64() * 1_000_000.0;
            flight.finish();
            Ok(timing)
        }

        fn pinned_output_slices(&self, n: usize) -> Result<TrtOutputSlices<'_>, BackendError> {
            self.pinned
                .as_ref()
                .map(|buffers| buffers.output_slices(n))
                .ok_or_else(|| BackendError::msg("pinned output requested on pageable session"))
        }

        fn pinned_bytes(&self) -> usize {
            self.pinned.as_ref().map_or(0, PinnedHostBuffers::bytes)
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
        host_io: TrtHostIoMode,
        profile_stages: bool,
        stats: Arc<TrtStats>,
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
            if config.host_io == TrtHostIoMode::Pinned && config.cuda_graphs {
                return Err(BackendError::msg(
                    "the pinned-I/O experiment requires CUDA graphs to remain off",
                ));
            }
            if config.host_io == TrtHostIoMode::Pinned && config.execution_contexts != 1 {
                return Err(BackendError::msg(
                    "the pinned-I/O experiment requires exactly one execution context",
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
                    config.host_io,
                    index == 0,
                )?);
            }
            let pinned_bytes = sessions.iter().map(TrtSession::pinned_bytes).sum();
            let sessions = LanePool::new(sessions)?;
            eprintln!(
                "[TensorRT] Profile: MIN=1, OPT={}, MAX={}; execution contexts: {}; CUDA graphs requested: {}; host I/O: {}; pinned bytes: {}; stage profiling: {}",
                config.opt_batch,
                config.max_batch,
                config.execution_contexts,
                config.cuda_graphs,
                config.host_io,
                pinned_bytes,
                config.profile_stages,
            );
            let stats = Arc::new(TrtStats::new(config.host_io, pinned_bytes));

            Ok(Self {
                sessions,
                encoder,
                host_io: config.host_io,
                profile_stages: config.profile_stages,
                stats,
            })
        }

        pub fn stats(&self) -> &Arc<TrtStats> {
            &self.stats
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

    impl<E: ObservationEncoder> TensorrtBackend<E> {
        /// Run inference on pre-encoded observations with per-phase timing.
        ///
        /// Unlike `evaluate_batch` (which encodes game states), this takes a
        /// flat f32 buffer that's already encoded. It deliberately calls the
        /// same session methods as production and only adds input staging plus
        /// output parsing around them.
        pub fn evaluate_encoded_timed(
            &self,
            encoded: &[f32],
            n: usize,
        ) -> Result<(Vec<EvalResult>, TrtTimingInfo), BackendError> {
            let call_start = Instant::now();
            let mut lease = self.sessions.lease()?;
            let session = lease.get_mut();
            session.validate_input(encoded.len(), n)?;

            let (results, mut timing) = match self.host_io {
                TrtHostIoMode::Pageable => {
                    let ((pp1, pp2, v1, v2), mut timing) =
                        session.infer_pageable(encoded, n, true)?;
                    let parse_start = Instant::now();
                    let results = parse_eval_results(&pp1, &pp2, &v1, &v2, n)?;
                    timing.parse_us = parse_start.elapsed().as_secs_f64() * 1_000_000.0;
                    (results, timing)
                }
                TrtHostIoMode::Pinned => {
                    let stage_start = Instant::now();
                    session.pinned_input_mut(n)?.copy_from_slice(encoded);
                    let input_stage_us = stage_start.elapsed().as_secs_f64() * 1_000_000.0;
                    let mut timing = session.infer_pinned_prepared(n, true)?;
                    timing.input_stage_us = input_stage_us;
                    let parse_start = Instant::now();
                    let (pp1, pp2, v1, v2) = session.pinned_output_slices(n)?;
                    let results = parse_eval_results(pp1, pp2, v1, v2, n)?;
                    timing.parse_us = parse_start.elapsed().as_secs_f64() * 1_000_000.0;
                    (results, timing)
                }
            };
            timing.total_us = call_start.elapsed().as_secs_f64() * 1_000_000.0;
            self.stats.record(n, &timing);
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
            let call_start = Instant::now();

            let (results, mut timing) = match self.host_io {
                TrtHostIoMode::Pageable => {
                    let allocation_start = Instant::now();
                    let mut buf = vec![0.0f32; n * obs_dim];
                    let input_stage_us = allocation_start.elapsed().as_secs_f64() * 1_000_000.0;
                    let encode_start = Instant::now();
                    for (i, game) in games.iter().enumerate() {
                        self.encoder.encode_into(game, &mut buf, i * obs_dim);
                    }
                    let encode_us = encode_start.elapsed().as_secs_f64() * 1_000_000.0;

                    let mut lease = self.sessions.lease()?;
                    let ((pp1, pp2, v1, v2), mut timing) =
                        lease
                            .get_mut()
                            .infer_pageable(&buf, n, self.profile_stages)?;
                    timing.input_stage_us = input_stage_us;
                    timing.encode_us = encode_us;
                    let parse_start = Instant::now();
                    let results = parse_eval_results(&pp1, &pp2, &v1, &v2, n)?;
                    timing.parse_us = parse_start.elapsed().as_secs_f64() * 1_000_000.0;
                    (results, timing)
                }
                TrtHostIoMode::Pinned => {
                    let mut lease = self.sessions.lease()?;
                    let session = lease.get_mut();
                    let encode_start = Instant::now();
                    {
                        let input = session.pinned_input_mut(n)?;
                        for (i, game) in games.iter().enumerate() {
                            self.encoder.encode_into(game, input, i * obs_dim);
                        }
                    }
                    let encode_us = encode_start.elapsed().as_secs_f64() * 1_000_000.0;
                    let mut timing = session.infer_pinned_prepared(n, self.profile_stages)?;
                    timing.encode_us = encode_us;
                    let parse_start = Instant::now();
                    let (pp1, pp2, v1, v2) = session.pinned_output_slices(n)?;
                    let results = parse_eval_results(pp1, pp2, v1, v2, n)?;
                    timing.parse_us = parse_start.elapsed().as_secs_f64() * 1_000_000.0;
                    (results, timing)
                }
            };

            timing.total_us = call_start.elapsed().as_secs_f64() * 1_000_000.0;
            if self.profile_stages {
                self.stats.record(n, &timing);
            }
            Ok(results)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{LanePool, OutputLayout, TensorrtBackend, TensorrtConfig, TrtHostIoMode};
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

        #[test]
        fn pinned_output_layout_uses_disjoint_max_batch_regions() {
            let layout = OutputLayout::new(128);
            assert_eq!(layout.policy_p1, 0);
            assert_eq!(layout.policy_p2, 640);
            assert_eq!(layout.value_p1, 1280);
            assert_eq!(layout.value_p2, 1408);
            assert_eq!(layout.total, 1536);
        }

        #[test]
        fn pinned_experiment_rejects_graphs_before_runtime_load() {
            let error = TensorrtBackend::new(
                "not-read.onnx",
                FlatEncoder::new(7, 7),
                TensorrtConfig {
                    host_io: TrtHostIoMode::Pinned,
                    cuda_graphs: true,
                    ..TensorrtConfig::default()
                },
            )
            .err()
            .expect("pinned host I/O plus graphs should fail");

            assert!(error
                .to_string()
                .contains("requires CUDA graphs to remain off"));
        }

        #[test]
        fn pinned_experiment_rejects_multiple_contexts_before_runtime_load() {
            let error = TensorrtBackend::new(
                "not-read.onnx",
                FlatEncoder::new(7, 7),
                TensorrtConfig {
                    host_io: TrtHostIoMode::Pinned,
                    execution_contexts: 2,
                    ..TensorrtConfig::default()
                },
            )
            .err()
            .expect("pinned host I/O plus multiple contexts should fail");

            assert!(error.to_string().contains("exactly one execution context"));
        }
    }
}

#[cfg(feature = "tensorrt")]
pub use inner::{
    load_trt_libs, parse_eval_results, TensorrtBackend, TensorrtConfig, TrtHostIoMode, TrtStats,
    TrtStatsSnapshot, TrtTimingInfo,
};
