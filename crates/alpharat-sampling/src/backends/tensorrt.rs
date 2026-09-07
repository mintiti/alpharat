#[cfg(feature = "tensorrt")]
mod inner {
    use crate::encoder::ObservationEncoder;
    use crate::inference_trace as trace;
    use alpharat_mcts::{Backend, BackendError, EvalResult};
    use pyrat::GameState;
    use sha2::{Digest, Sha256};
    use std::cell::{Cell, RefCell};
    use std::ffi::{c_void, CString};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
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
        fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
        fn cudaStreamDestroy(stream: *mut c_void) -> i32;
        fn cudaEventCreate(event: *mut *mut c_void) -> i32;
        fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
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
    const CUDA_EVENT_DISABLE_TIMING: u32 = 2;

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

    fn warn_cuda_drop(code: i32, op: &str) {
        if let Err(error) = cuda_check(code, op) {
            eprintln!("[TensorRT] Warning: {error}");
        }
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

        fn trt_create_session(
            engine_data: *const c_void,
            engine_len: usize,
            whole_graph: i32,
        ) -> *mut c_void;
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
        /// Preserve the legacy per-call pageable vectors and stage barriers.
        Pageable,
        /// Reuse one page-locked input/output slot and wait once after D2H.
        #[default]
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
        /// Defaults to `max_batch` when omitted.
        pub opt_batch: Option<usize>,
        /// Maximum batch size. GPU buffers are pre-allocated for this size.
        pub max_batch: usize,
        /// Keep the TensorRT execution shape at max_batch, padding unused rows.
        /// Requires a model whose rows are independent. Opt-in and pinned-only;
        /// returned outputs and statistics count real rows.
        pub pad_to_max: bool,
        /// Request TensorRT-managed whole-graph capture (the runtime may fall back).
        pub cuda_graph: bool,
        /// Optional strictly increasing fixed execution sizes, ending at max_batch.
        /// Each size owns a context; calls stay serialized and use the smallest fit.
        /// Requires pad_to_max and a row-independent model.
        pub execution_sizes: Vec<usize>,
        /// Independent groups of contexts, each with private streams and buffers.
        /// More than one requires concurrent callers to supply useful work.
        pub execution_lanes: usize,
        /// Serialize GPU submission/completion across lanes while allowing CPU
        /// encoding and result parsing to overlap another lane's device work.
        pub serialize_device: bool,
        /// Directory for cached serialized engines. `None` disables caching.
        pub cache_dir: Option<PathBuf>,
        /// Host I/O lifetime. Pinned is the measured RTX 5090 production default.
        pub host_io: TrtHostIoMode,
        /// Record CUDA-event and host-stage timing for production calls. Timing
        /// events are not allocated or touched when this is false.
        pub profile_stages: bool,
    }

    impl Default for TensorrtConfig {
        fn default() -> Self {
            Self {
                opt_batch: None,
                max_batch: 256,
                pad_to_max: false,
                cuda_graph: false,
                execution_sizes: Vec::new(),
                execution_lanes: 1,
                serialize_device: false,
                cache_dir: None,
                host_io: TrtHostIoMode::Pinned,
                profile_stages: false,
            }
        }
    }

    // -----------------------------------------------------------------------
    // GPU buffer management
    // -----------------------------------------------------------------------

    fn checked_elements(lhs: usize, rhs: usize, label: &str) -> Result<usize, BackendError> {
        lhs.checked_mul(rhs)
            .ok_or_else(|| BackendError::msg(format!("TensorRT {label} element count overflow")))
    }

    fn checked_bytes(elements: usize, label: &str) -> Result<usize, BackendError> {
        checked_elements(elements, std::mem::size_of::<f32>(), label)
    }

    fn elapsed_us(start: Option<Instant>) -> f64 {
        start.map_or(0.0, |start| start.elapsed().as_secs_f64() * 1_000_000.0)
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
        fn new(policy_elements: usize, value_elements: usize) -> Result<Self, BackendError> {
            let value_p1 = policy_elements.checked_mul(2).ok_or_else(|| {
                BackendError::msg("TensorRT output layout policy offset overflow")
            })?;
            let value_p2 = value_p1
                .checked_add(value_elements)
                .ok_or_else(|| BackendError::msg("TensorRT output layout value offset overflow"))?;
            let total = value_p2
                .checked_add(value_elements)
                .ok_or_else(|| BackendError::msg("TensorRT output layout total size overflow"))?;
            Ok(Self {
                policy_p1: 0,
                policy_p2: policy_elements,
                value_p1,
                value_p2,
                total,
            })
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct BatchLayout {
        batch_size: usize,
        input_elements: usize,
        policy_elements: usize,
        value_elements: usize,
        input_bytes: usize,
        policy_bytes: usize,
        value_bytes: usize,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct TensorLayout {
        max_batch: usize,
        obs_dim: usize,
        capacity: BatchLayout,
        output: OutputLayout,
        output_bytes: usize,
        pinned_bytes: usize,
    }

    impl TensorLayout {
        fn new(max_batch: usize, obs_dim: usize) -> Result<Self, BackendError> {
            if max_batch == 0 {
                return Err(BackendError::msg("TensorRT max_batch must be at least 1"));
            }
            if max_batch > i32::MAX as usize {
                return Err(BackendError::msg(format!(
                    "TensorRT max_batch must fit in i32 (got {max_batch})"
                )));
            }
            if obs_dim == 0 {
                return Err(BackendError::msg(
                    "TensorRT observation dimension must be at least 1",
                ));
            }
            if obs_dim > i64::MAX as usize {
                return Err(BackendError::msg(format!(
                    "TensorRT observation dimension must fit in i64 (got {obs_dim})"
                )));
            }

            let input_elements = checked_elements(max_batch, obs_dim, "input capacity")?;
            let policy_elements = checked_elements(max_batch, 5, "policy capacity")?;
            let value_elements = max_batch;
            let output = OutputLayout::new(policy_elements, value_elements)?;
            let capacity = BatchLayout {
                batch_size: max_batch,
                input_elements,
                policy_elements,
                value_elements,
                input_bytes: checked_bytes(input_elements, "input capacity")?,
                policy_bytes: checked_bytes(policy_elements, "policy capacity")?,
                value_bytes: checked_bytes(value_elements, "value capacity")?,
            };
            let output_bytes = checked_bytes(output.total, "output capacity")?;
            let pinned_bytes = capacity
                .input_bytes
                .checked_add(output_bytes)
                .ok_or_else(|| BackendError::msg("TensorRT total pinned byte size overflow"))?;
            Ok(Self {
                max_batch,
                obs_dim,
                capacity,
                output,
                output_bytes,
                pinned_bytes,
            })
        }

        fn batch(&self, n: usize) -> Result<BatchLayout, BackendError> {
            if n == 0 {
                return Err(BackendError::msg("TensorRT batch size must be at least 1"));
            }
            if n > self.max_batch {
                return Err(BackendError::msg(format!(
                    "batch size {n} exceeds max_batch {}",
                    self.max_batch
                )));
            }
            let input_elements = checked_elements(n, self.obs_dim, "input batch")?;
            let policy_elements = checked_elements(n, 5, "policy batch")?;
            let value_elements = n;
            Ok(BatchLayout {
                batch_size: n,
                input_elements,
                policy_elements,
                value_elements,
                input_bytes: checked_bytes(input_elements, "input batch")?,
                policy_bytes: checked_bytes(policy_elements, "policy batch")?,
                value_bytes: checked_bytes(value_elements, "value batch")?,
            })
        }
    }

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
        fn alloc(layout: TensorLayout) -> Result<Self, BackendError> {
            let mut b = Self {
                d_input: std::ptr::null_mut(),
                d_policy_p1: std::ptr::null_mut(),
                d_policy_p2: std::ptr::null_mut(),
                d_value_p1: std::ptr::null_mut(),
                d_value_p2: std::ptr::null_mut(),
            };
            unsafe {
                cuda_check(
                    cudaMalloc(&mut b.d_input, layout.capacity.input_bytes),
                    "cudaMalloc(input)",
                )?;
                cuda_check(
                    cudaMalloc(&mut b.d_policy_p1, layout.capacity.policy_bytes),
                    "cudaMalloc(policy_p1)",
                )?;
                cuda_check(
                    cudaMalloc(&mut b.d_policy_p2, layout.capacity.policy_bytes),
                    "cudaMalloc(policy_p2)",
                )?;
                cuda_check(
                    cudaMalloc(&mut b.d_value_p1, layout.capacity.value_bytes),
                    "cudaMalloc(value_p1)",
                )?;
                cuda_check(
                    cudaMalloc(&mut b.d_value_p2, layout.capacity.value_bytes),
                    "cudaMalloc(value_p2)",
                )?;
            }
            Ok(b)
        }
    }

    impl Drop for GpuBuffers {
        fn drop(&mut self) {
            unsafe {
                warn_cuda_drop(cudaFree(self.d_input), "cudaFree(input)");
                warn_cuda_drop(cudaFree(self.d_policy_p1), "cudaFree(policy_p1)");
                warn_cuda_drop(cudaFree(self.d_policy_p2), "cudaFree(policy_p2)");
                warn_cuda_drop(cudaFree(self.d_value_p1), "cudaFree(value_p1)");
                warn_cuda_drop(cudaFree(self.d_value_p2), "cudaFree(value_p2)");
            }
        }
    }

    struct PinnedAllocation {
        ptr: *mut c_void,
        len_f32: usize,
        bytes: usize,
    }

    // SAFETY: the allocation is exclusively owned by one mutex-serialized TrtSession.
    unsafe impl Send for PinnedAllocation {}

    impl PinnedAllocation {
        fn alloc(len_f32: usize, bytes: usize, label: &str) -> Result<Self, BackendError> {
            let mut ptr = std::ptr::null_mut();
            let operation = format!("{label} ({bytes} bytes of page-locked host memory)");
            cuda_check(unsafe { cudaMallocHost(&mut ptr, bytes) }, &operation)?;
            Ok(Self {
                ptr,
                len_f32,
                bytes,
            })
        }

        fn as_mut_slice(&mut self, len: usize) -> Result<&mut [f32], BackendError> {
            if len > self.len_f32 {
                return Err(BackendError::msg(format!(
                    "TensorRT pinned slice length {len} exceeds allocation {}",
                    self.len_f32
                )));
            }
            Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr.cast::<f32>(), len) })
        }

        fn ptr_at_mut(&mut self, offset: usize) -> Result<*mut c_void, BackendError> {
            if offset > self.len_f32 {
                return Err(BackendError::msg(format!(
                    "TensorRT pinned pointer offset {offset} exceeds allocation {}",
                    self.len_f32
                )));
            }
            Ok(unsafe { self.ptr.cast::<f32>().add(offset).cast::<c_void>() })
        }

        fn slice_at(&self, offset: usize, len: usize) -> Result<&[f32], BackendError> {
            let end = offset
                .checked_add(len)
                .ok_or_else(|| BackendError::msg("TensorRT pinned output slice range overflow"))?;
            if end > self.len_f32 {
                return Err(BackendError::msg(format!(
                    "TensorRT pinned output range {offset}..{end} exceeds allocation {}",
                    self.len_f32
                )));
            }
            Ok(unsafe { std::slice::from_raw_parts(self.ptr.cast::<f32>().add(offset), len) })
        }

        fn bytes(&self) -> usize {
            self.bytes
        }
    }

    impl Drop for PinnedAllocation {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                warn_cuda_drop(unsafe { cudaFreeHost(self.ptr) }, "cudaFreeHost");
            }
        }
    }

    struct PinnedHostBuffers {
        input: PinnedAllocation,
        output: PinnedAllocation,
        layout: TensorLayout,
    }

    impl PinnedHostBuffers {
        fn alloc(layout: TensorLayout) -> Result<Self, BackendError> {
            Ok(Self {
                input: PinnedAllocation::alloc(
                    layout.capacity.input_elements,
                    layout.capacity.input_bytes,
                    "cudaMallocHost(input)",
                )?,
                output: PinnedAllocation::alloc(
                    layout.output.total,
                    layout.output_bytes,
                    "cudaMallocHost(output)",
                )?,
                layout,
            })
        }

        fn input_mut(&mut self, batch: BatchLayout) -> Result<&mut [f32], BackendError> {
            self.input.as_mut_slice(batch.input_elements)
        }

        fn input_ptr(&self) -> *const c_void {
            self.input.ptr.cast_const()
        }

        fn output_ptrs_mut(&mut self) -> Result<[*mut c_void; 4], BackendError> {
            Ok([
                self.output.ptr_at_mut(self.layout.output.policy_p1)?,
                self.output.ptr_at_mut(self.layout.output.policy_p2)?,
                self.output.ptr_at_mut(self.layout.output.value_p1)?,
                self.output.ptr_at_mut(self.layout.output.value_p2)?,
            ])
        }

        fn output_slices(&self, batch: BatchLayout) -> Result<TrtOutputSlices<'_>, BackendError> {
            Ok((
                self.output
                    .slice_at(self.layout.output.policy_p1, batch.policy_elements)?,
                self.output
                    .slice_at(self.layout.output.policy_p2, batch.policy_elements)?,
                self.output
                    .slice_at(self.layout.output.value_p1, batch.value_elements)?,
                self.output
                    .slice_at(self.layout.output.value_p2, batch.value_elements)?,
            ))
        }

        fn bytes(&self) -> usize {
            debug_assert_eq!(
                self.input.bytes() + self.output.bytes(),
                self.layout.pinned_bytes
            );
            self.layout.pinned_bytes
        }
    }

    struct CudaEvent {
        handle: *mut c_void,
    }

    // SAFETY: events are only recorded and queried by their mutex-serialized session.
    unsafe impl Send for CudaEvent {}

    impl CudaEvent {
        fn timing() -> Result<Self, BackendError> {
            let mut handle = std::ptr::null_mut();
            cuda_check(unsafe { cudaEventCreate(&mut handle) }, "cudaEventCreate")?;
            Ok(Self { handle })
        }

        fn completion() -> Result<Self, BackendError> {
            let mut handle = std::ptr::null_mut();
            cuda_check(
                unsafe { cudaEventCreateWithFlags(&mut handle, CUDA_EVENT_DISABLE_TIMING) },
                "cudaEventCreateWithFlags(disable timing)",
            )?;
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
                if let Err(error) =
                    cuda_check(unsafe { cudaEventDestroy(self.handle) }, "cudaEventDestroy")
                {
                    eprintln!("[TensorRT] Warning: {error}");
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
                h2d_start: CudaEvent::timing()?,
                h2d_end: CudaEvent::timing()?,
                infer_start: CudaEvent::timing()?,
                infer_end: CudaEvent::timing()?,
                d2h_start: CudaEvent::timing()?,
                d2h_end: CudaEvent::timing()?,
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

    struct SessionHealth {
        poisoned: Cell<bool>,
        cleanup_error: RefCell<Option<String>>,
    }

    impl SessionHealth {
        fn new() -> Self {
            Self {
                poisoned: Cell::new(false),
                cleanup_error: RefCell::new(None),
            }
        }

        fn poison(&self, error: &BackendError) {
            self.poisoned.set(true);
            *self.cleanup_error.borrow_mut() = Some(error.to_string());
        }

        fn ensure_healthy(&self) -> Result<(), BackendError> {
            if !self.poisoned.get() {
                return Ok(());
            }
            let reason = self
                .cleanup_error
                .borrow()
                .clone()
                .unwrap_or_else(|| "unknown CUDA stream cleanup failure".to_string());
            Err(BackendError::msg(format!(
                "TensorRT session is poisoned and cannot be reused: {reason}"
            )))
        }
    }

    struct StreamFlight<'a> {
        stream: *mut c_void,
        health: &'a SessionHealth,
        armed: bool,
    }

    impl<'a> StreamFlight<'a> {
        fn new(stream: *mut c_void, health: &'a SessionHealth) -> Self {
            Self {
                stream,
                health,
                armed: true,
            }
        }

        fn finish(&mut self) {
            self.armed = false;
        }

        fn quiesce(&mut self) -> Result<(), BackendError> {
            self.armed = false;
            let result = cuda_check(
                unsafe { cudaStreamSynchronize(self.stream) },
                "cudaStreamSynchronize(error cleanup)",
            );
            if let Err(error) = &result {
                self.health.poison(error);
            }
            result
        }
    }

    impl Drop for StreamFlight<'_> {
        fn drop(&mut self) {
            if self.armed {
                if let Err(error) = self.quiesce() {
                    eprintln!("[TensorRT] Warning: {error}");
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
            -10 => "failed to allocate the serialized engine buffer",
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
        let opt_batch = i32::try_from(opt_batch)
            .map_err(|_| BackendError::msg("TensorRT opt_batch does not fit in i32"))?;
        let max_batch = i32::try_from(max_batch)
            .map_err(|_| BackendError::msg("TensorRT max_batch does not fit in i32"))?;
        let mut out_data: *mut c_void = std::ptr::null_mut();
        let mut out_len: usize = 0;

        let rc = unsafe {
            trt_build_engine(
                onnx_bytes.as_ptr() as *const c_void,
                onnx_bytes.len(),
                1, // min_batch
                opt_batch,
                max_batch,
                256, // workspace MB
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

    struct TrtSession {
        handle: *mut c_void, // opaque TrtSession from C++ shim
        stream: *mut c_void,
        input_name: CString,
        last_batch: Option<usize>,
        pad_to_max: bool,
        execution_size: Option<usize>,
        buffers: GpuBuffers,
        pinned: Option<PinnedHostBuffers>,
        completion: Option<CudaEvent>,
        stage_events: Option<StageEvents>,
        layout: TensorLayout,
        health: SessionHealth,
    }

    // SAFETY: all session access is serialized by `TensorrtBackend::sessions`.
    unsafe impl Send for TrtSession {}

    impl Drop for TrtSession {
        fn drop(&mut self) {
            // Host and device buffers are fields dropped after this method.
            // Quiesce first so no asynchronous work can outlive them.
            if let Err(error) = cuda_check(
                unsafe { cudaStreamSynchronize(self.stream) },
                "cudaStreamSynchronize(session drop)",
            ) {
                eprintln!("[TensorRT] Warning: {error}");
            }
            unsafe { trt_destroy_session(self.handle) };
            if let Err(error) = cuda_check(
                unsafe { cudaStreamDestroy(self.stream) },
                "cudaStreamDestroy",
            ) {
                eprintln!("[TensorRT] Warning: {error}");
            }
        }
    }

    impl TrtSession {
        fn new(
            engine_data: &[u8],
            layout: TensorLayout,
            host_io: TrtHostIoMode,
            profile_stages: bool,
            cuda_graph: bool,
        ) -> Result<Self, BackendError> {
            let handle = unsafe {
                trt_create_session(
                    engine_data.as_ptr().cast::<c_void>(),
                    engine_data.len(),
                    i32::from(cuda_graph),
                )
            };
            if handle.is_null() {
                return Err(BackendError::msg("Failed to create TensorRT session"));
            }

            match Self::init_session(handle, layout, host_io, profile_stages) {
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
            layout: TensorLayout,
            host_io: TrtHostIoMode,
            profile_stages: bool,
        ) -> Result<Self, BackendError> {
            let n_io = unsafe { trt_get_nb_io_tensors(handle) };
            let names: Vec<String> = (0..n_io)
                .map(|index| {
                    let ptr = unsafe { trt_get_tensor_name(handle, index) };
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
            let buffers = GpuBuffers::alloc(layout)?;

            // Bind tensor addresses
            Self::bind_tensors(handle, &buffers)?;

            // Lifecycle resources are created once, outside every timed call.
            let pinned = match host_io {
                TrtHostIoMode::Pageable => None,
                TrtHostIoMode::Pinned => Some(PinnedHostBuffers::alloc(layout)?),
            };
            let completion = match host_io {
                TrtHostIoMode::Pageable => None,
                TrtHostIoMode::Pinned => Some(CudaEvent::completion()?),
            };
            let stage_events = if profile_stages {
                Some(StageEvents::new()?)
            } else {
                None
            };

            let mut stream: *mut c_void = std::ptr::null_mut();
            cuda_check(unsafe { cudaStreamCreate(&mut stream) }, "cudaStreamCreate")?;
            let input_name = CString::new(crate::TENSOR_INPUT)
                .expect("TensorRT input tensor name must not contain NUL bytes");

            Ok(Self {
                handle,
                stream,
                input_name,
                last_batch: None,
                pad_to_max: false,
                execution_size: None,
                buffers,
                pinned,
                completion,
                stage_events,
                layout,
                health: SessionHealth::new(),
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

        fn ensure_healthy(&self) -> Result<(), BackendError> {
            self.health.ensure_healthy()
        }

        fn validate_input(&self, input_len: usize, batch: BatchLayout) -> Result<(), BackendError> {
            self.ensure_healthy()?;
            if input_len != batch.input_elements {
                return Err(BackendError::msg(format!(
                    "input length {input_len} != expected {}",
                    batch.input_elements
                )));
            }
            Ok(())
        }

        fn ensure_input_shape(&mut self, n: usize) -> Result<(), BackendError> {
            if self.last_batch == Some(n) {
                return Ok(());
            }
            let n = i64::try_from(n)
                .map_err(|_| BackendError::msg("TensorRT batch size does not fit in i64"))?;
            let obs_dim = i64::try_from(self.layout.obs_dim).map_err(|_| {
                BackendError::msg("TensorRT observation dimension does not fit in i64")
            })?;
            let shape = [n, obs_dim];
            let rc = unsafe {
                trt_set_input_shape(self.handle, self.input_name.as_ptr(), 2, shape.as_ptr())
            };
            if rc != 0 {
                return Err(BackendError::msg(format!(
                    "Failed to set input shape for batch size {n} (rc={rc})"
                )));
            }
            self.last_batch = Some(n as usize);
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
            let _trace = trace::range(
                if kind == CUDA_MEMCPY_H2D {
                    c"h2d.submit"
                } else {
                    c"d2h.submit"
                },
                trace::request_id(),
            );
            cuda_check(unsafe { cudaMemcpy(dst, src, count, kind) }, op)
        }

        fn copy_async(
            &self,
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: i32,
            op: &str,
        ) -> Result<(), BackendError> {
            let _trace = trace::range(
                if kind == CUDA_MEMCPY_H2D {
                    c"h2d.submit"
                } else {
                    c"d2h.submit"
                },
                trace::request_id(),
            );
            cuda_check(
                unsafe { cudaMemcpyAsync(dst, src, count, kind, self.stream) },
                op,
            )
        }

        /// Preserve the existing pageable allocation and synchronization path.
        fn infer_pageable(
            &mut self,
            input: &[f32],
            batch: BatchLayout,
            profile_stages: bool,
        ) -> Result<(OwnedTrtOutputs, TrtTimingInfo), BackendError> {
            self.validate_input(input.len(), batch)?;
            self.ensure_input_shape(batch.batch_size)?;
            let total_start = profile_stages.then(Instant::now);
            let mut timing = TrtTimingInfo::default();
            let events = if profile_stages {
                self.stage_events.as_ref()
            } else {
                None
            };
            let mut flight = StreamFlight::new(self.stream, &self.health);

            if let Some(events) = events {
                events
                    .h2d_start
                    .record(self.stream, "cudaEventRecord(H2D start)")?;
                self.copy_async(
                    self.buffers.d_input,
                    input.as_ptr().cast::<c_void>(),
                    batch.input_bytes,
                    CUDA_MEMCPY_H2D,
                    "input H2D async",
                )?;
                events
                    .h2d_end
                    .record(self.stream, "cudaEventRecord(H2D end)")?;
                // Match the control's existing H2D-before-enqueue barrier.
                events.h2d_end.synchronize("cudaEventSynchronize(H2D)")?;
                events
                    .infer_start
                    .record(self.stream, "cudaEventRecord(inference start)")?;
            } else {
                self.copy_control(
                    self.buffers.d_input,
                    input.as_ptr().cast::<c_void>(),
                    batch.input_bytes,
                    CUDA_MEMCPY_H2D,
                    "input H2D",
                )?;
            }

            let rc = {
                let _trace = trace::range(c"enqueue", trace::request_id());
                unsafe { trt_enqueue_v3(self.handle, self.stream) }
            };
            if rc != 0 {
                return Err(BackendError::msg(format!(
                    "TRT enqueue_v3 failed (rc={rc})"
                )));
            }
            if let Some(events) = events {
                events
                    .infer_end
                    .record(self.stream, "cudaEventRecord(inference end)")?;
                events
                    .infer_end
                    .synchronize("cudaEventSynchronize(inference)")?;
            } else {
                let _trace = trace::range(c"completion.wait", trace::request_id());
                cuda_check(
                    unsafe { cudaStreamSynchronize(self.stream) },
                    "cudaStreamSynchronize",
                )?;
            }

            let alloc_start = profile_stages.then(Instant::now);
            let mut pp1 = vec![0.0f32; batch.policy_elements];
            let mut pp2 = vec![0.0f32; batch.policy_elements];
            let mut v1 = vec![0.0f32; batch.value_elements];
            let mut v2 = vec![0.0f32; batch.value_elements];
            timing.output_alloc_us = elapsed_us(alloc_start);

            if let Some(events) = events {
                events
                    .d2h_start
                    .record(self.stream, "cudaEventRecord(D2H start)")?;
                self.copy_async(
                    pp1.as_mut_ptr().cast::<c_void>(),
                    self.buffers.d_policy_p1,
                    batch.policy_bytes,
                    CUDA_MEMCPY_D2H,
                    "policy_p1 D2H async",
                )?;
                self.copy_async(
                    pp2.as_mut_ptr().cast::<c_void>(),
                    self.buffers.d_policy_p2,
                    batch.policy_bytes,
                    CUDA_MEMCPY_D2H,
                    "policy_p2 D2H async",
                )?;
                self.copy_async(
                    v1.as_mut_ptr().cast::<c_void>(),
                    self.buffers.d_value_p1,
                    batch.value_bytes,
                    CUDA_MEMCPY_D2H,
                    "value_p1 D2H async",
                )?;
                self.copy_async(
                    v2.as_mut_ptr().cast::<c_void>(),
                    self.buffers.d_value_p2,
                    batch.value_bytes,
                    CUDA_MEMCPY_D2H,
                    "value_p2 D2H async",
                )?;
                events
                    .d2h_end
                    .record(self.stream, "cudaEventRecord(D2H end)")?;
                events.d2h_end.synchronize("cudaEventSynchronize(D2H)")?;
                let device = events.timing()?;
                timing.h2d_us = device.h2d_us;
                timing.infer_us = device.infer_us;
                timing.d2h_us = device.d2h_us;
            } else {
                self.copy_control(
                    pp1.as_mut_ptr().cast::<c_void>(),
                    self.buffers.d_policy_p1,
                    batch.policy_bytes,
                    CUDA_MEMCPY_D2H,
                    "policy_p1 D2H",
                )?;
                self.copy_control(
                    pp2.as_mut_ptr().cast::<c_void>(),
                    self.buffers.d_policy_p2,
                    batch.policy_bytes,
                    CUDA_MEMCPY_D2H,
                    "policy_p2 D2H",
                )?;
                self.copy_control(
                    v1.as_mut_ptr().cast::<c_void>(),
                    self.buffers.d_value_p1,
                    batch.value_bytes,
                    CUDA_MEMCPY_D2H,
                    "value_p1 D2H",
                )?;
                self.copy_control(
                    v2.as_mut_ptr().cast::<c_void>(),
                    self.buffers.d_value_p2,
                    batch.value_bytes,
                    CUDA_MEMCPY_D2H,
                    "value_p2 D2H",
                )?;
            }

            timing.total_us = elapsed_us(total_start);
            flight.finish();
            Ok(((pp1, pp2, v1, v2), timing))
        }

        fn pinned_input_mut(&mut self, batch: BatchLayout) -> Result<&mut [f32], BackendError> {
            self.ensure_healthy()?;
            self.pinned
                .as_mut()
                .ok_or_else(|| BackendError::msg("pinned input requested on pageable session"))?
                .input_mut(batch)
        }

        fn infer_pinned_prepared(
            &mut self,
            batch: BatchLayout,
            profile_stages: bool,
        ) -> Result<TrtTimingInfo, BackendError> {
            self.validate_input(batch.input_elements, batch)?;
            let execution = if self.pad_to_max {
                match self.execution_size {
                    Some(n) => self.layout.batch(n)?,
                    None => self.layout.capacity,
                }
            } else {
                batch
            };
            // Every submitted row is initialized, including after a larger real batch.
            // Only the real prefix is copied back and exposed to the caller below.
            if execution.input_elements > batch.input_elements {
                self.pinned_input_mut(execution)?[batch.input_elements..].fill(0.0);
            }
            self.ensure_input_shape(execution.batch_size)?;
            let total_start = profile_stages.then(Instant::now);
            let pinned = self.pinned.as_mut().ok_or_else(|| {
                BackendError::msg("pinned inference requested on pageable session")
            })?;
            let completion = self.completion.as_ref().ok_or_else(|| {
                BackendError::msg("pinned inference completion event is unavailable")
            })?;
            let events = if profile_stages {
                self.stage_events.as_ref()
            } else {
                None
            };
            let input_ptr = pinned.input_ptr();
            let [pp1_ptr, pp2_ptr, v1_ptr, v2_ptr] = pinned.output_ptrs_mut()?;
            let mut flight = StreamFlight::new(self.stream, &self.health);

            if let Some(events) = events {
                events
                    .h2d_start
                    .record(self.stream, "cudaEventRecord(H2D start)")?;
            }
            self.copy_async(
                self.buffers.d_input,
                input_ptr,
                execution.input_bytes,
                CUDA_MEMCPY_H2D,
                "pinned input H2D async",
            )?;
            if let Some(events) = events {
                events
                    .h2d_end
                    .record(self.stream, "cudaEventRecord(H2D end)")?;
                events
                    .infer_start
                    .record(self.stream, "cudaEventRecord(inference start)")?;
            }

            let rc = {
                let _trace = trace::range(c"enqueue", trace::request_id());
                unsafe { trt_enqueue_v3(self.handle, self.stream) }
            };
            if rc != 0 {
                return Err(BackendError::msg(format!(
                    "TRT enqueue_v3 failed (rc={rc})"
                )));
            }
            if let Some(events) = events {
                events
                    .infer_end
                    .record(self.stream, "cudaEventRecord(inference end)")?;
                events
                    .d2h_start
                    .record(self.stream, "cudaEventRecord(D2H start)")?;
            }

            self.copy_async(
                pp1_ptr,
                self.buffers.d_policy_p1,
                batch.policy_bytes,
                CUDA_MEMCPY_D2H,
                "pinned policy_p1 D2H async",
            )?;
            self.copy_async(
                pp2_ptr,
                self.buffers.d_policy_p2,
                batch.policy_bytes,
                CUDA_MEMCPY_D2H,
                "pinned policy_p2 D2H async",
            )?;
            self.copy_async(
                v1_ptr,
                self.buffers.d_value_p1,
                batch.value_bytes,
                CUDA_MEMCPY_D2H,
                "pinned value_p1 D2H async",
            )?;
            self.copy_async(
                v2_ptr,
                self.buffers.d_value_p2,
                batch.value_bytes,
                CUDA_MEMCPY_D2H,
                "pinned value_p2 D2H async",
            )?;
            if let Some(events) = events {
                events
                    .d2h_end
                    .record(self.stream, "cudaEventRecord(D2H end)")?;
            }
            completion.record(self.stream, "cudaEventRecord(completion)")?;
            {
                let _trace = trace::range(c"completion.wait", trace::request_id());
                completion.synchronize("cudaEventSynchronize(completion)")?;
            }

            let mut timing = if let Some(events) = events {
                events.timing()?
            } else {
                TrtTimingInfo::default()
            };
            timing.total_us = elapsed_us(total_start);
            flight.finish();
            Ok(timing)
        }

        fn pinned_output_slices(
            &self,
            batch: BatchLayout,
        ) -> Result<TrtOutputSlices<'_>, BackendError> {
            self.ensure_healthy()?;
            self.pinned
                .as_ref()
                .ok_or_else(|| BackendError::msg("pinned output requested on pageable session"))?
                .output_slices(batch)
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
    /// Thread safety: execution contexts, streams, and reusable host/device
    /// buffers are serialized within each lane. Optional fixed-size contexts
    /// retain their shapes across calls; explicitly configured lanes may overlap.
    pub struct TensorrtBackend<E: ObservationEncoder> {
        engine_sha256: String,
        sessions: Vec<Mutex<Vec<TrtSession>>>,
        next_lane: AtomicU64,
        device_gate: Option<Mutex<()>>,
        execution_sizes: Vec<usize>,
        encoder: E,
        layout: TensorLayout,
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
            let obs_dim = encoder.obs_dim();
            let layout = TensorLayout::new(config.max_batch, obs_dim)?;
            let opt_batch = config.opt_batch.unwrap_or(config.max_batch);

            if opt_batch == 0 || opt_batch > config.max_batch {
                return Err(BackendError::msg(format!(
                    "TensorRT opt_batch must be in 1..={} (got {})",
                    config.max_batch, opt_batch
                )));
            }
            if opt_batch > i32::MAX as usize {
                return Err(BackendError::msg(format!(
                    "TensorRT opt_batch must fit in i32 (got {})",
                    opt_batch
                )));
            }
            if config.pad_to_max && config.host_io != TrtHostIoMode::Pinned {
                return Err(BackendError::msg(
                    "TensorRT padding requires pinned host I/O",
                ));
            }
            if !config.execution_sizes.is_empty()
                && (!config.pad_to_max
                    || config.execution_sizes.first() == Some(&0)
                    || config.execution_sizes.last() != Some(&config.max_batch)
                    || config.execution_sizes.windows(2).any(|w| w[0] >= w[1]))
            {
                return Err(BackendError::msg("TensorRT execution_sizes requires padding and strictly increasing positive sizes ending at max_batch"));
            }
            if !(1..=8).contains(&config.execution_lanes) {
                return Err(BackendError::msg(
                    "TensorRT execution_lanes must be in 1..=8",
                ));
            }
            load_trt_libs()?;
            let onnx_path = model_path.as_ref();

            let onnx_bytes = fs::read(onnx_path).map_err(|e| {
                BackendError::msg(format!(
                    "Failed to read ONNX model at {}: {e}",
                    onnx_path.display()
                ))
            })?;

            let onnx_hash: [u8; 32] = Sha256::digest(&onnx_bytes).into();
            let key = cache_key(&onnx_hash, opt_batch, config.max_batch)?;

            // Load cached engine or build from scratch
            let engine_data = match &config.cache_dir {
                Some(dir) => match try_load_cache(dir, &key, &onnx_hash) {
                    Some(data) => {
                        eprintln!("[TensorRT] Loaded cached engine: {key}");
                        data
                    }
                    None => {
                        eprintln!("[TensorRT] Building engine from ONNX (this may take 10-30s)...");
                        let data = build_engine(&onnx_bytes, opt_batch, config.max_batch)?;
                        save_cache(dir, &key, &data, &onnx_hash);
                        eprintln!("[TensorRT] Engine cached as {key}");
                        data
                    }
                },
                None => {
                    eprintln!("[TensorRT] Building engine (no cache dir configured)...");
                    build_engine(&onnx_bytes, opt_batch, config.max_batch)?
                }
            };

            let engine_sha256 = format!("{:x}", Sha256::digest(&engine_data));
            let sizes: Vec<Option<usize>> = if config.execution_sizes.is_empty() {
                vec![None]
            } else {
                config.execution_sizes.iter().copied().map(Some).collect()
            };
            let mut sessions = Vec::with_capacity(config.execution_lanes);
            let mut pinned_bytes = 0;
            for _ in 0..config.execution_lanes {
                let mut lane = Vec::with_capacity(sizes.len());
                for &size in &sizes {
                    let mut session = TrtSession::new(
                        &engine_data,
                        layout,
                        config.host_io,
                        config.profile_stages,
                        config.cuda_graph,
                    )?;
                    session.pad_to_max = config.pad_to_max;
                    session.execution_size = size;
                    pinned_bytes += session.pinned_bytes();
                    lane.push(session);
                }
                sessions.push(Mutex::new(lane));
            }
            eprintln!(
                "[TensorRT] OPT={opt_batch} MAX={}; lanes={}; contexts={}; sizes={:?}; pad={}; graphs={}; pinned_bytes={pinned_bytes}",
                config.max_batch, sessions.len(), sessions.len()*sizes.len(), config.execution_sizes, config.pad_to_max, config.cuda_graph,
            );
            let stats = Arc::new(TrtStats::new(config.host_io, pinned_bytes));

            Ok(Self {
                engine_sha256,
                sessions,
                next_lane: AtomicU64::new(0),
                device_gate: config.serialize_device.then(|| Mutex::new(())),
                execution_sizes: config.execution_sizes,
                encoder,
                layout,
                host_io: config.host_io,
                profile_stages: config.profile_stages,
                stats,
            })
        }

        /// SHA-256 of the serialized engine actually loaded by this instance.
        pub fn engine_sha256(&self) -> &str {
            &self.engine_sha256
        }

        pub fn stats(&self) -> &Arc<TrtStats> {
            &self.stats
        }

        /// Number of physically created execution contexts, including fixed sizes.
        pub fn physical_contexts(&self) -> usize {
            self.sessions.len() * self.execution_sizes.len().max(1)
        }

        fn execution_slot(&self, n: usize) -> usize {
            // The caller validates n against max_batch before acquiring sessions.
            self.execution_sizes
                .iter()
                .position(|size| n <= *size)
                .unwrap_or(0)
        }

        fn lock_device(&self) -> Result<Option<MutexGuard<'_, ()>>, BackendError> {
            self.device_gate
                .as_ref()
                .map(|gate| {
                    let _trace = trace::range(c"device.lock_wait", trace::request_id());
                    gate.lock().map_err(|_| {
                        BackendError::msg("TensorRT device gate poisoned after a panic")
                    })
                })
                .transpose()
        }

        fn lock_sessions(&self) -> Result<MutexGuard<'_, Vec<TrtSession>>, BackendError> {
            let _trace = trace::range(c"session.lock_wait", trace::request_id());
            let lane_count = self.sessions.len();
            let start = if lane_count == 1 {
                0
            } else {
                self.next_lane.fetch_add(1, Ordering::Relaxed) as usize % lane_count
            };
            if lane_count > 1 {
                for offset in 0..lane_count {
                    match self.sessions[(start + offset) % lane_count].try_lock() {
                        Ok(guard) => return Ok(guard),
                        Err(TryLockError::WouldBlock) => {}
                        Err(TryLockError::Poisoned(_)) => {
                            return Err(BackendError::msg(
                                "TensorRT lane lock poisoned after a panic",
                            ))
                        }
                    }
                }
            }
            self.sessions[start].lock().map_err(|_| {
                BackendError::msg(
                    "TensorRT session lock poisoned after a panic; the session will not be reused",
                )
            })
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
        let policy_len = checked_elements(n, 5, "parsed policy output")?;
        for (name, actual, expected) in [
            ("policy_p1", pp1.len(), policy_len),
            ("policy_p2", pp2.len(), policy_len),
            ("value_p1", v1.len(), n),
            ("value_p2", v2.len(), n),
        ] {
            if actual < expected {
                return Err(BackendError::msg(format!(
                    "TensorRT {name} output length {actual} is shorter than expected {expected}"
                )));
            }
        }
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
            if !self.profile_stages {
                return Err(BackendError::msg(
                    "evaluate_encoded_timed requires TensorrtConfig::profile_stages = true",
                ));
            }
            let call_start = Instant::now();
            let batch = self.layout.batch(n)?;
            let mut sessions = self.lock_sessions()?;
            let session = &mut sessions[self.execution_slot(n)];
            session.validate_input(encoded.len(), batch)?;

            let (results, mut timing) = match self.host_io {
                TrtHostIoMode::Pageable => {
                    let ((pp1, pp2, v1, v2), mut timing) = {
                        let _device = self.lock_device()?;
                        session.infer_pageable(encoded, batch, true)?
                    };
                    let parse_start = Instant::now();
                    let results = parse_eval_results(&pp1, &pp2, &v1, &v2, n)?;
                    timing.parse_us = parse_start.elapsed().as_secs_f64() * 1_000_000.0;
                    (results, timing)
                }
                TrtHostIoMode::Pinned => {
                    let stage_start = Instant::now();
                    session.pinned_input_mut(batch)?.copy_from_slice(encoded);
                    let input_stage_us = stage_start.elapsed().as_secs_f64() * 1_000_000.0;
                    let mut timing = {
                        let _device = self.lock_device()?;
                        session.infer_pinned_prepared(batch, true)?
                    };
                    timing.input_stage_us = input_stage_us;
                    let parse_start = Instant::now();
                    let (pp1, pp2, v1, v2) = session.pinned_output_slices(batch)?;
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
            let batch = self.layout.batch(n)?;
            let call_start = self.profile_stages.then(Instant::now);

            let (results, mut timing) = match self.host_io {
                TrtHostIoMode::Pageable => {
                    let allocation_start = self.profile_stages.then(Instant::now);
                    let mut buf = vec![0.0f32; batch.input_elements];
                    let input_stage_us = elapsed_us(allocation_start);
                    let encode_trace = trace::range(c"encode", trace::request_id());
                    let encode_start = self.profile_stages.then(Instant::now);
                    for (i, game) in games.iter().enumerate() {
                        self.encoder.encode_into(game, &mut buf, i * obs_dim);
                    }
                    let encode_us = elapsed_us(encode_start);
                    drop(encode_trace);

                    let mut sessions = self.lock_sessions()?;
                    let session = &mut sessions[self.execution_slot(n)];
                    let ((pp1, pp2, v1, v2), mut timing) = {
                        let _device = self.lock_device()?;
                        session.infer_pageable(&buf, batch, self.profile_stages)?
                    };
                    timing.input_stage_us = input_stage_us;
                    timing.encode_us = encode_us;
                    let _parse_trace = trace::range(c"parse", trace::request_id());
                    let parse_start = self.profile_stages.then(Instant::now);
                    let results = parse_eval_results(&pp1, &pp2, &v1, &v2, n)?;
                    timing.parse_us = elapsed_us(parse_start);
                    (results, timing)
                }
                TrtHostIoMode::Pinned => {
                    let mut sessions = self.lock_sessions()?;
                    let session = &mut sessions[self.execution_slot(n)];
                    let encode_trace = trace::range(c"encode", trace::request_id());
                    let encode_start = self.profile_stages.then(Instant::now);
                    {
                        let input = session.pinned_input_mut(batch)?;
                        for (i, game) in games.iter().enumerate() {
                            self.encoder.encode_into(game, input, i * obs_dim);
                        }
                    }
                    let encode_us = elapsed_us(encode_start);
                    drop(encode_trace);
                    let mut timing = {
                        let _device = self.lock_device()?;
                        session.infer_pinned_prepared(batch, self.profile_stages)?
                    };
                    timing.encode_us = encode_us;
                    let _parse_trace = trace::range(c"parse", trace::request_id());
                    let parse_start = self.profile_stages.then(Instant::now);
                    let (pp1, pp2, v1, v2) = session.pinned_output_slices(batch)?;
                    let results = parse_eval_results(pp1, pp2, v1, v2, n)?;
                    timing.parse_us = elapsed_us(parse_start);
                    (results, timing)
                }
            };

            timing.total_us = elapsed_us(call_start);
            if self.profile_stages {
                self.stats.record(n, &timing);
            }
            Ok(results)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{
            parse_eval_results, OutputLayout, SessionHealth, TensorLayout, TensorrtBackend,
            TensorrtConfig, TrtHostIoMode,
        };
        use crate::FlatEncoder;
        use alpharat_mcts::BackendError;

        #[test]
        fn pinned_host_io_is_the_tensor_rt_default() {
            assert_eq!(TensorrtConfig::default().host_io, TrtHostIoMode::Pinned);
        }

        #[test]
        fn optimization_batch_defaults_to_the_configured_maximum() {
            let config = TensorrtConfig {
                max_batch: 128,
                ..TensorrtConfig::default()
            };
            assert_eq!(config.opt_batch.unwrap_or(config.max_batch), 128);
        }

        #[test]
        fn invalid_optimization_batch_is_rejected_before_runtime_load() {
            let error = TensorrtBackend::new(
                "not-read.onnx",
                FlatEncoder::new(7, 7),
                TensorrtConfig {
                    opt_batch: Some(129),
                    max_batch: 128,
                    ..TensorrtConfig::default()
                },
            )
            .err()
            .expect("invalid optimization point should fail");

            assert!(error.to_string().contains("opt_batch must be in 1..=128"));
        }

        #[test]
        fn invalid_execution_sizes_are_rejected_before_runtime_load() {
            for sizes in [
                vec![0, 128],
                vec![64, 32, 128],
                vec![32, 64],
                vec![32, 32, 128],
            ] {
                let error = TensorrtBackend::new(
                    "missing.onnx",
                    FlatEncoder::new(7, 7),
                    TensorrtConfig {
                        max_batch: 128,
                        pad_to_max: true,
                        execution_sizes: sizes,
                        ..TensorrtConfig::default()
                    },
                )
                .err()
                .expect("invalid sizes must fail");
                assert!(error.to_string().contains("execution_sizes"));
            }
            let error = TensorrtBackend::new(
                "missing.onnx",
                FlatEncoder::new(7, 7),
                TensorrtConfig {
                    max_batch: 128,
                    execution_sizes: vec![32, 128],
                    ..TensorrtConfig::default()
                },
            )
            .err()
            .expect("fixed sizes require padding");
            assert!(error.to_string().contains("execution_sizes"));
        }

        #[test]
        fn padding_requires_pinned_io_before_runtime_load() {
            let error = TensorrtBackend::new(
                "not-read.onnx",
                FlatEncoder::new(7, 7),
                TensorrtConfig {
                    pad_to_max: true,
                    host_io: TrtHostIoMode::Pageable,
                    ..TensorrtConfig::default()
                },
            )
            .err()
            .expect("pageable padding must fail explicitly");
            assert!(error.to_string().contains("padding requires pinned"));
        }

        #[test]
        fn padded_output_tail_is_not_exposed_or_validated_as_real_work() {
            let output = parse_eval_results(
                &[0.2, 0.2, 0.2, 0.2, 0.2, f32::NAN],
                &[0.2, 0.2, 0.2, 0.2, 0.2, f32::NAN],
                &[0.3, f32::NAN],
                &[0.4, f32::NAN],
                1,
            )
            .unwrap();
            assert_eq!(output.len(), 1);
            assert_eq!(output[0].value_p1, 0.3);
        }

        #[test]
        fn pinned_output_layout_uses_disjoint_max_batch_regions() {
            let layout = OutputLayout::new(640, 128).unwrap();
            assert_eq!(layout.policy_p1, 0);
            assert_eq!(layout.policy_p2, 640);
            assert_eq!(layout.value_p1, 1280);
            assert_eq!(layout.value_p2, 1408);
            assert_eq!(layout.total, 1536);
        }

        #[test]
        fn tensor_layout_accounts_for_the_full_pinned_allocation() {
            let layout = TensorLayout::new(128, 349).unwrap();
            assert_eq!(layout.capacity.input_elements, 44_672);
            assert_eq!(layout.output.total, 1_536);
            assert_eq!(layout.pinned_bytes, 184_832);
        }

        #[test]
        fn tensor_layout_rejects_overflow_before_loading_cuda() {
            let error = TensorLayout::new(3, i64::MAX as usize).unwrap_err();
            assert!(error.to_string().contains("overflow"));
        }

        #[test]
        fn invalid_max_batch_is_rejected_before_runtime_load() {
            let error = TensorrtBackend::new(
                "not-read.onnx",
                FlatEncoder::new(7, 7),
                TensorrtConfig {
                    max_batch: 0,
                    ..TensorrtConfig::default()
                },
            )
            .err()
            .expect("zero max batch should fail");

            assert!(error.to_string().contains("max_batch must be at least 1"));
        }

        #[test]
        fn short_output_buffers_are_rejected_without_indexing_them() {
            let error =
                parse_eval_results(&[0.0; 9], &[0.0; 10], &[0.0; 2], &[0.0; 2], 2).unwrap_err();
            assert!(error.to_string().contains("policy_p1 output length 9"));
        }

        #[test]
        fn failed_stream_cleanup_poison_is_fail_closed() {
            let health = SessionHealth::new();
            health.poison(&BackendError::msg("forced cleanup failure"));
            let error = health.ensure_healthy().unwrap_err();
            assert!(error.to_string().contains("forced cleanup failure"));
        }
    }
}

#[cfg(feature = "tensorrt")]
pub use inner::{
    load_trt_libs, parse_eval_results, TensorrtBackend, TensorrtConfig, TrtHostIoMode, TrtStats,
    TrtStatsSnapshot, TrtTimingInfo,
};
