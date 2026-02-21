//! Isolated NN inference throughput benchmark with detailed timing breakdown.
//!
//! Measures: tensor creation (+ H2D copy), GPU inference, output extraction (D2H copy).
//!
//! Usage:
//!   # ORT CPU
//!   cargo run --release -p alpharat-sampling --features onnx --bin bench_nn_throughput -- <model.onnx>
//!
//!   # ORT CUDA
//!   cargo run --release -p alpharat-sampling --features onnx-cuda --bin bench_nn_throughput -- <model.onnx> --device cuda
//!
//!   # TensorRT-RTX  (requires TENSORRT_RTX_ROOT + libs in LD_LIBRARY_PATH)
//!   cargo run --release -p alpharat-sampling --features tensorrt --bin bench_nn_throughput -- <model.onnx> --device tensorrt
//!
//! For NVIDIA Nsight profiling:
//!   nsys profile --stats=true target/release/bench_nn_throughput model.onnx --device tensorrt

use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .expect("usage: bench_nn_throughput <model.onnx> [--device cpu|cuda|tensorrt] [--width W] [--height H]");

    let mut width: u8 = 7;
    let mut height: u8 = 7;
    let mut device = "cpu";
    let mut max_batch: usize = 262144;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--width" => {
                width = args[i + 1].parse().expect("invalid width");
                i += 2;
            }
            "--height" => {
                height = args[i + 1].parse().expect("invalid height");
                i += 2;
            }
            "--device" => {
                device = Box::leak(args[i + 1].clone().into_boxed_str());
                i += 2;
            }
            "--max-batch" => {
                max_batch = args[i + 1].parse().expect("invalid max-batch");
                i += 2;
            }
            other => panic!("unknown arg: {other}"),
        }
    }

    // --- Pre-encode games ---
    let (encoded_buf, obs_dim) = pre_encode_games(width, height, max_batch);

    println!(
        "NN Inference Throughput — {}x{}, obs_dim={}, device={}",
        width, height, obs_dim, device
    );

    let mut batch_sizes: Vec<usize> = vec![
        1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
        262144,
    ];
    // Add max_batch to the list if it's not already there
    if !batch_sizes.contains(&max_batch) {
        batch_sizes.push(max_batch);
        batch_sizes.sort();
    }
    // Only keep batch sizes up to max_batch
    batch_sizes.retain(|&b| b <= max_batch);

    match device {
        #[cfg(feature = "onnx")]
        "cpu" => run_ort_benchmark(model_path, &encoded_buf, obs_dim, &batch_sizes, false),
        #[cfg(feature = "onnx-cuda")]
        "cuda" => run_ort_benchmark(model_path, &encoded_buf, obs_dim, &batch_sizes, true),
        #[cfg(feature = "tensorrt")]
        "tensorrt" => run_trt_benchmark(model_path, &encoded_buf, obs_dim, &batch_sizes, max_batch),
        other => {
            let mut supported = vec![];
            if cfg!(feature = "onnx") {
                supported.push("cpu");
            }
            if cfg!(feature = "onnx-cuda") {
                supported.push("cuda");
            }
            if cfg!(feature = "tensorrt") {
                supported.push("tensorrt");
            }
            eprintln!(
                "Device '{}' not available. Compiled with support for: {:?}",
                other, supported
            );
            std::process::exit(1);
        }
    }
}

fn pre_encode_games(width: u8, height: u8, max_batch: usize) -> (Vec<f32>, usize) {
    use alpharat_sampling::encoder::ObservationEncoder;
    use alpharat_sampling::FlatEncoder;
    use pyrat::{CheeseConfig, GameState, MazeConfig};

    let encoder = FlatEncoder::new(width, height);
    let obs_dim = encoder.obs_dim();
    let cheese_count = ((width as u16 * height as u16) as f64 * 0.2).round() as u16;
    let max_turns: u16 = if width <= 5 { 30 } else { 50 };

    let num_games: usize = max_batch;
    eprintln!("Generating {} random games...", num_games);
    let games: Vec<GameState> = (0..num_games)
        .map(|i| {
            let maze_config = MazeConfig {
                width,
                height,
                target_density: 0.0,
                connected: true,
                symmetry: true,
                mud_density: 0.0,
                mud_range: 0,
                seed: Some(i as u64),
            };
            let cheese_config = CheeseConfig {
                count: cheese_count,
                symmetry: true,
            };
            let mut g = GameState::new_random(width, height, maze_config, cheese_config);
            g.max_turns = max_turns;
            g
        })
        .collect();

    eprintln!("Pre-encoding {} games...", num_games);
    let mut buf = vec![0.0f32; num_games * obs_dim];
    let t0 = Instant::now();
    for (i, game) in games.iter().enumerate() {
        encoder.encode_into(game, &mut buf, i * obs_dim);
    }
    let encode_us = t0.elapsed().as_micros();
    println!(
        "Pre-encoded {} games in {:.1}ms ({:.0}ns/pos)\n",
        num_games,
        encode_us as f64 / 1000.0,
        (encode_us as f64 * 1000.0) / num_games as f64,
    );

    (buf, obs_dim)
}

fn format_throughput(pos_per_sec: f64) -> String {
    if pos_per_sec >= 1_000_000.0 {
        format!("{:.2}M", pos_per_sec / 1_000_000.0)
    } else if pos_per_sec >= 1_000.0 {
        format!("{:.1}k", pos_per_sec / 1_000.0)
    } else {
        format!("{:.0}", pos_per_sec)
    }
}

fn print_header() {
    println!(
        "  {:>6} {:>10} {:>10} {:>10} {:>10} {:>12} {:>10}",
        "batch", "h2d_µs", "infer_µs", "d2h_µs", "total_µs", "pos/s", "GPU_util%"
    );
    println!("  {:-<90}", "");
}

fn print_footer(peak_pos_s: f64) {
    println!(
        "\n  Peak throughput: {} pos/s\n",
        format_throughput(peak_pos_s)
    );
}

// ---------------------------------------------------------------------------
// ORT benchmark
// ---------------------------------------------------------------------------

#[cfg(feature = "onnx")]
fn run_ort_benchmark(
    model_path: &str,
    encoded_buf: &[f32],
    obs_dim: usize,
    batch_sizes: &[usize],
    use_cuda: bool,
) {
    use ort::session::Session;
    use ort::value::Tensor;

    let mut session = if use_cuda {
        #[cfg(feature = "onnx-cuda")]
        {
            let cuda_ep = ort::execution_providers::CUDAExecutionProvider::default();
            match cuda_ep.is_available() {
                Ok(true) => eprintln!("CUDA EP: available"),
                Ok(false) => eprintln!("WARNING: CUDA EP not available! Falling back to CPU."),
                Err(e) => eprintln!("WARNING: CUDA EP check failed: {e}"),
            }
            Session::builder()
                .expect("session builder")
                .with_execution_providers([cuda_ep.build()])
                .expect("register CUDA EP")
                .commit_from_file(model_path)
                .expect("load model")
        }
        #[cfg(not(feature = "onnx-cuda"))]
        {
            eprintln!("CUDA EP not compiled. Build with --features onnx-cuda");
            std::process::exit(1);
        }
    } else {
        Session::builder()
            .expect("session builder")
            .with_intra_threads(4)
            .expect("set threads")
            .commit_from_file(model_path)
            .expect("load model")
    };

    let num_games = encoded_buf.len() / obs_dim;
    print_header();
    let mut peak_pos_s: f64 = 0.0;

    for &batch_size in batch_sizes {
        if batch_size > num_games {
            break;
        }

        let iters = (5000 / batch_size).max(10).min(500);
        let warmup = (iters / 10).max(5);
        let batch_data = &encoded_buf[..batch_size * obs_dim];

        // Warmup
        for _ in 0..warmup {
            let input = Tensor::from_array(([batch_size, obs_dim], batch_data.to_vec()))
                .expect("tensor creation failed");
            let _outputs = session
                .run(ort::inputs!["observation" => input])
                .expect("inference failed");
        }

        // Timed runs
        let mut h2d_us_total: f64 = 0.0;
        let mut run_us_total: f64 = 0.0;
        let mut d2h_us_total: f64 = 0.0;

        for _ in 0..iters {
            let t0 = Instant::now();
            let input = Tensor::from_array(([batch_size, obs_dim], batch_data.to_vec()))
                .expect("tensor creation failed");
            h2d_us_total += t0.elapsed().as_micros() as f64;

            let t1 = Instant::now();
            let outputs = session
                .run(ort::inputs!["observation" => input])
                .expect("inference failed");
            run_us_total += t1.elapsed().as_micros() as f64;

            let t2 = Instant::now();
            let (_shape, data) = outputs["policy_p1"]
                .try_extract_tensor::<f32>()
                .expect("extract failed");
            std::hint::black_box(data[0]);
            d2h_us_total += t2.elapsed().as_micros() as f64;
        }

        let h2d_us = h2d_us_total / iters as f64;
        let run_us = run_us_total / iters as f64;
        let d2h_us = d2h_us_total / iters as f64;
        let total_us = h2d_us + run_us + d2h_us;
        let pos_per_s = batch_size as f64 / (total_us / 1_000_000.0);
        let gpu_util = (run_us / total_us) * 100.0;
        peak_pos_s = peak_pos_s.max(pos_per_s);

        println!(
            "  {:>6} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>12} {:>9.0}%",
            batch_size,
            h2d_us,
            run_us,
            d2h_us,
            total_us,
            format_throughput(pos_per_s),
            gpu_util,
        );
    }

    print_footer(peak_pos_s);
}

// ---------------------------------------------------------------------------
// TensorRT benchmark
// ---------------------------------------------------------------------------

#[cfg(feature = "tensorrt")]
fn run_trt_benchmark(
    model_path: &str,
    encoded_buf: &[f32],
    obs_dim: usize,
    batch_sizes: &[usize],
    max_batch: usize,
) {
    use std::ffi::{c_void, CString};

    // Raw CUDA + TRT shim FFI
    extern "C" {
        fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
        fn cudaFree(ptr: *mut c_void) -> i32;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
        fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
        fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
        fn cudaStreamDestroy(stream: *mut c_void) -> i32;

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
    }
    const H2D: i32 = 1;
    const D2H: i32 = 2;

    fn cuda_check(code: i32, op: &str) {
        assert_eq!(code, 0, "CUDA {op} failed with error code {code}");
    }

    // Load TRT-RTX dynamic libs (must be in LD_LIBRARY_PATH)
    alpharat_sampling::load_trt_libs();

    let trt_max_batch = max_batch;
    let num_games = encoded_buf.len() / obs_dim;
    let f = std::mem::size_of::<f32>();

    // Build engine
    eprintln!("Building TensorRT engine (max_batch={})...", trt_max_batch);
    let onnx_bytes = std::fs::read(model_path).expect("failed to read ONNX model");
    let mut out_data: *mut c_void = std::ptr::null_mut();
    let mut out_len: usize = 0;
    let rc = unsafe {
        trt_build_engine(
            onnx_bytes.as_ptr() as *const c_void,
            onnx_bytes.len(),
            1,
            trt_max_batch as i32,
            trt_max_batch as i32,
            2048,
            &mut out_data,
            &mut out_len,
        )
    };
    assert!(rc == 0 && !out_data.is_null(), "Engine build failed ({rc})");
    let engine_data = unsafe { std::slice::from_raw_parts(out_data as *const u8, out_len) };
    eprintln!("Engine built ({} bytes)", out_len);

    // Create session
    let session = unsafe { trt_create_session(engine_data.as_ptr() as *const c_void, out_len) };
    assert!(!session.is_null(), "Failed to create TRT session");
    unsafe { trt_free_buffer(out_data) };

    // Allocate GPU buffers for max batch
    let mut d_input: *mut c_void = std::ptr::null_mut();
    let mut d_pp1: *mut c_void = std::ptr::null_mut();
    let mut d_pp2: *mut c_void = std::ptr::null_mut();
    let mut d_v1: *mut c_void = std::ptr::null_mut();
    let mut d_v2: *mut c_void = std::ptr::null_mut();
    unsafe {
        cuda_check(cudaMalloc(&mut d_input, trt_max_batch * obs_dim * f), "malloc input");
        cuda_check(cudaMalloc(&mut d_pp1, trt_max_batch * 5 * f), "malloc pp1");
        cuda_check(cudaMalloc(&mut d_pp2, trt_max_batch * 5 * f), "malloc pp2");
        cuda_check(cudaMalloc(&mut d_v1, trt_max_batch * f), "malloc v1");
        cuda_check(cudaMalloc(&mut d_v2, trt_max_batch * f), "malloc v2");
    }

    // Bind tensor addresses
    let obs_name = CString::new("observation").unwrap();
    let pp1_name = CString::new("policy_p1").unwrap();
    let pp2_name = CString::new("policy_p2").unwrap();
    let vp1_name = CString::new("pred_value_p1").unwrap();
    let vp2_name = CString::new("pred_value_p2").unwrap();
    unsafe {
        assert_eq!(trt_set_tensor_address(session, obs_name.as_ptr(), d_input), 0);
        assert_eq!(trt_set_tensor_address(session, pp1_name.as_ptr(), d_pp1), 0);
        assert_eq!(trt_set_tensor_address(session, pp2_name.as_ptr(), d_pp2), 0);
        assert_eq!(trt_set_tensor_address(session, vp1_name.as_ptr(), d_v1), 0);
        assert_eq!(trt_set_tensor_address(session, vp2_name.as_ptr(), d_v2), 0);
    }

    // Create CUDA stream
    let mut stream: *mut c_void = std::ptr::null_mut();
    cuda_check(unsafe { cudaStreamCreate(&mut stream) }, "stream create");

    // Host output buffers
    let mut h_pp1 = vec![0.0f32; trt_max_batch * 5];

    print_header();
    let mut peak_pos_s: f64 = 0.0;

    for &batch_size in batch_sizes {
        if batch_size > num_games || batch_size > trt_max_batch {
            break;
        }

        let iters = (5000 / batch_size).max(10).min(500);
        let warmup = (iters / 10).max(5);
        let batch_data = &encoded_buf[..batch_size * obs_dim];
        let shape = [batch_size as i64, obs_dim as i64];

        // Warmup
        for _ in 0..warmup {
            unsafe {
                trt_set_input_shape(session, obs_name.as_ptr(), 2, shape.as_ptr());
                cudaMemcpy(d_input, batch_data.as_ptr() as *const c_void, batch_size * obs_dim * f, H2D);
                trt_enqueue_v3(session, stream);
                cudaStreamSynchronize(stream);
            }
        }

        // Timed runs
        let mut h2d_us_total: f64 = 0.0;
        let mut infer_us_total: f64 = 0.0;
        let mut d2h_us_total: f64 = 0.0;

        for _ in 0..iters {
            // 1. Set input shape + H2D copy
            let t0 = Instant::now();
            unsafe {
                trt_set_input_shape(session, obs_name.as_ptr(), 2, shape.as_ptr());
                cudaMemcpy(
                    d_input,
                    batch_data.as_ptr() as *const c_void,
                    batch_size * obs_dim * f,
                    H2D,
                );
            }
            h2d_us_total += t0.elapsed().as_micros() as f64;

            // 2. GPU inference + sync
            let t1 = Instant::now();
            unsafe {
                trt_enqueue_v3(session, stream);
                cudaStreamSynchronize(stream);
            }
            infer_us_total += t1.elapsed().as_micros() as f64;

            // 3. D2H copy (just policy_p1 to represent real extraction)
            let t2 = Instant::now();
            unsafe {
                cudaMemcpy(
                    h_pp1.as_mut_ptr() as *mut c_void,
                    d_pp1,
                    batch_size * 5 * f,
                    D2H,
                );
            }
            std::hint::black_box(h_pp1[0]);
            d2h_us_total += t2.elapsed().as_micros() as f64;
        }

        let h2d_us = h2d_us_total / iters as f64;
        let infer_us = infer_us_total / iters as f64;
        let d2h_us = d2h_us_total / iters as f64;
        let total_us = h2d_us + infer_us + d2h_us;
        let pos_per_s = batch_size as f64 / (total_us / 1_000_000.0);
        let gpu_util = (infer_us / total_us) * 100.0;
        peak_pos_s = peak_pos_s.max(pos_per_s);

        println!(
            "  {:>6} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>12} {:>9.0}%",
            batch_size, h2d_us, infer_us, d2h_us, total_us,
            format_throughput(pos_per_s),
            gpu_util,
        );
    }

    print_footer(peak_pos_s);

    // Cleanup
    unsafe {
        cudaStreamDestroy(stream);
        cudaFree(d_input);
        cudaFree(d_pp1);
        cudaFree(d_pp2);
        cudaFree(d_v1);
        cudaFree(d_v2);
        trt_destroy_session(session);
    }
}
