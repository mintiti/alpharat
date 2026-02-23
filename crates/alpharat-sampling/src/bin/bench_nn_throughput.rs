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
//!   # ORT CoreML (macOS)
//!   cargo run --release -p alpharat-sampling --features onnx-coreml --bin bench_nn_throughput -- <model.onnx> --device coreml
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
        "cpu" => run_ort_benchmark(model_path, &encoded_buf, obs_dim, &batch_sizes, "cpu"),
        #[cfg(feature = "onnx-cuda")]
        "cuda" => run_ort_benchmark(model_path, &encoded_buf, obs_dim, &batch_sizes, "cuda"),
        #[cfg(feature = "onnx-coreml")]
        "coreml" => run_ort_benchmark(model_path, &encoded_buf, obs_dim, &batch_sizes, "coreml"),
        #[cfg(feature = "tensorrt")]
        "tensorrt" => run_trt_benchmark(model_path, &encoded_buf, obs_dim, &batch_sizes, max_batch, width, height),
        other => {
            let mut supported = vec![];
            if cfg!(feature = "onnx") {
                supported.push("cpu");
            }
            if cfg!(feature = "onnx-cuda") {
                supported.push("cuda");
            }
            if cfg!(feature = "onnx-coreml") {
                supported.push("coreml");
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
    use pyrat::{GameBuilder, GameState, MazeParams};

    let encoder = FlatEncoder::new(width, height);
    let obs_dim = encoder.obs_dim();
    let cheese_count = ((width as u16 * height as u16) as f64 * 0.2).round() as u16;
    let max_turns: u16 = if width <= 5 { 30 } else { 50 };

    let num_games: usize = max_batch;
    eprintln!("Generating {} random games...", num_games);
    let games: Vec<GameState> = (0..num_games)
        .map(|i| {
            let config = GameBuilder::new(width, height)
                .with_max_turns(max_turns)
                .with_random_maze(MazeParams {
                    wall_density: 0.0,
                    mud_density: 0.0,
                    mud_range: 2,
                    connected: true,
                    symmetric: true,
                })
                .with_corner_positions()
                .with_random_cheese(cheese_count, true)
                .build();
            config.create(Some(i as u64)).unwrap()
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
    device: &str,
) {
    use ort::session::Session;
    use ort::value::Tensor;

    let mut session = match device {
        "cpu" => Session::builder()
            .expect("session builder")
            .with_intra_threads(4)
            .expect("set threads")
            .commit_from_file(model_path)
            .expect("load model"),
        #[cfg(feature = "onnx-cuda")]
        "cuda" => {
            let cuda_ep = ort::execution_providers::CUDAExecutionProvider::default();
            match cuda_ep.is_available() {
                Ok(true) => eprintln!("CUDA EP: available"),
                Ok(false) => eprintln!("WARNING: CUDA EP not available! Falling back to CPU."),
                Err(e) => eprintln!("WARNING: CUDA EP check failed: {e}"),
            }
            Session::builder()
                .expect("session builder")
                .with_execution_providers([cuda_ep.build().error_on_failure()])
                .expect("register CUDA EP")
                .commit_from_file(model_path)
                .expect("load model")
        }
        #[cfg(feature = "onnx-coreml")]
        "coreml" => {
            eprintln!("Registering CoreML EP...");
            Session::builder()
                .expect("session builder")
                .with_execution_providers([
                    ort::execution_providers::CoreMLExecutionProvider::default()
                        .with_profile_compute_plan(true)
                        .build()
                        .error_on_failure(),
                ])
                .expect("register CoreML EP")
                .commit_from_file(model_path)
                .expect("load model")
        }
        _ => unreachable!("unsupported device: {device}"),
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
    width: u8,
    height: u8,
) {
    use alpharat_sampling::{FlatEncoder, TensorrtBackend, TensorrtConfig};

    let encoder = FlatEncoder::new(width, height);
    let config = TensorrtConfig {
        max_batch,
        cache_dir: None,
    };
    let backend = TensorrtBackend::new(model_path, encoder, config)
        .expect("failed to create TensorRT backend");

    let num_games = encoded_buf.len() / obs_dim;
    print_header();
    let mut peak_pos_s: f64 = 0.0;

    for &batch_size in batch_sizes {
        if batch_size > num_games || batch_size > max_batch {
            break;
        }

        let iters = (5000 / batch_size).max(10).min(500);
        let warmup = (iters / 10).max(5);
        let batch_data = &encoded_buf[..batch_size * obs_dim];

        // Warmup
        for _ in 0..warmup {
            let _ = backend
                .evaluate_encoded_timed(batch_data, batch_size)
                .expect("TRT inference failed during warmup");
        }

        // Timed runs
        let mut h2d_us_total: f64 = 0.0;
        let mut infer_us_total: f64 = 0.0;
        let mut d2h_us_total: f64 = 0.0;

        for _ in 0..iters {
            let (_results, timing) = backend
                .evaluate_encoded_timed(batch_data, batch_size)
                .expect("TRT inference failed");
            h2d_us_total += timing.h2d_us;
            infer_us_total += timing.infer_us;
            d2h_us_total += timing.d2h_us;
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
}
