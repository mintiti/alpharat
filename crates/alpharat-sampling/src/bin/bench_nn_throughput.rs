//! Isolated ONNX inference throughput benchmark with detailed timing breakdown.
//!
//! Measures: tensor creation (+ H2D copy), GPU inference, output extraction (D2H copy).
//!
//! Usage:
//!   cargo run --release -p alpharat-sampling --features onnx-cuda --bin bench_nn_throughput -- <model.onnx>
//!
//! For NVIDIA Nsight profiling:
//!   nsys profile --stats=true target/release/bench_nn_throughput model.onnx

#[cfg(feature = "onnx")]
fn main() {
    use alpharat_sampling::encoder::ObservationEncoder;
    use alpharat_sampling::FlatEncoder;
    use ort::ep::ExecutionProvider;
    use ort::session::Session;
    use pyrat::{CheeseConfig, GameState, MazeConfig};
    use std::time::Instant;

    // --- Parse args ---
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .expect("usage: bench_nn_throughput <model.onnx> [--width W] [--height H]");

    let mut width: u8 = 7;
    let mut height: u8 = 7;
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
            other => panic!("unknown arg: {other}"),
        }
    }

    let encoder = FlatEncoder::new(width, height);
    let obs_dim = encoder.obs_dim();
    let cheese_count = ((width as u16 * height as u16) as f64 * 0.2).round() as u16;
    let max_turns: u16 = if width <= 5 { 30 } else { 50 };

    // --- Generate random games ---
    let num_games: usize = 262144; // 256k — enough for large GPU batches
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

    // --- Pre-encode all games into a reusable buffer ---
    eprintln!("Pre-encoding {} games...", num_games);
    let mut encoded_buf = vec![0.0f32; num_games * obs_dim];
    let encode_start = Instant::now();
    for (i, game) in games.iter().enumerate() {
        encoder.encode_into(game, &mut encoded_buf, i * obs_dim);
    }
    let encode_all_us = encode_start.elapsed().as_micros();
    let encode_per_pos_ns = (encode_all_us as f64 * 1000.0) / num_games as f64;

    println!(
        "ONNX Inference Throughput — {}x{}, obs_dim={}",
        width, height, obs_dim
    );
    println!(
        "Pre-encoded {} games in {:.1}ms ({:.0}ns/pos)\n",
        num_games,
        encode_all_us as f64 / 1000.0,
        encode_per_pos_ns
    );

    let batch_sizes: Vec<usize> = vec![
        1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
        262144,
    ];

    // --- Build CUDA session ---
    #[cfg(feature = "onnx-cuda")]
    {
        let cuda_ep = ort::execution_providers::CUDAExecutionProvider::default();
        match cuda_ep.is_available() {
            Ok(true) => eprintln!("CUDA EP: available"),
            Ok(false) => {
                eprintln!("WARNING: CUDA EP not available! Will fall back to CPU.");
                eprintln!("  Check: CUDA version, cuDNN, onnxruntime-gpu compatibility");
            }
            Err(e) => eprintln!("WARNING: CUDA EP check failed: {e}"),
        }

        let mut session = Session::builder()
            .expect("failed to create session builder")
            .with_execution_providers([cuda_ep.build()])
            .expect("failed to register CUDA EP")
            .commit_from_file(model_path)
            .expect("failed to load model");

        run_benchmark(&mut session, &encoded_buf, obs_dim, &batch_sizes);
    }

    #[cfg(all(feature = "onnx", not(feature = "onnx-cuda")))]
    {
        let mut session = Session::builder()
            .expect("failed to create session builder")
            .with_intra_threads(4)
            .expect("failed to set threads")
            .commit_from_file(model_path)
            .expect("failed to load model");

        eprintln!("Running CPU backend (no CUDA feature)");
        run_benchmark(&mut session, &encoded_buf, obs_dim, &batch_sizes);
    }
}

#[cfg(feature = "onnx")]
fn run_benchmark(
    session: &mut ort::session::Session,
    encoded_buf: &[f32],
    obs_dim: usize,
    batch_sizes: &[usize],
) {
    use ort::value::Tensor;
    use std::time::Instant;

    let num_games = encoded_buf.len() / obs_dim;

    println!(
        "  {:>6} {:>10} {:>10} {:>10} {:>10} {:>12} {:>10}",
        "batch", "tensor_µs", "run_µs", "extract_µs", "total_µs", "pos/s", "GPU_util%"
    );
    println!("  {:-<90}", "");

    let mut peak_pos_s: f64 = 0.0;

    for &batch_size in batch_sizes {
        if batch_size > num_games {
            break;
        }

        // Scale iterations inversely with batch size to keep total work reasonable
        let iters = (5000 / batch_size).max(10).min(500);
        let warmup = (iters / 10).max(5);

        let batch_data = &encoded_buf[..batch_size * obs_dim];

        // --- Warmup ---
        for _ in 0..warmup {
            let input = Tensor::from_array(([batch_size, obs_dim], batch_data.to_vec()))
                .expect("tensor creation failed");
            let _outputs = session
                .run(ort::inputs!["observation" => input])
                .expect("inference failed");
        }

        // --- Timed runs with breakdown ---
        let mut tensor_us_total: f64 = 0.0;
        let mut run_us_total: f64 = 0.0;
        let mut extract_us_total: f64 = 0.0;

        for _ in 0..iters {
            // 1. Tensor creation (includes memory alloc + potential H2D copy)
            let t0 = Instant::now();
            let input = Tensor::from_array(([batch_size, obs_dim], batch_data.to_vec()))
                .expect("tensor creation failed");
            tensor_us_total += t0.elapsed().as_micros() as f64;

            // 2. Inference (GPU kernels + sync)
            let t1 = Instant::now();
            let outputs = session
                .run(ort::inputs!["observation" => input])
                .expect("inference failed");
            run_us_total += t1.elapsed().as_micros() as f64;

            // 3. Output extraction (D2H copy when accessing data)
            let t2 = Instant::now();
            // Force materialization by accessing the tensor data
            let (_shape, data) = outputs["policy_p1"]
                .try_extract_tensor::<f32>()
                .expect("extract failed");
            // Touch the data to ensure D2H transfer completes
            std::hint::black_box(data[0]);
            extract_us_total += t2.elapsed().as_micros() as f64;
        }

        let tensor_us = tensor_us_total / iters as f64;
        let run_us = run_us_total / iters as f64;
        let extract_us = extract_us_total / iters as f64;
        let total_us = tensor_us + run_us + extract_us;

        let pos_per_s = batch_size as f64 / (total_us / 1_000_000.0);
        // GPU utilization estimate: run time as fraction of total
        let gpu_util = (run_us / total_us) * 100.0;

        if pos_per_s > peak_pos_s {
            peak_pos_s = pos_per_s;
        }

        println!(
            "  {:>6} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>12} {:>9.0}%",
            batch_size,
            tensor_us,
            run_us,
            extract_us,
            total_us,
            format_throughput(pos_per_s),
            gpu_util,
        );
    }

    println!("\n  Peak throughput: {} pos/s\n", format_throughput(peak_pos_s));

    println!("  Breakdown:");
    println!("    tensor_µs  = Tensor::from_array (CPU alloc + Vec clone + potential H2D copy)");
    println!("                 Includes per-batch allocation — representative of production encode cost.");
    println!("    run_µs     = session.run (GPU kernel dispatch + sync)");
    println!("                 For CUDA: may include GPU sync, making run/extract split approximate.");
    println!("    extract_µs = try_extract_tensor (D2H copy + data access)");
    println!("                 For CUDA: may show ~0 if sync already happened in run_µs.");
    println!("    total_µs   = sum of above — this is the reliable end-to-end number.");
    println!("\n  For detailed GPU profiling: nsys profile target/release/bench_nn_throughput model.onnx");
}

#[cfg(feature = "onnx")]
fn format_throughput(pos_per_sec: f64) -> String {
    if pos_per_sec >= 1_000_000.0 {
        format!("{:.2}M", pos_per_sec / 1_000_000.0)
    } else if pos_per_sec >= 1_000.0 {
        format!("{:.1}k", pos_per_sec / 1_000.0)
    } else {
        format!("{:.0}", pos_per_sec)
    }
}

#[cfg(not(feature = "onnx"))]
fn main() {
    eprintln!("Build with --features onnx (or onnx-cuda)");
}
