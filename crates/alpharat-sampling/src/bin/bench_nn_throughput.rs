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

#[cfg(feature = "tensorrt")]
use std::sync::{Arc, Barrier};
#[cfg(feature = "tensorrt")]
use std::time::Duration;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let _model_path = args
        .get(1)
        .expect("usage: bench_nn_throughput <model.onnx> [--device cpu|cuda|tensorrt] [--width W] [--height H]");

    let mut width: u8 = 7;
    let mut height: u8 = 7;
    let mut device = "cpu";
    let mut max_batch: usize = 262144;
    let mut opt_batch: Option<usize> = None;
    let mut intra_threads: usize = 4;
    let mut execution_contexts: usize = 1;
    let mut callers: usize = 1;
    let mut cuda_graphs = false;
    let mut tensorrt_cache_dir: Option<String> = None;
    let mut requested_batch_sizes = None;
    let mut benchmark_iters: Option<usize> = None;
    let mut verify_parity = false;
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
            "--opt-batch" => {
                opt_batch = Some(args[i + 1].parse().expect("invalid opt-batch"));
                i += 2;
            }
            "--intra-threads" => {
                intra_threads = args[i + 1].parse().expect("invalid intra-threads");
                i += 2;
            }
            "--contexts" => {
                execution_contexts = args[i + 1].parse().expect("invalid contexts");
                i += 2;
            }
            "--callers" => {
                callers = args[i + 1].parse().expect("invalid callers");
                i += 2;
            }
            "--cuda-graphs" => {
                cuda_graphs = true;
                i += 1;
            }
            "--cache-dir" => {
                tensorrt_cache_dir = Some(args[i + 1].clone());
                i += 2;
            }
            "--batches" => {
                requested_batch_sizes = Some(
                    args[i + 1]
                        .split(',')
                        .map(|value| value.parse().expect("invalid batch size"))
                        .collect::<Vec<_>>(),
                );
                i += 2;
            }
            "--iters" => {
                benchmark_iters = Some(args[i + 1].parse().expect("invalid iters"));
                i += 2;
            }
            "--verify-parity" => {
                verify_parity = true;
                i += 1;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    let opt_batch = opt_batch.unwrap_or(max_batch);

    #[cfg(not(feature = "tensorrt"))]
    let _ = (
        execution_contexts,
        callers,
        cuda_graphs,
        opt_batch,
        tensorrt_cache_dir,
        benchmark_iters,
        verify_parity,
    );

    // --- Pre-encode games ---
    let (_encoded_buf, obs_dim) = pre_encode_games(width, height, max_batch);

    println!(
        "NN Inference Throughput — {width}x{height}, obs_dim={obs_dim}, device={device}, intra_threads={intra_threads}",
    );

    let mut batch_sizes: Vec<usize> = requested_batch_sizes.unwrap_or_else(|| {
        vec![
            1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
            262144,
        ]
    });
    // Add max_batch to the list if it's not already there
    if !batch_sizes.contains(&max_batch) {
        batch_sizes.push(max_batch);
        batch_sizes.sort();
    }
    // Only keep batch sizes up to max_batch
    batch_sizes.retain(|&b| b <= max_batch);

    match device {
        #[cfg(feature = "onnx")]
        "cpu" => run_ort_benchmark(
            _model_path,
            &_encoded_buf,
            obs_dim,
            &batch_sizes,
            "cpu",
            intra_threads,
        ),
        #[cfg(feature = "onnx-cuda")]
        "cuda" => run_ort_benchmark(
            _model_path,
            &_encoded_buf,
            obs_dim,
            &batch_sizes,
            "cuda",
            intra_threads,
        ),
        #[cfg(feature = "onnx-coreml")]
        "coreml" => run_ort_benchmark(
            _model_path,
            &_encoded_buf,
            obs_dim,
            &batch_sizes,
            "coreml",
            intra_threads,
        ),
        #[cfg(feature = "tensorrt")]
        "tensorrt" => run_trt_benchmark(
            _model_path,
            &_encoded_buf,
            obs_dim,
            &batch_sizes,
            TrtBenchConfig {
                opt_batch,
                max_batch,
                width,
                height,
                execution_contexts,
                callers,
                cuda_graphs,
                cache_dir: tensorrt_cache_dir.map(Into::into),
                benchmark_iters,
                verify_parity,
            },
        ),
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
            eprintln!("Device '{other}' not available. Compiled with support for: {supported:?}",);
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
    eprintln!("Generating {num_games} random games...");
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

    eprintln!("Pre-encoding {num_games} games...");
    let mut buf = vec![0.0f32; num_games * obs_dim];
    let t0 = Instant::now();
    for (i, game) in games.iter().enumerate() {
        encoder.encode_into(game, &mut buf, i * obs_dim);
    }
    let encode_us = t0.elapsed().as_micros();
    println!(
        "Pre-encoded {num_games} games in {:.1}ms ({:.0}ns/pos)\n",
        encode_us as f64 / 1000.0,
        (encode_us as f64 * 1000.0) / num_games as f64,
    );

    (buf, obs_dim)
}

#[allow(dead_code)]
fn format_throughput(pos_per_sec: f64) -> String {
    if pos_per_sec >= 1_000_000.0 {
        format!("{:.2}M", pos_per_sec / 1_000_000.0)
    } else if pos_per_sec >= 1_000.0 {
        format!("{:.1}k", pos_per_sec / 1_000.0)
    } else {
        format!("{pos_per_sec:.0}")
    }
}

#[allow(dead_code)]
fn print_header() {
    println!(
        "  {:>6} {:>10} {:>10} {:>10} {:>10} {:>12} {:>10}",
        "batch", "h2d_µs", "infer_µs", "d2h_µs", "total_µs", "pos/s", "GPU_util%"
    );
    println!("  {:-<90}", "");
}

#[allow(dead_code)]
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
    intra_threads: usize,
) {
    use ort::session::Session;
    use ort::value::Tensor;

    let mut session = match device {
        "cpu" => Session::builder()
            .expect("session builder")
            .with_intra_threads(intra_threads)
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
                .with_intra_threads(intra_threads)
                .expect("set threads")
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
                .with_intra_threads(intra_threads)
                .expect("set threads")
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

        let iters = (5000 / batch_size).clamp(10, 500);
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
struct TrtBenchConfig {
    opt_batch: usize,
    max_batch: usize,
    width: u8,
    height: u8,
    execution_contexts: usize,
    callers: usize,
    cuda_graphs: bool,
    cache_dir: Option<std::path::PathBuf>,
    benchmark_iters: Option<usize>,
    verify_parity: bool,
}

#[cfg(feature = "tensorrt")]
fn run_trt_benchmark(
    model_path: &str,
    encoded_buf: &[f32],
    obs_dim: usize,
    batch_sizes: &[usize],
    config: TrtBenchConfig,
) {
    use alpharat_sampling::{FlatEncoder, TensorrtBackend, TensorrtConfig};

    let TrtBenchConfig {
        opt_batch,
        max_batch,
        width,
        height,
        execution_contexts,
        callers,
        cuda_graphs,
        cache_dir,
        benchmark_iters,
        verify_parity,
    } = config;
    assert!(callers > 0, "callers must be at least 1");

    let encoder = FlatEncoder::new(width, height);
    let config = TensorrtConfig {
        opt_batch,
        max_batch,
        cache_dir: cache_dir.clone(),
        execution_contexts,
        cuda_graphs,
    };
    let backend = TensorrtBackend::new(model_path, encoder, config)
        .expect("failed to create TensorRT backend");

    let num_games = encoded_buf.len() / obs_dim;
    if verify_parity {
        let parity_batch = batch_sizes.first().copied().unwrap_or(1).min(max_batch);
        verify_trt_parity(
            model_path,
            &backend,
            &encoded_buf[..parity_batch * obs_dim],
            parity_batch,
            TrtParityConfig {
                width,
                height,
                execution_contexts,
                max_batch,
                cache_dir,
            },
        );
    }
    println!(
        "  TensorRT profile=MIN1/OPT{opt_batch}/MAX{max_batch}, lanes={execution_contexts}, callers={callers}, cuda_graphs={cuda_graphs}"
    );
    print_trt_header();
    let mut peak_pos_s: f64 = 0.0;

    for &batch_size in batch_sizes {
        if batch_size > num_games || batch_size > max_batch {
            break;
        }

        let iters = benchmark_iters.unwrap_or_else(|| (5000 / batch_size).clamp(10, 500));
        let warmup = (iters / 10).clamp(5, 100);
        let batch_data = &encoded_buf[..batch_size * obs_dim];

        let timing = benchmark_trt_batch(&backend, batch_data, batch_size, callers, iters, warmup);
        let total_calls = callers * iters;
        let h2d_us = timing.h2d_us_total / total_calls as f64;
        let infer_us = timing.infer_us_total / total_calls as f64;
        let d2h_us = timing.d2h_us_total / total_calls as f64;
        let effective_us = timing.wall_time.as_secs_f64() * 1_000_000.0 / total_calls as f64;
        let pos_per_s = batch_size as f64 * total_calls as f64 / timing.wall_time.as_secs_f64();
        peak_pos_s = peak_pos_s.max(pos_per_s);

        println!(
            "  {:>6} {:>7} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>12}",
            batch_size,
            callers,
            h2d_us,
            infer_us,
            d2h_us,
            effective_us,
            format_throughput(pos_per_s),
        );
    }

    print_footer(peak_pos_s);
}

#[cfg(feature = "tensorrt")]
struct TrtParityConfig {
    width: u8,
    height: u8,
    execution_contexts: usize,
    max_batch: usize,
    cache_dir: Option<std::path::PathBuf>,
}

#[cfg(feature = "tensorrt")]
fn verify_trt_parity(
    model_path: &str,
    candidate: &alpharat_sampling::TensorrtBackend<alpharat_sampling::FlatEncoder>,
    batch_data: &[f32],
    batch_size: usize,
    config: TrtParityConfig,
) {
    use alpharat_sampling::{FlatEncoder, TensorrtBackend, TensorrtConfig};

    let TrtParityConfig {
        width,
        height,
        execution_contexts,
        max_batch,
        cache_dir,
    } = config;

    let baseline = TensorrtBackend::new(
        model_path,
        FlatEncoder::new(width, height),
        TensorrtConfig {
            opt_batch: max_batch,
            max_batch,
            cache_dir,
            execution_contexts: 1,
            cuda_graphs: false,
        },
    )
    .expect("failed to create TensorRT parity baseline");
    let (expected, _) = baseline
        .evaluate_encoded_timed(batch_data, batch_size)
        .expect("TensorRT parity baseline failed");

    let gate = Arc::new(Barrier::new(execution_contexts));
    let candidate_results = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(execution_contexts);
        for _ in 0..execution_contexts {
            let gate = Arc::clone(&gate);
            handles.push(scope.spawn(move || {
                gate.wait();
                candidate
                    .evaluate_encoded_timed(batch_data, batch_size)
                    .expect("TensorRT parity candidate failed")
                    .0
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("TensorRT parity caller panicked"))
            .collect::<Vec<_>>()
    });

    let mut max_abs_diff = 0.0_f32;
    for actual in &candidate_results {
        assert_eq!(
            actual.len(),
            expected.len(),
            "TensorRT parity length mismatch"
        );
        for (actual, expected) in actual.iter().zip(&expected) {
            for (actual, expected) in actual.policy_p1.iter().zip(expected.policy_p1) {
                max_abs_diff = max_abs_diff.max((actual - expected).abs());
            }
            for (actual, expected) in actual.policy_p2.iter().zip(expected.policy_p2) {
                max_abs_diff = max_abs_diff.max((actual - expected).abs());
            }
            max_abs_diff = max_abs_diff.max((actual.value_p1 - expected.value_p1).abs());
            max_abs_diff = max_abs_diff.max((actual.value_p2 - expected.value_p2).abs());
        }
    }
    assert!(
        max_abs_diff <= 1.0e-4,
        "TensorRT parity max abs diff {max_abs_diff} exceeds 1e-4"
    );
    println!(
        "  Parity: {execution_contexts} simultaneous lane(s), batch={batch_size}, max_abs_diff={max_abs_diff:.3e}"
    );

    verify_trt_root_behavior(candidate, &baseline, width, height);
}

#[cfg(feature = "tensorrt")]
fn verify_trt_root_behavior(
    candidate: &alpharat_sampling::TensorrtBackend<alpharat_sampling::FlatEncoder>,
    baseline: &alpharat_sampling::TensorrtBackend<alpharat_sampling::FlatEncoder>,
    width: u8,
    height: u8,
) {
    use alpharat_mcts::{run_search, Backend, MCTSTree, SearchConfig};
    use pyrat::GameBuilder;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    let game_config = GameBuilder::new(width, height)
        .with_max_turns(50)
        .with_open_maze()
        .with_corner_positions()
        .with_random_cheese(10, true)
        .build();
    let game = game_config
        .create(Some(20_260_813))
        .expect("failed to create deterministic root-parity game");
    let search_config = SearchConfig {
        c_puct: 0.512,
        fpu_reduction: 0.459,
        force_k: 0.103,
        noise_epsilon: 0.0,
        ..SearchConfig::default()
    };

    let run = |backend: &dyn Backend| {
        let mut tree = MCTSTree::new(&game);
        let mut rng = SmallRng::seed_from_u64(20_260_813);
        run_search(
            &mut tree,
            &game,
            backend,
            &search_config,
            2_048,
            16,
            &mut rng,
        )
        .expect("deterministic root-parity search failed")
    };

    let expected = run(baseline);
    let actual = run(candidate);
    let policy_l1_p1: f32 = expected
        .policy_p1
        .iter()
        .zip(actual.policy_p1)
        .map(|(expected, actual)| (expected - actual).abs())
        .sum();
    let policy_l1_p2: f32 = expected
        .policy_p2
        .iter()
        .zip(actual.policy_p2)
        .map(|(expected, actual)| (expected - actual).abs())
        .sum();
    let value_abs_p1 = (expected.value_p1 - actual.value_p1).abs();
    let value_abs_p2 = (expected.value_p2 - actual.value_p2).abs();
    println!(
        "  Root behavior: policy_l1_p1={policy_l1_p1:.6}, policy_l1_p2={policy_l1_p2:.6}, value_abs_p1={value_abs_p1:.6}, value_abs_p2={value_abs_p2:.6}"
    );
}

#[cfg(feature = "tensorrt")]
fn print_trt_header() {
    println!(
        "  {:>6} {:>7} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "batch", "callers", "h2d_us", "infer_us", "d2h_us", "eff_us", "pos/s"
    );
    println!("  {:-<86}", "");
}

#[cfg(feature = "tensorrt")]
struct ConcurrentTrtTiming {
    h2d_us_total: f64,
    infer_us_total: f64,
    d2h_us_total: f64,
    wall_time: Duration,
}

#[cfg(feature = "tensorrt")]
fn benchmark_trt_batch(
    backend: &alpharat_sampling::TensorrtBackend<alpharat_sampling::FlatEncoder>,
    batch_data: &[f32],
    batch_size: usize,
    callers: usize,
    iters: usize,
    warmup: usize,
) -> ConcurrentTrtTiming {
    let warmup_gate = Arc::new(Barrier::new(callers));
    let timed_gate = Arc::new(Barrier::new(callers));

    let per_caller = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(callers);
        for _ in 0..callers {
            let warmup_gate = Arc::clone(&warmup_gate);
            let timed_gate = Arc::clone(&timed_gate);
            handles.push(scope.spawn(move || {
                warmup_gate.wait();
                for _ in 0..warmup {
                    let _ = backend
                        .evaluate_encoded_timed(batch_data, batch_size)
                        .expect("TRT inference failed during warmup");
                }

                timed_gate.wait();
                let start = Instant::now();
                let mut h2d_us_total = 0.0;
                let mut infer_us_total = 0.0;
                let mut d2h_us_total = 0.0;
                for _ in 0..iters {
                    let (_results, timing) = backend
                        .evaluate_encoded_timed(batch_data, batch_size)
                        .expect("TRT inference failed");
                    h2d_us_total += timing.h2d_us;
                    infer_us_total += timing.infer_us;
                    d2h_us_total += timing.d2h_us;
                }
                (h2d_us_total, infer_us_total, d2h_us_total, start.elapsed())
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("TRT benchmark caller panicked"))
            .collect::<Vec<_>>()
    });

    ConcurrentTrtTiming {
        h2d_us_total: per_caller.iter().map(|result| result.0).sum(),
        infer_us_total: per_caller.iter().map(|result| result.1).sum(),
        d2h_us_total: per_caller.iter().map(|result| result.2).sum(),
        wall_time: per_caller
            .iter()
            .map(|result| result.3)
            .max()
            .expect("at least one TRT benchmark caller"),
    }
}
