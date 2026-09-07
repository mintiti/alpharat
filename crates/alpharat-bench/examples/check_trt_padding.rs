//! Explicit GPU regression check for opt-in TensorRT execution controls.
use alpharat_bench::inference::{driver::corpus_games, model::Corpus, Result};
use alpharat_eval_core::{Backend, EvalResult};
use alpharat_sampling::{FlatEncoder, ObservationEncoder, TensorrtBackend, TensorrtConfig};
use pyrat::Direction;
use serde_json::json;
use std::{fs, path::PathBuf};

#[derive(Clone, Copy, serde::Serialize)]
struct Execution {
    pad_to_max: bool,
    cuda_graph: bool,
}
impl Execution {
    const EXACT: Self = Self {
        pad_to_max: false,
        cuda_graph: false,
    };
    const PADDED: Self = Self {
        pad_to_max: true,
        ..Self::EXACT
    };
    const GRAPHS: Self = Self {
        cuda_graph: true,
        ..Self::PADDED
    };
    const EXACT_GRAPHS: Self = Self {
        cuda_graph: true,
        ..Self::EXACT
    };
}

fn delta(a: &[EvalResult], b: &[EvalResult]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(a, b)| {
            a.policy_p1
                .iter()
                .zip(&b.policy_p1)
                .chain(a.policy_p2.iter().zip(&b.policy_p2))
                .chain(std::iter::once((&a.value_p1, &b.value_p1)))
                .chain(std::iter::once((&a.value_p2, &b.value_p2)))
                .map(|(a, b)| {
                    assert!(a.is_finite() && b.is_finite());
                    (a - b).abs()
                })
        })
        .fold(0.0, f32::max)
}
fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    assert!(
        (5..=9).contains(&args.len()),
        "model corpus cache output [mode] [max_batch] [lanes] [sizes_csv]"
    );
    let mode = args.get(5).map_or("padding", String::as_str);
    let (baseline_execution, candidate_execution) = match mode {
        "padding" => (Execution::EXACT, Execution::PADDED),
        "graphs" => (Execution::PADDED, Execution::GRAPHS),
        "graphs-exact" => (Execution::EXACT, Execution::EXACT_GRAPHS),
        "combined" | "buckets" | "buckets64" | "lanes" | "bucket-lanes" | "pipeline" => {
            (Execution::EXACT, Execution::GRAPHS)
        }
        _ => return Err(format!("unknown execution check mode: {mode}").into()),
    };
    let max_batch = args.get(6).map_or(Ok(128), |s| s.parse::<usize>())?;
    assert!((1..=128).contains(&max_batch));
    let lane_count = args.get(7).map_or(Ok(2), |s| s.parse::<usize>())?;
    assert!((1..=8).contains(&lane_count));
    let custom_sizes: Option<Vec<usize>> = args
        .get(8)
        .map(|s| s.split(',').map(str::parse).collect())
        .transpose()?;
    if ["buckets", "buckets64", "bucket-lanes", "pipeline"].contains(&mode) && max_batch != 128 {
        return Err("bucket check modes require max_batch 128".into());
    }
    let corpus: Corpus = serde_json::from_slice(&fs::read(&args[2])?)?;
    let width = corpus.positions[0].width;
    let height = corpus.positions[0].height;
    let mut games = corpus_games(&corpus)?;
    // Retain late as well as opening states; fixed local action stream, no sampling RNG.
    let mut action_seed = 906_u64;
    for original in games.clone().into_iter().take(32) {
        let mut game = original;
        for turn in 0..100 {
            action_seed = action_seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let a = Direction::try_from(((action_seed >> 32) % 5) as u8).unwrap();
            action_seed = action_seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let b = Direction::try_from(((action_seed >> 32) % 5) as u8).unwrap();
            game.make_move(a, b);
            if game.check_game_over() {
                break;
            }
            if [11, 39, 79].contains(&turn) {
                games.push(game.clone());
            }
        }
    }
    let make = |execution: Execution, profile| {
        TensorrtBackend::new(
            &args[1],
            FlatEncoder::new(width, height),
            TensorrtConfig {
                max_batch,
                opt_batch: Some(max_batch),
                cache_dir: Some(PathBuf::from(&args[3])),
                pad_to_max: execution.pad_to_max,
                profile_stages: profile,
                cuda_graph: execution.cuda_graph,
                execution_sizes: if execution.cuda_graph && custom_sizes.is_some() {
                    custom_sizes.clone().unwrap()
                } else if execution.cuda_graph
                    && ["buckets", "bucket-lanes", "pipeline"].contains(&mode)
                {
                    vec![32, 64, max_batch]
                } else if execution.cuda_graph && mode == "buckets64" {
                    vec![64, max_batch]
                } else {
                    Vec::new()
                },
                execution_lanes: if execution.cuda_graph
                    && ["lanes", "bucket-lanes", "pipeline"].contains(&mode)
                {
                    lane_count
                } else {
                    1
                },
                serialize_device: execution.cuda_graph && mode == "pipeline",
                ..TensorrtConfig::default()
            },
        )
    };
    let control = make(baseline_execution, false)?;
    let candidate = make(candidate_execution, false)?;
    assert_eq!(control.engine_sha256(), candidate.engine_sha256());
    assert!(candidate.evaluate_batch(&[])?.is_empty());
    assert!(candidate
        .evaluate_batch(&vec![&games[0]; max_batch + 1])
        .is_err());
    let mut rows = Vec::new();
    let mut worst = 0_f32;
    for pass in 0..2 {
        for k in 1..=max_batch {
            let n = if pass == 0 { k } else { max_batch + 1 - k };
            let input = (0..n)
                .map(|i| &games[(i * 17 + k * 7 + pass * 31) % games.len()])
                .collect::<Vec<_>>();
            let (a, b) = if pass == 0 {
                (
                    control.evaluate_batch(&input)?,
                    candidate.evaluate_batch(&input)?,
                )
            } else {
                let b = candidate.evaluate_batch(&input)?;
                (control.evaluate_batch(&input)?, b)
            };
            let d = delta(&a, &b);
            worst = worst.max(d);
            rows.push(json!({"pass":pass,"real_rows":n,"max_abs_delta":d}));
        }
    }
    let timed = make(candidate_execution, true)?;
    let encoder = FlatEncoder::new(width, height);
    let mut counted = 0;
    for n in [1, 7, 31, 64, 127, 128, 1]
        .into_iter()
        .filter(|n| *n <= max_batch)
    {
        let input = (0..n)
            .map(|i| &games[(i * 17 + n) % games.len()])
            .collect::<Vec<_>>();
        let mut encoded = vec![0.0; n * encoder.obs_dim()];
        for (i, g) in input.iter().enumerate() {
            encoder.encode_into(g, &mut encoded, i * encoder.obs_dim());
        }
        let (b, _) = timed.evaluate_encoded_timed(&encoded, n)?;
        let d = delta(&control.evaluate_batch(&input)?, &b);
        worst = worst.max(d);
        counted += n;
        rows.push(json!({"path":"encoded_timed","real_rows":n,"max_abs_delta":d}));
    }
    assert_eq!(timed.stats().snapshot().positions, counted as u64);
    assert!(timed.evaluate_encoded_timed(&[0.0], 1).is_err());
    assert!(timed.evaluate_batch(&[])?.is_empty());
    let one = timed.evaluate_batch(&[&games[0]])?;
    assert_eq!(one.len(), 1);
    assert_eq!(timed.stats().snapshot().positions, (counted + 1) as u64);
    let concurrent = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker in 0..4 {
            let (games, control, candidate) = (&games, &control, &candidate);
            handles.push(scope.spawn(move || {
                let mut max_delta = 0_f32;
                for call in 0..12 {
                    let n = 1 + (worker * 7 + call * 13) % max_batch;
                    let input = (0..n)
                        .map(|i| &games[(i * 11 + call + worker) % games.len()])
                        .collect::<Vec<_>>();
                    let a = control.evaluate_batch(&input).unwrap();
                    let b = candidate.evaluate_batch(&input).unwrap();
                    max_delta = max_delta.max(delta(&a, &b));
                }
                max_delta
            }));
        }
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    for (worker, d) in concurrent.iter().enumerate() {
        worst = worst.max(*d);
        rows.push(json!({"path":"concurrent","worker":worker,"calls":12,"max_abs_delta":d}));
    }
    let result = json!({"model":args[1],"corpus":args[2],"baseline_execution":baseline_execution,"candidate_execution":candidate_execution,"max_batch":max_batch,"requested_lane_count":lane_count,"custom_execution_sizes":custom_sizes,"mode":mode,"engine_sha256":control.engine_sha256(),
        "states":games.len(),"guard":0.0001,"max_abs_delta":worst,"passed":worst<=0.0001,
        "checked_empty_oversize_recovery":true,"checked_stats_real_rows":true,"concurrent_calls":48,"comparisons":rows});
    fs::write(&args[4], serde_json::to_vec_pretty(&result)?)?;
    println!(
        "{}",
        json!({"states":games.len(),"comparisons":rows.len(),"max_abs_delta":worst,"passed":worst<=0.0001})
    );
    if worst > 0.0001 {
        return Err("TensorRT output regression guard exceeded".into());
    }
    Ok(())
}
