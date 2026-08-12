use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use alpharat_bench::calibration::{BackendRequest, CapacityRequest, OnnxProvider, TrialStatus};
use alpharat_bench::capacity::{
    build_capacity_workload, run_capacity_case, standard_capacity_workload, write_capacity_trials,
};
use alpharat_bench::production::{build_capacity_backend, ProductionBackendConfig};

#[derive(Clone, Copy, Debug)]
enum Engine {
    SmartUniform,
    Onnx,
    TensorRt,
}

struct Args {
    engine: Engine,
    provider: OnnxProvider,
    model: Option<PathBuf>,
    tensorrt_cache: Option<PathBuf>,
    width: u8,
    height: u8,
    batch_sizes: Vec<u32>,
    callers: Vec<u32>,
    calls_per_caller: u32,
    warmup_trials: u32,
    measured_trials: u32,
    case_prefix: String,
    output: Option<PathBuf>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("bench-backend-capacity: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let workload =
        standard_capacity_workload(args.width, args.height).map_err(|error| error.to_string())?;
    let games = build_capacity_workload(&workload).map_err(|error| error.to_string())?;
    let requested_backend = requested_backend(args.engine, args.provider);
    let max_batch = args.batch_sizes.iter().copied().max().unwrap();
    let config = ProductionBackendConfig {
        model: args.model.clone(),
        tensorrt_cache: args.tensorrt_cache.clone(),
    };
    let (backend, resolved_backend) = build_capacity_backend(
        &requested_backend,
        &config,
        args.width,
        args.height,
        max_batch,
    )
    .map_err(|error| error.to_string())?;

    let mut trials = Vec::new();
    let mut failed_trials = 0usize;
    for &batch_size in &args.batch_sizes {
        for &callers in &args.callers {
            let case_id = format!(
                "{}-{}-b{batch_size}-c{callers}",
                args.case_prefix,
                backend_label(&requested_backend)
            );
            let request = CapacityRequest {
                backend: requested_backend.clone(),
                batch_size,
                callers,
                calls_per_caller: args.calls_per_caller,
                warmup_trials: args.warmup_trials,
                measured_trials: args.measured_trials,
            };
            eprintln!(
                "measuring {case_id}: {} warmup + {} measured trial(s), {} call(s) per caller",
                request.warmup_trials, request.measured_trials, request.calls_per_caller
            );
            let result = run_capacity_case(
                &case_id,
                &request,
                resolved_backend.clone(),
                backend.as_ref(),
                &games,
            )
            .map_err(|error| error.to_string())?;
            failed_trials += result
                .trials
                .iter()
                .filter(|trial| trial.status == TrialStatus::Failed)
                .count();
            trials.extend(result.trials);
        }
    }

    match &args.output {
        Some(path) => {
            let file = File::create(path)
                .map_err(|error| format!("failed to create '{}': {error}", path.display()))?;
            let mut output = BufWriter::new(file);
            write_capacity_trials(&mut output, &trials)
                .map_err(|error| format!("failed to write '{}': {error}", path.display()))?;
            output
                .flush()
                .map_err(|error| format!("failed to flush '{}': {error}", path.display()))?;
        }
        None => {
            let stdout = std::io::stdout();
            let mut output = BufWriter::new(stdout.lock());
            write_capacity_trials(&mut output, &trials)
                .map_err(|error| format!("failed to write capacity trials: {error}"))?;
            output
                .flush()
                .map_err(|error| format!("failed to flush capacity trials: {error}"))?;
        }
    }

    if failed_trials > 0 {
        return Err(format!(
            "{failed_trials} trial(s) failed; their rows were preserved in the output"
        ));
    }
    Ok(())
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut engine = Engine::SmartUniform;
    let mut provider = OnnxProvider::Cpu;
    let mut model = None;
    let mut tensorrt_cache = None;
    let mut width = 7;
    let mut height = 7;
    let mut batch_sizes = vec![1, 8, 16, 32, 64, 128, 256, 512];
    let mut callers = vec![1, 2, 4];
    let mut calls_per_caller = 32;
    let mut warmup_trials = 2;
    let mut measured_trials = 10;
    let mut case_prefix = "capacity".to_owned();
    let mut output = None;

    let raw = std::env::args().collect::<Vec<_>>();
    let mut index = 1;
    while index < raw.len() {
        let flag = raw[index].as_str();
        let value = |index: &mut usize| -> Result<&str, String> {
            *index += 1;
            raw.get(*index)
                .map(String::as_str)
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag {
            "--backend" => {
                engine = match value(&mut index)? {
                    "smart-uniform" | "fake" => Engine::SmartUniform,
                    "onnx" => Engine::Onnx,
                    "tensorrt" => Engine::TensorRt,
                    other => {
                        return Err(format!(
                            "unknown backend '{other}'; expected smart-uniform, onnx, or tensorrt"
                        ))
                    }
                };
            }
            "--provider" => {
                provider = match value(&mut index)? {
                    "cpu" => OnnxProvider::Cpu,
                    "coreml" => OnnxProvider::Coreml,
                    "cuda" => OnnxProvider::Cuda,
                    other => {
                        return Err(format!(
                            "unknown ONNX provider '{other}'; expected cpu, coreml, or cuda"
                        ))
                    }
                };
            }
            "--model" => model = Some(PathBuf::from(value(&mut index)?)),
            "--tensorrt-cache" => {
                tensorrt_cache = Some(PathBuf::from(value(&mut index)?));
            }
            "--width" => width = parse(value(&mut index)?, "width")?,
            "--height" => height = parse(value(&mut index)?, "height")?,
            "--batch-sizes" => {
                batch_sizes = parse_csv(value(&mut index)?, "batch size")?;
            }
            "--callers" => callers = parse_csv(value(&mut index)?, "caller count")?,
            "--calls-per-caller" => {
                calls_per_caller = parse(value(&mut index)?, "calls per caller")?;
            }
            "--warmups" => warmup_trials = parse(value(&mut index)?, "warmup count")?,
            "--trials" => measured_trials = parse(value(&mut index)?, "trial count")?,
            "--case-prefix" => case_prefix = value(&mut index)?.to_owned(),
            "--output" => output = Some(PathBuf::from(value(&mut index)?)),
            "--help" | "-h" => {
                print_help();
                return Ok(None);
            }
            other => return Err(format!("unknown option '{other}'; use --help for usage")),
        }
        index += 1;
    }

    if width < 2 || height < 2 {
        return Err("width and height must both be at least 2".to_owned());
    }
    require_positive("batch sizes", &batch_sizes)?;
    require_positive("caller counts", &callers)?;
    if calls_per_caller == 0 || measured_trials == 0 {
        return Err("calls per caller and measured trials must be positive".to_owned());
    }
    if case_prefix.trim().is_empty() || case_prefix.contains(',') {
        return Err("case prefix must be non-empty and cannot contain a comma".to_owned());
    }
    if matches!(engine, Engine::Onnx | Engine::TensorRt) && model.is_none() {
        return Err("--model is required for ONNX and TensorRT backends".to_owned());
    }
    if matches!(engine, Engine::SmartUniform) && model.is_some() {
        return Err("--model requires --backend onnx or --backend tensorrt".to_owned());
    }
    if !matches!(engine, Engine::TensorRt) && tensorrt_cache.is_some() {
        return Err("--tensorrt-cache requires --backend tensorrt".to_owned());
    }

    Ok(Some(Args {
        engine,
        provider,
        model,
        tensorrt_cache,
        width,
        height,
        batch_sizes,
        callers,
        calls_per_caller,
        warmup_trials,
        measured_trials,
        case_prefix,
        output,
    }))
}

fn parse<T>(value: &str, label: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|error| format!("invalid {label} '{value}': {error}"))
}

fn parse_csv<T>(value: &str, label: &str) -> Result<Vec<T>, String>
where
    T: std::str::FromStr + Ord + Copy,
    T::Err: std::fmt::Display,
{
    let values = value
        .split(',')
        .map(|part| parse(part.trim(), label))
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err(format!("at least one {label} is required"));
    }
    if values.iter().copied().collect::<BTreeSet<_>>().len() != values.len() {
        return Err(format!("duplicate {label} values are not allowed"));
    }
    Ok(values)
}

fn require_positive(label: &str, values: &[u32]) -> Result<(), String> {
    if values.is_empty() || values.contains(&0) {
        Err(format!("{label} must contain positive values"))
    } else if values.iter().copied().collect::<BTreeSet<_>>().len() != values.len() {
        Err(format!("duplicate {label} are not allowed"))
    } else {
        Ok(())
    }
}

fn print_help() {
    eprintln!("Usage: bench-backend-capacity [OPTIONS]");
    eprintln!();
    eprintln!("Measures the production Backend boundary and writes calibration-v1 CSV rows.");
    eprintln!();
    eprintln!("  --backend smart-uniform|onnx|tensorrt  Backend (default: smart-uniform)");
    eprintln!("  --provider cpu|coreml|cuda              ONNX provider (default: cpu)");
    eprintln!("  --model PATH                            ONNX model for ONNX/TensorRT");
    eprintln!("  --tensorrt-cache DIR                    TensorRT engine cache directory");
    eprintln!("  --width N --height N                    Fixture dimensions (default: 7x7)");
    eprintln!("  --batch-sizes 1,8,...,512               Batch sweep");
    eprintln!("  --callers 1,2,4                         Concurrent caller sweep");
    eprintln!("  --calls-per-caller N                    Calls in each trial (default: 32)");
    eprintln!("  --warmups N                             Warmup trials per case (default: 2)");
    eprintln!("  --trials N                              Measured trials per case (default: 10)");
    eprintln!("  --case-prefix TEXT                      Case-id prefix (default: capacity)");
    eprintln!("  --output PATH                           CSV output (default: stdout)");
    eprintln!();
    eprintln!("Feature examples: --features onnx-coreml, onnx-cuda, or tensorrt");
}

fn requested_backend(engine: Engine, provider: OnnxProvider) -> BackendRequest {
    match engine {
        Engine::SmartUniform => BackendRequest::SmartUniform,
        Engine::Onnx => BackendRequest::Onnx { provider },
        Engine::TensorRt => BackendRequest::TensorRt,
    }
}

fn backend_label(backend: &BackendRequest) -> &'static str {
    match backend {
        BackendRequest::SmartUniform => "smart-uniform",
        BackendRequest::Onnx {
            provider: OnnxProvider::Cpu,
        } => "onnx-cpu",
        BackendRequest::Onnx {
            provider: OnnxProvider::Coreml,
        } => "onnx-coreml",
        BackendRequest::Onnx {
            provider: OnnxProvider::Cuda,
        } => "onnx-cuda",
        BackendRequest::TensorRt => "tensorrt",
    }
}
