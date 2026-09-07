//! Native inference experiment entry point. No shell interpolation of workload arguments.
use alpharat_bench::inference::{artifact, driver, model::Mode, report, supervisor, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
const HELP:&str="alpharat-infer
  check --plan PLAN
  run --plan PLAN --out NEW_DIRECTORY [--nsys PATH]
  diagnose --run RUN --attempt ID --mode latency|timeline|stages --out NEW_DIRECTORY [--nsys PATH]
  report --run RUN [--format markdown|json] [--out FILE]
  compare --left RUN --right RUN [--vary axis,axis] [--format markdown|json] [--out FILE]
  --describe
Relative paths inside plans resolve against the plan directory. Existing run directories are never overwritten.";
fn main() {
    if let Err(error) = entry() {
        eprintln!("alpharat-infer: {error}");
        std::process::exit(1);
    }
}
fn entry() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        println!("{HELP}");
        return Ok(());
    };
    if command == "--help" || command == "help" {
        println!("{HELP}");
        return Ok(());
    }
    if command == "--describe" {
        println!("{}", artifact::build_description());
        return Ok(());
    }
    if command == "_trial" {
        let request = PathBuf::from(args.next().ok_or("missing child request")?);
        let out = PathBuf::from(args.next().ok_or("missing child output")?);
        if args.next().is_some() {
            return Err("unexpected child argument".into());
        }
        let request = artifact::read_json(&request)?;
        supervisor::worker_handshake(&out)?;
        if let Err(error) = driver::execute(&request, &out) {
            artifact::write_json(
                &out.join("error.json"),
                &serde_json::json!({"error":error.to_string(),"at":artifact::now()}),
            )?;
            return Err(error);
        }
        return Ok(());
    }
    let mut flags = BTreeMap::new();
    while let Some(key) = args.next() {
        if !key.starts_with("--") || flags.contains_key(&key) {
            return Err(format!("invalid or duplicate option: {key}").into());
        }
        flags.insert(key, args.next().ok_or("option needs a value")?);
    }
    let allowed: &[&str] = match command.as_str() {
        "check" => &["--plan"],
        "run" => &["--plan", "--out", "--nsys"],
        "diagnose" => &["--run", "--attempt", "--mode", "--out", "--nsys"],
        "report" => &["--run", "--format", "--out"],
        "compare" => &["--left", "--right", "--vary", "--format", "--out"],
        _ => return Err(format!("unknown command: {command}\n{HELP}").into()),
    };
    if let Some(key) = flags.keys().find(|k| !allowed.contains(&k.as_str())) {
        return Err(format!("unknown option: {key}").into());
    }
    let get = |key: &str| {
        flags
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| format!("missing {key}"))
    };
    let nsys = flags.get("--nsys").map(Path::new);
    match command.as_str() {
        "check" => println!(
            "{}",
            serde_json::to_string_pretty(&supervisor::check(Path::new(get("--plan")?))?)?
        ),
        "run" | "diagnose" => {
            let record = if command == "run" {
                supervisor::run(Path::new(get("--plan")?), Path::new(get("--out")?), nsys)?
            } else {
                let mode = match get("--mode")? {
                    "latency" => Mode::Latency,
                    "timeline" => Mode::Timeline,
                    "stages" => Mode::Stages,
                    _ => return Err("diagnostic mode must be latency, timeline, or stages".into()),
                };
                supervisor::diagnose(
                    Path::new(get("--run")?),
                    get("--attempt")?,
                    mode,
                    Path::new(get("--out")?),
                    nsys,
                )?
            };
            println!(
                "Run {}: {:?}; {}/run.json",
                record.run_id,
                record.status,
                get("--out")?
            );
            if record.status != artifact::Status::Completed {
                return Err("run did not complete; failure artifacts retained".into());
            }
        }
        "report" | "compare" => {
            let value = if command == "report" {
                report::report(Path::new(get("--run")?))?
            } else {
                let vary = flags
                    .get("--vary")
                    .map(|v| {
                        v.split(',')
                            .filter(|s| !s.is_empty())
                            .map(str::to_owned)
                            .collect::<BTreeSet<_>>()
                    })
                    .unwrap_or_default();
                report::compare(Path::new(get("--left")?), Path::new(get("--right")?), &vary)?
            };
            let text = match flags
                .get("--format")
                .map(String::as_str)
                .unwrap_or("markdown")
            {
                "markdown" => report::markdown(&value),
                "json" => format!("{}\n", serde_json::to_string_pretty(&value)?),
                _ => return Err("format must be markdown or json".into()),
            };
            if let Some(out) = flags.get("--out") {
                use std::io::Write;
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(out)?
                    .write_all(text.as_bytes())?;
            } else {
                print!("{text}");
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}
