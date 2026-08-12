use std::fs;
use std::path::{Path, PathBuf};

use alpharat_bench::calibration::{
    compare_runs, load_run_folder, render_record_comparison, COMPARISON_FILE, COMPARISON_JSON_FILE,
};
use alpharat_bench::runner::execute_plan_file;

fn main() {
    if let Err(error) = run() {
        eprintln!("alpharat-calibrate: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let raw = std::env::args().skip(1).collect::<Vec<_>>();
    let Some(command) = raw.first().map(String::as_str) else {
        print_help();
        return Err("a command is required".to_owned());
    };
    match command {
        "run" => run_plan(&raw[1..]),
        "compare" => compare(&raw[1..]),
        "--help" | "-h" | "help" => {
            print_help();
            Ok(())
        }
        other => Err(format!(
            "unknown command '{other}'; expected run or compare"
        )),
    }
}

fn run_plan(args: &[String]) -> Result<(), String> {
    let mut plan = None;
    let mut output = None;
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = |index: &mut usize| -> Result<&str, String> {
            *index += 1;
            args.get(*index)
                .map(String::as_str)
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag {
            "--plan" => plan = Some(PathBuf::from(value(&mut index)?)),
            "--output" => output = Some(PathBuf::from(value(&mut index)?)),
            "--help" | "-h" => {
                print_run_help();
                return Ok(());
            }
            other => return Err(format!("unknown run option '{other}'")),
        }
        index += 1;
    }
    let plan = plan.ok_or_else(|| "run requires --plan <plan.json>".to_owned())?;
    let output = output.ok_or_else(|| "run requires --output <folder>".to_owned())?;
    let loaded = execute_plan_file(&plan, &output).map_err(|error| error.to_string())?;
    eprintln!(
        "wrote calibration run '{}' to {}",
        loaded.record.run_id,
        loaded.folder.display()
    );
    Ok(())
}

fn compare(args: &[String]) -> Result<(), String> {
    let mut left = None;
    let mut right = None;
    let mut output = None;
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = |index: &mut usize| -> Result<&str, String> {
            *index += 1;
            args.get(*index)
                .map(String::as_str)
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag {
            "--left" => left = Some(PathBuf::from(value(&mut index)?)),
            "--right" => right = Some(PathBuf::from(value(&mut index)?)),
            "--output" => output = Some(PathBuf::from(value(&mut index)?)),
            "--help" | "-h" => {
                print_compare_help();
                return Ok(());
            }
            other => return Err(format!("unknown compare option '{other}'")),
        }
        index += 1;
    }
    let left = left.ok_or_else(|| "compare requires --left <run-folder>".to_owned())?;
    let right = right.ok_or_else(|| "compare requires --right <run-folder>".to_owned())?;
    let left = load_run_folder(&left).map_err(|error| error.to_string())?;
    let right = load_run_folder(&right).map_err(|error| error.to_string())?;
    let comparison = compare_runs(&left, &right).map_err(|issues| {
        let details = issues
            .iter()
            .map(|issue| format!("- {issue:?}"))
            .collect::<Vec<_>>()
            .join("\n");
        format!("records are not comparable:\n{details}")
    })?;
    let markdown = render_record_comparison(&comparison);
    match output {
        Some(folder) => {
            create_output_folder(&folder)?;
            let mut json = serde_json::to_vec_pretty(&comparison)
                .map_err(|error| format!("failed to encode comparison JSON: {error}"))?;
            json.push(b'\n');
            fs::write(folder.join(COMPARISON_JSON_FILE), json)
                .map_err(|error| format!("failed to write comparison JSON: {error}"))?;
            fs::write(folder.join(COMPARISON_FILE), markdown)
                .map_err(|error| format!("failed to write comparison Markdown: {error}"))?;
            eprintln!("wrote comparison to {}", folder.display());
        }
        None => print!("{markdown}"),
    }
    Ok(())
}

fn create_output_folder(folder: &Path) -> Result<(), String> {
    if folder.exists() {
        return Err(format!(
            "output folder '{}' already exists; comparison never overwrites artifacts",
            folder.display()
        ));
    }
    if let Some(parent) = folder
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create '{}': {error}", parent.display()))?;
    }
    fs::create_dir(folder)
        .map_err(|error| format!("failed to create '{}': {error}", folder.display()))
}

fn print_help() {
    eprintln!("AlphaRat calibration v1");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  run      Execute an explicit plan into a self-contained run folder");
    eprintln!("  compare  Compare two compatible run folders");
    eprintln!();
    eprintln!("Use 'alpharat-calibrate <command> --help' for command options.");
}

fn print_run_help() {
    eprintln!("Usage: alpharat-calibrate run --plan <plan.json> --output <folder>");
}

fn print_compare_help() {
    eprintln!(
        "Usage: alpharat-calibrate compare --left <run-folder> --right <run-folder> [--output <folder>]"
    );
    eprintln!("Without --output, the Markdown comparison is written to stdout.");
}
