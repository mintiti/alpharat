#![cfg(feature = "mcgs-profile")]

use std::fs;
use std::path::Path;
use std::process::Command;

use alpharat_bench::calibration::{
    check_comparable, load_run_folder, CaseState, SourceState, COMPARISON_FILE, RUN_RECORD_FILE,
    SEARCH_TRIALS_FILE, SEARCH_WORKLOAD_FILE, SUMMARY_FILE,
};
use alpharat_bench::runner::{derive_run_artifacts, execute_plan_file};
use serde_json::json;

fn search_plan() -> serde_json::Value {
    json!({
        "protocol_version": 1,
        "plan_id": "runner-search-fixture",
        "model": { "label": "fixture model", "path": "model.bin" },
        "comparison_axes": ["workers"],
        "cases": [
            {
                "benchmark": "search",
                "id": "search-w1",
                "comparison_key": "search-workers-1-total-in-flight-4-direct",
                "label": "One worker — total in-flight 4, direct",
                "requested": {
                    "backend": { "engine": "smart_uniform" },
                    "workers": 1,
                    "total_in_flight": 4,
                    "mux_max_batch": null,
                    "productive_work": 8,
                    "warmup_trials": 0,
                    "measured_trials": 1
                }
            },
            {
                "benchmark": "search",
                "id": "search-w2",
                "comparison_key": "search-workers-2-total-in-flight-4-direct",
                "label": "Two workers — total in-flight 4, direct",
                "requested": {
                    "backend": { "engine": "smart_uniform" },
                    "workers": 2,
                    "total_in_flight": 4,
                    "mux_max_batch": null,
                    "productive_work": 8,
                    "warmup_trials": 0,
                    "measured_trials": 1
                }
            }
        ],
        "capacity_workload": null,
        "search_workload": { "source": "built_in", "label": "fixed 7x7 search" },
        "tensorrt_cache": null,
        "accelerators": [],
        "runtime_software": []
    })
}

fn run_cli(cwd: &Path, plan: &Path, output: &Path) {
    let result = Command::new(env!("CARGO_BIN_EXE_alpharat-calibrate"))
        .current_dir(cwd)
        .arg("run")
        .arg("--plan")
        .arg(plan)
        .arg("--output")
        .arg(output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "calibration CLI failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
}

fn mark_source_clean(folder: &Path) {
    let path = folder.join(RUN_RECORD_FILE);
    let mut record: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    record["context"]["source"]["state"] = json!({ "state": "clean" });
    fs::write(path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
}

#[test]
fn one_command_materializes_and_reloads_a_search_run() {
    let temporary = tempfile::tempdir().unwrap();
    let model = temporary.path().join("model.bin");
    fs::write(&model, b"runner fixture model identity").unwrap();
    let plan = temporary.path().join("plan.json");
    fs::write(&plan, serde_json::to_vec_pretty(&search_plan()).unwrap()).unwrap();
    let output = temporary.path().join("run");

    let run = execute_plan_file(&plan, &output).unwrap();

    assert_eq!(run.record.plan_id, "runner-search-fixture");
    assert!(run.record.run_id.starts_with("run-"));
    assert!(run
        .record
        .cases
        .iter()
        .all(|case| case.state() == CaseState::Completed));
    assert_eq!(run.search_trials.len(), 2);
    assert_eq!(
        run.record.context.build.rustc_version,
        env!("ALPHARAT_BUILD_RUSTC_VERSION")
    );
    assert_eq!(
        run.record.context.build.target,
        env!("ALPHARAT_BUILD_TARGET")
    );
    for file in [
        RUN_RECORD_FILE,
        SEARCH_WORKLOAD_FILE,
        SEARCH_TRIALS_FILE,
        SUMMARY_FILE,
        COMPARISON_FILE,
    ] {
        assert!(output.join(file).is_file(), "missing {file}");
    }
    let reloaded = load_run_folder(&output).unwrap();
    assert_eq!(reloaded.search_trials.len(), 2);
    let markdown = fs::read_to_string(output.join(COMPARISON_FILE)).unwrap();
    assert!(markdown.contains("One worker — total in-flight 4, direct"));
    assert!(markdown.contains("Two workers — total in-flight 4, direct"));
    assert!(markdown.contains("no ranking or recommendation"));
}

#[test]
fn cli_records_build_source_provenance_outside_the_repository() {
    let temporary = tempfile::tempdir().unwrap();
    fs::write(
        temporary.path().join("model.bin"),
        b"foreign cwd fixture model",
    )
    .unwrap();
    let plan = temporary.path().join("plan.json");
    fs::write(&plan, serde_json::to_vec_pretty(&search_plan()).unwrap()).unwrap();
    let output = temporary.path().join("run-from-foreign-cwd");

    run_cli(temporary.path(), &plan, &output);

    let run = load_run_folder(&output).unwrap();
    assert_eq!(
        run.record.context.source.revision,
        env!("ALPHARAT_BUILD_SOURCE_REVISION")
    );
    match env!("ALPHARAT_BUILD_SOURCE_STATE") {
        "clean" => assert_eq!(run.record.context.source.state, SourceState::Clean),
        "patched" => assert!(matches!(
            run.record.context.source.state,
            SourceState::Patched { .. }
        )),
        "non_reproducible" => assert_eq!(
            run.record.context.source.state,
            SourceState::NonReproducible {
                reason: env!("ALPHARAT_BUILD_SOURCE_REASON").to_owned()
            }
        ),
        state => panic!("unexpected embedded source state '{state}'"),
    }
}

#[test]
fn two_cli_runs_with_different_outputs_compare_as_the_same_build() {
    let temporary = tempfile::tempdir().unwrap();
    fs::write(temporary.path().join("model.bin"), b"repeat-run model").unwrap();
    let plan = temporary.path().join("plan.json");
    fs::write(&plan, serde_json::to_vec_pretty(&search_plan()).unwrap()).unwrap();
    let left_folder = temporary.path().join("run-a");
    let right_folder = temporary.path().join("run-b");

    run_cli(temporary.path(), &plan, &left_folder);
    run_cli(temporary.path(), &plan, &right_folder);
    // This test isolates build identity. Dirty developer builds may legitimately carry a
    // non-reproducible source state, which the comparison gate rejects independently.
    mark_source_clean(&left_folder);
    mark_source_clean(&right_folder);

    let left = load_run_folder(&left_folder).unwrap();
    let right = load_run_folder(&right_folder).unwrap();
    assert_eq!(left.record.plan_id, right.record.plan_id);
    assert!(left.record.run_id.starts_with("run-a-"));
    assert!(right.record.run_id.starts_with("run-b-"));
    assert_ne!(left.record.run_id, right.record.run_id);
    assert_ne!(
        left.record.context.build.command,
        right.record.context.build.command
    );

    let comparison = check_comparable(&left, &right).unwrap();
    assert!(!comparison
        .differing_axes
        .contains(&alpharat_bench::calibration::ComparisonAxis::Build));
}

#[test]
fn execution_refuses_to_overwrite_an_existing_record_folder() {
    let temporary = tempfile::tempdir().unwrap();
    fs::write(temporary.path().join("model.bin"), b"model").unwrap();
    let plan = temporary.path().join("plan.json");
    fs::write(&plan, serde_json::to_vec_pretty(&search_plan()).unwrap()).unwrap();
    let output = temporary.path().join("existing");
    fs::create_dir(&output).unwrap();

    let error = execute_plan_file(&plan, &output).unwrap_err();

    assert!(error.to_string().contains("never overwrites"));
}

#[test]
fn derive_regenerates_only_disposable_artifacts_from_a_valid_record() {
    let temporary = tempfile::tempdir().unwrap();
    fs::write(temporary.path().join("model.bin"), b"model").unwrap();
    let plan = temporary.path().join("plan.json");
    fs::write(&plan, serde_json::to_vec_pretty(&search_plan()).unwrap()).unwrap();
    let output = temporary.path().join("preserved-run");
    execute_plan_file(&plan, &output).unwrap();
    let record_before = fs::read(output.join(RUN_RECORD_FILE)).unwrap();
    let trials_before = fs::read(output.join(SEARCH_TRIALS_FILE)).unwrap();
    fs::remove_file(output.join(SUMMARY_FILE)).unwrap();
    fs::remove_file(output.join(COMPARISON_FILE)).unwrap();

    let loaded = derive_run_artifacts(&output).unwrap();

    assert!(loaded.record.run_id.starts_with("preserved-run-"));
    assert_eq!(
        fs::read(output.join(RUN_RECORD_FILE)).unwrap(),
        record_before
    );
    assert_eq!(
        fs::read(output.join(SEARCH_TRIALS_FILE)).unwrap(),
        trials_before
    );
    assert!(output.join(SUMMARY_FILE).is_file());
    assert!(output.join(COMPARISON_FILE).is_file());
}
