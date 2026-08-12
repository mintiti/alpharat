use std::fs;
use std::path::{Path, PathBuf};

use alpharat_bench::calibration::{
    check_comparable, compare_runs, load_run_folder, render_record_comparison, summarize_run,
    ComparisonAxis, ComparisonIssue, ProtocolError, CAPACITY_TRIALS_FILE, CAPACITY_WORKLOAD_FILE,
    RUN_RECORD_FILE, SEARCH_TRIALS_FILE, SEARCH_WORKLOAD_FILE,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

fn fixture_folder() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/calibration-v1/valid")
}

fn copy_fixture() -> TempDir {
    let temporary = tempfile::tempdir().unwrap();
    for file in [
        RUN_RECORD_FILE,
        CAPACITY_WORKLOAD_FILE,
        SEARCH_WORKLOAD_FILE,
        CAPACITY_TRIALS_FILE,
        SEARCH_TRIALS_FILE,
    ] {
        fs::copy(fixture_folder().join(file), temporary.path().join(file)).unwrap();
    }
    temporary
}

fn read_record(folder: &Path) -> Value {
    serde_json::from_slice(&fs::read(folder.join(RUN_RECORD_FILE)).unwrap()).unwrap()
}

fn update_record(folder: &Path, update: impl FnOnce(&mut Value)) {
    let mut record = read_record(folder);
    update(&mut record);
    fs::write(
        folder.join(RUN_RECORD_FILE),
        serde_json::to_vec_pretty(&record).unwrap(),
    )
    .unwrap();
}

fn refresh_trial_file(folder: &Path, key: &str, file: &str, rows: u64) {
    let bytes = fs::read(folder.join(file)).unwrap();
    let sha256 = format!("{:x}", Sha256::digest(bytes));
    update_record(folder, |record| {
        record["trial_files"][key]["sha256"] = json!(sha256);
        record["trial_files"][key]["rows"] = json!(rows);
    });
}

fn change_csv_cell(folder: &Path, file: &str, data_row: usize, column: &str, value: &str) {
    let path = folder.join(file);
    let contents = fs::read_to_string(&path).unwrap();
    let mut rows: Vec<Vec<String>> = contents
        .lines()
        .map(|line| line.split(',').map(str::to_owned).collect())
        .collect();
    let column_index = rows[0].iter().position(|header| header == column).unwrap();
    rows[data_row + 1][column_index] = value.to_owned();
    let mut updated = rows
        .into_iter()
        .map(|row| row.join(","))
        .collect::<Vec<_>>()
        .join("\n");
    updated.push('\n');
    fs::write(path, updated).unwrap();
}

fn error_message(error: ProtocolError) -> String {
    error.to_string()
}

#[test]
fn loads_the_checked_in_v1_example() {
    let run = load_run_folder(fixture_folder()).unwrap();

    assert_eq!(run.record.run_id, "synthetic-calibration-v1");
    assert_eq!(run.capacity_trials.len(), 2);
    assert_eq!(run.search_trials.len(), 2);
}

#[test]
fn rejects_unknown_record_fields() {
    let run = copy_fixture();
    update_record(run.path(), |record| {
        record["mystery"] = json!("schema drift");
    });

    let error = load_run_folder(run.path()).unwrap_err();

    assert!(matches!(error, ProtocolError::Json { .. }));
    assert!(error_message(error).contains("unknown field"));
}

#[test]
fn rejects_unknown_fields_inside_a_case_plan() {
    let run = copy_fixture();
    update_record(run.path(), |record| {
        record["plan"]["cases"][0]["requested"]["mystery"] = json!("schema drift");
    });

    let error = load_run_folder(run.path()).unwrap_err();

    assert!(matches!(error, ProtocolError::Json { .. }));
    assert!(error_message(error).contains("unknown field"));
}

#[test]
fn rejects_an_unsupported_protocol_version() {
    let run = copy_fixture();
    update_record(run.path(), |record| {
        record["protocol_version"] = json!(2);
    });

    let error = load_run_folder(run.path()).unwrap_err();

    assert!(error_message(error).contains("unsupported protocol version 2"));
}

#[test]
fn rejects_a_plan_without_an_outcome_for_every_case() {
    let run = copy_fixture();
    update_record(run.path(), |record| {
        record["cases"].as_array_mut().unwrap().pop();
    });

    let error = load_run_folder(run.path()).unwrap_err();

    assert!(error_message(error).contains("has no recorded outcome"));
}

#[test]
fn rejects_trial_header_drift_even_when_the_hash_matches() {
    let run = copy_fixture();
    let path = run.path().join(CAPACITY_TRIALS_FILE);
    let changed = fs::read_to_string(&path)
        .unwrap()
        .replacen("wall_ms", "elapsed_ms", 1);
    fs::write(path, changed).unwrap();
    refresh_trial_file(run.path(), "backend_capacity", CAPACITY_TRIALS_FILE, 2);

    let error = load_run_folder(run.path()).unwrap_err();

    assert!(error_message(error).contains("unexpected header"));
}

#[test]
fn rejects_a_tampered_trial_file() {
    let run = copy_fixture();
    change_csv_cell(run.path(), CAPACITY_TRIALS_FILE, 1, "wall_ms", "2.2");

    let error = load_run_folder(run.path()).unwrap_err();

    assert!(matches!(error, ProtocolError::HashMismatch { .. }));
}

#[test]
fn rejects_a_missing_workload_file() {
    let run = copy_fixture();
    fs::remove_file(run.path().join(SEARCH_WORKLOAD_FILE)).unwrap();

    let error = load_run_folder(run.path()).unwrap_err();

    assert!(matches!(&error, ProtocolError::Io { .. }));
    assert!(error_message(error).contains(SEARCH_WORKLOAD_FILE));
}

#[test]
fn rejects_a_tampered_workload_file() {
    let run = copy_fixture();
    let path = run.path().join(CAPACITY_WORKLOAD_FILE);
    let mut bytes = fs::read(&path).unwrap();
    bytes[0] = b'[';
    fs::write(path, bytes).unwrap();

    let error = load_run_folder(run.path()).unwrap_err();

    assert!(matches!(error, ProtocolError::HashMismatch { .. }));
}

#[test]
fn rejects_an_incomplete_completed_case() {
    let run = copy_fixture();
    let path = run.path().join(CAPACITY_TRIALS_FILE);
    let contents = fs::read_to_string(&path).unwrap();
    let mut shortened = contents.lines().take(2).collect::<Vec<_>>().join("\n");
    shortened.push('\n');
    fs::write(path, shortened).unwrap();
    refresh_trial_file(run.path(), "backend_capacity", CAPACITY_TRIALS_FILE, 1);

    let error = load_run_folder(run.path()).unwrap_err();

    assert!(error_message(error).contains("warmup/measured rows"));
}

#[test]
fn rejects_a_failed_trial_inside_a_completed_case() {
    let run = copy_fixture();
    change_csv_cell(run.path(), CAPACITY_TRIALS_FILE, 1, "status", "failed");
    change_csv_cell(
        run.path(),
        CAPACITY_TRIALS_FILE,
        1,
        "error",
        "fixture failure",
    );
    refresh_trial_file(run.path(), "backend_capacity", CAPACITY_TRIALS_FILE, 2);

    let error = load_run_folder(run.path()).unwrap_err();

    assert!(error_message(error).contains("completed case"));
}

#[test]
fn rejects_a_declared_trial_row_count_that_does_not_match_the_file() {
    let run = copy_fixture();
    update_record(run.path(), |record| {
        record["trial_files"]["search"]["rows"] = json!(3);
    });

    let error = load_run_folder(run.path()).unwrap_err();

    assert!(error_message(error).contains("declares 3 rows but contains 2"));
}

#[test]
fn rejects_search_work_that_does_not_balance() {
    let run = copy_fixture();
    change_csv_cell(run.path(), SEARCH_TRIALS_FILE, 1, "ledger_reserved", "104");
    refresh_trial_file(run.path(), "search", SEARCH_TRIALS_FILE, 2);

    let error = load_run_folder(run.path()).unwrap_err();

    assert!(error_message(error).contains("violates work or ledger accounting"));
}

#[test]
fn rejects_search_capacity_that_cannot_resolve_to_a_worker_batch() {
    let run = copy_fixture();
    update_record(run.path(), |record| {
        record["plan"]["cases"][1]["requested"]["total_in_flight"] = json!(127);
    });

    let error = load_run_folder(run.path()).unwrap_err();

    assert!(error_message(error).contains("divide evenly across workers"));
}

#[test]
fn rejects_case_differences_that_the_plan_did_not_name() {
    let run = copy_fixture();
    update_record(run.path(), |record| {
        record["plan"]["comparison_axes"]
            .as_array_mut()
            .unwrap()
            .retain(|axis| axis != "batch_size");
        let mut second_case = record["plan"]["cases"][0].clone();
        second_case["id"] = json!("capacity-cpu-b16-c2");
        second_case["requested"]["batch_size"] = json!(16);
        record["plan"]["cases"]
            .as_array_mut()
            .unwrap()
            .push(second_case);
    });

    let error = load_run_folder(run.path()).unwrap_err();

    assert!(error_message(error).contains("differ on undeclared axes"));
}

#[test]
fn accepts_two_equivalent_runs_for_comparison() {
    let left = copy_fixture();
    let right = copy_fixture();
    update_record(right.path(), |record| {
        record["run_id"] = json!("second-synthetic-run");
        record["created_at"] = json!("2026-08-11T13:00:00Z");
    });
    let left = load_run_folder(left.path()).unwrap();
    let right = load_run_folder(right.path()).unwrap();

    let check = check_comparable(&left, &right).unwrap();

    assert!(check.differing_axes.is_empty());
    assert_eq!(
        check.comparable_cases,
        ["capacity-cpu-b8-c2", "search-cpu-w2-cap128"]
    );
    assert!(check.unavailable_cases.is_empty());
}

#[test]
fn treats_execution_invocation_as_provenance_not_build_identity() {
    let left = copy_fixture();
    let right = copy_fixture();
    update_record(right.path(), |record| {
        record["run_id"] = json!("second-synthetic-run");
        record["context"]["build"]["command"] = json!([
            "/elsewhere/alpharat-calibrate",
            "run",
            "--output",
            "/different/run-folder"
        ]);
    });
    let left = load_run_folder(left.path()).unwrap();
    let right = load_run_folder(right.path()).unwrap();

    let check = check_comparable(&left, &right).unwrap();

    assert!(check.differing_axes.is_empty());
}

#[test]
fn still_treats_binary_content_as_build_identity() {
    let left = copy_fixture();
    let right = copy_fixture();
    update_record(right.path(), |record| {
        record["context"]["build"]["binary_sha256"] =
            json!("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee");
    });
    let left = load_run_folder(left.path()).unwrap();
    let right = load_run_folder(right.path()).unwrap();

    let issues = check_comparable(&left, &right).unwrap_err();

    assert!(issues.iter().any(|issue| matches!(
        issue,
        ComparisonIssue::UndeclaredAxes { axes, .. }
            if axes.contains(&ComparisonAxis::Build)
    )));
}

#[test]
fn derives_distributions_and_record_deltas_from_validated_trials() {
    let left = copy_fixture();
    let right = copy_fixture();
    update_record(right.path(), |record| {
        record["run_id"] = json!("second-synthetic-run");
        record["created_at"] = json!("2026-08-11T13:00:00Z");
    });
    let left = load_run_folder(left.path()).unwrap();
    let right = load_run_folder(right.path()).unwrap();

    let summary = summarize_run(&left);
    let capacity = summary
        .cases
        .iter()
        .find(|case| case.id == "capacity-cpu-b8-c2")
        .unwrap();
    assert_eq!(capacity.measured_trials.completed, 1);
    assert_eq!(capacity.metrics["positions_per_s"].median, 32_000.0);

    let comparison = compare_runs(&left, &right).unwrap();
    assert_eq!(comparison.cases.len(), 2);
    assert_eq!(
        comparison.cases[0].metrics["positions_per_s"].median_delta,
        0.0
    );
    let markdown = render_record_comparison(&comparison);
    assert!(markdown.contains("synthetic-calibration-v1 → second-synthetic-run"));
    assert!(markdown.contains("no ranking or recommendation"));
}

#[test]
fn treats_labels_paths_and_list_order_as_descriptions_not_identity() {
    let left = copy_fixture();
    let right = copy_fixture();
    update_record(right.path(), |record| {
        record["context"]["model"]["label"] = json!("same model, clearer label");
        record["context"]["model"]["path_hint"] = json!("elsewhere/model.onnx");
        record["context"]["runtime"]["software"]
            .as_array_mut()
            .unwrap()
            .reverse();
        for case in record["plan"]["cases"].as_array_mut().unwrap() {
            case["workload"]["label"] = json!("same workload, clearer label");
            case["workload"]["path_hint"] = json!("elsewhere/workload.bin");
        }
    });
    let left = load_run_folder(left.path()).unwrap();
    let right = load_run_folder(right.path()).unwrap();

    let check = check_comparable(&left, &right).unwrap();

    assert!(check.differing_axes.is_empty());
}

#[test]
fn allows_a_context_difference_only_when_both_plans_name_it() {
    let left = copy_fixture();
    let right = copy_fixture();
    update_record(right.path(), |record| {
        record["context"]["hardware"]["cpu"] = json!("second fixture CPU");
    });
    let left_loaded = load_run_folder(left.path()).unwrap();
    let right_loaded = load_run_folder(right.path()).unwrap();

    let issues = check_comparable(&left_loaded, &right_loaded).unwrap_err();
    assert!(issues.iter().any(|issue| matches!(
        issue,
        ComparisonIssue::UndeclaredAxes { axes, .. }
            if axes.contains(&ComparisonAxis::Hardware)
    )));

    for folder in [left.path(), right.path()] {
        update_record(folder, |record| {
            record["plan"]["comparison_axes"]
                .as_array_mut()
                .unwrap()
                .push(json!("hardware"));
        });
    }
    let left_loaded = load_run_folder(left.path()).unwrap();
    let right_loaded = load_run_folder(right.path()).unwrap();

    let check = check_comparable(&left_loaded, &right_loaded).unwrap();
    assert_eq!(check.differing_axes, [ComparisonAxis::Hardware].into());
}

#[test]
fn rejects_a_different_model_even_if_other_context_changes_are_allowed() {
    let left = copy_fixture();
    let right = copy_fixture();
    update_record(right.path(), |record| {
        record["context"]["model"]["sha256"] =
            json!("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee");
    });
    let left = load_run_folder(left.path()).unwrap();
    let right = load_run_folder(right.path()).unwrap();

    let issues = check_comparable(&left, &right).unwrap_err();

    assert!(issues.contains(&ComparisonIssue::ModelIdentity));
}

#[test]
fn rejects_a_comparison_when_the_source_cannot_be_reconstructed() {
    let left = copy_fixture();
    let right = copy_fixture();
    update_record(right.path(), |record| {
        record["context"]["source"]["state"] = json!({
            "state": "non_reproducible",
            "reason": "fixture omitted its local patch"
        });
    });
    let left = load_run_folder(left.path()).unwrap();
    let right = load_run_folder(right.path()).unwrap();

    let issues = check_comparable(&left, &right).unwrap_err();

    assert!(issues.iter().any(|issue| matches!(
        issue,
        ComparisonIssue::NonReproducibleSource { run_id }
            if run_id == "synthetic-calibration-v1"
    )));
}
