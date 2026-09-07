#![cfg(feature = "inference")]
use alpharat_bench::inference::{
    artifact::{self, Status},
    model::Plan,
};
use std::fs;
use std::path::Path;
use std::process::Command;

fn fixture(dir: &Path) -> std::path::PathBuf {
    fs::write(
        dir.join("corpus.json"),
        include_bytes!("../examples/inference/corpus.json"),
    )
    .unwrap();
    let mut p: Plan =
        serde_json::from_slice(include_bytes!("../examples/inference/cpu.json")).unwrap();
    p.variants.truncate(1);
    p.comparison = None;
    p.measurement.repetitions = 1;
    p.measurement.calls_per_caller = 20;
    p.measurement.max_samples_per_thread = 6;
    p.measurement.warmup.min_ms = 0;
    p.variants[0].executable = env!("CARGO_BIN_EXE_alpharat-infer").into();
    let path = dir.join("plan.json");
    artifact::write_json(&path, &p).unwrap();
    path
}
fn cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_alpharat-infer"))
        .args(args)
        .output()
        .unwrap()
}
fn ok(args: &[&str]) -> std::process::Output {
    let out = cli(args);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    out
}
#[test]
fn saved_capacity_mux_selfplay_and_diagnosis_roundtrip() {
    let temp = tempfile::tempdir().unwrap();
    let p = fixture(temp.path());
    let out = temp.path().join("run");
    ok(&["check", "--plan", p.to_str().unwrap()]);
    ok(&[
        "run",
        "--plan",
        p.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    let record = artifact::load_run(&out).unwrap();
    assert_eq!(record.status, Status::Completed);
    assert_eq!(record.attempts.len(), 3);
    let diag = temp.path().join("latency");
    ok(&[
        "diagnose",
        "--run",
        out.to_str().unwrap(),
        "--attempt",
        "a00002",
        "--mode",
        "latency",
        "--out",
        diag.to_str().unwrap(),
    ]);
    let record = artifact::load_run(&diag).unwrap();
    assert!(record.parent.is_some());
    assert_eq!(
        record.attempts[0]
            .result
            .as_ref()
            .unwrap()
            .metrics
            .latency
            .as_ref()
            .unwrap()
            .samples,
        24
    );
    assert_eq!(
        record.attempts[0]
            .result
            .as_ref()
            .unwrap()
            .metrics
            .latency
            .as_ref()
            .unwrap()
            .dropped,
        56
    );
    let report = ok(&[
        "report",
        "--run",
        diag.to_str().unwrap(),
        "--format",
        "json",
    ]);
    let value: serde_json::Value = serde_json::from_slice(&report.stdout).unwrap();
    assert_eq!(value["mode"], "latency");
    assert!(!cli(&[
        "run",
        "--plan",
        p.to_str().unwrap(),
        "--out",
        out.to_str().unwrap()
    ])
    .status
    .success());
    // Raw data corruption cannot be hidden behind a cached report.
    fs::write(diag.join("attempts/a00001/requests.csv"), "damaged").unwrap();
    assert!(artifact::load_run(&diag).is_err());
}
#[test]
fn invalid_or_unsupported_work_is_retained_without_trials() {
    let temp = tempfile::tempdir().unwrap();
    let path = fixture(temp.path());
    let mut plan: Plan = artifact::read_json(&path).unwrap();
    plan.variants[0].executable = "/bin/false".into();
    artifact::write_json(&path, &plan).unwrap();
    let out = temp.path().join("failure");
    assert!(!cli(&[
        "run",
        "--plan",
        path.to_str().unwrap(),
        "--out",
        out.to_str().unwrap()
    ])
    .status
    .success());
    let run = artifact::load_run(&out).unwrap();
    assert_eq!(run.status, Status::Unsupported);
    assert!(run.attempts.iter().all(|a| a.status == Status::NotRun));
    assert!(out.join("capabilities/control/stderr.log").is_file());
}
#[test]
fn diagnostic_rerun_rejects_changed_executable() {
    let temp = tempfile::tempdir().unwrap();
    let path = fixture(temp.path());
    let executable = temp.path().join("copy");
    fs::copy(env!("CARGO_BIN_EXE_alpharat-infer"), &executable).unwrap();
    let mut plan: Plan = artifact::read_json(&path).unwrap();
    plan.variants[0].executable = executable.clone();
    plan.cases.truncate(1);
    artifact::write_json(&path, &plan).unwrap();
    let out = temp.path().join("run");
    ok(&[
        "run",
        "--plan",
        path.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    fs::write(&executable, "changed").unwrap();
    assert!(!cli(&[
        "diagnose",
        "--run",
        out.to_str().unwrap(),
        "--attempt",
        "a00001",
        "--mode",
        "latency",
        "--out",
        temp.path().join("diag").to_str().unwrap()
    ])
    .status
    .success());
}
#[test]
fn interrupt_retains_status_and_reaps_owned_child() {
    use std::os::unix::fs::PermissionsExt;
    use std::time::{Duration, Instant};
    let temp = tempfile::tempdir().unwrap();
    let path = fixture(temp.path());
    let fake = temp.path().join("slow");
    fs::write(&fake,r#"#!/bin/sh
if [ "$1" = "--describe" ]; then
  echo '{"format":"alpharat.inference.capabilities","schema_version":1,"worker_handshake":true,"tensorrt":false,"timeline":false}'
else
  /usr/bin/setsid /bin/sh -c 'echo "{\"pid\":$$}" > "$1/worker.json"; echo $$ > "$1/owned-pid"; exec /bin/sleep 60' sh "$3" &
  wait
fi
"#).unwrap();
    fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).unwrap();
    let mut plan: Plan = artifact::read_json(&path).unwrap();
    plan.variants[0].executable = fake;
    artifact::write_json(&path, &plan).unwrap();
    let out = temp.path().join("interrupted");
    let mut parent = Command::new(env!("CARGO_BIN_EXE_alpharat-infer"))
        .args([
            "run",
            "--plan",
            path.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let pidfile = out.join("attempts/a00001/owned-pid");
    let began = Instant::now();
    while !out.join("attempts/a00001/worker-accepted.json").exists() {
        if began.elapsed() > Duration::from_secs(10) {
            let _ = parent.kill();
            panic!("child did not start");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let pid: i32 = fs::read_to_string(pidfile).unwrap().trim().parse().unwrap();
    assert_eq!(unsafe { libc::kill(parent.id() as i32, libc::SIGTERM) }, 0);
    assert!(!parent.wait().unwrap().success());
    let record = artifact::load_run(&out).unwrap();
    assert_eq!(record.status, Status::Interrupted);
    assert_eq!(record.attempts[0].status, Status::Interrupted);
    assert!(record.attempts[1..]
        .iter()
        .all(|a| a.status == Status::NotRun));
    let state = fs::read_to_string(format!("/proc/{pid}/stat")).ok();
    assert!(state.as_ref().is_none_or(|s| s
        .rsplit_once(')')
        .unwrap()
        .1
        .trim_start()
        .starts_with('Z')));
}

#[test]
fn mcgs_selfplay_roundtrip_preserves_explicit_search_and_tt_accounting() {
    use alpharat_bench::inference::model::{Case, SearchEngine, SearchParameters};
    let temp = tempfile::tempdir().unwrap();
    let path = fixture(temp.path());
    let mut plan: Plan = artifact::read_json(&path).unwrap();
    plan.cases.retain(|c| matches!(c, Case::Selfplay { .. }));
    let Case::Selfplay { config, .. } = &mut plan.cases[0] else {
        unreachable!()
    };
    config.engine = SearchEngine::Mcgs;
    assert!(plan.validate().is_err());
    let Case::Selfplay { config, .. } = &mut plan.cases[0] else {
        unreachable!()
    };
    config.search = Some(SearchParameters::default());
    config.simulations = 64;
    artifact::write_json(&path, &plan).unwrap();
    let out = temp.path().join("mcgs");
    ok(&[
        "run",
        "--plan",
        path.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    let run = artifact::load_run(&out).unwrap();
    assert_eq!(run.status, Status::Completed);
    let stats = run.attempts[0]
        .result
        .as_ref()
        .unwrap()
        .metrics
        .selfplay
        .as_ref()
        .unwrap();
    assert!(stats.tt_stop_hits > 0);
    assert!(!stats.bundle_files.is_empty());
    let mut mcts_plan = plan.clone();
    let Case::Selfplay { config, .. } = &mut mcts_plan.cases[0] else {
        unreachable!()
    };
    config.engine = SearchEngine::Mcts;
    assert_ne!(plan.cases, mcts_plan.cases);
    ok(&["report", "--run", out.to_str().unwrap()]);
}

#[cfg(unix)]
#[test]
fn comparison_rejects_known_bad_provenance_before_worker_launch() {
    use alpharat_bench::inference::model::Comparison;
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    let path = fixture(temp.path());
    let fake = temp.path().join("unreconstructable");
    let marker = temp.path().join("worker-was-launched");
    fs::write(&fake,format!(r#"#!/bin/sh
if [ "$1" = "--describe" ]; then
  echo '{{"format":"alpharat.inference.capabilities","schema_version":1,"worker_handshake":true,"tensorrt":false,"timeline":false,"source_state":"non_reproducible","source_reason":"untracked check program"}}'
else
  touch '{}'
  exit 99
fi
"#,marker.display())).unwrap();
    fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).unwrap();
    let mut plan: Plan = artifact::read_json(&path).unwrap();
    plan.cases.truncate(1);
    plan.variants[0].executable = fake;
    let mut candidate = plan.variants[0].clone();
    candidate.id = "candidate".into();
    plan.variants.push(candidate);
    plan.comparison = Some(Comparison {
        baseline: plan.variants[0].id.clone(),
        candidate: "candidate".into(),
        vary: Default::default(),
        order: vec!["AB".into()],
    });
    artifact::write_json(&path, &plan).unwrap();
    let checked = cli(&["check", "--plan", path.to_str().unwrap()]);
    assert!(!checked.status.success());
    assert!(String::from_utf8_lossy(&checked.stderr).contains("reconstructable source provenance"));
    let out = temp.path().join("run");
    assert!(!cli(&[
        "run",
        "--plan",
        path.to_str().unwrap(),
        "--out",
        out.to_str().unwrap()
    ])
    .status
    .success());
    let record = artifact::load_run(&out).unwrap();
    assert_eq!(record.status, Status::Unsupported);
    assert!(record.attempts.iter().all(|a| a.status == Status::NotRun));
    assert!(!marker.exists());
}
