//! Linux parent/child execution. Each attempt owns its process group and artifact directory.
use super::{artifact::*, model::*, Result};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
extern "C" fn interrupted(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::Relaxed);
}
fn signals() {
    INTERRUPTED.store(false, Ordering::Relaxed);
    // The handler only sets a lock-free flag; the supervisor owns cleanup.
    unsafe {
        libc::signal(libc::SIGINT, interrupted as libc::sighandler_t);
        libc::signal(libc::SIGTERM, interrupted as libc::sighandler_t);
    }
}
struct OwnedChild(Child, Option<OwnedFd>);

fn process_state(pid: u32) -> Result<(u32, u64)> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
    let fields = stat
        .rsplit_once(')')
        .ok_or("invalid process identity")?
        .1
        .split_whitespace()
        .collect::<Vec<_>>();
    Ok((
        fields.get(1).ok_or("missing parent pid")?.parse()?,
        fields.get(19).ok_or("missing start time")?.parse()?,
    ))
}
fn belongs_to(pid: u32, root: u32) -> Result<u64> {
    let original = process_state(pid)?.1;
    let mut current = pid;
    for _ in 0..100 {
        if current == root {
            return Ok(original);
        }
        if current <= 1 {
            break;
        }
        current = process_state(current)?.0;
    }
    Err("worker is not a descendant of the launched process".into())
}
impl OwnedChild {
    fn register_worker(&mut self, dir: &Path) -> Result<()> {
        let path = dir.join("worker.json");
        if self.1.is_some() || !path.exists() {
            return Ok(());
        }
        let receipt: serde_json::Value = read_json(&path)?;
        let pid = u32::try_from(receipt["pid"].as_u64().ok_or("worker pid absent")?)?;
        if pid <= 1 {
            return Err("invalid worker pid".into());
        }
        let birth = belongs_to(pid, self.0.id())?;
        // A pidfd binds the verified task, so PID reuse cannot redirect cleanup.
        let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) } as i32;
        if raw < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        if belongs_to(pid, self.0.id())? != birth {
            return Err("worker identity changed".into());
        }
        self.1 = Some(fd);
        write_json(
            &dir.join("worker-accepted.json"),
            &serde_json::json!({"pid":pid,"start_ticks":birth}),
        )?;
        Ok(())
    }
}

/// Wait for the parent to verify this exact process before loading GPU state.
pub fn worker_handshake(output: &Path) -> Result<()> {
    let pid = std::process::id();
    super::driver::stage(output, "awaiting_parent")?;
    write_json(&output.join("worker.json"), &serde_json::json!({"pid":pid}))?;
    let began = Instant::now();
    loop {
        if let Ok(ack) = read_json::<serde_json::Value>(&output.join("worker-accepted.json")) {
            if ack["pid"] != pid || ack["start_ticks"] != process_state(pid)?.1 {
                return Err("parent worker acknowledgement differs".into());
            }
            return Ok(());
        }
        if began.elapsed() > Duration::from_secs(10) {
            return Err("parent did not acknowledge worker ownership".into());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}
impl Drop for OwnedChild {
    fn drop(&mut self) {
        // Only the ancestry-verified worker is addressed through its stable
        // kernel handle. No adopted-process scan or unrelated PID is used.
        if let Some(fd) = &self.1 {
            unsafe {
                libc::syscall(
                    libc::SYS_pidfd_send_signal,
                    fd.as_raw_fd(),
                    libc::SIGKILL,
                    std::ptr::null::<libc::siginfo_t>(),
                    0,
                );
            }
        }
        // The launcher is always created in our own new process group.
        unsafe {
            libc::kill(-(self.0.id() as i32), libc::SIGKILL);
        }
        let _ = self.0.wait();
    }
}
fn launch(command: &mut Command, dir: &Path) -> Result<OwnedChild> {
    command
        .stdout(Stdio::from(File::create(dir.join("stdout.log"))?))
        .stderr(Stdio::from(File::create(dir.join("stderr.log"))?))
        .stdin(Stdio::null())
        .process_group(0);
    write_json(
        &dir.join("command.json"),
        &serde_json::json!({
            "program":command.get_program().to_string_lossy(),
            "args":command.get_args().map(|a|a.to_string_lossy()).collect::<Vec<_>>(),
            "environment_overrides":command.get_envs().map(|(k,v)|(k.to_string_lossy(),v.map(|v|v.to_string_lossy()))).collect::<BTreeMap<_,_>>()
        }),
    )?;
    Ok(OwnedChild(command.spawn()?, None))
}
fn current_stage(dir: &Path) -> String {
    read_json::<serde_json::Value>(&dir.join("progress.json"))
        .ok()
        .and_then(|v| v["stage"].as_str().map(str::to_owned))
        .unwrap_or_else(|| "starting".into())
}
fn wait(
    child: &mut OwnedChild,
    dir: &Path,
    limits: &Limits,
    total: Instant,
) -> Result<(Status, String, Option<String>)> {
    let began = Instant::now();
    let mut measured = None;
    loop {
        child.register_worker(dir)?;
        let stage = current_stage(dir);
        if INTERRUPTED.load(Ordering::Relaxed) {
            return Ok((
                Status::Interrupted,
                stage,
                Some("interrupted by signal".into()),
            ));
        }
        if let Some(status) = child.0.try_wait()? {
            return Ok((
                if status.success() {
                    Status::Completed
                } else {
                    Status::Failed
                },
                stage,
                (!status.success()).then(|| format!("child exited with {status}; see stderr.log")),
            ));
        }
        if stage == "measurement" || stage == "finalize" || stage == "completed" {
            measured.get_or_insert_with(Instant::now);
        }
        let reason = if total.elapsed().as_secs() >= limits.total_seconds {
            Some("total run time limit")
        } else if measured.is_none() && began.elapsed().as_secs() >= limits.setup_seconds {
            Some("setup/warmup time limit")
        } else if measured.is_some_and(|t| t.elapsed().as_secs() >= limits.trial_seconds) {
            Some("measurement/finalization time limit")
        } else {
            None
        };
        if let Some(reason) = reason {
            return Ok((Status::TimedOut, stage, Some(reason.into())));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}
fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    }
}
pub fn resolve_plan(path: &Path) -> Result<Plan> {
    let path = fs::canonicalize(path)?;
    let mut plan: Plan = read_json(&path)?;
    let base = path.parent().ok_or("plan has no directory")?;
    plan.corpus.path = fs::canonicalize(resolve(base, &plan.corpus.path))?;
    if let Some(m) = &mut plan.model {
        m.path = fs::canonicalize(resolve(base, &m.path))?;
    }
    for v in &mut plan.variants {
        v.executable = fs::canonicalize(resolve(base, &v.executable))?;
        if let BackendSpec::TensorRt { cache_dir, .. } = &mut v.backend {
            *cache_dir = resolve(base, cache_dir);
        }
    }
    plan.validate()?;
    super::driver::corpus_games(&read_json(&plan.corpus.path)?)?;
    Ok(plan)
}
fn capabilities(
    plan: &Plan,
    folder: &Path,
    total: Instant,
) -> Result<BTreeMap<String, FileIdentity>> {
    let mut identities = BTreeMap::new();
    fs::create_dir(folder.join("capabilities"))?;
    for v in &plan.variants {
        let identity = identify(&v.executable)?;
        let dir = folder.join("capabilities").join(&v.id);
        fs::create_dir(&dir)?;
        let mut command = Command::new(&v.executable);
        command.arg("--describe");
        let mut child = launch(&mut command, &dir)?;
        let limits = Limits {
            setup_seconds: 10,
            trial_seconds: 10,
            total_seconds: plan.limits.total_seconds,
        };
        let (status, _, error) = wait(&mut child, &dir, &limits, total)?;
        if status != Status::Completed {
            return Err(error
                .unwrap_or_else(|| "capability check failed".into())
                .into());
        }
        let info: serde_json::Value = read_json(&dir.join("stdout.log"))?;
        if info["format"] != "alpharat.inference.capabilities" || info["schema_version"] != VERSION
        {
            return Err("variant executable does not support this inference protocol".into());
        }
        if plan.comparison.is_some() && info["source_state"] == "non_reproducible" {
            return Err(format!(
                "variant {} lacks reconstructable source provenance: {}",
                v.id, info["source_reason"]
            )
            .into());
        }
        if info["worker_handshake"] != true {
            return Err("variant lacks verified worker ownership protocol".into());
        }
        if matches!(v.backend, BackendSpec::TensorRt { .. }) && info["tensorrt"] != true {
            return Err(format!("variant {} lacks TensorRT support", v.id).into());
        }
        if plan.cases.iter().any(|c| {
            matches!(c, Case::Selfplay { config, .. }
            if config.engine == SearchEngine::Mcgs)
        }) && info["mcgs_selfplay"] != true
        {
            return Err(format!("variant {} lacks MCGS self-play support", v.id).into());
        }
        if plan.measurement.mode == Mode::Timeline && info["timeline"] != true {
            return Err(format!("variant {} lacks timeline support", v.id).into());
        }
        verify(&identity)?;
        identities.insert(v.id.clone(), identity);
    }
    Ok(identities)
}
pub fn check(path: &Path) -> Result<serde_json::Value> {
    signals();
    let total = Instant::now();
    let plan = resolve_plan(path)?;
    let folder =
        std::env::temp_dir().join(format!("alpharat-infer-check-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&folder)?;
    let result=capabilities(&plan,&folder,total).and_then(|executables| {
        Ok(serde_json::json!({"valid":true,"plan":plan,"corpus":identify(&plan.corpus.path)?,
            "model":plan.model.as_ref().map(|m|identify(&m.path)).transpose()?,"executables":executables}))
    });
    // Only this freshly allocated temporary directory is removed.
    let _ = fs::remove_dir_all(&folder);
    result
}
fn attempts(plan: &Plan) -> Vec<Attempt> {
    let mut out = Vec::new();
    for case in &plan.cases {
        for repetition in 0..plan.measurement.repetitions {
            let variants = if let Some(c) = &plan.comparison {
                if c.order[repetition] == "AB" {
                    vec![&c.baseline, &c.candidate]
                } else {
                    vec![&c.candidate, &c.baseline]
                }
            } else {
                plan.variants.iter().map(|v| &v.id).collect()
            };
            for variant in variants {
                out.push(Attempt {
                    id: format!("a{:05}", out.len() + 1),
                    case_key: case.key().into(),
                    variant_id: variant.clone(),
                    repetition,
                    status: Status::Planned,
                    stage: "planned".into(),
                    error: None,
                    elapsed_ms: 0,
                    request_sha256: None,
                    result_sha256: None,
                    result: None,
                });
            }
        }
    }
    out
}
fn save(folder: &Path, record: &RunRecord) -> Result<()> {
    write_json(&folder.join("run.json"), record)
}
pub fn run(path: &Path, output: &Path, nsys: Option<&Path>) -> Result<RunRecord> {
    run_inner(path, output, nsys, None)
}
fn run_inner(
    path: &Path,
    output: &Path,
    nsys: Option<&Path>,
    parent: Option<(String, Identity)>,
) -> Result<RunRecord> {
    signals();
    let total = Instant::now();
    // Reserve a fresh run folder before validation, library loading, or GPU work.
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(output)?;
    let output = fs::canonicalize(output)?;
    fs::copy(path, output.join("submitted-plan.json"))?;
    let result: Result<RunRecord> = (|| {
        let plan = resolve_plan(path)?;
        if plan.measurement.mode == Mode::Timeline && nsys.is_none() {
            return Err("timeline requires --nsys PATH".into());
        }
        let nsys = nsys.map(fs::canonicalize).transpose()?;
        fs::copy(&plan.corpus.path, output.join("corpus.json"))?;
        let corpus = identify(&output.join("corpus.json"))?;
        let model = plan.model.as_ref().map(|m| identify(&m.path)).transpose()?;
        write_json(&output.join("plan.json"), &plan)?;
        fs::create_dir(output.join("attempts"))?;
        let mut record = RunRecord {
            format: RUN_FORMAT.into(),
            schema_version: VERSION,
            run_id: uuid::Uuid::new_v4().to_string(),
            created_at: now(),
            status: Status::Running,
            plan: plan.clone(),
            plan_sha256: hash(&fs::read(output.join("plan.json"))?),
            corpus: corpus.clone(),
            model: model.clone(),
            executables: BTreeMap::new(),
            parent: parent.as_ref().map(|p| p.0.clone()),
            attempts: attempts(&plan),
        };
        for a in &record.attempts {
            fs::create_dir(output.join("attempts").join(&a.id))?;
        }
        save(&output, &record)?;
        let execs = match capabilities(&plan, &output, total) {
            Ok(v) => v,
            Err(e) => {
                record.status = Status::Unsupported;
                for a in &mut record.attempts {
                    a.status = Status::NotRun;
                    a.stage = "preflight".into();
                    a.error = Some(e.to_string());
                }
                save(&output, &record)?;
                super::report::write_report(&output)?;
                return Ok(record);
            }
        };
        record.executables = execs;
        save(&output, &record)?;
        for index in 0..record.attempts.len() {
            if total.elapsed().as_secs() >= plan.limits.total_seconds
                || INTERRUPTED.load(Ordering::Relaxed)
            {
                record.status = if INTERRUPTED.load(Ordering::Relaxed) {
                    Status::Interrupted
                } else {
                    Status::TimedOut
                };
                break;
            }
            let a = &record.attempts[index];
            let dir = output.join("attempts").join(&a.id);
            let case = plan
                .cases
                .iter()
                .find(|c| c.key() == a.case_key)
                .unwrap()
                .clone();
            let variant = plan
                .variants
                .iter()
                .find(|v| v.id == a.variant_id)
                .unwrap()
                .clone();
            let request = TrialRequest {
                plan: plan.clone(),
                case,
                variant: variant.clone(),
                corpus: corpus.clone(),
                model: model.clone(),
                executable: record.executables[&variant.id].clone(),
                expected_identity: parent.as_ref().map(|p| p.1.clone()),
            };
            write_json(&dir.join("request.json"), &request)?;
            record.attempts[index].request_sha256 =
                Some(hash(&fs::read(dir.join("request.json"))?));
            record.attempts[index].status = Status::Running;
            save(&output, &record)?;
            let began = Instant::now();
            let outcome = (|| -> Result<(Status, String, Option<String>, Option<TrialResult>)> {
                verify(&request.executable)?;
                if let Some(m) = &model {
                    verify(m)?;
                }
                let mut command = if let Some(nsys) = nsys
                    .as_ref()
                    .filter(|_| plan.measurement.mode == Mode::Timeline)
                {
                    let mut c = Command::new(nsys);
                    c.args([
                        "profile",
                        "--trace=cuda,nvtx",
                        "--sample=none",
                        "--cpuctxsw=none",
                        "--capture-range=nvtx",
                        "--nvtx-capture=inference.measure@alpharat",
                        "--capture-range-end=stop",
                        "--kill=none",
                        "--wait=all",
                        "--force-overwrite=false",
                        "--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0",
                        "--output",
                    ])
                    .arg(dir.join("timeline"))
                    .arg(&variant.executable);
                    write_json(&dir.join("profiler.json"), &identify(nsys)?)?;
                    c
                } else {
                    Command::new(&variant.executable)
                };
                command
                    .arg("_trial")
                    .arg(dir.join("request.json"))
                    .arg(&dir);
                let mut child = launch(&mut command, &dir)?;
                let (status, stage, error) = wait(&mut child, &dir, &plan.limits, total)?;
                drop(child);
                if status != Status::Completed {
                    return Ok((status, stage, error, None));
                }
                let mut result: TrialResult = read_json(&dir.join("result.json"))?;
                if plan.measurement.mode == Mode::Timeline {
                    let trace = dir.join("timeline.nsys-rep");
                    let id = identify(&trace)?;
                    if id.bytes == 0 {
                        return Err("profiler produced an empty timeline".into());
                    }
                    result
                        .artifacts
                        .insert("timeline.nsys-rep".into(), id.sha256);
                    result.artifacts.insert(
                        "profiler.json".into(),
                        hash(&fs::read(dir.join("profiler.json"))?),
                    );
                    write_json(&dir.join("result.json"), &result)?;
                }
                Ok((Status::Completed, "completed".into(), None, Some(result)))
            })();
            let (status, stage, error, result) = outcome.unwrap_or_else(|e| {
                (
                    Status::Failed,
                    current_stage(&dir),
                    Some(e.to_string()),
                    None,
                )
            });
            let a = &mut record.attempts[index];
            a.status = status.clone();
            a.stage = stage;
            a.error = error;
            a.result = result;
            a.elapsed_ms = began.elapsed().as_millis() as u64;
            if a.result.is_some() {
                a.result_sha256 = Some(hash(&fs::read(dir.join("result.json"))?));
            }
            save(&output, &record)?;
            if status != Status::Completed {
                record.status = status;
                break;
            }
            // Validate the saved result, work counts, and artifacts before another attempt.
            if let Err(e) = load_run(&output) {
                record.attempts[index].status = Status::Failed;
                record.attempts[index].error = Some(format!("result validation: {e}"));
                record.status = Status::Failed;
                break;
            }
        }
        if record.status == Status::Running {
            record.status = Status::Completed;
        }
        for a in &mut record.attempts {
            if a.status == Status::Planned {
                a.status = Status::NotRun;
                a.error = Some("run stopped before this attempt".into());
            }
        }
        save(&output, &record)?;
        super::report::write_report(&output)?;
        Ok(record)
    })();
    if let Err(e) = &result {
        write_json(
            &output.join("failure.json"),
            &serde_json::json!({"status":"failed","error":e.to_string(),"at":now()}),
        )?;
    }
    result
}
pub fn diagnose(
    parent: &Path,
    attempt: &str,
    mode: Mode,
    output: &Path,
    nsys: Option<&Path>,
) -> Result<RunRecord> {
    if mode == Mode::Clean {
        return Err("diagnose needs latency, timeline, or stages".into());
    }
    let parent = fs::canonicalize(parent)?;
    let record = load_run(&parent)?;
    let a = record
        .attempts
        .iter()
        .find(|a| a.id == attempt && a.status == Status::Completed)
        .ok_or("completed parent attempt required")?;
    let result = a.result.as_ref().ok_or("parent result absent")?;
    let mut plan = record.plan.clone();
    plan.corpus.path = parent.join("corpus.json");
    plan.variants.retain(|v| v.id == a.variant_id);
    plan.cases.retain(|c| c.key() == a.case_key);
    plan.measurement.mode = mode;
    plan.measurement.repetitions = 1;
    plan.comparison = None;
    // Input/build identities are checked in the child against the exact selected parent.
    verify(&result.identity.executable)?;
    if let Some(m) = &record.model {
        verify(m)?;
    }
    let temp =
        std::env::temp_dir().join(format!("alpharat-diagnose-{}.json", uuid::Uuid::new_v4()));
    write_json(&temp, &plan)?;
    let parent_link = format!(
        "{}#{}@{}",
        parent.display(),
        attempt,
        hash(&fs::read(parent.join("run.json"))?)
    );
    let result = run_inner(
        &temp,
        output,
        nsys,
        Some((parent_link, result.identity.clone())),
    );
    let _ = fs::remove_file(temp);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn supervisor_kills_its_child_group_on_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let mut c = Command::new("sleep");
        c.arg("20");
        let mut child = launch(&mut c, dir.path()).unwrap();
        let pid = child.0.id();
        let result = wait(
            &mut child,
            dir.path(),
            &Limits {
                setup_seconds: 1,
                trial_seconds: 1,
                total_seconds: 1,
            },
            Instant::now(),
        )
        .unwrap();
        assert_eq!(result.0, Status::TimedOut);
        drop(child);
        assert_eq!(unsafe { libc::kill(pid as i32, 0) }, -1);
    }
    #[test]
    fn worker_receipt_cannot_address_an_unrelated_process() {
        let dir = tempfile::tempdir().unwrap();
        let mut command = Command::new("sleep");
        command.arg("20");
        let mut child = launch(&mut command, dir.path()).unwrap();
        write_json(
            &dir.path().join("worker.json"),
            &serde_json::json!({"pid":std::process::id()}),
        )
        .unwrap();
        assert!(child
            .register_worker(dir.path())
            .unwrap_err()
            .to_string()
            .contains("not a descendant"));
        assert!(child.1.is_none());
        assert!(!dir.path().join("worker-accepted.json").exists());
    }
}
