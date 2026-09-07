//! Reports are derived from validated, immutable attempt records.
use super::{artifact::*, model::*, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

fn require(equal: bool, axis: &str, vary: &BTreeSet<String>) -> Result<()> {
    if !equal && !vary.contains(axis) {
        return Err(format!("comparison differs on undeclared axis: {axis}").into());
    }
    Ok(())
}
type TrialCoordinate<'a> = (&'a TrialResult, &'a Case, &'a Variant, &'a Plan);
fn compatible(
    left: TrialCoordinate<'_>,
    right: TrialCoordinate<'_>,
    vary: &BTreeSet<String>,
) -> Result<()> {
    let (a, ca, va, pa) = left;
    let (b, cb, vb, pb) = right;
    if a.mode != Mode::Clean || b.mode != Mode::Clean {
        return Err("speed comparisons require clean attempts".into());
    }
    if pa.measurement != pb.measurement {
        return Err("measurement/warmup policies differ".into());
    }
    let (x, y) = (&a.identity, &b.identity);
    if [x, y].iter().any(|i| {
        i.source_state == "non_reproducible"
            || i.source_revision == "unknown"
            || i.cargo_lock_sha256 == "unavailable"
    }) {
        return Err("comparison requires reconstructable source/build provenance".into());
    }
    if x.model_sha256 != y.model_sha256
        || x.corpus_sha256 != y.corpus_sha256
        || x.encoded_sha256 != y.encoded_sha256
    {
        return Err("model or corpus identity differs".into());
    }
    if x.physical_contexts != y.physical_contexts {
        return Err("physical context counts differ".into());
    }
    require(
        (
            x.source_revision.as_str(),
            x.source_state.as_str(),
            x.source_patch_sha256.as_str(),
        ) == (
            y.source_revision.as_str(),
            y.source_state.as_str(),
            y.source_patch_sha256.as_str(),
        ),
        "source",
        vary,
    )?;
    require(
        (
            &x.rustc,
            &x.target,
            &x.profile,
            &x.features,
            &x.rustflags,
            &x.cargo_lock_sha256,
        ) == (
            &y.rustc,
            &y.target,
            &y.profile,
            &y.features,
            &y.rustflags,
            &y.cargo_lock_sha256,
        ),
        "build",
        vary,
    )?;
    if x.executable.sha256 != y.executable.sha256
        && !vary.contains("source")
        && !vary.contains("build")
    {
        return Err("executable identity differs without source/build axis".into());
    }
    // Paths may differ across hosts; contents identify runtime libraries.
    let libs = |i: &Identity| {
        i.libraries
            .iter()
            .map(|f| (f.sha256.clone(), f.bytes))
            .collect::<BTreeSet<_>>()
    };
    require(libs(x) == libs(y), "runtime", vary)?;
    require(
        x.runtime_environment == y.runtime_environment,
        "runtime",
        vary,
    )?;
    require(x.hardware == y.hardware, "hardware", vary)?;
    match (&va.backend, &vb.backend) {
        (BackendSpec::SmartUniform {}, BackendSpec::SmartUniform {}) => {}
        (
            BackendSpec::TensorRt {
                host_io: ah,
                opt_batch: ao,
                max_batch: am,
                ..
            },
            BackendSpec::TensorRt {
                host_io: bh,
                opt_batch: bo,
                max_batch: bm,
                ..
            },
        ) => {
            require(ah == bh, "host_io", vary)?;
            require((ao, am) == (bo, bm), "profile", vary)?;
            if x.engine_sha256 != y.engine_sha256
                && !["profile", "runtime", "source", "build", "hardware"]
                    .iter()
                    .any(|k| vary.contains(*k))
            {
                return Err("serialized engine identity differs".into());
            }
        }
        _ => return Err("backend kinds differ".into()),
    }
    require(ca.topology() == cb.topology(), "topology", vary)?;
    match (ca, cb) {
        (Case::Capacity { requests: a, .. }, Case::Capacity { requests: b, .. }) => {
            require(a == b, "requests", vary)?
        }
        (Case::Selfplay { config: a, .. }, Case::Selfplay { config: b, .. }) => {
            if a != b {
                return Err("self-play game/search configuration differs".into());
            }
        }
        _ => return Err("driver kinds differ".into()),
    }
    Ok(())
}
fn rate(r: &TrialResult) -> f64 {
    if let Some(s) = &r.metrics.selfplay {
        s.games as f64 * 1e9 / r.metrics.wall_ns as f64
    } else {
        r.metrics.evaluations_per_second()
    }
}
fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) / 2.
    } else {
        values[n / 2]
    }
}
#[derive(Debug, Serialize)]
pub struct ComparisonRow {
    case_key: String,
    unit: String,
    pairs: usize,
    vary: BTreeSet<String>,
    baseline_median_rate: f64,
    candidate_median_rate: f64,
    median_paired_speedup: f64,
    min_paired_speedup: f64,
    max_paired_speedup: f64,
    paired_speedups: Vec<f64>,
}
fn paired(
    left: &RunRecord,
    right: &RunRecord,
    baseline: &str,
    candidate: &str,
    vary: &BTreeSet<String>,
) -> Result<Vec<ComparisonRow>> {
    if left.status != Status::Completed || right.status != Status::Completed {
        return Err(
            "comparison requires completed runs; failed/incomplete attempts are not discarded"
                .into(),
        );
    }
    let allowed = [
        "host_io", "profile", "source", "build", "runtime", "hardware", "topology", "requests",
    ];
    if vary.iter().any(|s| !allowed.contains(&s.as_str())) {
        return Err("unknown comparison axis".into());
    }
    let av = left
        .plan
        .variants
        .iter()
        .find(|v| v.id == baseline)
        .ok_or("baseline variant missing")?;
    let bv = right
        .plan
        .variants
        .iter()
        .find(|v| v.id == candidate)
        .ok_or("candidate variant missing")?;
    if left.plan.cases.len() != right.plan.cases.len() {
        return Err("case sets differ".into());
    }
    let mut rows = Vec::new();
    for ac in &left.plan.cases {
        let bc = right
            .plan
            .cases
            .iter()
            .find(|c| c.key() == ac.key())
            .ok_or("case sets differ")?;
        let trials = |r: &RunRecord, v: &str| -> Result<BTreeMap<usize, TrialResult>> {
            let mut trials = BTreeMap::new();
            for a in r
                .attempts
                .iter()
                .filter(|a| a.case_key == ac.key() && a.variant_id == v)
            {
                if a.status != Status::Completed
                    || trials
                        .insert(a.repetition, a.result.clone().ok_or("result missing")?)
                        .is_some()
                {
                    return Err("missing, duplicated or incomplete repetition".into());
                }
            }
            Ok(trials)
        };
        let a = trials(left, baseline)?;
        let b = trials(right, candidate)?;
        if a.len() != left.plan.measurement.repetitions || a.keys().ne(b.keys()) {
            return Err("repetition sets differ".into());
        }
        let mut ar = Vec::new();
        let mut br = Vec::new();
        let mut ratios = Vec::new();
        for (rep, x) in &a {
            let y = &b[rep];
            compatible((x, ac, av, &left.plan), (y, bc, bv, &right.plan), vary)?;
            // No drift within an arm, even when that axis varies between arms.
            compatible(
                (&a[&0], ac, av, &left.plan),
                (x, ac, av, &left.plan),
                &BTreeSet::new(),
            )?;
            compatible(
                (&b[&0], bc, bv, &right.plan),
                (y, bc, bv, &right.plan),
                &BTreeSet::new(),
            )?;
            ar.push(rate(x));
            br.push(rate(y));
            ratios.push(rate(y) / rate(x));
        }
        rows.push(ComparisonRow {
            case_key: ac.key().into(),
            unit: if matches!(ac, Case::Selfplay { .. }) {
                "games/s"
            } else {
                "evaluations/s"
            }
            .into(),
            pairs: ratios.len(),
            vary: vary.clone(),
            baseline_median_rate: median(&ar),
            candidate_median_rate: median(&br),
            median_paired_speedup: median(&ratios),
            min_paired_speedup: ratios.iter().copied().reduce(f64::min).unwrap(),
            max_paired_speedup: ratios.iter().copied().reduce(f64::max).unwrap(),
            paired_speedups: ratios,
        });
    }
    Ok(rows)
}
pub fn compare(left: &Path, right: &Path, vary: &BTreeSet<String>) -> Result<serde_json::Value> {
    let a = load_run(left)?;
    let b = load_run(right)?;
    if a.plan.variants.len() != 1 || b.plan.variants.len() != 1 {
        return Err(
            "cross-run compare requires one variant per run; paired plans compare in report".into(),
        );
    }
    let rows = paired(&a, &b, &a.plan.variants[0].id, &b.plan.variants[0].id, vary)?;
    Ok(
        serde_json::json!({"format":"alpharat.inference.comparison","schema_version":VERSION,
        "left":a.run_id,"right":b.run_id,"pairing":"equal repetition index; independent runs are not interleaved",
        "summary":"median and range of per-pair candidate/baseline rates; no automatic winner",
        "rows":rows}),
    )
}
pub fn report(folder: &Path) -> Result<serde_json::Value> {
    let r = load_run(folder)?;
    let mut comparison = serde_json::Value::Null;
    let mut comparison_error = None;
    if let Some(c) = &r.plan.comparison {
        match paired(&r, &r, &c.baseline, &c.candidate, &c.vary) {
            Ok(rows) => comparison = serde_json::to_value(rows)?,
            Err(e) => comparison_error = Some(e.to_string()),
        }
    }
    let attempts=r.attempts.iter().map(|a|serde_json::json!({
        "id":a.id,"case":a.case_key,"variant":a.variant_id,"repetition":a.repetition,
        "status":a.status,"stage":a.stage,"error":a.error,
        "measurement":a.result.as_ref().map(|r|serde_json::json!({
            "mode":r.mode,"wall_ns":r.metrics.wall_ns,"evaluations":r.metrics.completed_evaluations,
            "evaluations_per_second":r.metrics.evaluations_per_second(),
            "games_per_second":r.metrics.selfplay.as_ref().map(|s|s.games as f64*1e9/r.metrics.wall_ns as f64),
            "calls":r.metrics.completed_calls,"latency":r.metrics.latency,
            "mux":r.metrics.mux,"stages":r.metrics.stages,"selfplay":r.metrics.selfplay,
            "physical_contexts":r.identity.physical_contexts
        }))
    })).collect::<Vec<_>>();
    Ok(
        serde_json::json!({"format":"alpharat.inference.report","schema_version":VERSION,
        "run_id":r.run_id,"status":r.status,"mode":r.plan.measurement.mode,"parent":r.parent,
        "attempts":attempts,"comparison":comparison,"comparison_error":comparison_error,
        "definitions":{
            "clean":"Production Backend::evaluate_batch capacity, or native self-play including bundle output; no per-request recorder.",
            "latency":"Caller wall time including encode, queue/lock wait, GPU completion and parse. Quantiles are recomputed from retained request samples.",
            "mux":"Actual production merged-batch histogram. worker_wait_drain_ns includes wait, queue lock and draining; it is not request queue residence.",
            "stages":"Diagnostic CUDA-event observations. Stage sums are not critical-path wall time or clean throughput.",
            "speedup":"Median and range of per-repetition candidate/baseline rates. Median rates are not pooled-work/pooled-time rates."
        }}),
    )
}
pub fn markdown(value: &serde_json::Value) -> String {
    if value["format"] == "alpharat.inference.comparison" {
        return format!(
            "Inference comparison\n\n{}\n\n{}\n",
            value["summary"].as_str().unwrap_or(""),
            comparison_markdown(&value["rows"])
        );
    }
    let mut out = format!(
        "Inference run {}\n\nStatus: **{}**. Mode: **{}**.\n\n",
        value["run_id"].as_str().unwrap_or("?"),
        value["status"].as_str().unwrap_or("?"),
        value["mode"].as_str().unwrap_or("?")
    );
    out.push_str("| Attempt | Case | Variant | Status | Evaluations/s | Games/s | Wall ms |\n|---|---|---|---|---:|---:|---:|\n");
    if let Some(attempts) = value["attempts"].as_array() {
        for a in attempts {
            let m = &a["measurement"];
            let n = |key: &str, div: f64| {
                m[key]
                    .as_f64()
                    .map(|x| format!("{:.3}", x / div))
                    .unwrap_or_else(|| "-".into())
            };
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} |\n",
                a["id"].as_str().unwrap_or(""),
                a["case"].as_str().unwrap_or(""),
                a["variant"].as_str().unwrap_or(""),
                a["status"].as_str().unwrap_or(""),
                n("evaluations_per_second", 1.),
                n("games_per_second", 1.),
                n("wall_ns", 1e6)
            ));
            if let Some(e) = a["error"].as_str() {
                out.push_str(&format!("\n{}: {}\n\n", a["id"], e));
            }
        }
    }
    if let Some(error) = value["comparison_error"].as_str() {
        out.push_str(&format!("\nComparison rejected: {error}.\n"));
    }
    out.push_str(&comparison_markdown(&value["comparison"]));
    out.push_str("\nQuantiles, mux histograms, stage totals and native search counters are in report.json and the raw attempt artifacts. Diagnostic rates are not clean capacity.\n");
    if let Some(defs) = value["definitions"].as_object() {
        for (key, val) in defs {
            out.push_str(&format!("\n- {key}: {}\n", val.as_str().unwrap_or("")));
        }
    }
    out
}
fn comparison_markdown(rows: &serde_json::Value) -> String {
    let Some(rows) = rows.as_array() else {
        return String::new();
    };
    let mut out = String::from(
        "\nPaired comparisons: median speedup [minimum, maximum]; no automatic winner.\n\n",
    );
    for row in rows {
        out.push_str(&format!(
            "- {}: {:.4}x [{:.4}, {:.4}], {} pairs; {}.\n",
            row["case_key"].as_str().unwrap_or(""),
            row["median_paired_speedup"].as_f64().unwrap_or(0.),
            row["min_paired_speedup"].as_f64().unwrap_or(0.),
            row["max_paired_speedup"].as_f64().unwrap_or(0.),
            row["pairs"],
            row["unit"].as_str().unwrap_or("")
        ));
    }
    out
}
pub fn write_report(folder: &Path) -> Result<()> {
    let value = report(folder)?;
    write_json(&folder.join("report.json"), &value)?;
    std::fs::write(folder.join("report.md"), markdown(&value))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sample_median_is_not_pooled_rate() {
        assert_eq!(median(&[1., 2., 9., 12.]), 5.5);
        assert_eq!(median(&[9., 1., 2.]), 2.);
    }
    fn result() -> TrialResult {
        TrialResult {
            schema_version: VERSION,
            case_key: "direct".into(),
            variant_id: "control".into(),
            mode: Mode::Clean,
            identity: Identity {
                executable: FileIdentity {
                    path: "/bin/test".into(),
                    sha256: "exe".into(),
                    bytes: 1,
                },
                source_revision: "revision".into(),
                source_state: "clean".into(),
                source_patch_sha256: "patch".into(),
                rustc: "rustc".into(),
                target: "linux".into(),
                profile: "release".into(),
                features: "inference".into(),
                rustflags: String::new(),
                cargo_lock_sha256: "lock".into(),
                libraries: vec![],
                hardware: "test".into(),
                runtime_environment: BTreeMap::new(),
                model_sha256: None,
                corpus_sha256: "corpus".into(),
                encoded_sha256: "encoded".into(),
                engine_sha256: None,
                physical_contexts: 0,
            },
            setup_ms: 0,
            warmup_ms: 0,
            warmup_passes: 1,
            warmed_shapes: vec![8],
            metrics: Metrics {
                wall_ns: 100,
                completed_calls: Some(1),
                completed_evaluations: 8,
                latency: None,
                mux: None,
                stages: None,
                selfplay: None,
            },
            artifacts: BTreeMap::new(),
        }
    }
    #[test]
    fn compatibility_checks_diagnostics_inputs_and_declared_axes() {
        let plan: Plan =
            serde_json::from_str(include_str!("../../examples/inference/cpu.json")).unwrap();
        let c = &plan.cases[0];
        let v = &plan.variants[0];
        let a = result();
        let mut b = a.clone();
        let same = |b: &TrialResult, vary: &BTreeSet<String>| {
            compatible((&a, c, v, &plan), (b, c, v, &plan), vary)
        };
        assert!(same(&b, &BTreeSet::new()).is_ok());
        b.identity.hardware = "different".into();
        assert!(same(&b, &BTreeSet::new()).is_err());
        let axis = BTreeSet::from(["hardware".into()]);
        assert!(same(&b, &axis).is_ok());
        b.identity.corpus_sha256 = "changed".into();
        assert!(same(&b, &axis).is_err());
        b = a.clone();
        b.mode = Mode::Latency;
        assert!(same(&b, &BTreeSet::new()).is_err());
        b = a.clone();
        b.identity.executable.sha256 = "changed".into();
        assert!(same(&b, &BTreeSet::new()).is_err());
        assert!(same(&b, &BTreeSet::from(["build".into()])).is_ok());
    }
}
