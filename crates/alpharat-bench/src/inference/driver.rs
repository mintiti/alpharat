use super::{artifact::*, model::*, Result};
use alpharat_eval_core::{Backend, BackendError, EvalResult, SmartUniformBackend};
use alpharat_sampling::inference_trace as trace;
use alpharat_sampling::{FlatEncoder, MuxBackend, MuxConfig, MuxStatsSnapshot, ObservationEncoder};
use pyrat::{Coordinates, Direction, GameBuilder, GameState};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};
use thread_local::ThreadLocal;

fn nanos(d: Duration) -> u64 {
    d.as_nanos().min(u64::MAX as u128) as u64
}
fn millis(d: Duration) -> u64 {
    d.as_millis().min(u64::MAX as u128) as u64
}
pub fn stage(output: &Path, name: &str) -> Result<()> {
    use std::io::Write;
    let event = serde_json::json!({"stage":name,"updated_at":now()});
    let mut log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output.join("progress.jsonl"))?;
    writeln!(log, "{event}")?;
    log.sync_data()?;
    write_json(&output.join("progress.json"), &event)
}
pub fn corpus_games(corpus: &Corpus) -> Result<Vec<GameState>> {
    if corpus.schema_version != 1 || corpus.positions.is_empty() || corpus.positions.len() > 4096 {
        return Err("invalid corpus version/positions".into());
    }
    let dims = (corpus.positions[0].width, corpus.positions[0].height);
    if dims.0 as usize * dims.1 as usize * corpus.positions.len() > 1_000_000
        || corpus
            .positions
            .iter()
            .map(|p| p.actions.len())
            .sum::<usize>()
            > 1_000_000
    {
        return Err("corpus resource bound exceeded".into());
    }
    let mut games = Vec::new();
    for p in &corpus.positions {
        if (p.width, p.height) != dims
            || p.width < 2
            || p.height < 2
            || p.max_turns == 0
            || p.cheese.is_empty()
        {
            return Err("corpus requires consistent nonempty board dimensions and cheese".into());
        }
        let inside = |p: [u8; 2]| p[0] < dims.0 && p[1] < dims.1;
        if !inside(p.player_1)
            || !inside(p.player_2)
            || p.cheese.iter().any(|p| !inside(*p))
            || p.cheese
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != p.cheese.len()
        {
            return Err("invalid corpus coordinates".into());
        }
        let adjacent = |a: [u8; 2], b: [u8; 2]| {
            inside(a) && inside(b) && a[0].abs_diff(b[0]) as u16 + a[1].abs_diff(b[1]) as u16 == 1
        };
        let mut edges = std::collections::BTreeSet::new();
        let mut walls = std::collections::HashMap::<Coordinates, Vec<Coordinates>>::new();
        for &[a, b] in &p.walls {
            if !adjacent(a, b) || !edges.insert((a.min(b), a.max(b))) {
                return Err("invalid or duplicate wall".into());
            }
            walls
                .entry(Coordinates::new(a[0], a[1]))
                .or_default()
                .push(Coordinates::new(b[0], b[1]));
            walls
                .entry(Coordinates::new(b[0], b[1]))
                .or_default()
                .push(Coordinates::new(a[0], a[1]));
        }
        let mut mud = pyrat::MudMap::new();
        for edge in &p.mud {
            let (a, b) = (edge.from, edge.to);
            if !adjacent(a, b) || edge.cost < 2 || !edges.insert((a.min(b), a.max(b))) {
                return Err("invalid, duplicate, or walled mud edge".into());
            }
            mud.insert(
                Coordinates::new(a[0], a[1]),
                Coordinates::new(b[0], b[1]),
                edge.cost,
            );
        }
        let mut game = GameBuilder::new(p.width, p.height)
            .with_custom_maze(walls, mud)
            .with_custom_positions(
                Coordinates::new(p.player_1[0], p.player_1[1]),
                Coordinates::new(p.player_2[0], p.player_2[1]),
            )
            .with_custom_cheese(
                p.cheese
                    .iter()
                    .map(|p| Coordinates::new(p[0], p[1]))
                    .collect(),
            )
            .with_max_turns(p.max_turns)
            .build()
            .create(Some(p.creation_seed))?;
        for [a, b] in &p.actions {
            if game.check_game_over() {
                return Err("corpus action prefix continues after termination".into());
            }
            game.make_move(
                Direction::try_from(*a).map_err(|_| "invalid action")?,
                Direction::try_from(*b).map_err(|_| "invalid action")?,
            );
        }
        if game.check_game_over() {
            return Err("inference corpus contains a terminal state".into());
        }
        games.push(game);
    }
    Ok(games)
}
struct Dyn(Arc<dyn Backend>);
impl Backend for Dyn {
    fn evaluate(&self, g: &GameState) -> std::result::Result<EvalResult, BackendError> {
        self.0.evaluate(g)
    }
    fn evaluate_batch(
        &self,
        g: &[&GameState],
    ) -> std::result::Result<Vec<EvalResult>, BackendError> {
        self.0.evaluate_batch(g)
    }
}
struct Built {
    backend: Arc<dyn Backend>,
    mux: Option<Arc<alpharat_sampling::MuxStats>>,
    #[cfg(feature = "tensorrt")]
    trt: Option<Arc<alpharat_sampling::TrtStats>>,
    engine: Option<String>,
}
fn build(request: &TrialRequest, width: u8, height: u8) -> Result<Built> {
    #[cfg(feature = "tensorrt")]
    let mut engine = None;
    #[cfg(not(feature = "tensorrt"))]
    let engine = None;
    #[cfg(feature = "tensorrt")]
    let mut trt = None;
    let inner: Arc<dyn Backend> = match &request.variant.backend {
        BackendSpec::SmartUniform {} => Arc::new(SmartUniformBackend),
        BackendSpec::TensorRt {
            host_io,
            opt_batch,
            max_batch,
            pad_to_max,
            cuda_graph,
            cache_dir,
        } => {
            #[cfg(feature = "tensorrt")]
            {
                fs::create_dir_all(cache_dir)?;
                let probe =
                    cache_dir.join(format!(".inference-write-probe-{}", uuid::Uuid::new_v4()));
                fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&probe)?;
                fs::remove_file(probe)?;
                let backend = alpharat_sampling::TensorrtBackend::new(
                    &request.model.as_ref().ok_or("model required")?.path,
                    FlatEncoder::new(width, height),
                    alpharat_sampling::TensorrtConfig {
                        opt_batch: Some(*opt_batch),
                        max_batch: *max_batch,
                        pad_to_max: *pad_to_max,
                        cuda_graph: *cuda_graph,
                        cache_dir: Some(cache_dir.clone()),
                        host_io: if host_io == "pinned" {
                            alpharat_sampling::TrtHostIoMode::Pinned
                        } else {
                            alpharat_sampling::TrtHostIoMode::Pageable
                        },
                        profile_stages: request.plan.measurement.mode == Mode::Stages,
                    },
                )?;
                engine = Some(backend.engine_sha256().to_owned());
                trt = Some(backend.stats().clone());
                Arc::new(backend)
            }
            #[cfg(not(feature = "tensorrt"))]
            {
                let _ = (
                    host_io, opt_batch, max_batch, pad_to_max, cuda_graph, cache_dir, width, height,
                );
                return Err("TensorRT support is not compiled".into());
            }
        }
    };
    let (backend, mux): (Arc<dyn Backend>, _) = match request.case.topology() {
        Topology::Direct {} => (inner, None),
        Topology::EagerMux { max_batch } => {
            let mux = MuxBackend::new(
                Dyn(inner),
                MuxConfig {
                    max_batch_size: *max_batch,
                },
            );
            let stats = Some(mux.stats().clone());
            (Arc::new(mux), stats)
        }
    };
    Ok(Built {
        backend,
        mux,
        engine,
        #[cfg(feature = "tensorrt")]
        trt,
    })
}
#[derive(Default)]
struct Worker {
    calls: u64,
    positions: u64,
    samples: Vec<Sample>,
    dropped: u64,
}
fn observe(mode: Mode) -> bool {
    matches!(mode, Mode::Latency | Mode::Timeline)
}
fn sample(
    worker: &mut Worker,
    limit: usize,
    thread: usize,
    ordinal: usize,
    positions: usize,
    start: Instant,
    epoch: Instant,
) {
    if worker.samples.len() < limit {
        worker.samples.push(Sample {
            trace_id: trace::request_id(),
            worker: thread,
            request: ordinal,
            positions,
            started_ns: nanos(start.duration_since(epoch)),
            elapsed_ns: nanos(start.elapsed()),
        });
    } else {
        worker.dropped += 1;
    }
}
fn validate_results(results: &[EvalResult], n: usize) -> Result<()> {
    if results.len() != n {
        return Err("backend returned wrong output count".into());
    }
    if results.iter().any(|r| {
        r.policy_p1
            .iter()
            .chain(&r.policy_p2)
            .chain([&r.value_p1, &r.value_p2])
            .any(|x| !x.is_finite())
    }) {
        return Err("backend returned nonfinite outputs".into());
    }
    Ok(())
}
fn capacity(
    backend: &dyn Backend,
    games: &[GameState],
    requests: &Requests,
    m: &Measurement,
) -> Result<(u64, Vec<Worker>)> {
    let count = requests.callers();
    let references = games
        .len()
        .checked_mul(requests.sizes().iter().sum::<usize>())
        .and_then(|n| n.checked_mul(count))
        .ok_or("prepared workload overflow")?;
    if references > 8_000_000 {
        return Err("prepared workload exceeds 8 million state references".into());
    }
    let ready = Barrier::new(count + 1);
    let start = Barrier::new(count + 1);
    let done = Barrier::new(count + 1);
    let epoch = Instant::now();
    let observe = observe(m.mode);
    let result = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for caller in 0..count {
            let (ready, start, done) = (&ready, &start, &done);
            handles.push(scope.spawn(move || {
                let cycle = games.len() * requests.sizes().len();
                let batches = (0..cycle)
                    .map(|i| {
                        (0..requests.batch(i))
                            .map(|j| {
                                &games[(requests.offset() % games.len() + caller + i + j)
                                    % games.len()]
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let mut worker = Worker {
                    samples: if observe {
                        Vec::with_capacity(m.calls_per_caller.min(m.max_samples_per_thread))
                    } else {
                        Vec::new()
                    },
                    ..Worker::default()
                };
                ready.wait();
                start.wait();
                let result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                        for i in 0..m.calls_per_caller {
                            let batch = &batches[i % cycle];
                            let began = observe.then(Instant::now);
                            let _trace = if m.mode == Mode::Timeline {
                                Some(trace::request())
                            } else {
                                None
                            };
                            let output = backend.evaluate_batch(batch)?;
                            if output.len() != batch.len() {
                                return Err("backend returned wrong output count".into());
                            }
                            worker.calls += 1;
                            worker.positions += batch.len() as u64;
                            if let Some(began) = began {
                                sample(
                                    &mut worker,
                                    m.max_samples_per_thread,
                                    caller,
                                    i,
                                    batch.len(),
                                    began,
                                    epoch,
                                );
                            }
                            drop(output);
                        }
                        Ok(())
                    }))
                    .unwrap_or_else(|_| Err("backend panicked".into()));
                done.wait();
                result.map(|_| worker)
            }));
        }
        ready.wait();
        let _trace = trace::range(c"inference.measure", 0);
        let began = Instant::now();
        start.wait();
        done.wait();
        let elapsed = nanos(began.elapsed());
        drop(_trace);
        let workers = handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| "caller panicked".into())
                    .and_then(|r| r)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>((elapsed, workers))
    });
    result
}
struct Observed {
    inner: Arc<dyn Backend>,
    mode: Mode,
    limit: usize,
    epoch: Instant,
    workers: ThreadLocal<RefCell<Worker>>,
}
impl Backend for Observed {
    fn evaluate(&self, g: &GameState) -> std::result::Result<EvalResult, BackendError> {
        Ok(self.evaluate_batch(&[g])?[0])
    }
    fn evaluate_batch(
        &self,
        g: &[&GameState],
    ) -> std::result::Result<Vec<EvalResult>, BackendError> {
        let mut worker = self
            .workers
            .get_or(|| {
                RefCell::new(Worker {
                    samples: Vec::with_capacity(self.limit),
                    ..Worker::default()
                })
            })
            .borrow_mut();
        let began = Instant::now();
        let _trace = if self.mode == Mode::Timeline {
            Some(trace::request())
        } else {
            None
        };
        let result = self.inner.evaluate_batch(g)?;
        if result.len() != g.len() {
            return Err(BackendError::msg("backend output count differs"));
        }
        let request = worker.calls as usize;
        worker.calls += 1;
        worker.positions += g.len() as u64;
        sample(
            &mut worker,
            self.limit,
            0,
            request,
            g.len(),
            began,
            self.epoch,
        );
        Ok(result)
    }
}
fn mux_difference(end: MuxStatsSnapshot, start: MuxStatsSnapshot) -> MuxObservation {
    let before = start
        .batch_histogram
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    MuxObservation {
        calls: end.total_batches - start.total_batches,
        positions: end.total_positions - start.total_positions,
        worker_backend_ns: end.nn_time_ns - start.nn_time_ns,
        worker_wait_drain_ns: end.wait_time_ns - start.wait_time_ns,
        batch_histogram: end
            .batch_histogram
            .into_iter()
            .filter_map(|(b, n)| {
                let n = n - before.get(&b).copied().unwrap_or(0);
                (n > 0).then_some((b, n))
            })
            .collect(),
    }
}
pub fn execute(request: &TrialRequest, output: &Path) -> Result<TrialResult> {
    request.plan.validate()?;
    verify(&request.executable)?;
    verify(&request.corpus)?;
    if let Some(m) = &request.model {
        verify(m)?;
    }
    let began = Instant::now();
    stage(output, "setup")?;
    let corpus: Corpus = read_json(&request.corpus.path)?;
    let games = corpus_games(&corpus)?;
    let encoder = FlatEncoder::new(games[0].width, games[0].height);
    let mut encoded = vec![0f32; games.len() * encoder.obs_dim()];
    for (i, g) in games.iter().enumerate() {
        encoder.encode_into(g, &mut encoded, i * encoder.obs_dim());
    }
    let encoded_bytes = encoded
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect::<Vec<_>>();
    // All output paths are ready before loading the backend.
    fs::create_dir(output.join("bundles"))?;
    let built = build(request, games[0].width, games[0].height)?;
    let ident = identity(
        &request.corpus.sha256,
        &hash(&encoded_bytes),
        request.model.as_ref(),
        built.engine.clone(),
        output,
    )?;
    if ident.executable.sha256 != request.executable.sha256 {
        return Err("child executable identity differs".into());
    }
    if let Some(expected) = &request.expected_identity {
        if expected != &ident {
            return Err("diagnostic runtime/build/input identity differs from original".into());
        }
    }
    let setup_ms = millis(began.elapsed());
    stage(output, "warmup")?;
    let warm_start = Instant::now();
    let shapes = request.case.warm_shapes();
    let mut passes = 0;
    loop {
        for &n in &shapes {
            let inputs = (0..n).map(|i| &games[i % games.len()]).collect::<Vec<_>>();
            validate_results(&built.backend.evaluate_batch(&inputs)?, n)?;
            if millis(warm_start.elapsed()) > request.plan.measurement.warmup.max_ms {
                return Err("warmup limit exceeded".into());
            }
        }
        passes += 1;
        if passes >= request.plan.measurement.warmup.passes
            && millis(warm_start.elapsed()) >= request.plan.measurement.warmup.min_ms
        {
            break;
        }
    }
    let warmup_ms = millis(warm_start.elapsed());
    let gpu_before = if built.engine.is_some() {
        gpu_query("uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm")
    } else {
        None
    };
    let before_mux = built.mux.as_ref().map(|s| s.snapshot());
    #[cfg(feature = "tensorrt")]
    let before_trt = built.trt.as_ref().map(|s| s.snapshot());
    let mode = request.plan.measurement.mode;
    if mode == Mode::Timeline {
        trace::enable()?;
    }
    stage(output, "measurement")?;
    let (wall_ns, mut workers, selfplay) = match &request.case {
        Case::Capacity { requests, .. } => {
            let (elapsed, workers) = capacity(
                built.backend.as_ref(),
                &games,
                requests,
                &request.plan.measurement,
            )?;
            (elapsed, workers, None)
        }
        Case::Selfplay { config, .. } => {
            let games = (0..config.games)
                .map(|i| games[i as usize % games.len()].clone())
                .collect::<Vec<_>>();
            let observed = observe(mode).then(|| Observed {
                inner: built.backend.clone(),
                mode,
                limit: request.plan.measurement.max_samples_per_thread,
                epoch: Instant::now(),
                workers: ThreadLocal::new(),
            });
            let backend: &dyn Backend = observed
                .as_ref()
                .map_or(built.backend.as_ref(), |b| b as &dyn Backend);
            let _range = trace::range(c"inference.measure", 0);
            let play_config = alpharat_sampling::SelfPlayConfig {
                n_sims: config.simulations,
                batch_size: config.batch_size,
                num_threads: config.workers,
                seed: Some(config.seed),
            };
            let r = match config.engine {
                SearchEngine::Mcts => alpharat_sampling::run_self_play_to_disk(
                    &games,
                    backend,
                    &config
                        .search
                        .as_ref()
                        .map(SearchParameters::mcts)
                        .unwrap_or_default(),
                    &play_config,
                    &output.join("bundles"),
                    config.games_per_bundle,
                    None,
                )?,
                SearchEngine::Mcgs => alpharat_sampling::run_mcgs_self_play_to_disk(
                    &games,
                    backend,
                    &config
                        .search
                        .as_ref()
                        .ok_or("MCGS search parameters missing")?
                        .mcgs(),
                    &play_config,
                    &output.join("bundles"),
                    config.games_per_bundle,
                    None,
                )?,
            };
            drop(_range);
            let workers = observed
                .map(|o| {
                    o.workers
                        .into_iter()
                        .enumerate()
                        .map(|(id, w)| {
                            let mut w = w.into_inner();
                            for s in &mut w.samples {
                                s.worker = id;
                            }
                            w
                        })
                        .collect()
                })
                .unwrap_or_default();
            let stats = r.stats;
            let files = r
                .written_paths
                .iter()
                .map(|p| {
                    p.strip_prefix(output)
                        .map(|r| r.to_string_lossy().into_owned())
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            (
                nanos(Duration::from_secs_f64(stats.elapsed_secs)),
                workers,
                Some(SelfplayObservation {
                    games: stats.total_games,
                    positions: stats.total_positions,
                    simulations: stats.total_simulations,
                    nn_evals: stats.total_nn_evals,
                    terminals: stats.total_terminals,
                    collisions: stats.total_collisions,
                    tt_stop_hits: stats.total_tt_stop_hits,
                    bundle_files: files,
                }),
            )
        }
    };
    stage(output, "finalize")?;
    let mux = built
        .mux
        .as_ref()
        .zip(before_mux)
        .map(|(s, b)| mux_difference(s.snapshot(), b));
    #[cfg(feature = "tensorrt")]
    let mut stages = None;
    #[cfg(not(feature = "tensorrt"))]
    let stages = None;
    #[cfg(feature = "tensorrt")]
    if let Some((end, begin)) = built.trt.as_ref().map(|s| s.snapshot()).zip(before_trt) {
        if mode == Mode::Stages {
            stages = Some(StageObservation {
                calls: end.calls - begin.calls,
                positions: end.positions - begin.positions,
                encode_ns: end.encode_ns - begin.encode_ns,
                input_stage_ns: end.input_stage_ns - begin.input_stage_ns,
                h2d_ns: end.h2d_ns - begin.h2d_ns,
                infer_ns: end.infer_ns - begin.infer_ns,
                d2h_ns: end.d2h_ns - begin.d2h_ns,
                output_alloc_ns: end.output_alloc_ns - begin.output_alloc_ns,
                parse_ns: end.parse_ns - begin.parse_ns,
                total_ns: end.total_ns - begin.total_ns,
            });
        }
    }
    let calls = if matches!(request.case, Case::Capacity { .. }) || observe(mode) {
        Some(workers.iter().map(|w| w.calls).sum())
    } else {
        None
    };
    let positions = selfplay
        .as_ref()
        .map_or_else(|| workers.iter().map(|w| w.positions).sum(), |s| s.nn_evals);
    let dropped = workers.iter().map(|w| w.dropped).sum();
    let samples = workers
        .iter_mut()
        .flat_map(|w| std::mem::take(&mut w.samples))
        .collect::<Vec<_>>();
    let latency = latency_summary(samples.iter().map(|s| s.elapsed_ns).collect(), dropped);
    let mut artifacts = BTreeMap::new();
    if built.engine.is_some() {
        let after =
            gpu_query("uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm");
        write_json(
            &output.join("gpu-state.json"),
            &serde_json::json!({
            "fields":"uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm",
            "before":gpu_before,"after":after,"note":"Snapshots outside measurement; no attribution or exclusivity claim."}),
        )?;
        artifacts.insert(
            "gpu-state.json".into(),
            hash(&fs::read(output.join("gpu-state.json"))?),
        );
    }
    if observe(mode) {
        let path = output.join("requests.csv");
        let mut writer = csv::Writer::from_path(&path)?;
        for s in &samples {
            writer.serialize(s)?;
        }
        writer.flush()?;
        artifacts.insert("requests.csv".into(), hash(&fs::read(path)?));
    }
    if mode == Mode::Timeline {
        let (links, dropped) = trace::links();
        let links = links
            .iter()
            .map(|l| {
                serde_json::json!({"request_id":l.request_id,"batch_id":l.batch_id,
            "positions":l.positions,"queued_ns":l.queued_ns,"dequeued_ns":l.dequeued_ns})
            })
            .collect::<Vec<_>>();
        write_json(
            &output.join("batch-links.json"),
            &serde_json::json!({"links":links,"dropped":dropped}),
        )?;
        artifacts.insert(
            "batch-links.json".into(),
            hash(&fs::read(output.join("batch-links.json"))?),
        );
    }
    if let Some(s) = &selfplay {
        for file in &s.bundle_files {
            artifacts.insert(file.clone(), hash(&fs::read(output.join(file))?));
        }
    }
    artifacts.insert(
        "source.patch".into(),
        hash(&fs::read(output.join("source.patch"))?),
    );
    if wall_ns == 0 || positions == 0 {
        return Err("measurement produced no timed work".into());
    }
    let result = TrialResult {
        schema_version: VERSION,
        case_key: request.case.key().into(),
        variant_id: request.variant.id.clone(),
        mode,
        identity: ident,
        setup_ms,
        warmup_ms,
        warmup_passes: passes,
        warmed_shapes: shapes.into_iter().collect(),
        metrics: Metrics {
            wall_ns,
            completed_calls: calls,
            completed_evaluations: positions,
            latency,
            mux,
            stages,
            selfplay,
        },
        artifacts,
    };
    write_json(&output.join("result.json"), &result)?;
    stage(output, "completed")?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn corpus() -> Corpus {
        Corpus {
            schema_version: 1,
            label: "test".into(),
            positions: vec![GameRecipe {
                width: 5,
                height: 5,
                player_1: [0, 0],
                player_2: [4, 4],
                cheese: vec![[2, 2]],
                max_turns: 20,
                creation_seed: 1,
                walls: vec![],
                mud: vec![],
                actions: vec![],
            }],
        }
    }
    #[test]
    fn action_prefix_changes_production_state() {
        let original = corpus_games(&corpus()).unwrap();
        let mut c = corpus();
        c.positions[0].actions.push([1, 3]);
        let next = corpus_games(&c).unwrap();
        let encoder = FlatEncoder::new(5, 5);
        let mut a = vec![0.; encoder.obs_dim()];
        let mut b = a.clone();
        encoder.encode_into(&original[0], &mut a, 0);
        encoder.encode_into(&next[0], &mut b, 0);
        assert_ne!(a, b);
    }
    #[test]
    fn mud_prefix_uses_production_timer_and_walls_are_checked() {
        let mut c = corpus();
        c.positions[0].mud.push(MudEdge {
            from: [0, 0],
            to: [0, 1],
            cost: 3,
        });
        c.positions[0].actions.push([0, 4]);
        assert!(corpus_games(&c).unwrap()[0].player1.mud_timer > 0);
        c.positions[0].walls.push([[0, 0], [0, 1]]);
        assert!(corpus_games(&c).is_err());
    }
    #[test]
    fn invalid_action_prefix_rejected() {
        let mut c = corpus();
        c.positions[0].actions.push([9, 0]);
        assert!(corpus_games(&c).is_err());
    }
    #[test]
    fn clean_capacity_has_exact_counts_without_samples() {
        let games = corpus_games(&corpus()).unwrap();
        let m = Measurement {
            mode: Mode::Clean,
            warmup: Warmup {
                passes: 1,
                min_ms: 0,
                max_ms: 1000,
            },
            calls_per_caller: 7,
            repetitions: 1,
            max_samples_per_thread: 10,
        };
        let (_, w) = capacity(
            &SmartUniformBackend,
            &games,
            &Requests::Sequence {
                batches: vec![1, 3],
                callers: 3,
                corpus_offset: 0,
            },
            &m,
        )
        .unwrap();
        assert_eq!(w.iter().map(|w| w.calls).sum::<u64>(), 21);
        assert_eq!(w.iter().map(|w| w.positions).sum::<u64>(), 39);
        assert!(w.iter().all(|w| w.samples.capacity() == 0));
    }
    struct Failing;
    impl Backend for Failing {
        fn evaluate(&self, _: &GameState) -> std::result::Result<EvalResult, BackendError> {
            Err(BackendError::msg("deliberate failure"))
        }
    }
    #[test]
    fn concurrent_failure_releases_completion_barrier() {
        let games = corpus_games(&corpus()).unwrap();
        let m = Measurement {
            mode: Mode::Clean,
            warmup: Warmup {
                passes: 1,
                min_ms: 0,
                max_ms: 1000,
            },
            calls_per_caller: 2,
            repetitions: 1,
            max_samples_per_thread: 10,
        };
        assert!(capacity(
            &Failing,
            &games,
            &Requests::Constant {
                batch_size: 1,
                callers: 4
            },
            &m
        )
        .is_err());
    }
}
