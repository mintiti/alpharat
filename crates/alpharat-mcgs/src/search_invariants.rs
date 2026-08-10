//! Executable quiescent-DAG contract for search refactors.
//!
//! These checks deliberately target AlphaRat's graph semantics rather than
//! LC0's chess-specific value, bounds, and repetition rules. The caller must
//! provide the [`GameState`] represented by the current root and must call the
//! audit only when search workers have stopped mutating the graph.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

use alpharat_eval_core::compute_outcomes;
use pyrat::{Coordinates, Direction, GameBuilder, GameState};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::access::ExclusiveAccess;
use crate::observer::NodeHandle;
use crate::tree::{compute_rewards, MCGSTree};
use crate::{
    run_search, Backend, BackendError, ConstantValueBackend, EvalResult, SearchConfig,
    SmartUniformBackend,
};

const FLOAT_TOLERANCE: f32 = 1e-4;

/// Whether asynchronous reclamation is allowed to retain unreachable owners.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AuditMode {
    /// The reachable DAG must be coherent, but old queued edges may still own
    /// nodes and keep transposition-table entries alive.
    PendingGcAllowed,
    /// GC has drained, so live TT nodes and incoming-edge counts must match the
    /// graph reachable from the current root exactly.
    GcDrained,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AuditReport {
    pub(crate) nodes: usize,
    pub(crate) edges: usize,
    pub(crate) transposition_nodes: usize,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct AuditViolation {
    pub(crate) rule: &'static str,
    pub(crate) detail: String,
}

impl AuditViolation {
    fn new(rule: &'static str, detail: impl Into<String>) -> Self {
        Self {
            rule,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for AuditViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.rule, self.detail)
    }
}

type AuditResult<T> = Result<T, AuditViolation>;

macro_rules! require {
    ($condition:expr, $rule:literal, $($arg:tt)*) => {
        if !$condition {
            return Err(AuditViolation::new($rule, format!($($arg)*)));
        }
    };
}

#[derive(Clone)]
struct SeenNode<'session> {
    handle: NodeHandle<'session>,
    hash: u64,
    turn: u16,
}

struct WalkState<'session> {
    seen: HashMap<usize, SeenNode<'session>>,
    node_for_hash: HashMap<u64, usize>,
    incoming: HashMap<usize, usize>,
    active: HashSet<usize>,
    edges: usize,
}

impl<'session> WalkState<'session> {
    fn new() -> Self {
        Self {
            seen: HashMap::new(),
            node_for_hash: HashMap::new(),
            incoming: HashMap::new(),
            active: HashSet::new(),
            edges: 0,
        }
    }
}

struct EdgeSnapshot<'session> {
    p1_outcome: u8,
    p2_outcome: u8,
    reward_p1: f32,
    reward_p2: f32,
    visits: u32,
    q1: f32,
    q2: f32,
    child: NodeHandle<'session>,
}

/// Validate the graph rooted at `root_game` after a search or cleanup boundary.
///
/// The audit assumes collision-free `GameState::state_hash` values and a
/// well-formed backend numeric domain. It checks the assumptions where the
/// graph retains enough information to do so.
pub(crate) fn audit_quiescent_dag(
    tree: &mut MCGSTree,
    root_game: &GameState,
    mode: AuditMode,
) -> AuditResult<AuditReport> {
    let live_tt_nodes = tree.stats().transpositions.live_entries;

    tree.with_exclusive(|access| {
        let root = access.root();
        let root_id = access.test_node_id(&root);
        let mut state = WalkState::new();

        walk_node(&access, root, root_game.clone(), &mut state)?;

        for (node_id, seen) in &state.seen {
            let reachable_incoming = state.incoming.get(node_id).copied().unwrap_or(0);
            let recorded_incoming = access.num_parents(&seen.handle) as usize;
            match mode {
                AuditMode::PendingGcAllowed => require!(
                    recorded_incoming >= reachable_incoming,
                    "parent-accounting",
                    "node {node_id:#x} records {recorded_incoming} parents but has at least {reachable_incoming} reachable incoming edges"
                ),
                AuditMode::GcDrained => require!(
                    recorded_incoming == reachable_incoming,
                    "parent-accounting",
                    "node {node_id:#x} records {recorded_incoming} parents but has {reachable_incoming} reachable incoming edges"
                ),
            }
        }

        if mode == AuditMode::GcDrained {
            require!(
                state.incoming.get(&root_id).copied().unwrap_or(0) == 0,
                "parent-accounting",
                "root {root_id:#x} has a reachable incoming edge"
            );
            require!(
                access.num_parents(&state.seen[&root_id].handle) == 0,
                "parent-accounting",
                "drained root {root_id:#x} still records incoming parents"
            );
            require!(
                live_tt_nodes == state.seen.len(),
                "tt-reachability",
                "TT has {live_tt_nodes} live nodes but rooted DAG has {}",
                state.seen.len()
            );
        } else {
            require!(
                live_tt_nodes >= state.seen.len(),
                "tt-reachability",
                "TT has {live_tt_nodes} live nodes but rooted DAG already has {}",
                state.seen.len()
            );
        }

        let transposition_nodes = state
            .incoming
            .values()
            .filter(|&&parents| parents > 1)
            .count();
        Ok(AuditReport {
            nodes: state.seen.len(),
            edges: state.edges,
            transposition_nodes,
        })
    })
}

fn walk_node<'tree, 'session>(
    access: &ExclusiveAccess<'tree, 'session>,
    node: NodeHandle<'session>,
    game: GameState,
    state: &mut WalkState<'session>,
) -> AuditResult<()> {
    let node_id = access.test_node_id(&node);
    let hash = game.state_hash();
    let turn = game.turn;

    require!(
        !state.active.contains(&node_id),
        "acyclic",
        "node {node_id:#x} is reachable from itself"
    );

    if let Some(seen) = state.seen.get(&node_id) {
        require!(
            seen.hash == hash && seen.turn == turn,
            "single-state-per-node",
            "node {node_id:#x} represents ({:#x}, turn {}) and ({hash:#x}, turn {turn})",
            seen.hash,
            seen.turn
        );
        return Ok(());
    }

    if let Some(existing_id) = state.node_for_hash.insert(hash, node_id) {
        require!(
            existing_id == node_id,
            "canonical-tt-identity",
            "hash {hash:#x} maps to nodes {existing_id:#x} and {node_id:#x}"
        );
    }

    let canonical = access.test_tt_lookup(hash).ok_or_else(|| {
        AuditViolation::new(
            "canonical-tt-identity",
            format!("hash {hash:#x} for node {node_id:#x} is absent from the TT"),
        )
    })?;
    require!(
        access.same_node(&node, &canonical),
        "canonical-tt-identity",
        "TT hash {hash:#x} does not resolve to node {node_id:#x}"
    );

    state.seen.insert(
        node_id,
        SeenNode {
            handle: node.clone(),
            hash,
            turn,
        },
    );
    state.active.insert(node_id);

    let edges = snapshot_and_validate_node(access, &node, &game)?;
    state.edges += edges.len();

    for edge in edges {
        let mut child_game = game.clone();
        let p1_action = access
            .node(&node)
            .p1_outcome_action(edge.p1_outcome as usize);
        let p2_action = access
            .node(&node)
            .p2_outcome_action(edge.p2_outcome as usize);
        let p1_direction = Direction::try_from(p1_action).map_err(|_| {
            AuditViolation::new(
                "outcome-mapping",
                format!("node {node_id:#x} has invalid P1 action {p1_action}"),
            )
        })?;
        let p2_direction = Direction::try_from(p2_action).map_err(|_| {
            AuditViolation::new(
                "outcome-mapping",
                format!("node {node_id:#x} has invalid P2 action {p2_action}"),
            )
        })?;

        let scores_before = (child_game.player1_score(), child_game.player2_score());
        let parent_turn = child_game.turn;
        let _undo = child_game.make_move(p1_direction, p2_direction);
        require!(
            child_game.turn == parent_turn + 1,
            "edge-transition",
            "edge ({}, {}) from node {node_id:#x} did not advance exactly one turn",
            edge.p1_outcome,
            edge.p2_outcome
        );
        let (expected_r1, expected_r2) = compute_rewards(&child_game, scores_before);
        require!(
            approx_eq(edge.reward_p1, expected_r1)
                && approx_eq(edge.reward_p2, expected_r2),
            "edge-transition",
            "edge ({}, {}) from node {node_id:#x} stores rewards ({}, {}) but replay gives ({expected_r1}, {expected_r2})",
            edge.p1_outcome,
            edge.p2_outcome,
            edge.reward_p1,
            edge.reward_p2
        );

        let child_id = access.test_node_id(&edge.child);
        *state.incoming.entry(child_id).or_insert(0) += 1;

        let child_low = access.node(&edge.child);
        require!(
            edge.visits <= child_low.total_visits(),
            "visit-accounting",
            "edge ({}, {}) from node {node_id:#x} has {} visits but child {child_id:#x} has {}",
            edge.p1_outcome,
            edge.p2_outcome,
            edge.visits,
            child_low.total_visits()
        );
        if edge.visits > 0 && edge.visits == child_low.total_visits() {
            require!(
                approx_eq(edge.q1, edge.reward_p1 + child_low.v1())
                    && approx_eq(edge.q2, edge.reward_p2 + child_low.v2()),
                "visit-accounting",
                "caught-up edge ({}, {}) from node {node_id:#x} has Q ({}, {}) but reward + child V is ({}, {})",
                edge.p1_outcome,
                edge.p2_outcome,
                edge.q1,
                edge.q2,
                edge.reward_p1 + child_low.v1(),
                edge.reward_p2 + child_low.v2()
            );
        }

        walk_node(access, edge.child, child_game, state)?;
    }

    state.active.remove(&node_id);
    Ok(())
}

fn snapshot_and_validate_node<'tree, 'session>(
    access: &ExclusiveAccess<'tree, 'session>,
    node: &NodeHandle<'session>,
    game: &GameState,
) -> AuditResult<Vec<EdgeSnapshot<'session>>> {
    let node_id = access.test_node_id(node);
    let low = access.node(node);
    let (expected_p1, expected_n1, expected_map_p1) = compute_outcomes(game.effective_actions_p1());
    let (expected_p2, expected_n2, expected_map_p2) = compute_outcomes(game.effective_actions_p2());

    require!(
        low.n1() == expected_n1 as usize && low.n2() == expected_n2 as usize,
        "outcome-mapping",
        "node {node_id:#x} stores ({}, {}) outcomes but game requires ({expected_n1}, {expected_n2})",
        low.n1(),
        low.n2()
    );
    for (i, &expected) in expected_p1.iter().enumerate().take(low.n1()) {
        require!(
            low.p1_outcome_action(i) == expected,
            "outcome-mapping",
            "node {node_id:#x} P1 outcome {i} is {}, expected {}",
            low.p1_outcome_action(i),
            expected
        );
    }
    for (j, &expected) in expected_p2.iter().enumerate().take(low.n2()) {
        require!(
            low.p2_outcome_action(j) == expected,
            "outcome-mapping",
            "node {node_id:#x} P2 outcome {j} is {}, expected {}",
            low.p2_outcome_action(j),
            expected
        );
    }
    for action in 0..5u8 {
        require!(
            low.p1_action_to_outcome_idx(action) == expected_map_p1[action as usize]
                && low.p2_action_to_outcome_idx(action) == expected_map_p2[action as usize],
            "outcome-mapping",
            "node {node_id:#x} has an incorrect reverse mapping for action {action}"
        );
    }

    let expected_scale = game.cheese.remaining_cheese().max(1) as f32;
    require!(
        low.value_scale().is_finite() && approx_eq(low.value_scale(), expected_scale),
        "finite-statistics",
        "node {node_id:#x} has value scale {}, expected {expected_scale}",
        low.value_scale()
    );
    require!(
        low.v1().is_finite() && low.v2().is_finite(),
        "finite-statistics",
        "node {node_id:#x} has non-finite value ({}, {})",
        low.v1(),
        low.v2()
    );
    require!(
        low.n_in_flight() == 0,
        "quiescent-reservations",
        "node {node_id:#x} has {} node reservations",
        low.n_in_flight()
    );

    let mut p1_prior_sum = 0.0;
    for i in 0..low.n1() {
        let prior = low.p1_prior(i);
        require!(
            prior.is_finite() && prior >= 0.0,
            "finite-statistics",
            "node {node_id:#x} has invalid P1 prior {prior} at {i}"
        );
        p1_prior_sum += prior;
    }
    let mut p2_prior_sum = 0.0;
    for j in 0..low.n2() {
        let prior = low.p2_prior(j);
        require!(
            prior.is_finite() && prior >= 0.0,
            "finite-statistics",
            "node {node_id:#x} has invalid P2 prior {prior} at {j}"
        );
        p2_prior_sum += prior;
    }
    if low.is_evaluated() {
        require!(
            approx_eq(p1_prior_sum, 1.0) && approx_eq(p2_prior_sum, 1.0),
            "node-lifecycle",
            "evaluated node {node_id:#x} has prior sums ({p1_prior_sum}, {p2_prior_sum})"
        );
    } else {
        require!(
            p1_prior_sum == 0.0 && p2_prior_sum == 0.0,
            "node-lifecycle",
            "unevaluated node {node_id:#x} has nonzero prior mass ({p1_prior_sum}, {p2_prior_sum})"
        );
    }

    let mut matrix_visits = 0u32;
    let mut p1_marginal_total = 0u32;
    let mut p2_marginal_total = 0u32;
    for i in 0..low.n1() {
        p1_marginal_total += low.marginal_visits_p1(i);
        for j in 0..low.n2() {
            let visits = low.edge_visits(i, j);
            let q1 = low.edge_q_p1(i, j);
            let q2 = low.edge_q_p2(i, j);
            require!(
                low.edge_in_flight(i, j) == 0,
                "quiescent-reservations",
                "node {node_id:#x} cell ({i}, {j}) has {} reservations",
                low.edge_in_flight(i, j)
            );
            require!(
                q1.is_finite() && q2.is_finite(),
                "finite-statistics",
                "node {node_id:#x} cell ({i}, {j}) has non-finite Q ({q1}, {q2})"
            );
            if visits == 0 {
                require!(
                    q1 == 0.0 && q2 == 0.0,
                    "visit-accounting",
                    "unvisited cell ({i}, {j}) at node {node_id:#x} has Q ({q1}, {q2})"
                );
            }
            matrix_visits = matrix_visits.checked_add(visits).ok_or_else(|| {
                AuditViolation::new(
                    "visit-accounting",
                    format!("edge-visit total overflow at node {node_id:#x}"),
                )
            })?;
        }
    }
    for j in 0..low.n2() {
        p2_marginal_total += low.marginal_visits_p2(j);
    }
    require!(
        matrix_visits == low.total_edge_visits()
            && p1_marginal_total == matrix_visits
            && p2_marginal_total == matrix_visits,
        "visit-accounting",
        "node {node_id:#x} disagrees on matrix/marginal visit totals: matrix={matrix_visits}, p1={p1_marginal_total}, p2={p2_marginal_total}, stored={} ",
        low.total_edge_visits()
    );

    let mut edges = Vec::new();
    let mut outcomes = HashSet::new();
    let mut current = low.first_child();
    while let Some(edge) = current {
        let (i, j) = edge.parent_outcome();
        require!(
            (i as usize) < low.n1() && (j as usize) < low.n2(),
            "outcome-mapping",
            "node {node_id:#x} edge outcome ({i}, {j}) is out of range"
        );
        require!(
            outcomes.insert((i, j)),
            "unique-parent-outcome",
            "node {node_id:#x} has more than one edge for outcome ({i}, {j})"
        );
        edges.push(EdgeSnapshot {
            p1_outcome: i,
            p2_outcome: j,
            reward_p1: edge.r1(),
            reward_p2: edge.r2(),
            visits: low.edge_visits(i as usize, j as usize),
            q1: low.edge_q_p1(i as usize, j as usize),
            q2: low.edge_q_p2(i as usize, j as usize),
            child: NodeHandle::new(edge.low_node().clone()),
        });
        current = edge.next_sibling();
    }

    for i in 0..low.n1() {
        for j in 0..low.n2() {
            require!(
                low.edge_visits(i, j) == 0 || outcomes.contains(&(i as u8, j as u8)),
                "visit-accounting",
                "node {node_id:#x} cell ({i}, {j}) has visits but no child edge"
            );
        }
    }

    if low.total_visits() == 0 {
        require!(
            low.total_edge_visits() == 0 && low.v1() == 0.0 && low.v2() == 0.0 && edges.is_empty(),
            "node-lifecycle",
            "zero-visit node {node_id:#x} has values, outgoing visits, or children"
        );
    } else {
        require!(
            low.is_evaluated() || low.is_terminal(),
            "node-lifecycle",
            "visited node {node_id:#x} is neither evaluated nor terminal"
        );
    }

    if low.is_terminal() {
        require!(
            game.check_game_over(),
            "node-lifecycle",
            "node {node_id:#x} is marked terminal but replayed game is live"
        );
        require!(
            low.total_edge_visits() == 0 && edges.is_empty() && low.v1() == 0.0 && low.v2() == 0.0,
            "node-lifecycle",
            "terminal node {node_id:#x} has children, outgoing visits, or nonzero future value"
        );
    } else if low.total_visits() > 0 {
        require!(
            !game.check_game_over(),
            "node-lifecycle",
            "visited node {node_id:#x} represents a terminal game but is not marked terminal"
        );
        require!(
            low.total_visits() == low.total_edge_visits() + 1,
            "visit-accounting",
            "visited nonterminal node {node_id:#x} has {} node visits and {} outgoing visits",
            low.total_visits(),
            low.total_edge_visits()
        );
    }

    Ok(edges)
}

fn approx_eq(left: f32, right: f32) -> bool {
    let scale = 1.0f32.max(left.abs()).max(right.abs());
    (left - right).abs() <= FLOAT_TOLERANCE * scale
}

fn open_game(max_turns: u16) -> GameState {
    GameBuilder::new(5, 5)
        .with_open_maze()
        .with_custom_positions(Coordinates::new(2, 2), Coordinates::new(2, 2))
        .with_custom_cheese(vec![Coordinates::new(0, 0), Coordinates::new(4, 4)])
        .with_max_turns(max_turns)
        .build()
        .create(None)
        .unwrap()
}

fn root_visits(tree: &MCGSTree) -> u32 {
    tree.observe(|view| view.with_root(|root| root.stats().total_visits))
}

#[derive(Clone, Copy)]
enum BatchLengthSkew {
    Short,
    Long,
}

struct WrongBatchLengthBackend {
    skew: BatchLengthSkew,
    requested: AtomicUsize,
}

impl WrongBatchLengthBackend {
    fn new(skew: BatchLengthSkew) -> Self {
        Self {
            skew,
            requested: AtomicUsize::new(0),
        }
    }

    fn requested(&self) -> usize {
        self.requested.load(Ordering::Relaxed)
    }
}

impl Backend for WrongBatchLengthBackend {
    fn evaluate(&self, game: &GameState) -> Result<EvalResult, BackendError> {
        SmartUniformBackend.evaluate(game)
    }

    fn evaluate_batch(&self, games: &[&GameState]) -> Result<Vec<EvalResult>, BackendError> {
        self.requested.store(games.len(), Ordering::Relaxed);
        let mut results = SmartUniformBackend.evaluate_batch(games)?;
        match self.skew {
            BatchLengthSkew::Short => {
                results.pop();
            }
            BatchLengthSkew::Long => {
                let game = games
                    .first()
                    .expect("search should not evaluate an empty backend batch");
                results.push(SmartUniformBackend.evaluate(game)?);
            }
        }
        Ok(results)
    }
}

struct ErrorAfterOneBatchBackend {
    calls: AtomicUsize,
}

impl ErrorAfterOneBatchBackend {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }
}

impl Backend for ErrorAfterOneBatchBackend {
    fn evaluate(&self, game: &GameState) -> Result<EvalResult, BackendError> {
        SmartUniformBackend.evaluate(game)
    }

    fn evaluate_batch(&self, games: &[&GameState]) -> Result<Vec<EvalResult>, BackendError> {
        if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
            SmartUniformBackend.evaluate_batch(games)
        } else {
            Err(BackendError::msg("intentional backend failure"))
        }
    }
}

struct PanicAfterOneBatchBackend {
    calls: AtomicUsize,
}

impl PanicAfterOneBatchBackend {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }
}

impl Backend for PanicAfterOneBatchBackend {
    fn evaluate(&self, game: &GameState) -> Result<EvalResult, BackendError> {
        SmartUniformBackend.evaluate(game)
    }

    fn evaluate_batch(&self, games: &[&GameState]) -> Result<Vec<EvalResult>, BackendError> {
        if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
            SmartUniformBackend.evaluate_batch(games)
        } else {
            panic!("intentional backend panic")
        }
    }
}

struct NonFiniteTailBackend {
    requested: AtomicUsize,
}

impl NonFiniteTailBackend {
    fn new() -> Self {
        Self {
            requested: AtomicUsize::new(0),
        }
    }

    fn requested(&self) -> usize {
        self.requested.load(Ordering::Relaxed)
    }
}

impl Backend for NonFiniteTailBackend {
    fn evaluate(&self, game: &GameState) -> Result<EvalResult, BackendError> {
        SmartUniformBackend.evaluate(game)
    }

    fn evaluate_batch(&self, games: &[&GameState]) -> Result<Vec<EvalResult>, BackendError> {
        self.requested.store(games.len(), Ordering::Relaxed);
        let mut results = SmartUniformBackend.evaluate_batch(games)?;
        results
            .last_mut()
            .expect("search should not evaluate an empty backend batch")
            .value_p2 = f32::NAN;
        Ok(results)
    }
}

#[test]
fn fresh_tree_satisfies_quiescent_invariants() {
    let game = open_game(20);
    let mut tree = MCGSTree::new(&game);

    let report = audit_quiescent_dag(&mut tree, &game, AuditMode::GcDrained).unwrap();

    assert_eq!(report.nodes, 1);
    assert_eq!(report.edges, 0);
}

#[test]
fn searched_tree_satisfies_quiescent_invariants() {
    for batch_size in [1, 16] {
        let game = open_game(20);
        let mut tree = MCGSTree::new(&game);
        let mut rng = SmallRng::seed_from_u64(0x00A1_1CE5 + batch_size as u64);
        let before = root_visits(&tree);

        let result = run_search(
            &mut tree,
            &game,
            &SmartUniformBackend,
            &SearchConfig::default(),
            400,
            batch_size,
            &mut rng,
        )
        .unwrap();
        let after = root_visits(&tree);

        assert_eq!(result.total_visits, after);
        assert_eq!(
            after - before,
            result.nn_evals + result.terminals + result.tt_stop_hits
        );
        let report = audit_quiescent_dag(&mut tree, &game, AuditMode::GcDrained).unwrap();
        assert!(report.nodes > 1);
        assert!(report.edges > 0);
        assert!(
            report.transposition_nodes > 0,
            "fixture should exercise a shared DAG node at batch size {batch_size}"
        );
    }
}

#[test]
fn asymmetric_nonzero_values_preserve_player_identity() {
    let game = open_game(20);
    let mut tree = MCGSTree::new(&game);
    let mut rng = SmallRng::seed_from_u64(0xA51_0E7);
    let backend = ConstantValueBackend {
        value_p1: 0.75,
        value_p2: 0.125,
    };

    let result = run_search(
        &mut tree,
        &game,
        &backend,
        &SearchConfig::default(),
        2,
        1,
        &mut rng,
    )
    .unwrap();

    assert!(approx_eq(result.value_p1, 0.75));
    assert!(approx_eq(result.value_p2, 0.125));
    tree.observe(|view| {
        let (root, transition, child) = view.with_root(|root| {
            let mut children = root.children();
            let edge = children
                .next()
                .expect("the second simulation should publish an edge");
            assert!(children.next().is_none());
            (root.stats(), edge.transition(), edge.child())
        });
        let child = view.with_node(&child, |child| child.stats());
        let q1 = view.with_root(|root| {
            root.outcome(crate::SearchPlayer::Player1, transition.p1_outcome)
                .unwrap()
                .q
        });
        let q2 = view.with_root(|root| {
            root.outcome(crate::SearchPlayer::Player2, transition.p2_outcome)
                .unwrap()
                .q
        });

        assert_eq!(root.total_visits, 2);
        assert_eq!(root.total_edge_visits, 1);
        assert_eq!(child.total_visits, 1);
        assert!(approx_eq(child.value_p1, 0.75));
        assert!(approx_eq(child.value_p2, 0.125));
        assert!(approx_eq(q1, transition.reward_p1 + 0.75));
        assert!(approx_eq(q2, transition.reward_p2 + 0.125));
        assert!(approx_eq(root.value_p1, 0.75 + transition.reward_p1 / 2.0));
        assert!(approx_eq(root.value_p2, 0.125 + transition.reward_p2 / 2.0));
    });
    audit_quiescent_dag(&mut tree, &game, AuditMode::GcDrained).unwrap();
}

#[test]
fn terminal_root_satisfies_quiescent_invariants() {
    let mut game = open_game(1);
    let _undo = game.make_move(Direction::Stay, Direction::Stay);
    assert!(game.check_game_over());
    let mut tree = MCGSTree::new(&game);
    let mut rng = SmallRng::seed_from_u64(7);

    let result = run_search(
        &mut tree,
        &game,
        &SmartUniformBackend,
        &SearchConfig::default(),
        8,
        4,
        &mut rng,
    )
    .unwrap();

    assert_eq!(result.terminals, 8);
    assert_eq!(result.total_visits, 8);
    audit_quiescent_dag(&mut tree, &game, AuditMode::GcDrained).unwrap();
}

#[test]
fn wrong_backend_batch_lengths_are_rejected_before_returned_eval_commit() {
    for (skew, seed) in [(BatchLengthSkew::Short, 11), (BatchLengthSkew::Long, 12)] {
        let game = open_game(20);
        let mut tree = MCGSTree::new(&game);
        let mut rng = SmallRng::seed_from_u64(seed);
        run_search(
            &mut tree,
            &game,
            &SmartUniformBackend,
            &SearchConfig::default(),
            1,
            1,
            &mut rng,
        )
        .unwrap();
        let visits_before_error = root_visits(&tree);
        let backend = WrongBatchLengthBackend::new(skew);

        let error = run_search(
            &mut tree,
            &game,
            &backend,
            &SearchConfig::default(),
            8,
            8,
            &mut rng,
        )
        .unwrap_err();

        assert!(
            backend.requested() > 1,
            "fixture must exercise a multi-result backend batch"
        );
        assert!(error.to_string().contains("result count"));
        assert_eq!(root_visits(&tree), visits_before_error);
        audit_quiescent_dag(&mut tree, &game, AuditMode::GcDrained).unwrap();
        run_search(
            &mut tree,
            &game,
            &SmartUniformBackend,
            &SearchConfig::default(),
            20,
            8,
            &mut rng,
        )
        .unwrap();
        audit_quiescent_dag(&mut tree, &game, AuditMode::GcDrained).unwrap();
    }
}

#[test]
fn non_finite_tail_result_is_rejected_before_returned_eval_commit() {
    let game = open_game(20);
    let mut tree = MCGSTree::new(&game);
    let mut rng = SmallRng::seed_from_u64(21);
    run_search(
        &mut tree,
        &game,
        &SmartUniformBackend,
        &SearchConfig::default(),
        1,
        1,
        &mut rng,
    )
    .unwrap();
    let visits_before_error = root_visits(&tree);
    let backend = NonFiniteTailBackend::new();

    let error = run_search(
        &mut tree,
        &game,
        &backend,
        &SearchConfig::default(),
        8,
        8,
        &mut rng,
    )
    .unwrap_err();

    assert!(
        backend.requested() > 1,
        "fixture must put valid results before the malformed tail"
    );
    assert!(error.to_string().contains("non-finite"));
    assert_eq!(root_visits(&tree), visits_before_error);
    audit_quiescent_dag(&mut tree, &game, AuditMode::GcDrained).unwrap();
    run_search(
        &mut tree,
        &game,
        &SmartUniformBackend,
        &SearchConfig::default(),
        20,
        8,
        &mut rng,
    )
    .unwrap();
}

#[test]
fn backend_error_after_in_call_commit_preserves_commit_and_restores_quiescence() {
    let game = open_game(20);
    let mut tree = MCGSTree::new(&game);
    let mut rng = SmallRng::seed_from_u64(12);
    let backend = ErrorAfterOneBatchBackend::new();

    assert_eq!(root_visits(&tree), 0);

    let error = run_search(
        &mut tree,
        &game,
        &backend,
        &SearchConfig::default(),
        2,
        1,
        &mut rng,
    )
    .unwrap_err();

    assert!(error.to_string().contains("intentional backend failure"));
    assert_eq!(root_visits(&tree), 1);
    audit_quiescent_dag(&mut tree, &game, AuditMode::GcDrained).unwrap();

    run_search(
        &mut tree,
        &game,
        &SmartUniformBackend,
        &SearchConfig::default(),
        20,
        8,
        &mut rng,
    )
    .unwrap();
    audit_quiescent_dag(&mut tree, &game, AuditMode::GcDrained).unwrap();
}

#[test]
fn backend_panic_after_in_call_commit_preserves_commit_and_restores_quiescence() {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    let game = open_game(20);
    let mut tree = MCGSTree::new(&game);
    let mut rng = SmallRng::seed_from_u64(13);
    let backend = PanicAfterOneBatchBackend::new();

    assert_eq!(root_visits(&tree), 0);

    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = run_search(
            &mut tree,
            &game,
            &backend,
            &SearchConfig::default(),
            2,
            1,
            &mut rng,
        );
    }));

    assert!(panic.is_err());
    assert_eq!(root_visits(&tree), 1);
    audit_quiescent_dag(&mut tree, &game, AuditMode::GcDrained).unwrap();
    run_search(
        &mut tree,
        &game,
        &SmartUniformBackend,
        &SearchConfig::default(),
        20,
        8,
        &mut rng,
    )
    .unwrap();
}

#[test]
fn oracle_rejects_wrong_transition_reward() {
    let game = open_game(20);
    let mut tree = MCGSTree::new(&game);
    let mut rng = SmallRng::seed_from_u64(17);
    run_search(
        &mut tree,
        &game,
        &SmartUniformBackend,
        &SearchConfig::default(),
        1,
        1,
        &mut rng,
    )
    .unwrap();
    tree.with_exclusive(|mut access| {
        let root = access.root();
        let (p1_action, p2_action) = {
            let root_low = access.node(&root);
            (root_low.p1_outcome_action(0), root_low.p2_outcome_action(0))
        };
        let mut child_game = game.clone();
        let _undo = child_game.make_move(
            Direction::try_from(p1_action).unwrap(),
            Direction::try_from(p2_action).unwrap(),
        );
        access.find_or_create_child(&root, 0, 0, &child_game, 7.0, 9.0);
    });

    let violation = audit_quiescent_dag(&mut tree, &game, AuditMode::GcDrained).unwrap_err();
    assert_eq!(violation.rule, "edge-transition");
}

#[test]
fn advance_and_reuse_satisfy_pending_then_drained_gc_invariants() {
    crate::gc::init();
    crate::gc::stop();
    crate::gc::wait();
    let mut game = open_game(20);
    let mut tree = MCGSTree::new(&game);
    let mut rng = SmallRng::seed_from_u64(19);
    run_search(
        &mut tree,
        &game,
        &SmartUniformBackend,
        &SearchConfig::default(),
        50,
        8,
        &mut rng,
    )
    .unwrap();

    let (p1_action, p2_action) = tree.observe(|view| {
        view.with_root(|root| {
            let mut children = root.children();
            let transition = children
                .next()
                .expect("searched root should have a child")
                .transition();
            assert!(
                children.next().is_some(),
                "fixture must have a sibling for root advancement to prune"
            );
            (
                root.outcome(crate::SearchPlayer::Player1, transition.p1_outcome)
                    .unwrap()
                    .action,
                root.outcome(crate::SearchPlayer::Player2, transition.p2_outcome)
                    .unwrap()
                    .action,
            )
        })
    });
    let _undo = game.make_move(
        Direction::try_from(p1_action).unwrap(),
        Direction::try_from(p2_action).unwrap(),
    );
    tree.advance_root(&game, p1_action, p2_action);

    let report = audit_quiescent_dag(&mut tree, &game, AuditMode::PendingGcAllowed).unwrap();
    assert!(report.nodes >= 1);

    run_search(
        &mut tree,
        &game,
        &SmartUniformBackend,
        &SearchConfig::default(),
        20,
        8,
        &mut rng,
    )
    .unwrap();
    audit_quiescent_dag(&mut tree, &game, AuditMode::PendingGcAllowed).unwrap();

    crate::gc::start();
    crate::gc::stop();
    crate::gc::wait();
    tree.evict_expired();
    audit_quiescent_dag(&mut tree, &game, AuditMode::GcDrained).unwrap();
}
