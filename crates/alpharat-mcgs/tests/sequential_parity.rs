use alpharat_mcgs::{run_search, MCGSTree, SearchConfig, SearchResult, SmartUniformBackend};
use pyrat::{Coordinates, Direction, GameBuilder, GameState};
use rand::rngs::SmallRng;
use rand::SeedableRng;

// Exact noise-off behavior captured from pre-Chunk-2 commit 22a90fd.
const FIRST_RESULT: u64 = 0xca96_aca0_3a20_e93d;
const FIRST_STATS: u64 = 0x3d1c_b749_c27f_1e74;
const ADVANCED_STATS: u64 = 0x6c89_e5ab_c6be_5517;
const SECOND_RESULT: u64 = 0x41e4_87d1_51d9_0f8b;
const SECOND_STATS: u64 = 0xf610_a113_f1be_c408;

fn open_7x7_game() -> GameState {
    let mut cheese = Vec::new();
    'outer: for y in 0..7 {
        for x in 0..7 {
            if (x + y) % 2 == 1 && (x, y) != (0, 0) && (x, y) != (6, 6) {
                cheese.push(Coordinates::new(x, y));
                if cheese.len() == 10 {
                    break 'outer;
                }
            }
        }
    }

    GameBuilder::new(7, 7)
        .with_open_maze()
        .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(6, 6))
        .with_custom_cheese(cheese)
        .with_max_turns(50)
        .build()
        .create(None)
        .unwrap()
}

fn mix(hash: &mut u64, word: u64) {
    *hash ^= word;
    *hash = hash.wrapping_mul(0x100_0000_01b3);
}

fn result_fingerprint(result: &SearchResult) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for values in [
        &result.policy_p1,
        &result.policy_p2,
        &result.visit_counts_p1,
        &result.visit_counts_p2,
        &result.prior_p1,
        &result.prior_p2,
        &result.q_values_p1,
        &result.q_values_p2,
    ] {
        for value in values {
            mix(&mut hash, value.to_bits() as u64);
        }
    }
    mix(&mut hash, result.value_p1.to_bits() as u64);
    mix(&mut hash, result.value_p2.to_bits() as u64);
    for value in [
        result.total_visits,
        result.nn_evals,
        result.terminals,
        result.collisions,
        result.tt_stop_hits,
    ] {
        mix(&mut hash, value as u64);
    }
    hash
}

fn stats_fingerprint(tree: &MCGSTree) -> u64 {
    let stats = tree.stats();
    let mut hash = 0xcbf2_9ce4_8422_2325;
    mix(&mut hash, stats.node_count as u64);
    mix(&mut hash, stats.transpositions.entries as u64);
    mix(&mut hash, stats.transpositions.live_entries as u64);
    mix(&mut hash, stats.transpositions.expired_entries as u64);
    hash
}

fn best_action(visits: &[f32; 5]) -> u8 {
    visits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap())
        .unwrap()
        .0 as u8
}

#[test]
fn fixed_seed_fresh_and_reuse_match_pre_capability_baseline() {
    let mut game = open_7x7_game();
    let mut tree = MCGSTree::new(&game);
    let backend = SmartUniformBackend;
    let config = SearchConfig::default();
    assert_eq!(config.noise_epsilon, 0.0);

    let mut rng = SmallRng::seed_from_u64(42);
    let first = run_search(&mut tree, &game, &backend, &config, 8_000, 64, &mut rng).unwrap();
    assert_eq!(result_fingerprint(&first), FIRST_RESULT);
    assert_eq!(stats_fingerprint(&tree), FIRST_STATS);

    let p1_action = best_action(&first.visit_counts_p1);
    let p2_action = best_action(&first.visit_counts_p2);
    assert_eq!((p1_action, p2_action), (1, 2));
    game.make_move(
        Direction::try_from(p1_action).unwrap(),
        Direction::try_from(p2_action).unwrap(),
    );
    tree.advance_root(&game, p1_action, p2_action);
    tree.evict_expired();
    assert_eq!(stats_fingerprint(&tree), ADVANCED_STATS);

    let mut rng = SmallRng::seed_from_u64(123);
    let second = run_search(&mut tree, &game, &backend, &config, 8_000, 64, &mut rng).unwrap();
    assert_eq!(result_fingerprint(&second), SECOND_RESULT);
    assert_eq!(stats_fingerprint(&tree), SECOND_STATS);
}
