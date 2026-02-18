//! Serialize `GameRecord`s to the bundle `.npz` format consumed by the Python
//! training pipeline.
//!
//! Two layers:
//! - `write_bundle`: converts a slice of `GameRecord`s into one `.npz` file (atomic write).
//! - `BundleWriter`: streaming accumulator that flushes bundles when full.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::npz_writer::NpzWriter;
use crate::selfplay::GameRecord;

// ---------------------------------------------------------------------------
// write_bundle — GameRecord slice → bundle .npz
// ---------------------------------------------------------------------------

/// Write a bundle `.npz` from a slice of games. Atomic (tmp → rename).
///
/// All games must have the same width/height. The resulting file matches
/// the format expected by `load_game_bundle()` in Python.
pub fn write_bundle(games: &[GameRecord], path: &Path) -> io::Result<()> {
    // Validate
    if games.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "no games to write"));
    }
    let w = games[0].width;
    let h = games[0].height;
    for (i, g) in games.iter().enumerate() {
        if g.width != w || g.height != h {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "game {} has dimensions {}x{}, expected {}x{}",
                    i, g.width, g.height, w, h
                ),
            ));
        }
        if g.positions.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("game {} has no positions", i),
            ));
        }
    }

    let hw = h as usize * w as usize;
    let k = games.len(); // number of games
    let total_positions: usize = games.iter().map(|g| g.positions.len()).sum();

    // --- Build flat arrays ---

    // Game-level
    let mut game_lengths = Vec::with_capacity(k);
    let mut maze = Vec::with_capacity(k * hw * 4);
    let mut initial_cheese = Vec::with_capacity(k * hw);
    let mut cheese_outcomes = Vec::with_capacity(k * hw);
    let mut max_turns = Vec::with_capacity(k);
    let mut result = Vec::with_capacity(k);
    let mut final_p1_score = Vec::with_capacity(k);
    let mut final_p2_score = Vec::with_capacity(k);

    // Position-level
    let mut p1_pos = Vec::with_capacity(total_positions * 2);
    let mut p2_pos = Vec::with_capacity(total_positions * 2);
    let mut p1_score = Vec::with_capacity(total_positions);
    let mut p2_score = Vec::with_capacity(total_positions);
    let mut p1_mud = Vec::with_capacity(total_positions);
    let mut p2_mud = Vec::with_capacity(total_positions);
    let mut cheese_mask = Vec::with_capacity(total_positions * hw);
    let mut turn = Vec::with_capacity(total_positions);
    let mut value_p1 = Vec::with_capacity(total_positions);
    let mut value_p2 = Vec::with_capacity(total_positions);
    let mut visit_counts_p1 = Vec::with_capacity(total_positions * 5);
    let mut visit_counts_p2 = Vec::with_capacity(total_positions * 5);
    let mut prior_p1 = Vec::with_capacity(total_positions * 5);
    let mut prior_p2 = Vec::with_capacity(total_positions * 5);
    let mut policy_p1 = Vec::with_capacity(total_positions * 5);
    let mut policy_p2 = Vec::with_capacity(total_positions * 5);
    let mut action_p1 = Vec::with_capacity(total_positions);
    let mut action_p2 = Vec::with_capacity(total_positions);

    for g in games {
        // Game-level
        game_lengths.push(g.positions.len() as i32);
        maze.extend_from_slice(&g.maze);
        initial_cheese.extend_from_slice(&g.initial_cheese);
        cheese_outcomes.extend(g.cheese_outcomes.iter().map(|&v| v as i8));
        max_turns.push(g.max_turns as i16);
        result.push(g.result as u8 as i8);
        final_p1_score.push(g.final_p1_score);
        final_p2_score.push(g.final_p2_score);

        // Position-level
        for p in &g.positions {
            p1_pos.push(p.p1_pos[0] as i8);
            p1_pos.push(p.p1_pos[1] as i8);
            p2_pos.push(p.p2_pos[0] as i8);
            p2_pos.push(p.p2_pos[1] as i8);
            p1_score.push(p.p1_score);
            p2_score.push(p.p2_score);
            p1_mud.push(p.p1_mud as i8);
            p2_mud.push(p.p2_mud as i8);
            cheese_mask.extend_from_slice(&p.cheese_mask);
            turn.push(p.turn as i16);
            value_p1.push(p.value_p1);
            value_p2.push(p.value_p2);
            visit_counts_p1.extend_from_slice(&p.visit_counts_p1);
            visit_counts_p2.extend_from_slice(&p.visit_counts_p2);
            prior_p1.extend_from_slice(&p.prior_p1);
            prior_p2.extend_from_slice(&p.prior_p2);
            policy_p1.extend_from_slice(&p.policy_p1);
            policy_p2.extend_from_slice(&p.policy_p2);
            action_p1.push(p.action_p1 as i8);
            action_p2.push(p.action_p2 as i8);
        }
    }

    // --- Write to temp file, then rename ---
    let tmp_path = path.with_extension("npz.tmp");
    {
        let mut npz = NpzWriter::new(&tmp_path)?;
        let n = total_positions;
        let hh = h as usize;
        let ww = w as usize;

        // Game-level
        npz.add("game_lengths", &[k], &game_lengths)?;
        npz.add("maze", &[k, hh, ww, 4], &maze)?;
        npz.add_bool("initial_cheese", &[k, hh, ww], &initial_cheese)?;
        npz.add("cheese_outcomes", &[k, hh, ww], &cheese_outcomes)?;
        npz.add("max_turns", &[k], &max_turns)?;
        npz.add("result", &[k], &result)?;
        npz.add("final_p1_score", &[k], &final_p1_score)?;
        npz.add("final_p2_score", &[k], &final_p2_score)?;

        // Position-level
        npz.add("p1_pos", &[n, 2], &p1_pos)?;
        npz.add("p2_pos", &[n, 2], &p2_pos)?;
        npz.add("p1_score", &[n], &p1_score)?;
        npz.add("p2_score", &[n], &p2_score)?;
        npz.add("p1_mud", &[n], &p1_mud)?;
        npz.add("p2_mud", &[n], &p2_mud)?;
        npz.add_bool("cheese_mask", &[n, hh, ww], &cheese_mask)?;
        npz.add("turn", &[n], &turn)?;
        npz.add("value_p1", &[n], &value_p1)?;
        npz.add("value_p2", &[n], &value_p2)?;
        npz.add("visit_counts_p1", &[n, 5], &visit_counts_p1)?;
        npz.add("visit_counts_p2", &[n, 5], &visit_counts_p2)?;
        npz.add("prior_p1", &[n, 5], &prior_p1)?;
        npz.add("prior_p2", &[n, 5], &prior_p2)?;
        npz.add("policy_p1", &[n, 5], &policy_p1)?;
        npz.add("policy_p2", &[n, 5], &policy_p2)?;
        npz.add("action_p1", &[n], &action_p1)?;
        npz.add("action_p2", &[n], &action_p2)?;

        npz.finish()?;
    }

    fs::rename(&tmp_path, path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// BundleWriter — streaming accumulator
// ---------------------------------------------------------------------------

/// Accumulates games and flushes bundles to disk when the buffer is full.
pub struct BundleWriter {
    output_dir: PathBuf,
    _width: u8,
    _height: u8,
    max_games_per_bundle: usize,
    buffer: Vec<GameRecord>,
    written_paths: Vec<PathBuf>,
}

impl BundleWriter {
    pub fn new(output_dir: &Path, width: u8, height: u8, max_games_per_bundle: usize) -> Self {
        Self {
            output_dir: output_dir.to_path_buf(),
            _width: width,
            _height: height,
            max_games_per_bundle,
            buffer: Vec::with_capacity(max_games_per_bundle),
            written_paths: Vec::new(),
        }
    }

    /// Add a game. Flushes to disk if buffer reaches capacity.
    pub fn add_game(&mut self, game: GameRecord) -> io::Result<Option<PathBuf>> {
        self.buffer.push(game);
        if self.buffer.len() >= self.max_games_per_bundle {
            return self.flush();
        }
        Ok(None)
    }

    /// Flush buffered games to disk. Returns path if anything was written.
    pub fn flush(&mut self) -> io::Result<Option<PathBuf>> {
        if self.buffer.is_empty() {
            return Ok(None);
        }

        let filename = format!("bundle_{}.npz", uuid::Uuid::new_v4());
        let path = self.output_dir.join(&filename);
        write_bundle(&self.buffer, &path)?;
        self.written_paths.push(path.clone());
        self.buffer.clear();
        Ok(Some(path))
    }

    /// Flush remaining games and return all written paths.
    pub fn finish(mut self) -> io::Result<Vec<PathBuf>> {
        self.flush()?;
        Ok(self.written_paths)
    }

    /// All bundle paths written so far (not including current buffer).
    pub fn paths(&self) -> &[PathBuf] {
        &self.written_paths
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selfplay::{CheeseOutcome, GameOutcome, PositionRecord};

    /// Build a minimal synthetic GameRecord for testing.
    fn synthetic_game(index: u32, num_positions: usize) -> GameRecord {
        let w: u8 = 3;
        let h: u8 = 3;
        let hw = w as usize * h as usize;

        let mut maze = vec![1i8; hw * 4];
        // Block DOWN and LEFT at (0,0)
        maze[0 * 4 + 2] = -1; // (0,0) DOWN
        maze[0 * 4 + 3] = -1; // (0,0) LEFT

        let mut initial_cheese = vec![0u8; hw];
        initial_cheese[1 * 3 + 1] = 1; // cheese at (1,1)

        let positions: Vec<PositionRecord> = (0..num_positions)
            .map(|t| PositionRecord {
                p1_pos: [0, 0],
                p2_pos: [2, 2],
                p1_score: t as f32 * 0.5,
                p2_score: t as f32 * 0.25,
                p1_mud: 0,
                p2_mud: if t == 1 { 2 } else { 0 },
                turn: t as u16,
                cheese_mask: if t == 0 {
                    initial_cheese.clone()
                } else {
                    vec![0u8; hw]
                },
                value_p1: 1.5,
                value_p2: 0.5,
                visit_counts_p1: [10.0, 5.0, 3.0, 2.0, 0.0],
                visit_counts_p2: [0.0, 8.0, 4.0, 4.0, 4.0],
                prior_p1: [0.2, 0.2, 0.2, 0.2, 0.2],
                prior_p2: [0.2, 0.2, 0.2, 0.2, 0.2],
                policy_p1: [0.5, 0.25, 0.15, 0.1, 0.0],
                policy_p2: [0.0, 0.4, 0.2, 0.2, 0.2],
                action_p1: 0,
                action_p2: 1,
            })
            .collect();

        let mut cheese_outcomes = vec![CheeseOutcome::Uncollected as u8; hw];
        cheese_outcomes[1 * 3 + 1] = CheeseOutcome::P1Win as u8;

        GameRecord {
            width: w,
            height: h,
            max_turns: 30,
            maze,
            initial_cheese,
            positions,
            final_p1_score: 1.0,
            final_p2_score: 0.0,
            result: GameOutcome::P1Win,
            total_simulations: 100,
            cheese_available: 1,
            game_index: index,
            cheese_outcomes,
        }
    }

    #[test]
    fn write_bundle_roundtrip() {
        let dir = std::env::temp_dir().join("bundle_roundtrip_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_bundle.npz");

        let games = vec![synthetic_game(0, 3), synthetic_game(1, 2)];
        write_bundle(&games, &path).unwrap();

        // Read back with zip crate and verify array shapes
        let file = std::fs::File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();

        // Should have all expected entries
        let entry_names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();
        assert!(entry_names.contains(&"game_lengths.npy".to_string()));
        assert!(entry_names.contains(&"maze.npy".to_string()));
        assert!(entry_names.contains(&"policy_p1.npy".to_string()));
        assert!(entry_names.contains(&"cheese_mask.npy".to_string()));

        // Verify game_lengths data
        {
            let mut entry = archive.by_name("game_lengths.npy").unwrap();
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut entry, &mut buf).unwrap();

            let header_str = std::str::from_utf8(&buf[10..256]).unwrap();
            assert!(header_str.contains("'descr':'<i4'"));
            assert!(header_str.contains("'shape':(2,)"));

            // Data: 2 * 4 = 8 bytes
            let data = &buf[256..];
            assert_eq!(data.len(), 8);
            let lengths: Vec<i32> = data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(lengths, vec![3, 2]); // 3 positions in game 0, 2 in game 1
        }

        // Verify maze shape: [2, 3, 3, 4]
        {
            let mut entry = archive.by_name("maze.npy").unwrap();
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut entry, &mut buf).unwrap();

            let header_str = std::str::from_utf8(&buf[10..256]).unwrap();
            assert!(header_str.contains("'shape':(2,3,3,4)"));
            assert_eq!(buf.len() - 256, 2 * 3 * 3 * 4); // i8, 1 byte each

            // Check first game's (0,0) has DOWN=-1, LEFT=-1
            let game0_start = 256;
            assert_eq!(buf[game0_start + 2] as i8, -1); // DOWN
            assert_eq!(buf[game0_start + 3] as i8, -1); // LEFT
        }

        // Verify position-level shapes: total 5 positions
        {
            let mut entry = archive.by_name("p1_pos.npy").unwrap();
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut entry, &mut buf).unwrap();

            let header_str = std::str::from_utf8(&buf[10..256]).unwrap();
            assert!(header_str.contains("'shape':(5,2)")); // 5 total positions
        }

        // Verify policy_p1 shape and sample values
        {
            let mut entry = archive.by_name("policy_p1.npy").unwrap();
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut entry, &mut buf).unwrap();

            let header_str = std::str::from_utf8(&buf[10..256]).unwrap();
            assert!(header_str.contains("'shape':(5,5)"));

            // First position's policy
            let data = &buf[256..];
            let first_policy: Vec<f32> = data[..20]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(first_policy, vec![0.5, 0.25, 0.15, 0.1, 0.0]);
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_bundle_validates_dimensions() {
        let dir = std::env::temp_dir().join("bundle_dim_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.npz");

        let mut game1 = synthetic_game(0, 2);
        let game2 = synthetic_game(1, 2);
        game1.width = 5; // mismatch

        let err = write_bundle(&[game1, game2], &path).unwrap_err();
        assert!(err.to_string().contains("dimensions"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_bundle_validates_empty() {
        let dir = std::env::temp_dir().join("bundle_empty_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.npz");

        let err = write_bundle(&[], &path).unwrap_err();
        assert!(err.to_string().contains("no games"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_bundle_validates_no_positions() {
        let dir = std::env::temp_dir().join("bundle_nopos_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("nopos.npz");

        let mut game = synthetic_game(0, 2);
        game.positions.clear();

        let err = write_bundle(&[game], &path).unwrap_err();
        assert!(err.to_string().contains("no positions"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn bundle_writer_flushes_at_capacity() {
        let dir = std::env::temp_dir().join("bundle_writer_cap_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let mut writer = BundleWriter::new(&dir, 3, 3, 2);

        // Add 3 games with max_per_bundle=2, should produce 2 files (2+1)
        let r1 = writer.add_game(synthetic_game(0, 2)).unwrap();
        assert!(r1.is_none()); // not full yet

        let r2 = writer.add_game(synthetic_game(1, 2)).unwrap();
        assert!(r2.is_some()); // flushed!

        let r3 = writer.add_game(synthetic_game(2, 2)).unwrap();
        assert!(r3.is_none()); // not full yet

        let paths = writer.finish().unwrap();
        assert_eq!(paths.len(), 2);

        for p in &paths {
            assert!(p.exists());
            assert!(p.file_name().unwrap().to_str().unwrap().starts_with("bundle_"));
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn bundle_writer_finish_flushes_remainder() {
        let dir = std::env::temp_dir().join("bundle_writer_finish_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let mut writer = BundleWriter::new(&dir, 3, 3, 100);

        writer.add_game(synthetic_game(0, 2)).unwrap();
        writer.add_game(synthetic_game(1, 3)).unwrap();

        let paths = writer.finish().unwrap();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn bundle_writer_finish_empty() {
        let dir = std::env::temp_dir().join("bundle_writer_empty_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let writer = BundleWriter::new(&dir, 3, 3, 100);
        let paths = writer.finish().unwrap();
        assert!(paths.is_empty());

        let _ = fs::remove_dir_all(&dir);
    }
}
