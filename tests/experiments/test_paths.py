"""Tests for experiment path utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from alpharat.experiments import paths


class TestParseBatchId:
    """Tests for parse_batch_id()."""

    def test_valid_format(self) -> None:
        """Valid batch_id returns (group, uuid) tuple."""
        group, uuid = paths.parse_batch_id("my_group/abc123")
        assert group == "my_group"
        assert uuid == "abc123"

    def test_no_slash_raises(self) -> None:
        """Batch ID without slash raises ValueError."""
        with pytest.raises(ValueError, match="Invalid batch_id format"):
            paths.parse_batch_id("no_slash_here")

    def test_multiple_slashes_raises(self) -> None:
        """Batch ID with multiple slashes raises ValueError."""
        with pytest.raises(ValueError, match="Invalid batch_id format"):
            paths.parse_batch_id("group/sub/uuid")

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid batch_id format"):
            paths.parse_batch_id("")


class TestParseShardId:
    """Tests for parse_shard_id()."""

    def test_valid_format(self) -> None:
        """Valid shard_id returns (group, uuid) tuple."""
        group, uuid = paths.parse_shard_id("5x5_uniform/def456")
        assert group == "5x5_uniform"
        assert uuid == "def456"

    def test_no_slash_raises(self) -> None:
        """Shard ID without slash raises ValueError."""
        with pytest.raises(ValueError, match="Invalid shard_id format"):
            paths.parse_shard_id("no_slash_here")

    def test_multiple_slashes_raises(self) -> None:
        """Shard ID with multiple slashes raises ValueError."""
        with pytest.raises(ValueError, match="Invalid shard_id format"):
            paths.parse_shard_id("group/sub/uuid")

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid shard_id format"):
            paths.parse_shard_id("")


class TestBatchIdFromPath:
    """Tests for batch_id_from_path()."""

    def test_extracts_group_and_uuid(self) -> None:
        """Extracts group/uuid from batch path."""
        batch_path = Path("/experiments/batches/my_group/abc123")
        batch_id = paths.batch_id_from_path(batch_path)
        assert batch_id == "my_group/abc123"


class TestShardIdFromPath:
    """Tests for shard_id_from_path()."""

    def test_extracts_group_and_uuid(self) -> None:
        """Extracts group/uuid from shard path."""
        shard_path = Path("/experiments/shards/5x5_uniform/def456")
        shard_id = paths.shard_id_from_path(shard_path)
        assert shard_id == "5x5_uniform/def456"
