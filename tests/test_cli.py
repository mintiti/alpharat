"""Smoke tests for CLI entry points."""

import subprocess

import pytest

COMMANDS = [
    "alpharat-sample",
    "alpharat-train",
    "alpharat-manifest",
    "alpharat-benchmark",
    "alpharat-prepare-shards",
    "alpharat-train-and-benchmark",
]


@pytest.mark.parametrize("cmd", COMMANDS)
def test_cli_help(cmd: str) -> None:
    """CLI entry points are installed and respond to --help."""
    result = subprocess.run([cmd, "--help"], capture_output=True)
    assert result.returncode == 0, f"{cmd} --help failed: {result.stderr.decode()}"
