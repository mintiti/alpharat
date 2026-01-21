"""Shared fixtures for config tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def mlp_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal MLP checkpoint for testing."""
    from alpharat.nn.models.mlp import PyRatMLP

    width, height = 5, 5
    obs_dim = width * height * 7 + 6

    model = PyRatMLP(obs_dim=obs_dim, hidden_dim=128, dropout=0.0)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "width": width,
        "height": height,
        "config": {
            "model": {
                "architecture": "mlp",
                "hidden_dim": 128,
                "dropout": 0.0,
                "p_augment": 0.5,
            },
            "optim": {"lr": 0.001},
        },
    }

    checkpoint_path = tmp_path / "mlp_model.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def symmetric_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal Symmetric checkpoint for testing."""
    from alpharat.nn.models.symmetric import SymmetricMLP

    width, height = 5, 5

    model = SymmetricMLP(width=width, height=height, hidden_dim=128, dropout=0.0)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "width": width,
        "height": height,
        "config": {
            "model": {
                "architecture": "symmetric",
                "hidden_dim": 128,
                "dropout": 0.0,
            },
            "optim": {"lr": 0.001},
        },
    }

    checkpoint_path = tmp_path / "symmetric_model.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def local_value_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal LocalValue checkpoint for testing."""
    from alpharat.nn.models.local_value import LocalValueMLP

    width, height = 5, 5
    obs_dim = width * height * 7 + 6

    model = LocalValueMLP(obs_dim=obs_dim, width=width, height=height, hidden_dim=128, dropout=0.0)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "width": width,
        "height": height,
        "config": {
            "model": {
                "architecture": "local_value",
                "hidden_dim": 128,
                "dropout": 0.0,
                "p_augment": 0.5,
            },
            "optim": {"lr": 0.001},
        },
    }

    checkpoint_path = tmp_path / "local_value_model.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path
