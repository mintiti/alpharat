"""Tests for checkpoint loading with ModelConfig.build_model()."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from alpharat.config.checkpoint import load_model_from_checkpoint

if TYPE_CHECKING:
    from pathlib import Path


class TestLoadModelFromCheckpoint:
    """Tests for load_model_from_checkpoint function."""

    @pytest.fixture
    def mlp_checkpoint(self, tmp_path: Path) -> Path:
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
    def symmetric_checkpoint(self, tmp_path: Path) -> Path:
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

    def test_loads_mlp_model(self, mlp_checkpoint: Path) -> None:
        """load_model_from_checkpoint loads MLP model."""
        model, builder, width, height = load_model_from_checkpoint(
            mlp_checkpoint, device="cpu", compile_model=False
        )

        assert width == 5
        assert height == 5
        assert model is not None
        # Check it's a PyRatMLP by checking attribute
        assert hasattr(model, "predict")

    def test_loads_symmetric_model(self, symmetric_checkpoint: Path) -> None:
        """load_model_from_checkpoint loads Symmetric model."""
        model, builder, width, height = load_model_from_checkpoint(
            symmetric_checkpoint, device="cpu", compile_model=False
        )

        assert width == 5
        assert height == 5
        assert model is not None

    def test_returns_observation_builder(self, mlp_checkpoint: Path) -> None:
        """load_model_from_checkpoint returns observation builder."""
        model, builder, width, height = load_model_from_checkpoint(
            mlp_checkpoint, device="cpu", compile_model=False
        )

        assert builder is not None
        # Builder should have a build method
        assert hasattr(builder, "build")

    def test_model_in_eval_mode(self, mlp_checkpoint: Path) -> None:
        """Loaded model is in eval mode."""
        model, _, _, _ = load_model_from_checkpoint(
            mlp_checkpoint, device="cpu", compile_model=False
        )

        assert not model.training

    def test_raises_on_missing_dimensions(self, tmp_path: Path) -> None:
        """load_model_from_checkpoint raises if dimensions missing."""
        checkpoint = {
            "model_state_dict": {},
            # Missing width and height
            "config": {"model": {"architecture": "mlp"}},
        }
        checkpoint_path = tmp_path / "bad_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        with pytest.raises(ValueError, match="missing width/height"):
            load_model_from_checkpoint(checkpoint_path)

    def test_raises_on_missing_config(self, tmp_path: Path) -> None:
        """load_model_from_checkpoint raises if config.model missing."""
        checkpoint = {
            "model_state_dict": {},
            "width": 5,
            "height": 5,
            "config": {},  # Missing 'model' key
        }
        checkpoint_path = tmp_path / "bad_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        with pytest.raises(ValueError, match="missing config.model"):
            load_model_from_checkpoint(checkpoint_path)

    def test_raises_on_missing_architecture(self, tmp_path: Path) -> None:
        """load_model_from_checkpoint raises if architecture missing."""
        checkpoint = {
            "model_state_dict": {},
            "width": 5,
            "height": 5,
            "config": {
                "model": {
                    "hidden_dim": 128,
                    # Missing 'architecture' field
                },
            },
        }
        checkpoint_path = tmp_path / "bad_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        with pytest.raises(ValueError, match="missing config.model.architecture"):
            load_model_from_checkpoint(checkpoint_path)


class TestLoadModelInference:
    """Tests that loaded models can run inference."""

    @pytest.fixture
    def mlp_checkpoint(self, tmp_path: Path) -> Path:
        """Create MLP checkpoint for inference tests."""
        from alpharat.nn.models.mlp import PyRatMLP

        width, height = 5, 5
        obs_dim = width * height * 7 + 6

        model = PyRatMLP(obs_dim=obs_dim, hidden_dim=64, dropout=0.0)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "width": width,
            "height": height,
            "config": {
                "model": {
                    "architecture": "mlp",
                    "hidden_dim": 64,
                    "dropout": 0.0,
                    "p_augment": 0.5,
                },
            },
        }

        checkpoint_path = tmp_path / "inference_model.pt"
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def test_model_can_forward(self, mlp_checkpoint: Path) -> None:
        """Loaded model can run forward pass."""
        import numpy as np

        model, builder, width, height = load_model_from_checkpoint(
            mlp_checkpoint, device="cpu", compile_model=False
        )

        # Create dummy observation
        obs_dim = width * height * 7 + 6
        obs = np.random.randn(obs_dim).astype(np.float32)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)

        with torch.inference_mode():
            output = model.forward(obs_tensor)

        # Forward returns logits (see ModelOutput enum)
        assert "logits_p1" in output
        assert "logits_p2" in output
        assert "payout" in output

    def test_model_can_predict(self, mlp_checkpoint: Path) -> None:
        """Loaded model can run predict (with softmax)."""
        import numpy as np

        model, builder, width, height = load_model_from_checkpoint(
            mlp_checkpoint, device="cpu", compile_model=False
        )

        obs_dim = width * height * 7 + 6
        obs = np.random.randn(obs_dim).astype(np.float32)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)

        with torch.inference_mode():
            output = model.predict(obs_tensor)  # type: ignore[operator]

        # Predict returns probabilities, not logits
        assert "policy_p1" in output
        assert "policy_p2" in output

        # Policies should be valid distributions
        p1 = output["policy_p1"].numpy()
        assert np.allclose(p1.sum(), 1.0, rtol=1e-5)
        assert np.all(p1 >= 0)

    def test_builder_works_with_model(self, mlp_checkpoint: Path) -> None:
        """Builder output is compatible with model."""
        from alpharat.config.game import GameConfig
        from alpharat.data.maze import build_maze_array
        from alpharat.nn.extraction import from_pyrat_game

        model, builder, width, height = load_model_from_checkpoint(
            mlp_checkpoint, device="cpu", compile_model=False
        )

        # Create a real game
        game_config = GameConfig(
            width=width,
            height=height,
            max_turns=30,
            cheese_count=5,
            wall_density=0.0,
            mud_density=0.0,
        )
        game = game_config.build(seed=42)
        maze = build_maze_array(game, width, height)

        # Build observation
        obs_input = from_pyrat_game(game, maze, game.max_turns)
        obs = builder.build(obs_input)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)

        # Run inference
        with torch.inference_mode():
            output = model.predict(obs_tensor)  # type: ignore[operator]

        assert output["policy_p1"].shape == (1, 5)
        assert output["policy_p2"].shape == (1, 5)
