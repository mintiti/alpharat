"""Tests for checkpoint loading with ModelConfig.build_model()."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from alpharat.config.checkpoint import load_model_from_checkpoint

if TYPE_CHECKING:
    from pathlib import Path


class TestLoadModelFromCheckpoint:
    """Tests for load_model_from_checkpoint function.

    Fixtures (mlp_checkpoint, symmetric_checkpoint, local_value_checkpoint)
    are defined in tests/config/conftest.py.
    """

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

    def test_loads_local_value_model(self, local_value_checkpoint: Path) -> None:
        """load_model_from_checkpoint loads LocalValue model."""
        model, builder, width, height = load_model_from_checkpoint(
            local_value_checkpoint, device="cpu", compile_model=False
        )

        assert width == 5
        assert height == 5
        assert model is not None
        # LocalValueMLP should have ownership head
        assert hasattr(model, "ownership_head")

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

    def test_raises_on_unknown_architecture(self, tmp_path: Path) -> None:
        """load_model_from_checkpoint raises on unknown architecture."""
        from pydantic import ValidationError

        checkpoint = {
            "model_state_dict": {},
            "width": 5,
            "height": 5,
            "config": {
                "model": {
                    "architecture": "nonexistent_arch",
                    "hidden_dim": 128,
                },
            },
        }
        checkpoint_path = tmp_path / "bad_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Pydantic discriminated union raises ValidationError for unknown discriminator
        with pytest.raises(ValidationError) as exc_info:
            load_model_from_checkpoint(checkpoint_path)
        assert "nonexistent_arch" in str(exc_info.value)

    def test_raises_on_state_dict_mismatch(self, tmp_path: Path) -> None:
        """load_model_from_checkpoint raises when state dict doesn't match config."""
        from alpharat.nn.models.mlp import PyRatMLP

        width, height = 5, 5
        obs_dim = width * height * 7 + 6

        # Create model with hidden_dim=64
        model = PyRatMLP(obs_dim=obs_dim, hidden_dim=64, dropout=0.0)

        # But config says hidden_dim=256 (mismatch!)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "width": width,
            "height": height,
            "config": {
                "model": {
                    "architecture": "mlp",
                    "hidden_dim": 256,  # Different from actual model
                    "dropout": 0.0,
                    "p_augment": 0.5,
                },
            },
        }
        checkpoint_path = tmp_path / "mismatched_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # PyTorch raises RuntimeError on state dict size mismatch
        with pytest.raises(RuntimeError, match="size mismatch"):
            load_model_from_checkpoint(checkpoint_path)


class TestLoadModelInference:
    """Tests that loaded models can run inference.

    Uses mlp_checkpoint fixture from tests/config/conftest.py.
    """

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
        assert "pred_value_p1" in output
        assert "pred_value_p2" in output

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
        from alpharat.config.game import CheeseConfig, GameConfig
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
            cheese=CheeseConfig(count=5),
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
