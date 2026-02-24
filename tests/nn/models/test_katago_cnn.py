"""Tests for KataGoCNN model."""

from __future__ import annotations

import torch

from alpharat.nn.architectures.cnn.blocks import ResBlockConfig, TrunkConfig
from alpharat.nn.models.cnn.katago import KataGoCNN
from alpharat.nn.training.keys import ModelOutput


def _make_model(width: int = 5, height: int = 5, channels: int = 32) -> KataGoCNN:
    """Build a small KataGoCNN for testing."""
    trunk_cfg = TrunkConfig(channels=channels, blocks=[ResBlockConfig()])
    stem, blocks = trunk_cfg.build(in_channels=7)
    return KataGoCNN(
        width=width,
        height=height,
        stem=stem,
        blocks=blocks,
        hidden_channels=channels,
        hidden_dim=32,
    )


class TestForwardPass:
    """Forward pass shape tests."""

    def test_output_shapes_5x5(self) -> None:
        model = _make_model(5, 5)
        model.eval()
        obs_dim = 5 * 5 * 7 + 6
        x = torch.randn(4, obs_dim)
        with torch.no_grad():
            out = model(x)
        assert out[ModelOutput.LOGITS_P1].shape == (4, 5)
        assert out[ModelOutput.LOGITS_P2].shape == (4, 5)
        assert out[ModelOutput.VALUE_P1].shape == (4,)
        assert out[ModelOutput.VALUE_P2].shape == (4,)

    def test_output_shapes_7x7(self) -> None:
        model = _make_model(7, 7)
        model.eval()
        obs_dim = 7 * 7 * 7 + 6
        x = torch.randn(2, obs_dim)
        with torch.no_grad():
            out = model(x)
        assert out[ModelOutput.LOGITS_P1].shape == (2, 5)
        assert out[ModelOutput.VALUE_P1].shape == (2,)

    def test_batch_size_one(self) -> None:
        model = _make_model(5, 5)
        model.eval()
        obs_dim = 5 * 5 * 7 + 6
        x = torch.randn(1, obs_dim)
        with torch.no_grad():
            out = model(x)
        assert out[ModelOutput.LOGITS_P1].shape == (1, 5)


class TestPredictPass:
    """Inference pass tests."""

    def test_predict_returns_probabilities(self) -> None:
        model = _make_model(5, 5)
        model.eval()
        obs_dim = 5 * 5 * 7 + 6
        x = torch.randn(2, obs_dim)
        with torch.no_grad():
            out = model.predict(x)
        assert ModelOutput.POLICY_P1 in out
        assert ModelOutput.POLICY_P2 in out
        # Probabilities sum to 1
        assert torch.allclose(out[ModelOutput.POLICY_P1].sum(dim=-1), torch.ones(2), atol=1e-5)


class TestValueProperties:
    """Value head behavior."""

    def test_values_non_negative(self) -> None:
        """Softplus ensures non-negative values."""
        model = _make_model(5, 5)
        model.eval()
        obs_dim = 5 * 5 * 7 + 6
        x = torch.randn(8, obs_dim)
        with torch.no_grad():
            out = model(x)
        assert (out[ModelOutput.VALUE_P1] >= 0).all()
        assert (out[ModelOutput.VALUE_P2] >= 0).all()


class TestOutputKeys:
    """TrainableModel protocol compliance."""

    def test_forward_has_required_keys(self) -> None:
        model = _make_model(5, 5)
        model.eval()
        obs_dim = 5 * 5 * 7 + 6
        x = torch.randn(2, obs_dim)
        with torch.no_grad():
            out = model(x)
        for key in [
            ModelOutput.LOGITS_P1,
            ModelOutput.LOGITS_P2,
            ModelOutput.VALUE_P1,
            ModelOutput.VALUE_P2,
        ]:
            assert key in out

    def test_predict_has_required_keys(self) -> None:
        model = _make_model(5, 5)
        model.eval()
        obs_dim = 5 * 5 * 7 + 6
        x = torch.randn(2, obs_dim)
        with torch.no_grad():
            out = model.predict(x)
        for key in [
            ModelOutput.POLICY_P1,
            ModelOutput.POLICY_P2,
            ModelOutput.VALUE_P1,
            ModelOutput.VALUE_P2,
        ]:
            assert key in out
