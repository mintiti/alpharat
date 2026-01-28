"""Tests for architecture dependency inversion compliance.

Verifies that all architectures properly implement the protocols and
that their components are self-consistent.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from alpharat.nn.architectures.local_value import LocalValueModelConfig, LocalValueOptimConfig
from alpharat.nn.architectures.mlp import MLPModelConfig, MLPOptimConfig
from alpharat.nn.architectures.symmetric import SymmetricModelConfig, SymmetricOptimConfig
from alpharat.nn.training.keys import ArchitectureType, LossKey, ModelOutput
from alpharat.nn.training.protocols import (
    TrainableModel,
)

# --- Test fixtures for each architecture ---


def _make_mlp_config() -> MLPModelConfig:
    """Create MLPModelConfig with dimensions set."""
    config = MLPModelConfig(obs_dim=181, hidden_dim=64)  # 5x5 maze
    return config


def _make_symmetric_config() -> SymmetricModelConfig:
    """Create SymmetricModelConfig with dimensions set."""
    config = SymmetricModelConfig(width=5, height=5, hidden_dim=64)
    return config


def _make_local_value_config() -> LocalValueModelConfig:
    """Create LocalValueModelConfig with dimensions set."""
    config = LocalValueModelConfig(obs_dim=181, width=5, height=5, hidden_dim=64)
    return config


def _make_mlp_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    """Create a valid batch for MLP training."""
    obs_dim = 181  # 5x5 maze
    return {
        "observation": torch.randn(batch_size, obs_dim),
        "policy_p1": torch.softmax(torch.randn(batch_size, 5), dim=-1),
        "policy_p2": torch.softmax(torch.randn(batch_size, 5), dim=-1),
        "action_p1": torch.randint(0, 5, (batch_size, 1)),
        "action_p2": torch.randint(0, 5, (batch_size, 1)),
        "p1_value": torch.rand(batch_size) * 5,
        "p2_value": torch.rand(batch_size) * 5,
        "payout_matrix": torch.rand(batch_size, 2, 5, 5) * 5,
    }


def _make_local_value_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    """Create a valid batch for LocalValue training (includes cheese_outcomes)."""
    batch = _make_mlp_batch(batch_size)
    # cheese_outcomes: -1 for inactive, 0-3 for outcome classes
    cheese_outcomes = torch.randint(-1, 4, (batch_size, 5, 5))
    batch["cheese_outcomes"] = cheese_outcomes
    return batch


# --- Parametrized test data ---


ARCHITECTURE_CONFIGS = [
    pytest.param(
        ArchitectureType.MLP,
        _make_mlp_config,
        MLPOptimConfig,
        _make_mlp_batch,
        id="mlp",
    ),
    pytest.param(
        ArchitectureType.SYMMETRIC,
        _make_symmetric_config,
        SymmetricOptimConfig,
        _make_mlp_batch,  # Same batch format as MLP
        id="symmetric",
    ),
    pytest.param(
        ArchitectureType.LOCAL_VALUE,
        _make_local_value_config,
        LocalValueOptimConfig,
        _make_local_value_batch,  # Needs cheese_outcomes
        id="local_value",
    ),
]


class TestBuildModel:
    """Tests for build_model() returning TrainableModel."""

    @pytest.mark.parametrize(
        "arch_type,config_factory,optim_cls,batch_factory", ARCHITECTURE_CONFIGS
    )
    def test_build_model_returns_trainable_model(
        self, arch_type: Any, config_factory: Any, optim_cls: Any, batch_factory: Any
    ) -> None:
        """build_model() should return a TrainableModel instance."""
        config = config_factory()
        model = config.build_model()

        # Runtime check via isinstance (Protocol is @runtime_checkable)
        assert isinstance(model, TrainableModel), (
            f"{arch_type}: build_model() did not return TrainableModel"
        )

    @pytest.mark.parametrize(
        "arch_type,config_factory,optim_cls,batch_factory", ARCHITECTURE_CONFIGS
    )
    def test_model_has_forward_method(
        self, arch_type: Any, config_factory: Any, optim_cls: Any, batch_factory: Any
    ) -> None:
        """Model should have forward() method."""
        config = config_factory()
        model = config.build_model()

        assert hasattr(model, "forward"), f"{arch_type}: model missing forward()"
        assert callable(model.forward), f"{arch_type}: forward is not callable"

    @pytest.mark.parametrize(
        "arch_type,config_factory,optim_cls,batch_factory", ARCHITECTURE_CONFIGS
    )
    def test_model_has_predict_method(
        self, arch_type: Any, config_factory: Any, optim_cls: Any, batch_factory: Any
    ) -> None:
        """Model should have predict() method."""
        config = config_factory()
        model = config.build_model()

        assert hasattr(model, "predict"), f"{arch_type}: model missing predict()"
        assert callable(model.predict), f"{arch_type}: predict is not callable"


class TestBuildLossFn:
    """Tests for build_loss_fn() returning LossFunction."""

    @pytest.mark.parametrize(
        "arch_type,config_factory,optim_cls,batch_factory", ARCHITECTURE_CONFIGS
    )
    def test_build_loss_fn_returns_callable(
        self, arch_type: Any, config_factory: Any, optim_cls: Any, batch_factory: Any
    ) -> None:
        """build_loss_fn() should return a callable."""
        config = config_factory()
        loss_fn = config.build_loss_fn()

        assert callable(loss_fn), f"{arch_type}: build_loss_fn() did not return callable"

    @pytest.mark.parametrize(
        "arch_type,config_factory,optim_cls,batch_factory", ARCHITECTURE_CONFIGS
    )
    def test_loss_fn_returns_dict_with_total(
        self, arch_type: Any, config_factory: Any, optim_cls: Any, batch_factory: Any
    ) -> None:
        """Loss function should return dict with LossKey.TOTAL."""
        config = config_factory()
        model = config.build_model()
        model.eval()
        loss_fn = config.build_loss_fn()
        optim_config = optim_cls()
        batch = batch_factory()

        with torch.no_grad():
            model_output = model(batch["observation"])

        losses = loss_fn(model_output, batch, optim_config)

        assert isinstance(losses, dict), f"{arch_type}: loss_fn did not return dict"
        assert LossKey.TOTAL in losses, f"{arch_type}: loss_fn missing LossKey.TOTAL"

    @pytest.mark.parametrize(
        "arch_type,config_factory,optim_cls,batch_factory", ARCHITECTURE_CONFIGS
    )
    def test_loss_fn_total_is_scalar(
        self, arch_type: Any, config_factory: Any, optim_cls: Any, batch_factory: Any
    ) -> None:
        """LossKey.TOTAL should be a scalar tensor."""
        config = config_factory()
        model = config.build_model()
        model.eval()
        loss_fn = config.build_loss_fn()
        optim_config = optim_cls()
        batch = batch_factory()

        with torch.no_grad():
            model_output = model(batch["observation"])

        losses = loss_fn(model_output, batch, optim_config)
        total = losses[LossKey.TOTAL]

        assert total.dim() == 0, f"{arch_type}: LossKey.TOTAL is not scalar (dim={total.dim()})"


class TestBuildAugmentation:
    """Tests for build_augmentation() returning AugmentationStrategy."""

    @pytest.mark.parametrize(
        "arch_type,config_factory,optim_cls,batch_factory", ARCHITECTURE_CONFIGS
    )
    def test_build_augmentation_returns_callable(
        self, arch_type: Any, config_factory: Any, optim_cls: Any, batch_factory: Any
    ) -> None:
        """build_augmentation() should return a callable."""
        config = config_factory()
        aug = config.build_augmentation()

        assert callable(aug), f"{arch_type}: build_augmentation() did not return callable"

    @pytest.mark.parametrize(
        "arch_type,config_factory,optim_cls,batch_factory", ARCHITECTURE_CONFIGS
    )
    def test_augmentation_has_needs_augmentation_property(
        self, arch_type: Any, config_factory: Any, optim_cls: Any, batch_factory: Any
    ) -> None:
        """Augmentation should have needs_augmentation property."""
        config = config_factory()
        aug = config.build_augmentation()

        assert hasattr(aug, "needs_augmentation"), (
            f"{arch_type}: augmentation missing needs_augmentation"
        )
        assert isinstance(aug.needs_augmentation, bool), (
            f"{arch_type}: needs_augmentation is not bool"
        )

    @pytest.mark.parametrize(
        "arch_type,config_factory,optim_cls,batch_factory", ARCHITECTURE_CONFIGS
    )
    def test_augmentation_returns_dict(
        self, arch_type: Any, config_factory: Any, optim_cls: Any, batch_factory: Any
    ) -> None:
        """Augmentation should return a dict."""
        config = config_factory()
        aug = config.build_augmentation()
        batch = batch_factory()

        result = aug(batch, width=5, height=5)

        assert isinstance(result, dict), f"{arch_type}: augmentation did not return dict"


class TestSelfConsistency:
    """Tests that model outputs are compatible with loss functions."""

    @pytest.mark.parametrize(
        "arch_type,config_factory,optim_cls,batch_factory", ARCHITECTURE_CONFIGS
    )
    def test_model_output_has_required_keys_for_loss(
        self, arch_type: Any, config_factory: Any, optim_cls: Any, batch_factory: Any
    ) -> None:
        """Model forward() output should have keys required by loss function."""
        config = config_factory()
        model = config.build_model()
        model.eval()
        batch = batch_factory()

        with torch.no_grad():
            output = model(batch["observation"])

        # All models must have these core keys
        required_keys = [
            ModelOutput.LOGITS_P1,
            ModelOutput.LOGITS_P2,
            ModelOutput.VALUE_P1,
            ModelOutput.VALUE_P2,
        ]
        for key in required_keys:
            assert key in output, f"{arch_type}: model output missing {key}"

    @pytest.mark.parametrize(
        "arch_type,config_factory,optim_cls,batch_factory", ARCHITECTURE_CONFIGS
    )
    def test_model_predict_has_required_keys(
        self, arch_type: Any, config_factory: Any, optim_cls: Any, batch_factory: Any
    ) -> None:
        """Model predict() output should have keys required for inference."""
        config = config_factory()
        model = config.build_model()
        model.eval()
        batch = batch_factory()

        with torch.no_grad():
            output = model.predict(batch["observation"])

        # All models must have these core keys
        required_keys = [
            ModelOutput.POLICY_P1,
            ModelOutput.POLICY_P2,
            ModelOutput.VALUE_P1,
            ModelOutput.VALUE_P2,
        ]
        for key in required_keys:
            assert key in output, f"{arch_type}: model predict output missing {key}"

    @pytest.mark.parametrize(
        "arch_type,config_factory,optim_cls,batch_factory", ARCHITECTURE_CONFIGS
    )
    def test_loss_computes_without_error(
        self, arch_type: Any, config_factory: Any, optim_cls: Any, batch_factory: Any
    ) -> None:
        """Loss function should compute without errors on model output."""
        config = config_factory()
        model = config.build_model()
        loss_fn = config.build_loss_fn()
        optim_config = optim_cls()
        batch = batch_factory()

        # Training mode forward
        model.train()
        output = model(batch["observation"])
        losses = loss_fn(output, batch, optim_config)

        # Should have computed successfully
        assert LossKey.TOTAL in losses
        assert not torch.isnan(losses[LossKey.TOTAL]), f"{arch_type}: loss is NaN"
        assert not torch.isinf(losses[LossKey.TOTAL]), f"{arch_type}: loss is Inf"

    @pytest.mark.parametrize(
        "arch_type,config_factory,optim_cls,batch_factory", ARCHITECTURE_CONFIGS
    )
    def test_loss_has_gradients(
        self, arch_type: Any, config_factory: Any, optim_cls: Any, batch_factory: Any
    ) -> None:
        """Loss should be differentiable (gradients flow)."""
        config = config_factory()
        model = config.build_model()
        loss_fn = config.build_loss_fn()
        optim_config = optim_cls()
        batch = batch_factory()

        model.train()
        output = model(batch["observation"])
        losses = loss_fn(output, batch, optim_config)

        # Backprop
        losses[LossKey.TOTAL].backward()

        # At least some parameters should have gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, f"{arch_type}: no gradients after backward()"


class TestArchitectureSpecificBehavior:
    """Tests for architecture-specific properties."""

    def test_symmetric_no_augmentation_needed(self) -> None:
        """SymmetricMLP should not need augmentation (structural symmetry)."""
        config = _make_symmetric_config()
        aug = config.build_augmentation()

        assert not aug.needs_augmentation, "SymmetricMLP should not need augmentation"

    def test_mlp_needs_augmentation(self) -> None:
        """MLP should need augmentation (player swap for symmetry)."""
        config = _make_mlp_config()
        aug = config.build_augmentation()

        assert aug.needs_augmentation, "MLP should need augmentation"

    def test_local_value_needs_augmentation(self) -> None:
        """LocalValueMLP should need augmentation (player swap for symmetry)."""
        config = _make_local_value_config()
        aug = config.build_augmentation()

        assert aug.needs_augmentation, "LocalValueMLP should need augmentation"

    def test_local_value_model_has_ownership_output(self) -> None:
        """LocalValueMLP should output ownership logits."""
        config = _make_local_value_config()
        model: torch.nn.Module = config.build_model()  # type: ignore[assignment]
        model.eval()

        batch = _make_local_value_batch()
        with torch.no_grad():
            output = model(batch["observation"])

        assert ModelOutput.OWNERSHIP_LOGITS in output, "LocalValueMLP missing ownership_logits"
        assert ModelOutput.OWNERSHIP_VALUE in output, "LocalValueMLP missing ownership_value"

    def test_local_value_loss_has_ownership_component(self) -> None:
        """LocalValueMLP loss should include ownership loss."""
        config = _make_local_value_config()
        model: torch.nn.Module = config.build_model()  # type: ignore[assignment]
        loss_fn = config.build_loss_fn()
        optim_config = LocalValueOptimConfig()
        batch = _make_local_value_batch()

        model.train()
        output = model(batch["observation"])
        losses = loss_fn(output, batch, optim_config)

        assert LossKey.OWNERSHIP in losses, "LocalValueMLP loss missing ownership component"


class TestBuildObservationBuilder:
    """Tests for build_observation_builder() returning ObservationBuilder."""

    @pytest.mark.parametrize(
        "arch_type,config_factory,optim_cls,batch_factory", ARCHITECTURE_CONFIGS
    )
    def test_build_observation_builder_returns_builder(
        self, arch_type: Any, config_factory: Any, optim_cls: Any, batch_factory: Any
    ) -> None:
        """build_observation_builder() should return an ObservationBuilder."""

        config = config_factory()
        builder = config.build_observation_builder(width=5, height=5)

        # Check protocol compliance
        assert hasattr(builder, "version"), f"{arch_type}: builder missing version property"
        assert hasattr(builder, "build"), f"{arch_type}: builder missing build method"
        assert callable(builder.build), f"{arch_type}: build is not callable"

    @pytest.mark.parametrize(
        "arch_type,config_factory,optim_cls,batch_factory", ARCHITECTURE_CONFIGS
    )
    def test_builder_has_correct_dimensions(
        self, arch_type: Any, config_factory: Any, optim_cls: Any, batch_factory: Any
    ) -> None:
        """Builder should be configured with the given dimensions."""
        config = config_factory()
        builder = config.build_observation_builder(width=7, height=9)

        # FlatObservationBuilder stores width/height
        assert builder.width == 7, f"{arch_type}: builder has wrong width"
        assert builder.height == 9, f"{arch_type}: builder has wrong height"
