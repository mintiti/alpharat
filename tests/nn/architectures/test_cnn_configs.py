"""Tests for CNN architecture Pydantic configs."""

from __future__ import annotations

import torch

from alpharat.nn.architectures.cnn.blocks import (
    GPoolBlockConfig,
    InterleavedBlockConfig,
    ResBlockConfig,
    TrunkConfig,
)
from alpharat.nn.architectures.cnn.config import CNNModelConfig, KataGoCNNModelConfig
from alpharat.nn.architectures.cnn.heads import (
    PooledValueHeadConfig,
)
from alpharat.nn.models.cnn.blocks import GPoolResBlock, ResBlock


class TestBlockConfigSerialization:
    """Config round-trip tests."""

    def test_res_block_round_trip(self) -> None:
        cfg = ResBlockConfig()
        data = cfg.model_dump()
        restored = ResBlockConfig(**data)
        assert restored.type == "res"

    def test_gpool_block_round_trip(self) -> None:
        cfg = GPoolBlockConfig(gpool_channels=16)
        data = cfg.model_dump()
        restored = GPoolBlockConfig(**data)
        assert restored.gpool_channels == 16

    def test_interleaved_round_trip(self) -> None:
        cfg = InterleavedBlockConfig(
            block=ResBlockConfig(),
            inject=GPoolBlockConfig(gpool_channels=16),
            every_n=3,
            count=6,
        )
        data = cfg.model_dump()
        restored = InterleavedBlockConfig(**data)
        assert restored.count == 6
        assert restored.every_n == 3

    def test_trunk_config_round_trip(self) -> None:
        cfg = TrunkConfig(channels=32, blocks=[ResBlockConfig(), GPoolBlockConfig()])
        data = cfg.model_dump()
        restored = TrunkConfig(**data)
        assert restored.channels == 32
        assert len(restored.blocks) == 2

    def test_cnn_model_config_round_trip(self) -> None:
        cfg = CNNModelConfig(
            width=5,
            height=5,
            trunk=TrunkConfig(channels=32),
            value_head=PooledValueHeadConfig(),
        )
        data = cfg.model_dump()
        restored = CNNModelConfig(**data)
        assert restored.trunk.channels == 32
        assert restored.value_head.type == "pooled"


class TestTrunkConfigBuild:
    """Tests for TrunkConfig.build()."""

    def test_stem_in_channels(self) -> None:
        cfg = TrunkConfig(channels=32)
        stem, _ = cfg.build(in_channels=5)
        assert stem.in_channels == 5

    def test_stem_in_channels_7(self) -> None:
        cfg = TrunkConfig(channels=32)
        stem, _ = cfg.build(in_channels=7)
        assert stem.in_channels == 7

    def test_builds_correct_number_of_blocks(self) -> None:
        cfg = TrunkConfig(channels=32, blocks=[ResBlockConfig(), ResBlockConfig()])
        _, blocks = cfg.build(in_channels=5)
        assert len(blocks) == 2

    def test_default_single_res_block(self) -> None:
        cfg = TrunkConfig()
        _, blocks = cfg.build(in_channels=5)
        assert len(blocks) == 1
        assert isinstance(blocks[0], ResBlock)


class TestInterleavedBlockConfig:
    """Tests for InterleavedBlockConfig composite pattern."""

    def test_interleaved_pattern(self) -> None:
        """count=6, every_n=3: [block, block, inject, block, block, inject]."""
        cfg = InterleavedBlockConfig(
            block=ResBlockConfig(),
            inject=GPoolBlockConfig(gpool_channels=16),
            every_n=3,
            count=6,
        )
        modules = cfg.build(channels=32)
        assert len(modules) == 6
        # Positions 0, 1, 3, 4 should be ResBlock
        assert isinstance(modules[0], ResBlock)
        assert isinstance(modules[1], ResBlock)
        assert isinstance(modules[3], ResBlock)
        assert isinstance(modules[4], ResBlock)
        # Positions 2, 5 should be GPoolResBlock
        assert isinstance(modules[2], GPoolResBlock)
        assert isinstance(modules[5], GPoolResBlock)

    def test_interleaved_every_2(self) -> None:
        """count=4, every_n=2: [block, inject, block, inject]."""
        cfg = InterleavedBlockConfig(
            block=ResBlockConfig(),
            inject=GPoolBlockConfig(),
            every_n=2,
            count=4,
        )
        modules = cfg.build(channels=32)
        assert len(modules) == 4
        assert isinstance(modules[0], ResBlock)
        assert isinstance(modules[1], GPoolResBlock)
        assert isinstance(modules[2], ResBlock)
        assert isinstance(modules[3], GPoolResBlock)


class TestEndToEnd:
    """End-to-end: config -> model -> forward -> correct shapes."""

    def test_default_config_forward(self) -> None:
        cfg = CNNModelConfig(
            width=5,
            height=5,
            trunk=TrunkConfig(channels=32, blocks=[ResBlockConfig()]),
            player_dim=16,
            hidden_dim=32,
        )
        model: torch.nn.Module = cfg.build_model()  # type: ignore[assignment]
        model.eval()

        obs_dim = 5 * 5 * 7 + 6
        x = torch.randn(2, obs_dim)
        with torch.no_grad():
            out = model(x)

        assert out["logits_p1"].shape == (2, 5)
        assert out["logits_p2"].shape == (2, 5)
        assert out["pred_value_p1"].shape == (2,)
        assert out["pred_value_p2"].shape == (2,)

    def test_gpool_trunk_forward(self) -> None:
        cfg = CNNModelConfig(
            width=5,
            height=5,
            trunk=TrunkConfig(channels=32, blocks=[GPoolBlockConfig(gpool_channels=16)]),
            player_dim=16,
            hidden_dim=32,
        )
        model: torch.nn.Module = cfg.build_model()  # type: ignore[assignment]
        model.eval()

        obs_dim = 5 * 5 * 7 + 6
        x = torch.randn(2, obs_dim)
        with torch.no_grad():
            out = model(x)

        assert out["logits_p1"].shape == (2, 5)
        assert out["pred_value_p1"].shape == (2,)

    def test_pooled_value_head_forward(self) -> None:
        cfg = CNNModelConfig(
            width=5,
            height=5,
            trunk=TrunkConfig(channels=32, blocks=[ResBlockConfig()]),
            value_head=PooledValueHeadConfig(),
            player_dim=16,
            hidden_dim=32,
        )
        model: torch.nn.Module = cfg.build_model()  # type: ignore[assignment]
        model.eval()

        obs_dim = 5 * 5 * 7 + 6
        x = torch.randn(2, obs_dim)
        with torch.no_grad():
            out = model(x)

        assert out["pred_value_p1"].shape == (2,)
        assert out["pred_value_p2"].shape == (2,)
        # Softplus ensures non-negative
        assert (out["pred_value_p1"] >= 0).all()

    def test_cnn_no_augmentation(self) -> None:
        """DeepSet CNN should not need augmentation (structural symmetry)."""
        cfg = CNNModelConfig(width=5, height=5)
        aug = cfg.build_augmentation()
        assert not aug.needs_augmentation

    def test_7x7_forward(self) -> None:
        cfg = CNNModelConfig(
            width=7,
            height=7,
            trunk=TrunkConfig(channels=32, blocks=[ResBlockConfig()]),
            player_dim=16,
            hidden_dim=32,
        )
        model: torch.nn.Module = cfg.build_model()  # type: ignore[assignment]
        model.eval()

        obs_dim = 7 * 7 * 7 + 6
        x = torch.randn(2, obs_dim)
        with torch.no_grad():
            out = model(x)

        assert out["logits_p1"].shape == (2, 5)
        assert out["pred_value_p1"].shape == (2,)


class TestKataGoCNNConfig:
    """Tests for KataGoCNNModelConfig."""

    def test_round_trip(self) -> None:
        cfg = KataGoCNNModelConfig(
            width=5,
            height=5,
            trunk=TrunkConfig(channels=32),
            hidden_dim=64,
        )
        data = cfg.model_dump()
        restored = KataGoCNNModelConfig(**data)
        assert restored.architecture == "cnn_katago"
        assert restored.trunk.channels == 32

    def test_forward(self) -> None:
        cfg = KataGoCNNModelConfig(
            width=5,
            height=5,
            trunk=TrunkConfig(channels=32, blocks=[ResBlockConfig()]),
            hidden_dim=32,
        )
        model: torch.nn.Module = cfg.build_model()  # type: ignore[assignment]
        model.eval()

        obs_dim = 5 * 5 * 7 + 6
        x = torch.randn(2, obs_dim)
        with torch.no_grad():
            out = model(x)

        assert out["logits_p1"].shape == (2, 5)
        assert out["logits_p2"].shape == (2, 5)
        assert out["pred_value_p1"].shape == (2,)
        assert out["pred_value_p2"].shape == (2,)

    def test_needs_augmentation(self) -> None:
        """KataGoCNN should need augmentation (no structural symmetry)."""
        cfg = KataGoCNNModelConfig(width=5, height=5)
        aug = cfg.build_augmentation()
        assert aug.needs_augmentation
