"""Block and trunk configuration for CNN architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Discriminator, Field

from alpharat.config.base import StrictBaseModel

if TYPE_CHECKING:
    import torch.nn as nn


class ResBlockConfig(StrictBaseModel):
    """Config for a standard pre-activation residual block."""

    type: Literal["res"] = "res"

    def build(self, channels: int) -> list[nn.Module]:
        from alpharat.nn.models.cnn.blocks import ResBlock

        return [ResBlock(channels)]


class GPoolBlockConfig(StrictBaseModel):
    """Config for a residual block with global pooling branch."""

    type: Literal["gpool"] = "gpool"
    gpool_channels: int = 32

    def build(self, channels: int) -> list[nn.Module]:
        from alpharat.nn.models.cnn.blocks import GPoolResBlock

        return [GPoolResBlock(channels, self.gpool_channels)]


class InterleavedBlockConfig(StrictBaseModel):
    """Composite config that interleaves two block types.

    Produces a sequence where `inject` is placed every `every_n` blocks,
    and `block` fills the rest.

    Example: count=6, every_n=3 with block=res, inject=gpool:
        [res, res, gpool, res, res, gpool]
    """

    type: Literal["interleaved"] = "interleaved"
    block: BlockConfig
    inject: BlockConfig
    every_n: int = 3
    count: int

    def build(self, channels: int) -> list[nn.Module]:
        modules: list[nn.Module] = []
        for i in range(1, self.count + 1):
            if i % self.every_n == 0:
                modules.extend(self.inject.build(channels))
            else:
                modules.extend(self.block.build(channels))
        return modules


BlockConfig = Annotated[
    ResBlockConfig | GPoolBlockConfig | InterleavedBlockConfig,
    Discriminator("type"),
]

# Update forward ref for InterleavedBlockConfig's recursive fields
InterleavedBlockConfig.model_rebuild()


def _default_blocks() -> list[ResBlockConfig | GPoolBlockConfig | InterleavedBlockConfig]:
    return [ResBlockConfig()]


class TrunkConfig(StrictBaseModel):
    """Trunk configuration: stem + sequence of blocks.

    `channels` is trunk-wide — all blocks share the same channel count.
    The caller decides `in_channels` (e.g. 5 for DeepSet, 7 for KataGo).
    """

    channels: int = 64
    blocks: list[BlockConfig] = Field(default_factory=_default_blocks)

    def build(self, in_channels: int) -> tuple[nn.Module, nn.ModuleList]:
        """Build stem conv and block modules.

        Args:
            in_channels: Number of input channels for the stem conv.

        Returns:
            (stem, blocks) where stem is Conv2d and blocks is ModuleList.
        """
        import torch.nn as nn_mod

        stem = nn_mod.Conv2d(in_channels, self.channels, kernel_size=3, padding=1, bias=False)
        modules: list[nn.Module] = []
        for block_cfg in self.blocks:
            modules.extend(block_cfg.build(self.channels))
        return stem, nn_mod.ModuleList(modules)
