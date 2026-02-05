"""Protocols for the generic training loop.

Defines the contracts that models, loss functions, and augmentation strategies
must follow to work with the generic trainer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


@runtime_checkable
class TrainableModel(Protocol):
    """Protocol for models compatible with the generic trainer.

    Models must implement both forward (for training) and predict (for inference).
    Returns are dict-based for extensibility and self-documentation.
    """

    def forward(self, x: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            x: Input observation tensor, shape (batch, obs_dim).
            **kwargs: Model-specific arguments (e.g., cheese_mask for LocalValueMLP).

        Returns:
            Dict with at least:
            - ModelOutput.LOGITS_P1: (batch, 5) policy logits for P1
            - ModelOutput.LOGITS_P2: (batch, 5) policy logits for P2
            - ModelOutput.VALUE_P1: (batch,) scalar value for P1
            - ModelOutput.VALUE_P2: (batch,) scalar value for P2

            May include model-specific outputs (e.g., ownership_logits).
        """
        ...

    def predict(self, x: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
        """Inference forward pass.

        Args:
            x: Input observation tensor, shape (batch, obs_dim).
            **kwargs: Model-specific arguments.

        Returns:
            Dict with at least:
            - ModelOutput.POLICY_P1: (batch, 5) probabilities for P1
            - ModelOutput.POLICY_P2: (batch, 5) probabilities for P2
            - ModelOutput.VALUE_P1: (batch,) scalar value for P1
            - ModelOutput.VALUE_P2: (batch,) scalar value for P2
        """
        ...

    def __call__(self, x: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
        """Alias for forward (nn.Module compatibility)."""
        ...


class LossFunction(Protocol):
    """Protocol for model-specific loss computation.

    Each architecture defines its own loss function that composes shared losses
    from alpharat.nn.losses.
    """

    def __call__(
        self,
        model_output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        config: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute losses from model output and batch targets.

        Args:
            model_output: Dict from model.forward().
            batch: Dict with training targets (policies, values, etc.).
            config: Architecture-specific optimization config.

        Returns:
            Dict with:
            - LossKey.TOTAL: scalar combined loss (required, for backward)
            - Individual loss components for logging (e.g., LossKey.POLICY_P1)
        """
        ...


class AugmentationStrategy(Protocol):
    """Protocol for batch-level data augmentation.

    Augmentation is applied during training before the forward pass.
    """

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        width: int,
        height: int,
    ) -> dict[str, torch.Tensor]:
        """Apply augmentation to a batch.

        Args:
            batch: Training batch dict. May be modified in-place.
            width: Maze width (for spatial operations).
            height: Maze height (for spatial operations).

        Returns:
            Augmented batch dict.
        """
        ...

    @property
    def needs_augmentation(self) -> bool:
        """Whether this strategy actually augments data.

        Returns False for no-op strategies (e.g., SymmetricMLP doesn't need
        player swap augmentation due to structural symmetry).
        """
        ...
