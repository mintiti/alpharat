"""Enum keys for model outputs and losses.

Provides type-safe keys for dict-based model interfaces. Using StrEnum so
they work naturally as dict keys and serialize to strings.
"""

from __future__ import annotations

from enum import StrEnum


class ModelOutput(StrEnum):
    """Keys for model forward/predict returns."""

    # Policy logits (from forward)
    LOGITS_P1 = "logits_p1"
    LOGITS_P2 = "logits_p2"

    # Policy probabilities (from predict)
    POLICY_P1 = "policy_p1"
    POLICY_P2 = "policy_p2"

    # Scalar value outputs
    VALUE_P1 = "value_p1"
    VALUE_P2 = "value_p2"

    # LocalValueMLP-specific
    OWNERSHIP_LOGITS = "ownership_logits"
    OWNERSHIP_PROBS = "ownership_probs"
    OWNERSHIP_VALUE = "ownership_value"


class LossKey(StrEnum):
    """Keys for loss function returns."""

    # Total combined loss (required)
    TOTAL = "loss"

    # Policy losses
    POLICY_P1 = "loss_p1"
    POLICY_P2 = "loss_p2"

    # Value losses
    VALUE = "loss_value"
    VALUE_P1 = "loss_value_p1"
    VALUE_P2 = "loss_value_p2"

    # Auxiliary losses
    OWNERSHIP = "loss_ownership"


class BatchKey(StrEnum):
    """Keys for training batch dicts."""

    OBSERVATION = "observation"
    POLICY_P1 = "policy_p1"
    POLICY_P2 = "policy_p2"
    ACTION_P1 = "action_p1"
    ACTION_P2 = "action_p2"
    P1_VALUE = "p1_value"
    P2_VALUE = "p2_value"
    CHEESE_OUTCOMES = "cheese_outcomes"


class ArchitectureType(StrEnum):
    """Available model architectures.

    Each architecture bundles a model, loss function, and augmentation strategy.
    Use with the architecture registry to get the appropriate config class.

    Architectures:
        MLP: Basic shared-trunk MLP with separate P1/P2 policy heads.
            Requires player-swap augmentation for symmetry. Fast, simple baseline.

        SYMMETRIC: DeepSet-based architecture with structural P1/P2 symmetry.
            Weight sharing guarantees swap(input) → swap(output). No augmentation
            needed. Currently best performing.

        LOCAL_VALUE: MLP with auxiliary per-cheese ownership prediction head.
            KataGo-inspired: predicts who collects each cheese, derives value
            from ownership. Experimental, sharper gradients for value learning.
    """

    MLP = "mlp"
    SYMMETRIC = "symmetric"
    LOCAL_VALUE = "local_value"
