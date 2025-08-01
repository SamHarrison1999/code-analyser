# ✅ Best Practice: Modular reward functions for RL fine-tuning of annotation model
# 🧠 ML Signal: Reward shaping based on annotation correctness, confidence, and coverage

import torch
from typing import Union


def compute_reward(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes reward for a single annotation step.

    Args:
        predicted (torch.Tensor): Binary prediction vector (multi-label).
        target (torch.Tensor): Ground truth label vector.

    Returns:
        float: Reward value.
    """
    # ✅ Match reward: positive if prediction overlaps with true labels
    match_reward = (predicted.int() & target.int()).sum().item()

    # ⚠️ Penalty for false positives
    false_positive_penalty = ((predicted.int() == 1) & (target.int() == 0)).sum().item()

    # ✅ Coverage reward: bonus for hitting at least one correct label
    coverage_bonus = 1.0 if match_reward > 0 else 0.0

    # Final reward formula
    reward = 1.5 * match_reward - 0.5 * false_positive_penalty + coverage_bonus
    return reward


def confidence_scaled_reward(
    predicted: torch.Tensor, target: torch.Tensor, confidences: torch.Tensor
) -> float:
    """
    Computes a reward scaled by prediction confidence.

    Args:
        predicted (torch.Tensor): Binary predictions (e.g. 0 or 1).
        target (torch.Tensor): True binary labels.
        confidences (torch.Tensor): Raw probabilities or confidence values (0–1).

    Returns:
        float: Scaled reward.
    """
    agreement = (predicted.int() & target.int()).float()
    weighted_reward = (agreement * confidences).sum().item()
    return weighted_reward


def diversity_penalty(
    prediction_history: list, new_pred: Union[list, torch.Tensor], min_unique: int = 2
) -> float:
    """
    Penalises repetitive predictions if they lack diversity.

    Args:
        prediction_history (list): Previous predictions (list of tensors or lists).
        new_pred (list or tensor): Current prediction.
        min_unique (int): Minimum distinct predictions to avoid penalty.

    Returns:
        float: Diversity bonus or penalty.
    """
    flattened = [
        tuple(p.tolist() if isinstance(p, torch.Tensor) else p)
        for p in prediction_history
    ]
    flattened.append(
        tuple(new_pred.tolist() if isinstance(new_pred, torch.Tensor) else new_pred)
    )

    unique = len(set(flattened))
    if unique < min_unique:
        return -1.0  # Penalise redundancy
    return +0.5  # Bonus for maintaining diversity
