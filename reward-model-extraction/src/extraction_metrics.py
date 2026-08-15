"""Metrics shared by model-extraction evaluation scripts."""

import numpy as np


def compute_positive_pair_agreement(target_scores, student_scores, positive_indices):
    """Return the fraction of pairs for which both models rank y+ above y-."""
    target_scores = np.asarray(target_scores, dtype=np.float64)
    student_scores = np.asarray(student_scores, dtype=np.float64)
    positive_indices = np.asarray(positive_indices, dtype=np.int64)

    if target_scores.shape != student_scores.shape:
        raise ValueError("target_scores and student_scores must have the same shape")
    if target_scores.ndim != 2 or target_scores.shape[1] != 2:
        raise ValueError("scores must have shape [n, 2]")
    if len(positive_indices) != len(target_scores):
        raise ValueError("positive_indices must contain one index per pair")
    if len(target_scores) == 0:
        return None
    if not np.isin(positive_indices, [0, 1]).all():
        raise ValueError("each positive index must be 0 or 1")

    rows = np.arange(len(target_scores))
    negative_indices = 1 - positive_indices
    target_correct = target_scores[rows, positive_indices] > target_scores[rows, negative_indices]
    student_correct = student_scores[rows, positive_indices] > student_scores[rows, negative_indices]
    return float(np.mean(target_correct & student_correct))
