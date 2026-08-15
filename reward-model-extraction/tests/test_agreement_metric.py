import numpy as np
from src.extraction_metrics import compute_positive_pair_agreement


def test_agreement_requires_both_models_to_rank_y_plus_higher():
    teacher = np.array([[2.0, 1.0], [2.0, 1.0], [1.0, 3.0], [1.0, 1.0]])
    student = np.array([[3.0, 0.0], [0.0, 3.0], [0.0, 2.0], [2.0, 1.0]])

    value = compute_positive_pair_agreement(
        teacher, student, positive_indices=[0, 0, 1, 0]
    )

    # Items 1 and 3 agree. Item 2 fails for the student; item 4 is a teacher tie.
    assert value == 0.5


def test_empty_agreement_is_none():
    empty = np.empty((0, 2))
    assert compute_positive_pair_agreement(empty, empty, []) is None
