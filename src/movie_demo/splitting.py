from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_array

DEFAULT_SPLIT_SEED = 42
DEFAULT_VALIDATION_FRACTION = 0.1
DEFAULT_TEST_FRACTION = 0.1


@dataclass(frozen=True)
class ViewerSplit:
    """Sparse matrices and original column IDs for disjoint viewer groups."""

    train: csc_array
    validation: csc_array
    test: csc_array
    train_viewer_ids: np.ndarray
    validation_viewer_ids: np.ndarray
    test_viewer_ids: np.ndarray


def split_viewers(
    interactions: csc_array,
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> ViewerSplit:
    """Split whole viewer columns so each viewer belongs to exactly one group."""
    for name, fraction in (
        ("validation_fraction", validation_fraction),
        ("test_fraction", test_fraction),
    ):
        if type(fraction) not in (int, float):
            raise TypeError(f"{name} must be a number.")
        if not np.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError(f"{name} must be finite and in [0, 1).")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1.")
    if type(seed) is not int:
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    num_viewers = interactions.shape[1]
    validation_count = (
        max(1, int(num_viewers * validation_fraction)) if validation_fraction > 0 else 0
    )
    test_count = max(1, int(num_viewers * test_fraction)) if test_fraction > 0 else 0
    train_count = num_viewers - validation_count - test_count
    if train_count < 1:
        raise ValueError("Split fractions leave no viewers for training.")

    shuffled = np.random.default_rng(seed).permutation(num_viewers)
    train_ids = np.sort(shuffled[:train_count])
    validation_ids = np.sort(shuffled[train_count : train_count + validation_count])
    test_ids = np.sort(shuffled[train_count + validation_count :])
    return ViewerSplit(
        train=interactions[:, train_ids],
        validation=interactions[:, validation_ids],
        test=interactions[:, test_ids],
        train_viewer_ids=train_ids,
        validation_viewer_ids=validation_ids,
        test_viewer_ids=test_ids,
    )
