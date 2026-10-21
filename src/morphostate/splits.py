"""Donor-level partitioning. Repeated slides from a donor always stay together."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import _check_identifiers


def make_donor_split(
    cells: pd.DataFrame,
    seed: int = 7,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    explicit: dict | None = None,
) -> dict[str, np.ndarray]:
    """Return positional row indices; explicit partitions must be exhaustive."""
    if "donor_id" not in cells:
        raise ValueError("cells must contain donor_id")
    donor_values = _check_identifiers(cells["donor_id"], "donor_id")
    donors = sorted(set(donor_values))
    if len(donors) < 3:
        raise ValueError("at least three donors are required for nonempty train/val/test splits")
    names = ("train", "val", "test")
    if explicit is not None:
        if not isinstance(explicit, dict) or set(explicit) != set(names):
            raise ValueError("explicit split must have exactly train, val, test keys")
        partitions = {}
        for name in names:
            if not isinstance(explicit[name], (list, tuple)):
                raise ValueError(f"explicit {name} donors must be a list of strings")
            partitions[name] = _check_identifiers(explicit[name], f"{name} donor IDs", unique=True)
            if not partitions[name]:
                raise ValueError(f"{name} donor split must not be empty")
        flattened = [donor for name in names for donor in partitions[name]]
        if len(flattened) != len(set(flattened)):
            raise ValueError("donor overlap between explicit train/val/test splits")
        unknown = set(flattened) - set(donors)
        omitted = set(donors) - set(flattened)
        if unknown or omitted:
            raise ValueError(f"explicit split has unknown donors {sorted(unknown)} and missing donors {sorted(omitted)}")
    else:
        if (not np.isfinite(val_fraction) or not np.isfinite(test_fraction)
                or val_fraction <= 0 or test_fraction <= 0
                or val_fraction + test_fraction >= 1):
            raise ValueError("validation and test fractions must be positive and sum to less than one")
        shuffled = np.random.default_rng(seed).permutation(donors)
        n_val = max(1, int(np.floor(len(donors) * val_fraction)))
        n_test = max(1, int(np.floor(len(donors) * test_fraction)))
        if n_val + n_test >= len(donors):
            raise ValueError("fractions leave no training donors")
        partitions = {
            "val": shuffled[:n_val].tolist(),
            "test": shuffled[n_val:n_val + n_test].tolist(),
            "train": shuffled[n_val + n_test:].tolist(),
        }
    labels = np.asarray(donor_values)
    return {name: np.flatnonzero(np.isin(labels, partitions[name])) for name in names}
