"""Tests for the offline experiment harness.

These use SYNTHETIC numbers only, to check the mechanics (splitting, baselines,
metrics). They say nothing about real relationships.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.analysis import PAIR_FEATURE_NAMES
from app.research import (
    Dataset,
    connected_groups,
    grouped_split,
    load_csv,
    roc_auc,
    run_experiment,
)
from tests.conftest import ROOT


def synthetic(n_groups=60, per_group=5, signal=True, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    n = n_groups * per_group
    X = rng.normal(size=(n, len(PAIR_FEATURE_NAMES)))
    if signal:
        y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    else:
        y = rng.integers(0, 2, size=n)
    groups = np.repeat(np.arange(n_groups), per_group)
    return Dataset(X, y, groups)


def test_grouped_split_has_no_overlap():
    groups = np.repeat(np.arange(20), 4)
    train, test = grouped_split(groups, 0.3, seed=1)
    assert set(groups[train]).isdisjoint(groups[test])
    assert len(train) + len(test) == len(groups)


def test_connected_groups_link_shared_people_and_photos():
    rows = [
        {"photo_id": "p1", "person_a_id": "u1", "person_b_id": "u2"},
        {"photo_id": "p2", "person_a_id": "u2", "person_b_id": "u3"},  # shares u2 with p1
        {"photo_id": "p3", "person_a_id": "u8", "person_b_id": "u9"},  # unrelated
    ]
    g = connected_groups(rows)
    assert g[0] == g[1] != g[2]


def test_auc_known_values():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert roc_auc(np.array([1, 1]), np.array([0.3, 0.4])) is None


def test_model_beats_baselines_when_signal_exists():
    r = run_experiment(synthetic(signal=True))
    assert not r["train_test_groups_overlap"]
    assert r["logistic_regression"]["roc_auc"] > 0.9
    assert r["logistic_regression"]["balanced_accuracy"] > r["baseline_majority"]["balanced_accuracy"]


def test_model_does_not_beat_chance_on_random_labels():
    r = run_experiment(synthetic(signal=False, seed=3))
    assert 0.3 < r["logistic_regression"]["roc_auc"] < 0.7


def test_template_csv_has_required_columns_but_no_data():
    with pytest.raises(ValueError, match="no data rows"):
        load_csv(str(ROOT / "research" / "pairs_template.csv"))
