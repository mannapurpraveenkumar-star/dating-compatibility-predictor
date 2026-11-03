"""Offline experiment harness for the research question.

This is NOT used by the website. It lets you test, honestly, whether the pair
features measured by the app predict an independently collected label, once
you have properly consented, labelled data.

Input: a CSV with one row per pair (see research/pairs_template.csv):

    photo_id, person_a_id, person_b_id, <PAIR_FEATURE_NAMES...>, label

* ``label`` is 1 or 0 and must come from the participants themselves (e.g. a
  consented self-report that they are a couple), never from looking at the photo.
* ``photo_id`` and the person ids are used for a *grouped* split so the same
  photo or person never appears in both train and test (prevents leakage).

The harness compares a logistic-regression model against two baselines
(majority class and a stratified-random guesser) and reports accuracy,
balanced accuracy, ROC AUC and a confusion matrix. A model that does not
clearly beat the baselines on held-out data has NOT shown predictive ability.

Pure NumPy so it adds no dependencies. Run:
    .\\.venv\\Scripts\\python.exe -m app.research research\\your_data.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass

import numpy as np

from app.analysis import PAIR_FEATURE_NAMES

ID_COLUMNS = ["photo_id", "person_a_id", "person_b_id"]
LABEL_COLUMN = "label"
REQUIRED_COLUMNS = ID_COLUMNS + PAIR_FEATURE_NAMES + [LABEL_COLUMN]


@dataclass
class Dataset:
    X: np.ndarray            # (n, d) features
    y: np.ndarray            # (n,) 0/1
    groups: np.ndarray       # (n,) group id per row (connected photos/people)


# ---------------------------------------------------------------------------
# Loading and grouping
# ---------------------------------------------------------------------------
def load_csv(path: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no data rows")
    X = np.array([[float(r[c]) for c in PAIR_FEATURE_NAMES] for r in rows], dtype=float)
    y = np.array([int(r[LABEL_COLUMN]) for r in rows], dtype=int)
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("label must be 0 or 1")
    return rows, X, y


def connected_groups(rows: list[dict]) -> np.ndarray:
    """Union-find over photos and people: rows sharing a photo or person share a group."""
    parent: dict[str, str] = {}

    def find(a: str) -> str:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for r in rows:
        p = "photo:" + r["photo_id"]
        union(p, "person:" + r["person_a_id"])
        union(p, "person:" + r["person_b_id"])
    roots = [find("photo:" + r["photo_id"]) for r in rows]
    index = {root: i for i, root in enumerate(dict.fromkeys(roots))}
    return np.array([index[r] for r in roots], dtype=int)


def grouped_split(groups: np.ndarray, test_fraction: float = 0.3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices so that no group appears on both sides."""
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(groups))
    n_test = max(1, round(len(unique) * test_fraction))
    test_groups = set(unique[:n_test].tolist())
    test_mask = np.array([g in test_groups for g in groups])
    return np.where(~test_mask)[0], np.where(test_mask)[0]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class LogisticRegression:
    """L2-regularised logistic regression trained by gradient descent."""

    def __init__(self, l2: float = 1.0, lr: float = 0.1, epochs: int = 2000):
        self.l2, self.lr, self.epochs = l2, lr, epochs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        Z = (X - self.mu) / self.sd
        n, d = Z.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = self._sigmoid(Z @ self.w + self.b)
            grad_w = Z.T @ (p - y) / n + self.l2 * self.w / n
            grad_b = float(np.mean(p - y))
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(((X - self.mu) / self.sd) @ self.w + self.b)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.sum((pred == 1) & (y == 1))),
        "fp": int(np.sum((pred == 1) & (y == 0))),
        "tn": int(np.sum((pred == 0) & (y == 0))),
        "fn": int(np.sum((pred == 0) & (y == 1))),
    }


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    c = confusion(y, pred)
    tpr = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    tnr = c["tn"] / (c["tn"] + c["fp"]) if (c["tn"] + c["fp"]) else 0.0
    return (tpr + tnr) / 2


def roc_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    """Mann-Whitney formulation; None if only one class is present."""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def evaluate(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (score >= threshold).astype(int)
    auc = roc_auc(y, score)
    return {
        "n": int(len(y)),
        "accuracy": round(float(np.mean(pred == y)), 4),
        "balanced_accuracy": round(balanced_accuracy(y, pred), 4),
        "roc_auc": None if auc is None else round(auc, 4),
        "confusion": confusion(y, pred),
    }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_experiment(data: Dataset, test_fraction: float = 0.3, seed: int = 0) -> dict:
    train, test = grouped_split(data.groups, test_fraction, seed)
    if len(np.unique(data.y[train])) < 2:
        raise ValueError("Training split contains only one class; collect more varied data.")
    Xtr, ytr, Xte, yte = data.X[train], data.y[train], data.X[test], data.y[test]

    majority = int(round(ytr.mean()))
    rng = np.random.default_rng(seed + 1)
    results = {
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "train_test_groups_overlap": bool(set(data.groups[train]) & set(data.groups[test])),
        "positive_rate_train": round(float(ytr.mean()), 4),
        "baseline_majority": evaluate(yte, np.full(len(yte), float(majority))),
        "baseline_random": evaluate(yte, (rng.random(len(yte)) < ytr.mean()).astype(float)),
        "logistic_regression": evaluate(yte, LogisticRegression().fit(Xtr, ytr).predict_proba(Xte)),
        "features": PAIR_FEATURE_NAMES,
    }
    lr_auc = results["logistic_regression"]["roc_auc"]
    results["interpretation"] = (
        "Insufficient test data to judge." if lr_auc is None else
        "The model beats chance on held-out groups by this measure; this is evidence about these "
        "features in this dataset only, not proof of 'compatibility'." if lr_auc >= 0.6 else
        "The model does not meaningfully beat chance on held-out groups (ROC AUC < 0.6)."
    )
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate pair features against consented labels.")
    ap.add_argument("csv", help="CSV in the format of research/pairs_template.csv")
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    try:
        rows, X, y = load_csv(args.csv)
        data = Dataset(X, y, connected_groups(rows))
        print(json.dumps(run_experiment(data, args.test_fraction, args.seed), indent=2))
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
