"""Small helpers so the package can be used without scikit-learn."""

from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd


def train_test_split(X, y, test_size: float = 0.25, random_state: int | None = None):
    """Shuffle and split ``X`` and ``y`` into train and test parts."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    n = len(X)
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_size)))
    test, train = idx[:n_test], idx[n_test:]

    def take(a, rows):
        return a.iloc[rows].reset_index(drop=True) if isinstance(a, (pd.DataFrame, pd.Series)) else np.asarray(a)[rows]

    return take(X, train), take(X, test), take(y, train), take(y, test)


def accuracy_score(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred) -> pd.DataFrame:
    """Confusion matrix as a DataFrame (rows: true class, columns: predicted class)."""
    return pd.crosstab(pd.Series(np.asarray(y_true), name="true"),
                       pd.Series(np.asarray(y_pred), name="predicted"))


def _load(name: str) -> pd.DataFrame:
    with resources.files("entropy_tree.datasets").joinpath(name).open("r", encoding="utf-8") as f:
        return pd.read_csv(f)


def load_play_tennis() -> pd.DataFrame:
    """Quinlan's classic 14-row "play tennis" dataset (categorical features, target ``play``)."""
    return _load("play_tennis.csv")


def load_loan_approval() -> pd.DataFrame:
    """Synthetic 600-row loan dataset mixing numeric and categorical features (target ``approved``)."""
    return _load("loan_approval.csv")
