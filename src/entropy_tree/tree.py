"""Decision tree classifier implemented from scratch with entropy and information gain.

Numeric features are split with a binary threshold test (``x <= t``), categorical
features with a multiway split (one branch per category, as in ID3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Impurity measures
# ---------------------------------------------------------------------------

def entropy(y) -> float:
    """Shannon entropy (in bits) of a sequence of labels."""
    p = pd.Series(y).value_counts(normalize=True).to_numpy()
    return float(-(p * np.log2(p)).sum()) + 0.0


def information_gain(parent, children) -> float:
    """Entropy of ``parent`` minus the weighted entropy of the ``children`` partitions."""
    n = len(parent)
    weighted = sum(len(c) / n * entropy(c) for c in children if len(c))
    return entropy(parent) - weighted


def _entropy_from_counts(counts: np.ndarray) -> np.ndarray:
    """Row-wise entropy of a 2-D array of class counts (also accepts 1-D)."""
    counts = np.atleast_2d(counts).astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(totals > 0, counts / totals, 0.0)
        terms = np.where(p > 0, p * np.log2(p), 0.0)
    return -terms.sum(axis=1) + 0.0  # + 0.0 turns -0.0 into 0.0


# ---------------------------------------------------------------------------
# Tree structure
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """A node of the tree. A node without ``feature`` is a leaf."""

    prediction: object
    class_counts: np.ndarray
    entropy: float
    n_samples: int
    depth: int
    feature: str | None = None
    threshold: float | None = None
    gain: float = 0.0
    # numeric split: {"<=": Node, ">": Node}; categorical split: {category: Node}
    children: dict = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class DecisionTreeClassifier:
    """Entropy-based decision tree classifier.

    Parameters controlling the size of the tree
    -------------------------------------------
    max_depth : int or None
        Maximum depth of the tree (root has depth 0). ``None`` means unlimited.
    min_samples_split : int
        Minimum number of samples a node needs to be split.
    min_samples_leaf : int
        Minimum number of samples every child of a split must contain.
    min_information_gain : float
        A split is only made if its information gain is at least this value.
    max_leaf_nodes : int or None
        Maximum number of leaves. When set, the tree is grown best-first
        (the split with the highest weighted gain is expanded first).
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_information_gain: float = 0.0,
        max_leaf_nodes: int | None = None,
    ):
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be >= 0 or None")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if max_leaf_nodes is not None and max_leaf_nodes < 1:
            raise ValueError("max_leaf_nodes must be >= 1 or None")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.max_leaf_nodes = max_leaf_nodes
        self.root_: Node | None = None

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={getattr(self, k)!r}"
            for k in ("max_depth", "min_samples_split", "min_samples_leaf",
                      "min_information_gain", "max_leaf_nodes")
        )
        return f"DecisionTreeClassifier({params})"

    # ------------------------------------------------------------------ fit

    def fit(self, X, y):
        """Build the tree from ``X`` (DataFrame or array) and labels ``y``.

        ``y`` can also be the name of a column of ``X``, which is then used as
        the target and removed from the features.
        """
        X = self._to_frame(X)
        if isinstance(y, str) and y in X.columns:
            X, y = X.drop(columns=y), X[y]
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)} labels")
        if len(y) == 0:
            raise ValueError("cannot fit on an empty dataset")
        if X.isna().any().any() or pd.isna(y).any():
            raise ValueError("missing values are not supported; drop or impute them first")

        self.feature_names_ = list(X.columns)
        self.numeric_features_ = {
            c for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])
        }
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self._columns = {
            c: X[c].to_numpy(dtype=float if c in self.numeric_features_ else object)
            for c in X.columns
        }
        self._y = y_enc
        self._importance = dict.fromkeys(self.feature_names_, 0.0)

        all_idx = np.arange(len(y_enc))
        self.root_ = self._make_node(all_idx, depth=0)
        if self.max_leaf_nodes is None:
            self._grow_depth_first(self.root_, all_idx)
        else:
            self._grow_best_first(all_idx)

        total = sum(self._importance.values())
        self.feature_importances_ = {
            f: (v / total if total > 0 else 0.0) for f, v in self._importance.items()
        }
        del self._columns, self._y, self._importance
        return self

    def _make_node(self, idx: np.ndarray, depth: int) -> Node:
        counts = np.bincount(self._y[idx], minlength=len(self.classes_))
        return Node(
            prediction=self.classes_[counts.argmax()],
            class_counts=counts,
            entropy=float(_entropy_from_counts(counts)[0]),
            n_samples=len(idx),
            depth=depth,
        )

    def _grow_depth_first(self, node: Node, idx: np.ndarray) -> None:
        split = self._find_best_split(node, idx)
        if split is None:
            return
        for child, child_idx in self._apply_split(node, idx, split):
            self._grow_depth_first(child, child_idx)

    def _grow_best_first(self, idx: np.ndarray) -> None:
        # Frontier of candidate splits; always expand the one that reduces the
        # total (sample-weighted) entropy the most, until the leaf budget is used.
        n_leaves = 1
        frontier = []

        def push(node, node_idx):
            split = self._find_best_split(node, node_idx)
            if split is not None:
                frontier.append((split["gain"] * node.n_samples, node, node_idx, split))

        push(self.root_, idx)
        while frontier:
            frontier.sort(key=lambda item: item[0])
            _, node, node_idx, split = frontier.pop()
            n_new = self._n_branches(split) - 1
            if n_leaves + n_new > self.max_leaf_nodes:
                continue  # does not fit the leaf budget; try the next best split
            n_leaves += n_new
            for child, child_idx in self._apply_split(node, node_idx, split):
                push(child, child_idx)

    @staticmethod
    def _n_branches(split: dict) -> int:
        return 2 if "threshold" in split else len(split["categories"])

    def _find_best_split(self, node: Node, idx: np.ndarray) -> dict | None:
        """Return the best admissible split of ``node`` or ``None`` (node stays a leaf)."""
        if node.entropy == 0.0:
            return None
        if self.max_depth is not None and node.depth >= self.max_depth:
            return None
        if node.n_samples < self.min_samples_split:
            return None

        best = None
        y = self._y[idx]
        for feature in self.feature_names_:
            x = self._columns[feature][idx]
            if feature in self.numeric_features_:
                candidate = self._best_threshold(x, y, node.entropy)
            else:
                candidate = self._categorical_split(x, y, node.entropy)
            if candidate is not None and (best is None or candidate["gain"] > best["gain"]):
                best = {**candidate, "feature": feature}

        if best is None or best["gain"] <= 1e-12 or best["gain"] < self.min_information_gain:
            return None
        return best

    def _best_threshold(self, x: np.ndarray, y: np.ndarray, parent_entropy: float) -> dict | None:
        """Scan every midpoint between consecutive distinct values in one pass."""
        n, k = len(x), len(self.classes_)
        order = np.argsort(x, kind="mergesort")
        xs, ys = x[order], y[order]

        onehot = np.zeros((n, k))
        onehot[np.arange(n), ys] = 1
        left = np.cumsum(onehot, axis=0)[:-1]  # row i: counts of xs[:i + 1]
        right = left[-1] + onehot[-1] - left
        n_left = np.arange(1, n)
        n_right = n - n_left

        valid = ((xs[1:] != xs[:-1])
                 & (n_left >= self.min_samples_leaf)
                 & (n_right >= self.min_samples_leaf))
        if not valid.any():
            return None

        child_entropy = (n_left * _entropy_from_counts(left)
                         + n_right * _entropy_from_counts(right)) / n
        gains = np.where(valid, parent_entropy - child_entropy, -np.inf)
        i = int(gains.argmax())
        return {"gain": float(gains[i]), "threshold": float((xs[i] + xs[i + 1]) / 2)}

    def _categorical_split(self, x: np.ndarray, y: np.ndarray, parent_entropy: float) -> dict | None:
        values, inverse = np.unique(x.astype(str), return_inverse=True)
        if len(values) < 2:
            return None
        counts = np.zeros((len(values), len(self.classes_)))
        np.add.at(counts, (inverse, y), 1)
        sizes = counts.sum(axis=1)
        if sizes.min() < self.min_samples_leaf:
            return None
        child_entropy = float((sizes * _entropy_from_counts(counts)).sum() / len(x))
        return {"gain": parent_entropy - child_entropy, "categories": values}

    def _apply_split(self, node: Node, idx: np.ndarray, split: dict):
        """Turn ``node`` into an internal node and return its (child, indices) pairs."""
        feature = split["feature"]
        x = self._columns[feature][idx]
        if "threshold" in split:
            mask = x <= split["threshold"]
            parts = {"<=": idx[mask], ">": idx[~mask]}
            node.threshold = split["threshold"]
        else:
            x = x.astype(str)
            parts = {v: idx[x == v] for v in split["categories"]}

        node.feature = feature
        node.gain = split["gain"]
        self._importance[feature] += node.n_samples * node.gain
        for key, part in parts.items():
            node.children[key] = self._make_node(part, node.depth + 1)
        return [(node.children[key], part) for key, part in parts.items()]

    # -------------------------------------------------------------- predict

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities (columns ordered as ``classes_``) for every row of ``X``."""
        self._check_fitted()
        X = self._prepare(X)
        proba = np.zeros((len(X), len(self.classes_)))
        self._route(self.root_, X, np.arange(len(X)), proba)
        return proba

    def predict(self, X) -> np.ndarray:
        """Predicted class for every row of ``X``."""
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

    def score(self, X, y) -> float:
        """Accuracy on ``(X, y)``."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _route(self, node: Node, X: pd.DataFrame, idx: np.ndarray, proba: np.ndarray) -> None:
        if len(idx) == 0:
            return
        if node.is_leaf:
            proba[idx] = node.class_counts / node.n_samples
            return
        x = X[node.feature].to_numpy()[idx]
        if node.threshold is not None:
            mask = x.astype(float) <= node.threshold
            self._route(node.children["<="], X, idx[mask], proba)
            self._route(node.children[">"], X, idx[~mask], proba)
            return
        x = x.astype(str)
        unseen = np.ones(len(idx), dtype=bool)
        for value, child in node.children.items():
            mask = x == value
            unseen &= ~mask
            self._route(child, X, idx[mask], proba)
        # category never seen during training: use this node's class distribution
        proba[idx[unseen]] = node.class_counts / node.n_samples

    # ------------------------------------------------------- introspection

    def get_depth(self) -> int:
        self._check_fitted()
        return max(n.depth for n in self._nodes())

    def get_n_leaves(self) -> int:
        self._check_fitted()
        return sum(n.is_leaf for n in self._nodes())

    def _nodes(self):
        stack = [self.root_]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    # -------------------------------------------------------------- display

    def export_text(self, decimals: int = 3) -> str:
        """Readable text rendering of the tree."""
        self._check_fitted()
        lines = [self._describe(self.root_, decimals)]

        def walk(node: Node, prefix: str):
            items = list(node.children.items())
            for i, (key, child) in enumerate(items):
                last = i == len(items) - 1
                branch = "└── " if last else "├── "
                lines.append(f"{prefix}{branch}{self._edge_label(node, key, decimals)}"
                             f" → {self._describe(child, decimals)}")
                walk(child, prefix + ("    " if last else "│   "))

        walk(self.root_, "")
        return "\n".join(lines)

    def print_tree(self, decimals: int = 3) -> None:
        print(self.export_text(decimals))

    def to_dot(self, decimals: int = 3) -> str:
        """Graphviz DOT source of the tree (render with ``dot -Tpng tree.dot -o tree.png``)."""
        self._check_fitted()
        lines = ["digraph DecisionTree {", '  node [shape=box, style="rounded,filled", fontname=Helvetica];']
        ids = {}

        def visit(node: Node):
            ids[id(node)] = len(ids)
            color = "#d9ead3" if node.is_leaf else "#cfe2f3"
            label = self._node_label(node, decimals).replace('"', '\\"').replace("\n", "\\n")
            lines.append(f'  n{ids[id(node)]} [label="{label}", fillcolor="{color}"];')
            for key, child in node.children.items():
                visit(child)
                edge = self._edge_label(node, key, decimals).replace('"', '\\"')
                lines.append(f'  n{ids[id(node)]} -> n{ids[id(child)]} [label="{edge}"];')

        visit(self.root_)
        lines.append("}")
        return "\n".join(lines)

    def plot(self, ax=None, decimals: int = 2, fontsize: int = 9, figsize=None):
        """Draw the tree with matplotlib (``pip install entropy-tree[plot]``). Returns the axes."""
        self._check_fitted()
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError("plot() needs matplotlib: pip install entropy-tree[plot]") from exc

        # leaves are spread evenly on x, parents are centred above their children
        positions, next_x = {}, [0]

        def layout(node: Node) -> float:
            if node.is_leaf:
                x = next_x[0]
                next_x[0] += 1
            else:
                xs = [layout(c) for c in node.children.values()]
                x = (xs[0] + xs[-1]) / 2
            positions[id(node)] = (x, -node.depth)
            return x

        layout(self.root_)
        n_leaves, depth = next_x[0], self.get_depth()
        if ax is None:
            figsize = figsize or (max(6, 2.2 * n_leaves), max(3, 1.8 * (depth + 1)))
            _, ax = plt.subplots(figsize=figsize)

        def draw(node: Node):
            x, y = positions[id(node)]
            for key, child in node.children.items():
                cx, cy = positions[id(child)]
                ax.plot([x, cx], [y, cy], color="#888888", lw=1, zorder=1)
                ax.text((x + cx) / 2, (y + cy) / 2, self._edge_label(node, key, decimals, short=True),
                        fontsize=fontsize - 1, ha="center", va="center", zorder=3,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
                draw(child)
            ax.text(x, y, self._node_label(node, decimals), fontsize=fontsize,
                    ha="center", va="center", zorder=2,
                    bbox=dict(boxstyle="round,pad=0.4",
                              fc="#d9ead3" if node.is_leaf else "#cfe2f3", ec="#555555"))

        draw(self.root_)
        ax.set_xlim(-0.7, n_leaves - 0.3)
        ax.set_ylim(-depth - 0.6, 0.6)
        ax.axis("off")
        return ax

    def _describe(self, node: Node, decimals: int) -> str:
        stats = f"n={node.n_samples}, H={node.entropy:.{decimals}f}"
        if node.is_leaf:
            return f"predict {node.prediction!r} ({stats}, counts={self._counts_str(node)})"
        return f"split on {node.feature!r} ({stats}, IG={node.gain:.{decimals}f})"

    def _node_label(self, node: Node, decimals: int) -> str:
        head = f"{node.prediction}" if node.is_leaf else f"{node.feature}?"
        return (f"{head}\nH={node.entropy:.{decimals}f}  n={node.n_samples}\n"
                f"{self._counts_str(node)}"
                + ("" if node.is_leaf else f"\nIG={node.gain:.{decimals}f}"))

    def _counts_str(self, node: Node) -> str:
        return " ".join(f"{c}:{int(n)}" for c, n in zip(self.classes_, node.class_counts))

    @staticmethod
    def _edge_label(node: Node, key, decimals: int, short: bool = False) -> str:
        if node.threshold is not None:
            test = f"{key} {node.threshold:.{decimals}f}"
        else:
            test = f"= {key}"
        return test if short else f"{node.feature} {test}"

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _to_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])]).infer_objects()

    def _prepare(self, X) -> pd.DataFrame:
        X = self._to_frame(X)
        if set(self.feature_names_) <= set(X.columns):
            return X[self.feature_names_]
        if X.shape[1] != len(self.feature_names_):
            missing = sorted(set(self.feature_names_) - set(X.columns))
            raise ValueError(f"missing feature columns: {missing}")
        # same number of columns but other names (e.g. a plain array): match by position
        X.columns = self.feature_names_
        return X

    def _check_fitted(self) -> None:
        if self.root_ is None:
            raise RuntimeError("the tree is not fitted yet; call fit() first")


def any_df_decision_tree(data: pd.DataFrame, target_column: str, **params) -> DecisionTreeClassifier:
    """Build a decision tree on ``data`` predicting ``target_column``.

    Any keyword argument is forwarded to :class:`DecisionTreeClassifier`
    (``max_depth``, ``min_samples_split``, ``min_samples_leaf``, ...).
    """
    if target_column not in data.columns:
        raise KeyError(f"target column {target_column!r} not found in data")
    return DecisionTreeClassifier(**params).fit(data, target_column)
