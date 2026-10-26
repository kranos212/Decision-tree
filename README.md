# entropy-tree

A decision tree classifier written from scratch with **numpy** and **pandas**. It picks splits by **entropy** and **information gain**.

- Numeric features get binary threshold splits (`x <= t`). Categorical features get multiway splits (one branch per category, as in ID3).
- The size of the tree is controlled by hyperparameters: `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_information_gain`, `max_leaf_nodes`.
- Three ways to display the tree: text (`print_tree()`), matplotlib (`plot()`), Graphviz (`to_dot()`).
- Two example datasets are included: `load_play_tennis()` and `load_loan_approval()`.

## Installation

```bash
pip install entropy-tree            # from PyPI
pip install "entropy-tree[plot]"    # with matplotlib, for tree.plot()
# or, from a clone of this repository:
pip install -e ".[plot]"
```

## Quick start

```python
from entropy_tree import DecisionTreeClassifier, load_play_tennis

df = load_play_tennis()
tree = DecisionTreeClassifier(max_depth=3).fit(df, "play")   # "play" is the target column
tree.print_tree()
```

```
split on 'outlook' (n=14, H=0.940, IG=0.247)
├── outlook = overcast → predict 'yes' (n=4, H=0.000, counts=no:0 yes:4)
├── outlook = rainy → split on 'windy' (n=5, H=0.971, IG=0.971)
│   ├── windy = False → predict 'yes' (n=3, H=0.000, counts=no:0 yes:3)
│   └── windy = True → predict 'no' (n=2, H=0.000, counts=no:2 yes:0)
└── outlook = sunny → split on 'humidity' (n=5, H=0.971, IG=0.971)
    ├── humidity = high → predict 'no' (n=3, H=0.000, counts=no:3 yes:0)
    └── humidity = normal → predict 'yes' (n=2, H=0.000, counts=no:0 yes:2)
```

```python
tree.predict(df.drop(columns="play"))       # labels
tree.predict_proba(df.drop(columns="play")) # class probabilities (order: tree.classes_)
tree.score(df.drop(columns="play"), df["play"])
tree.plot()                                 # matplotlib figure
open("tree.dot", "w").write(tree.to_dot())  # Graphviz
```

`fit(X, y)` also takes a numpy array or a separate label vector:
`DecisionTreeClassifier().fit(X, y)`.

## API

| Item | Description |
|---|---|
| `DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, min_information_gain=0.0, max_leaf_nodes=None)` | The model |
| `.fit(X, y)` / `.predict(X)` / `.predict_proba(X)` / `.score(X, y)` | Train, predict, evaluate |
| `.get_depth()`, `.get_n_leaves()`, `.feature_importances_`, `.classes_`, `.root_` | Inspect the tree |
| `.print_tree()`, `.export_text()`, `.plot()`, `.to_dot()` | Display the tree |
| `any_df_decision_tree(df, target_column, **params)` | One-call helper: build a tree from a DataFrame |
| `entropy(y)`, `information_gain(parent, children)` | Impurity functions |
| `train_test_split`, `accuracy_score`, `confusion_matrix` | Small helpers, so scikit-learn is not needed |
| `load_play_tennis()`, `load_loan_approval()` | Example datasets |

## Examples and tests

```bash
python examples/demo.py --plot     # full walkthrough on both datasets
python -m unittest discover -s tests
```

See [report.md](report.md) for a description of the algorithm and results.
