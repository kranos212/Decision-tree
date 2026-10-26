"""Demo of entropy_tree on the two bundled datasets.

Run from the repository root:  python examples/demo.py
(add --plot to also save PNG drawings of the trees; needs matplotlib)
"""

import sys

from entropy_tree import (
    DecisionTreeClassifier,
    any_df_decision_tree,
    confusion_matrix,
    entropy,
    information_gain,
    load_loan_approval,
    load_play_tennis,
    train_test_split,
)

PLOT = "--plot" in sys.argv


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# 1. Play tennis: the textbook ID3 example ------------------------------------
section("1. Play tennis (14 rows, categorical features)")
tennis = load_play_tennis()
print(tennis.to_string(index=False))

print(f"\nEntropy of 'play': {entropy(tennis['play']):.3f} bits")
for col in ["outlook", "temperature", "humidity", "windy"]:
    groups = [g["play"] for _, g in tennis.groupby(col)]
    print(f"  IG(play, {col:<11}) = {information_gain(tennis['play'], groups):.3f}")

tree = any_df_decision_tree(tennis, "play")  # original entry point, still works
print("\nLearned tree:")
tree.print_tree()
print(f"\ndepth={tree.get_depth()}  leaves={tree.get_n_leaves()}  "
      f"training accuracy={tree.score(tennis.drop(columns='play'), tennis['play']):.2f}")

new_day = tennis.drop(columns="play").iloc[[0]].assign(outlook="overcast")
print(f"Prediction for {new_day.to_dict('records')[0]} -> {tree.predict(new_day)[0]}")

# 2. Loan approval: numeric + categorical, effect of the size hyperparameters --
section("2. Loan approval (600 rows, numeric + categorical features)")
loans = load_loan_approval()
X, y = loans.drop(columns="approved"), loans["approved"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
print(f"fit={len(X_fit)} rows, validation={len(X_val)} rows, test={len(X_test)} rows")

# choose the tree size on the validation split, the test split stays untouched
print(f"\n{'max_depth':>9} | {'leaves':>6} | {'fit acc':>7} | {'val acc':>7}")
print("-" * 40)
scores = {}
for depth in [1, 2, 3, 4, 5, 6, 8, None]:
    model = DecisionTreeClassifier(max_depth=depth).fit(X_fit, y_fit)
    scores[depth] = model.score(X_val, y_val)
    print(f"{str(depth):>9} | {model.get_n_leaves():>6} | "
          f"{model.score(X_fit, y_fit):>7.3f} | {scores[depth]:>7.3f}")
best_depth = max(scores, key=scores.get)  # first (smallest) depth reaching the best score
print("-> training accuracy keeps rising with depth (the tree memorises the 5% label noise),")
print(f"   validation accuracy does not: best max_depth = {best_depth}")

best = DecisionTreeClassifier(max_depth=best_depth).fit(X_train, y_train)
print(f"\nFinal model: {best}")
print(f"depth={best.get_depth()}  leaves={best.get_n_leaves()}  "
      f"test accuracy={best.score(X_test, y_test):.3f}")
print("Confusion matrix (test):")
print(confusion_matrix(y_test, best.predict(X_test)))
print("\nFeature importances (share of total information gain):")
for name, value in sorted(best.feature_importances_.items(), key=lambda kv: -kv[1]):
    print(f"  {name:<16} {value:.3f}")

small = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5).fit(X_train, y_train)
print(f"\nA small, readable tree (max_depth=3, min_samples_leaf=5): "
      f"test accuracy={small.score(X_test, y_test):.3f}")
small.print_tree(decimals=2)

tiny = DecisionTreeClassifier(max_leaf_nodes=4).fit(X_train, y_train)
print(f"\nWith max_leaf_nodes=4 (best-first growth): leaves={tiny.get_n_leaves()}, "
      f"test accuracy={tiny.score(X_test, y_test):.3f}")
tiny.print_tree(decimals=2)

with open("loan_tree.dot", "w", encoding="utf-8") as f:
    f.write(small.to_dot())
print("\nGraphviz source written to loan_tree.dot (render: dot -Tpng loan_tree.dot -o loan_tree.png)")

if PLOT:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tree.plot().figure.savefig("play_tennis_tree.png", dpi=120, bbox_inches="tight")
    small.plot(fontsize=8).figure.savefig("loan_tree.png", dpi=120, bbox_inches="tight")
    plt.close("all")
    print("Plots saved to play_tennis_tree.png and loan_tree.png")
