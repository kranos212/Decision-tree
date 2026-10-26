"""entropy_tree: a from-scratch decision tree classifier based on entropy and information gain."""

from .tree import DecisionTreeClassifier, Node, any_df_decision_tree, entropy, information_gain
from .utils import (
    accuracy_score,
    confusion_matrix,
    load_loan_approval,
    load_play_tennis,
    train_test_split,
)

__version__ = "0.1.2"

__all__ = [
    "DecisionTreeClassifier",
    "Node",
    "any_df_decision_tree",
    "entropy",
    "information_gain",
    "accuracy_score",
    "confusion_matrix",
    "load_loan_approval",
    "load_play_tennis",
    "train_test_split",
]
