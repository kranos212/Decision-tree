import math
import unittest

import numpy as np
import pandas as pd

from entropy_tree import (
    DecisionTreeClassifier,
    any_df_decision_tree,
    entropy,
    information_gain,
    load_loan_approval,
    load_play_tennis,
    train_test_split,
)


class TestImpurity(unittest.TestCase):
    def test_entropy(self):
        self.assertEqual(entropy(["a", "a", "a"]), 0.0)
        self.assertAlmostEqual(entropy(["a", "b"]), 1.0)
        self.assertAlmostEqual(entropy(list("aabbccdd")), 2.0)

    def test_play_tennis_gains_match_textbook(self):
        df = load_play_tennis()
        self.assertAlmostEqual(entropy(df["play"]), 0.940, places=3)
        gain = lambda col: information_gain(df["play"], [g["play"] for _, g in df.groupby(col)])
        self.assertAlmostEqual(gain("outlook"), 0.247, places=3)
        self.assertAlmostEqual(gain("humidity"), 0.152, places=3)
        self.assertAlmostEqual(gain("windy"), 0.048, places=3)
        self.assertAlmostEqual(gain("temperature"), 0.029, places=3)


class TestTree(unittest.TestCase):
    def test_play_tennis_tree(self):
        df = load_play_tennis()
        tree = any_df_decision_tree(df, "play")
        self.assertEqual(tree.root_.feature, "outlook")
        self.assertTrue(tree.root_.children["overcast"].is_leaf)
        self.assertEqual(tree.root_.children["sunny"].feature, "humidity")
        self.assertEqual(tree.root_.children["rainy"].feature, "windy")
        self.assertEqual(tree.score(df.drop(columns="play"), df["play"]), 1.0)
        self.assertEqual(tree.get_depth(), 2)
        self.assertEqual(tree.get_n_leaves(), 5)

    def test_numeric_threshold(self):
        X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTreeClassifier().fit(X, y)
        self.assertAlmostEqual(tree.root_.threshold, 6.5)
        self.assertAlmostEqual(tree.root_.gain, 1.0)
        np.testing.assert_array_equal(tree.predict([[0.0], [6.0], [7.0], [100.0]]), [0, 0, 1, 1])

    def test_max_depth(self):
        df = load_loan_approval()
        for depth in [0, 1, 2, 3]:
            tree = DecisionTreeClassifier(max_depth=depth).fit(df, "approved")
            self.assertLessEqual(tree.get_depth(), depth)
        stump = DecisionTreeClassifier(max_depth=0).fit(df, "approved")
        self.assertTrue(stump.root_.is_leaf)

    def test_min_samples_leaf(self):
        df = load_loan_approval()
        tree = DecisionTreeClassifier(min_samples_leaf=20).fit(df, "approved")
        self.assertTrue(all(n.n_samples >= 20 for n in tree._nodes()))

    def test_min_samples_split(self):
        df = load_loan_approval()
        tree = DecisionTreeClassifier(min_samples_split=50).fit(df, "approved")
        self.assertTrue(all(n.n_samples >= 50 for n in tree._nodes() if not n.is_leaf))

    def test_min_information_gain(self):
        df = load_loan_approval()
        tree = DecisionTreeClassifier(min_information_gain=0.05).fit(df, "approved")
        self.assertTrue(all(n.gain >= 0.05 for n in tree._nodes() if not n.is_leaf))

    def test_max_leaf_nodes(self):
        df = load_loan_approval()
        for k in [1, 2, 5, 10]:
            tree = DecisionTreeClassifier(max_leaf_nodes=k).fit(df, "approved")
            self.assertLessEqual(tree.get_n_leaves(), k)

    def test_generalises_on_loans(self):
        df = load_loan_approval()
        X, y = df.drop(columns="approved"), df["approved"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
        tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5).fit(X_tr, y_tr)
        self.assertGreater(tree.score(X_te, y_te), 0.8)
        self.assertAlmostEqual(sum(tree.feature_importances_.values()), 1.0)

    def test_predict_proba_rows_sum_to_one(self):
        df = load_loan_approval()
        tree = DecisionTreeClassifier(max_depth=3).fit(df, "approved")
        proba = tree.predict_proba(df.drop(columns="approved"))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_unseen_category_falls_back_to_parent(self):
        df = load_play_tennis()
        tree = any_df_decision_tree(df, "play")
        row = df.drop(columns="play").iloc[[0]].assign(outlook="foggy")
        self.assertEqual(tree.predict(row)[0], "yes")  # majority class at the root

    def test_display(self):
        tree = any_df_decision_tree(load_play_tennis(), "play")
        text = tree.export_text()
        self.assertIn("split on 'outlook'", text)
        self.assertIn("outlook = overcast → predict 'yes'", text)
        ascii_text = tree.export_text(ascii=True)
        ascii_text.encode("ascii")  # must not raise
        self.assertIn("outlook = overcast -> predict 'yes'", ascii_text)
        dot = tree.to_dot()
        self.assertTrue(dot.startswith("digraph"))
        self.assertEqual(dot.count("->"), 7)

    def test_print_tree_on_cp1252_console(self):
        import io
        from contextlib import redirect_stdout

        tree = any_df_decision_tree(load_play_tennis(), "play")
        buffer = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        with redirect_stdout(buffer):
            tree.print_tree()  # used to raise UnicodeEncodeError
        buffer.seek(0)
        self.assertIn("|-- outlook = overcast", buffer.read())

    def test_errors(self):
        with self.assertRaises(RuntimeError):
            DecisionTreeClassifier().predict([[1]])
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(min_samples_split=1)
        with self.assertRaises(ValueError):
            DecisionTreeClassifier().fit(pd.DataFrame({"a": [1.0, math.nan]}), [0, 1])
        with self.assertRaises(KeyError):
            any_df_decision_tree(load_play_tennis(), "nope")


if __name__ == "__main__":
    unittest.main()
