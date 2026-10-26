"""Regenerate the synthetic loan_approval.csv dataset shipped with the package.

The label follows known rules (plus 5% label noise), so we can check that the
tree recovers them:
    - credit_score < 580                              -> rejected
    - otherwise income >= 3 * monthly_payment needed  -> approved
    - employment == "unemployed" with no guarantor    -> rejected
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 600

df = pd.DataFrame({
    "age": rng.integers(19, 70, n),
    "income": rng.normal(3200, 1100, n).clip(800).round(-1),
    "credit_score": rng.normal(660, 75, n).clip(350, 850).round().astype(int),
    "monthly_payment": rng.uniform(300, 1600, n).round(-1),
    "employment": rng.choice(["employed", "self-employed", "unemployed"], n, p=[0.6, 0.25, 0.15]),
    "guarantor": rng.choice(["yes", "no"], n, p=[0.3, 0.7]),
})

approved = (
    (df["credit_score"] >= 580)
    & (df["income"] >= 3 * df["monthly_payment"])
    & ~((df["employment"] == "unemployed") & (df["guarantor"] == "no"))
)
noise = rng.random(n) < 0.05
df["approved"] = np.where(approved ^ noise, "yes", "no")

out = Path(__file__).resolve().parents[1] / "src" / "entropy_tree" / "datasets" / "loan_approval.csv"
df.to_csv(out, index=False)
print(f"wrote {len(df)} rows to {out}")
print(df["approved"].value_counts())
