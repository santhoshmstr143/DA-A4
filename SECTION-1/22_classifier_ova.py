"""
22_classifier_ova.py
One-vs-All (one-vs-rest) multiclass classifier built from binary SVMs.

Usage:
    python 22_classifier_ova.py <path_to_test_file> [--retune]

Output:
    ova.csv (single column `predicted`, one row per test row, same order)

Design:
    - Same leak-free preprocessing Pipeline as the OvO script (imported from
      22_classifier_ovo.py located next to this file): imputation,
      ordinal Spending_Score, one-hot nominals and scaling are fitted inside
      each CV fold and on the full training set for the final model.
    - K binary SVC models, model k trained on target (y == class_k) vs rest.
    - Prediction: class with the highest decision score; exact ties broken by
      the larger training prior, then by a seeded random choice among the
      remaining tied classes (reproducible, not biased toward any label).
    - class_weight="balanced" here re-weights the 1-vs-rest binary problem,
      which is exactly where OvA suffers most from imbalance (each binary
      problem is 1:2..1:4 skewed).
    - Grid (kernel, C, gamma, class_weight) searched with stratified 5-fold
      CV, selected by macro-F1; table in ova_grid_results.csv, chosen params
      cached in ova_best_params.json (re-run with --retune).
"""

import os
import sys
import importlib.util
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "22_classifier_ovo", os.path.join(HERE, "22_classifier_ovo.py"))
ovo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ovo)

TRAIN_FILE = ovo.TRAIN_FILE
GRID_CSV = os.path.join(HERE, "ova_grid_results.csv")
PARAMS_JSON = os.path.join(HERE, "ova_best_params.json")
OUT_CSV = "ova.csv"
build_preprocessor = ovo.build_preprocessor
load_xy = ovo.load_xy


class OneVsAllSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel="rbf", C=1.0, gamma="scale", degree=3,
                 class_weight=None, random_state=0):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.prior_ = np.array([(y == c).mean() for c in self.classes_])
        self.models_ = []
        for c in self.classes_:
            yb = (y == c).astype(int)            # binary target: 1 = class c
            svm = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma,
                      degree=self.degree, class_weight=self.class_weight)
            svm.fit(X, yb)
            self.models_.append(svm)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        # positive score => svm.classes_[1] == 1 == "is class c"
        return np.column_stack([m.decision_function(X) for m in self.models_])

    def predict(self, X):
        S = self.decision_function(X)
        out = np.empty(S.shape[0], dtype=self.classes_.dtype)
        rng = np.random.default_rng(42)
        for k, row in enumerate(S):
            cand = np.flatnonzero(np.isclose(row, row.max()))
            if len(cand) > 1:   # tie -> larger training prior
                p = self.prior_[cand]
                cand = cand[np.isclose(p, p.max())]
            if len(cand) > 1:   # still tied -> seeded random choice
                cand = [rng.choice(cand)]
            out[k] = self.classes_[cand[0]]
        return out


def make_pipeline(**params):
    return Pipeline([("prep", build_preprocessor()),
                     ("clf", OneVsAllSVM(**params))])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print("Usage: python 22_classifier_ova.py <path_to_test_file> [--retune]")
        sys.exit(1)
    X, y = load_xy(TRAIN_FILE)
    params = ovo.get_best_params(X, y, make_pipeline, GRID_CSV, PARAMS_JSON,
                                 "OvA", "--retune" in sys.argv)
    model = make_pipeline(**params).fit(X, y)
    X_test, _ = load_xy(args[0])
    preds = model.predict(X_test)
    pd.DataFrame({"predicted": preds}).to_csv(OUT_CSV, index=False)
    print(f"[OvA] wrote {OUT_CSV} ({len(preds)} rows); distribution: "
          f"{pd.Series(preds).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
