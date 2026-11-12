"""
teamId_classifier_ovo.py
One-vs-One multiclass classifier built from binary SVMs.

Usage:
    python teamId_classifier_ovo.py <path_to_test_file> [--retune]

Output:
    ovo.csv (single column `predicted`, one row per test row, same order)

Design:
    - Preprocessing is a sklearn ColumnTransformer (median/most-frequent
      imputation, ordinal Spending_Score Low<Average<High, one-hot nominals,
      standard scaling) wrapped in a Pipeline together with the classifier, so
      it is re-fitted inside every CV fold and on the full training set for the
      final model (no leakage of fold-validation rows into the imputers/scaler).
    - K(K-1)/2 binary SVC models, each trained only on the rows of its two
      classes (every SVM sees a binary target).
    - Prediction: majority vote. Ties are broken by the summed signed decision
      scores (sum over all pairwise SVMs of the margin in favour of each class);
      if still tied, by the class with the larger training prior, and finally by
      a seeded random choice among the remaining tied classes (reproducible via
      random_state, and not systematically biased toward any label).
    - Hyperparameters (kernel, C, gamma, class_weight) are selected by stratified
      5-fold CV using macro-F1 (the classes are imbalanced); accuracy is reported
      too. The full grid table is written to ovo_grid_results.csv and the chosen
      parameters are cached in ovo_best_params.json (re-run the search with
      --retune).
"""

import os
import sys
import json
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(HERE, "Customer_train.csv")
GRID_CSV = os.path.join(HERE, "ovo_grid_results.csv")
PARAMS_JSON = os.path.join(HERE, "ovo_best_params.json")
OUT_CSV = "ovo.csv"

TARGET_COL = "Segmentation"
NUMERIC_COLS = ["Age", "Work_Experience", "Family_Size"]
ORDINAL_COLS = ["Spending_Score"]
ORDINAL_ORDER = [["Low", "Average", "High"]]
NOMINAL_COLS = ["Gender", "Ever_Married", "Graduated", "Profession", "Var_1"]
FEATURE_COLS = NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS


# ─────────────────────── preprocessing ───────────────────────
def build_preprocessor():
    """Unfitted ColumnTransformer; fit it on training rows only."""
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    ordi = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                     ("encode", OrdinalEncoder(categories=ORDINAL_ORDER)),
                     ("scale", StandardScaler())])
    nom = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore",
                                             sparse_output=False)),
                    ("scale", StandardScaler())])
    return ColumnTransformer([("num", num, NUMERIC_COLS),
                              ("ord", ordi, ORDINAL_COLS),
                              ("nom", nom, NOMINAL_COLS)])


def load_xy(path):
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(str).to_numpy() if TARGET_COL in df else None
    return X, y


# ─────────────────────── OvO classifier ──────────────────────
class OneVsOneSVM(BaseEstimator, ClassifierMixin):
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
        self.models_ = {}
        for ia, ib in itertools.combinations(range(len(self.classes_)), 2):
            mask = (y == self.classes_[ia]) | (y == self.classes_[ib])
            svm = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma,
                      degree=self.degree, class_weight=self.class_weight)
            svm.fit(X[mask], y[mask])          # binary target: two labels only
            self.models_[(ia, ib)] = svm
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        n, K = X.shape[0], len(self.classes_)
        votes = np.zeros((n, K))
        score_sums = np.zeros((n, K))
        for (ia, ib), svm in self.models_.items():
            # sklearn binary SVC: decision_function > 0  =>  svm.classes_[1].
            # svm.classes_ is sorted, so classes_[1] == self.classes_[ib].
            assert svm.classes_[1] == self.classes_[ib]
            df = svm.decision_function(X)
            votes[df > 0, ib] += 1
            votes[df <= 0, ia] += 1
            score_sums[:, ib] += df     # margin in favour of cb
            score_sums[:, ia] -= df     # margin in favour of ca
        out = np.empty(n, dtype=self.classes_.dtype)
        rng = np.random.default_rng(42)
        for k in range(n):
            cand = np.flatnonzero(votes[k] == votes[k].max())
            if len(cand) > 1:   # tie on votes -> summed decision score
                s = score_sums[k, cand]
                cand = cand[np.isclose(s, s.max())]
            if len(cand) > 1:   # still tied -> larger training prior
                p = self.prior_[cand]
                cand = cand[np.isclose(p, p.max())]
            # Final tie-break: seeded random choice among remaining candidates.
            if len(cand) > 1:
                cand = [rng.choice(cand)]
            out[k] = self.classes_[cand[0]]
        return out


def make_pipeline(**params):
    return Pipeline([("prep", build_preprocessor()),
                     ("clf", OneVsOneSVM(**params))])


# ──────────────────── hyperparameter search ────────────────────
C_VALUES = [0.1, 1.0, 10.0]
GAMMA_VALUES = [0.01, 0.1, 1.0]
CW_VALUES = [None, "balanced"]


def param_grid():
    grid = []
    for cw in CW_VALUES:
        for C in C_VALUES:
            grid.append(dict(kernel="linear", C=C, gamma="scale", degree=3,
                             class_weight=cw))
            for g in GAMMA_VALUES:
                grid.append(dict(kernel="rbf", C=C, gamma=g, degree=3,
                                 class_weight=cw))
            grid.append(dict(kernel="poly", C=C, gamma="scale", degree=2,
                             class_weight=cw))
    return grid


def cv_evaluate(make_model, X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for tr, va in skf.split(X, y):
        m = make_model().fit(X.iloc[tr], y[tr])
        p = m.predict(X.iloc[va])
        accs.append(accuracy_score(y[va], p))
        f1s.append(f1_score(y[va], p, average="macro"))
    return np.array(accs), np.array(f1s)


def tune(X, y, pipeline_fn, grid_csv, name):
    rows = []
    for params in param_grid():
        accs, f1s = cv_evaluate(lambda: pipeline_fn(**params), X, y)
        row = dict(params, class_weight=str(params["class_weight"]),
                   macro_f1_mean=f1s.mean(), macro_f1_std=f1s.std(),
                   acc_mean=accs.mean(), acc_std=accs.std())
        rows.append(row)
        print(f"  {params['kernel']:6s} C={params['C']:<5} "
              f"gamma={str(params['gamma']):<6} cw={str(params['class_weight']):8s} "
              f"macroF1={f1s.mean():.4f}±{f1s.std():.4f} "
              f"acc={accs.mean():.4f}±{accs.std():.4f}", flush=True)
    res = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
    res.to_csv(grid_csv, index=False)
    print(f"[{name}] grid results saved to {os.path.basename(grid_csv)}")
    best = res.iloc[0]
    return dict(kernel=best["kernel"], C=float(best["C"]),
                gamma=(best["gamma"] if best["gamma"] == "scale"
                       else float(best["gamma"])),
                degree=int(best["degree"]),
                class_weight=None if best["class_weight"] == "None"
                else best["class_weight"])


def get_best_params(X, y, pipeline_fn, grid_csv, params_json, name, retune):
    if os.path.exists(params_json) and not retune:
        with open(params_json) as f:
            params = json.load(f)
        print(f"[{name}] using cached params from {os.path.basename(params_json)}"
              " (pass --retune to redo the grid search)")
        if os.path.exists(grid_csv):
            # keep_default_na=False so class_weight "None" is shown as None, not NaN
            print(pd.read_csv(grid_csv, keep_default_na=False).to_string(index=False))
    else:
        print(f"[{name}] grid search, stratified 5-fold CV, selection by macro-F1")
        params = tune(X, y, pipeline_fn, grid_csv, name)
        with open(params_json, "w") as f:
            json.dump(params, f, indent=2)
    print(f"[{name}] chosen params: {params}")
    return params


# ─────────────────────────── main ────────────────────────────
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print("Usage: python teamId_classifier_ovo.py <path_to_test_file> [--retune]")
        sys.exit(1)
    X, y = load_xy(TRAIN_FILE)
    params = get_best_params(X, y, make_pipeline, GRID_CSV, PARAMS_JSON,
                             "OvO", "--retune" in sys.argv)
    model = make_pipeline(**params).fit(X, y)
    X_test, _ = load_xy(args[0])
    preds = model.predict(X_test)
    pd.DataFrame({"predicted": preds}).to_csv(OUT_CSV, index=False)
    print(f"[OvO] wrote {OUT_CSV} ({len(preds)} rows); distribution: "
          f"{pd.Series(preds).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
