"""
ml_model.py -- Phases 8, 9, 10: ML prediction of simulation-derived influence.

    X = SNA features (structure only)   ->   y = mean_spread (Monte Carlo label)

Leakage control: labelled nodes are split into TRAIN/TEST. Models are fitted on train
labels only and evaluated on held-out test labels. The target never appears in X and
raw community_id is excluded.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .feature_engineering import ML_FEATURES, assert_no_leakage

TARGET = "mean_spread"


def build_ml_dataset(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join features with labels on node_id (only labelled candidate nodes are kept)."""
    assert_no_leakage(ML_FEATURES)
    return features.merge(labels[["node_id", TARGET, "std_spread"]], on="node_id", how="inner")


def split_data(ds: pd.DataFrame, test_size: float, random_state: int):
    """Reproducible train/test split by node."""
    train, test = train_test_split(ds, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _models(random_state: int, quick: bool) -> dict:
    n_est = 150 if quick else 300
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=n_est, min_samples_leaf=2,
                                               random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=n_est, max_depth=3,
                                                       learning_rate=0.05, subsample=0.8,
                                                       random_state=random_state),
    }


def topk_overlap(actual: np.ndarray, pred: np.ndarray, k: int) -> float:
    """|top-k(actual) ∩ top-k(pred)| / k  (stable tie-breaking by position)."""
    k = min(k, len(actual))
    a = set(np.argsort(-actual, kind="stable")[:k])
    b = set(np.argsort(-pred, kind="stable")[:k])
    return len(a & b) / k


def _spearman(a, b) -> float:
    r = spearmanr(a, b)[0]
    return float(r) if np.isfinite(r) else 0.0


def train_and_evaluate(train: pd.DataFrame, test: pd.DataFrame, random_state: int,
                       quick: bool = False, topk_values=(5, 10, 20)):
    """Fit all models on TRAIN, score on held-out TEST. Returns (table, fitted, preds, topk_df)."""
    Xtr, ytr = train[ML_FEATURES], train[TARGET]
    Xte, yte = test[ML_FEATURES], test[TARGET].to_numpy()
    fitted, preds, rows, topk_rows = {}, {}, [], []
    for name, model in _models(random_state, quick).items():
        model.fit(Xtr, ytr)
        yp = model.predict(Xte)
        fitted[name], preds[name] = model, yp
        rows.append({"Model": name,
                     "MAE": mean_absolute_error(yte, yp),
                     "RMSE": float(np.sqrt(mean_squared_error(yte, yp))),
                     "R2": r2_score(yte, yp),
                     "Spearman": _spearman(yte, yp)})
        for k in topk_values:
            topk_rows.append({"Method": name, "K": k, "TopK_overlap": topk_overlap(yte, yp, k)})
    # Single-feature baselines on the same held-out nodes: ranking power of one centrality.
    for feat in ("degree", "pagerank", "betweenness"):
        v = test[feat].to_numpy()
        rows.append({"Model": f"[baseline] {feat} only", "MAE": np.nan, "RMSE": np.nan,
                     "R2": np.nan, "Spearman": _spearman(yte, v)})
        for k in topk_values:
            topk_rows.append({"Method": f"[baseline] {feat}", "K": k,
                              "TopK_overlap": topk_overlap(yte, v, k)})
    return pd.DataFrame(rows), fitted, preds, pd.DataFrame(topk_rows)


def select_best_model(table: pd.DataFrame) -> str:
    """
    Best model = lowest average rank over (RMSE asc, MAE asc, R2 desc, Spearman desc).
    Not hard-coded: whichever model really performs best on held-out nodes wins.
    """
    t = table[~table.Model.str.startswith("[baseline]")].set_index("Model")
    ranks = pd.DataFrame({"MAE": t.MAE.rank(), "RMSE": t.RMSE.rank(),
                          "R2": t.R2.rank(ascending=False), "Spearman": t.Spearman.rank(ascending=False)})
    return ranks.mean(axis=1).idxmin()


def feature_importance(fitted: dict, test: pd.DataFrame, best_name: str,
                       random_state: int) -> pd.DataFrame:
    """Impurity importance of tree models + permutation importance of the best model (test set)."""
    rows = []
    for name, m in fitted.items():
        if hasattr(m, "feature_importances_"):
            for f, imp in zip(ML_FEATURES, m.feature_importances_):
                rows.append({"model": name, "feature": f, "importance": imp, "type": "impurity"})
    pi = permutation_importance(fitted[best_name], test[ML_FEATURES], test[TARGET],
                                n_repeats=10, random_state=random_state)
    for f, imp in zip(ML_FEATURES, pi.importances_mean):
        rows.append({"model": best_name, "feature": f, "importance": imp, "type": "permutation(test)"})
    return pd.DataFrame(rows)


def save_model(model, name: str, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "name": name, "features": ML_FEATURES}, path)


def load_model(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Saved model '{path}' not found. Run main.py to train and save it.")
    return joblib.load(path)


def predict_all_nodes(bundle: dict, features: pd.DataFrame) -> pd.DataFrame:
    """AI influence score for EVERY node of the network (structure features only)."""
    pred = bundle["model"].predict(features[bundle["features"]])
    return pd.DataFrame({"node_id": features.node_id, "predicted_influence_score": pred})
