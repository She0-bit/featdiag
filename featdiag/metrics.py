"""
Metric computation for the three-metric diagnostic framework.

Each function is independent and can be used separately.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ── SHAP ─────────────────────────────────────────────────────────────────────

def compute_shap(clf, X: pd.DataFrame, feature: str) -> dict:
    """
    Compute mean |SHAP| values and rank for all features.

    Requires the model to have been fitted and `shap` to be installed.
    Uses TreeExplainer for XGBoost/LightGBM/sklearn trees; falls back to
    KernelExplainer (slower) for other model types.

    Returns
    -------
    dict with keys:
        feature_shap : float — mean |SHAP| for the target feature
        feature_rank : int   — 1-based rank (1 = most important)
        all_shap     : dict  — {feature_name: mean_abs_shap} for all features
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap is required: pip install shap")

    try:
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X)
    except Exception:
        explainer = shap.KernelExplainer(clf.predict_proba, shap.sample(X, 100))
        shap_vals = explainer.shap_values(X)[1]

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0), index=X.columns)
    ranked = mean_abs.sort_values(ascending=False)
    rank = list(ranked.index).index(feature) + 1

    return {
        "feature_shap": float(mean_abs[feature]),
        "feature_rank": rank,
        "all_shap": mean_abs.to_dict(),
        "shap_values": shap_vals,
    }


# ── Ablation ΔAUC ─────────────────────────────────────────────────────────────

def compute_ablation_auc(
    X: pd.DataFrame,
    y: np.ndarray,
    feature: str,
    model_class,
    model_params: dict,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Compute 5-fold CV AUC with and without the target feature.

    The full model and the model without the feature are both trained from
    scratch within each fold to ensure fair comparison.

    Returns
    -------
    dict with keys:
        auc_full     : float — mean 5-fold CV AUC with all features
        auc_no_feat  : float — mean 5-fold CV AUC without the target feature
        delta_auc    : float — auc_full - auc_no_feat (positive = feature helps)
        fold_deltas  : list  — per-fold ΔAUC values
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    features_no_target = [f for f in X.columns if f != feature]

    aucs_full, aucs_drop = [], []

    for train_idx, val_idx in cv_splitter.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        m_full = model_class(**model_params)
        m_full.fit(X_tr, y_tr)
        aucs_full.append(roc_auc_score(y_val, m_full.predict_proba(X_val)[:, 1]))

        m_drop = model_class(**model_params)
        m_drop.fit(X_tr[features_no_target], y_tr)
        aucs_drop.append(roc_auc_score(y_val, m_drop.predict_proba(X_val[features_no_target])[:, 1]))

    fold_deltas = [a - b for a, b in zip(aucs_full, aucs_drop)]
    return {
        "auc_full": float(np.mean(aucs_full)),
        "auc_no_feat": float(np.mean(aucs_drop)),
        "delta_auc": float(np.mean(fold_deltas)),
        "fold_deltas": fold_deltas,
        "delta_auc_se": float(np.std(fold_deltas, ddof=1) / np.sqrt(cv)),
    }


# ── Permutation importance ────────────────────────────────────────────────────

def compute_permutation_rank(
    clf,
    X: pd.DataFrame,
    y: np.ndarray,
    feature: str,
    n_repeats: int = 30,
    random_state: int = 42,
) -> dict:
    """
    Compute permutation importance for all features and return rank of target.

    Returns
    -------
    dict with keys:
        feature_perm_delta : float — mean ΔAUC from permuting the target feature
        feature_perm_rank  : int   — 1-based rank among all features
        all_perm           : dict  — {feature_name: mean_delta_auc}
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        clf, X, y, n_repeats=n_repeats,
        random_state=random_state, scoring="roc_auc"
    )
    means = pd.Series(result.importances_mean, index=X.columns)
    ranked = means.sort_values(ascending=False)
    rank = list(ranked.index).index(feature) + 1

    return {
        "feature_perm_delta": float(means[feature]),
        "feature_perm_rank": rank,
        "all_perm": means.to_dict(),
    }


# ── Logistic regression OR ────────────────────────────────────────────────────

def compute_logistic_or(
    X: pd.DataFrame,
    y: np.ndarray,
    feature: str,
    alpha: float = 0.05,
) -> dict:
    """
    Fit a logistic regression on standardised features and return OR, 95% CI,
    and p-value for the target feature using statsmodels (Wald inference).

    Returns
    -------
    dict with keys:
        or_value    : float — odds ratio for the target feature
        ci_lower    : float — 95% CI lower bound
        ci_upper    : float — 95% CI upper bound
        p_value     : float — two-sided Wald p-value
        significant : bool  — p_value < alpha
        coef        : float — log-odds coefficient
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels is required: pip install statsmodels")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_sm = sm.add_constant(X_scaled)

    model = sm.Logit(y, X_sm)
    result = model.fit(disp=False)

    feat_idx = list(X.columns).index(feature) + 1  # +1 for constant
    coef = result.params[feat_idx]
    se = result.bse[feat_idx]
    p = result.pvalues[feat_idx]
    z = 1.96  # 95% CI

    return {
        "or_value": float(np.exp(coef)),
        "ci_lower": float(np.exp(coef - z * se)),
        "ci_upper": float(np.exp(coef + z * se)),
        "p_value": float(p),
        "significant": bool(p < alpha),
        "coef": float(coef),
    }


# ── LIME stability ────────────────────────────────────────────────────────────

def compute_lime_stability(
    clf,
    X: pd.DataFrame,
    feature: str,
    n_seeds: int = 10,
    n_samples: int = 300,
    lime_samples: int = 300,
    class_names: Optional[List[str]] = None,
    verbose: bool = False,
) -> dict:
    """
    Run LIME across multiple random seeds and report rank stability for the
    target feature. Each seed uses a different random subsample of training
    instances and a different LIME random state.

    Returns
    -------
    dict with keys:
        rank_mean   : float — mean rank across seeds (1 = most important)
        rank_sd     : float — standard deviation of ranks
        rank_min    : int
        rank_max    : int
        ranks       : list  — per-seed ranks
        stable      : bool  — True if rank_sd <= 1.0
    """
    try:
        import lime
        import lime.lime_tabular
    except ImportError:
        raise ImportError("lime is required: pip install lime")

    features = list(X.columns)
    if class_names is None:
        class_names = ["negative", "positive"]

    ranks = []
    for seed in range(n_seeds):
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=features,
            class_names=class_names,
            mode="classification",
            discretize_continuous=True,
            random_state=seed,
        )
        rng = np.random.RandomState(seed + 100)
        idxs = rng.choice(len(X), size=n_samples, replace=False)

        weights = {f: [] for f in features}
        for idx in idxs:
            exp = explainer.explain_instance(
                X.values[idx],
                clf.predict_proba,
                num_features=len(features),
                num_samples=lime_samples,
            )
            for feat_label, w in exp.as_list():
                for f in features:
                    if f in feat_label:
                        weights[f].append(abs(w))
                        break

        mean_w = {f: np.mean(v) if v else 0.0 for f, v in weights.items()}
        ranked = sorted(mean_w, key=lambda f: mean_w[f], reverse=True)
        rank = ranked.index(feature) + 1
        ranks.append(rank)
        if verbose:
            print(f"  LIME seed={seed}: {feature} rank {rank}/{len(features)}")

    rank_sd = float(np.std(ranks, ddof=1)) if len(set(ranks)) > 1 else 0.0

    return {
        "rank_mean": float(np.mean(ranks)),
        "rank_sd": rank_sd,
        "rank_min": int(min(ranks)),
        "rank_max": int(max(ranks)),
        "ranks": ranks,
        "stable": rank_sd <= 1.0,
    }
