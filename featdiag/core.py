"""
Core DiagnosticFramework class and DiagnosticResult dataclass.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .metrics import (
    compute_ablation_auc,
    compute_lime_stability,
    compute_logistic_or,
    compute_permutation_rank,
    compute_shap,
)
from .plot import plot_result
from .rules import classify, interpret

warnings.filterwarnings("ignore")

_XGB_DEFAULTS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42,
    verbosity=0,
    use_label_encoder=False,
)


@dataclass
class DiagnosticResult:
    """All metrics and the final classification for one feature."""

    # Identifiers
    feature: str
    outcome: str = "outcome"

    # Sample info
    n_samples: int = 0
    n_events: int = 0
    n_features: int = 0

    # Model performance
    auc_full: float = float("nan")
    auc_no_feat: float = float("nan")

    # SHAP
    shap_rank: int = 0
    shap_value: float = float("nan")
    all_shap: Dict[str, float] = field(default_factory=dict)
    shap_values: Optional[np.ndarray] = field(default=None, repr=False)
    X: Optional[pd.DataFrame] = field(default=None, repr=False)

    # Permutation importance
    perm_rank: int = 0
    perm_delta_auc: float = float("nan")
    all_perm: Dict[str, float] = field(default_factory=dict)

    # Ablation ΔAUC (used in diagnostic rule)
    delta_auc_ablation: float = float("nan")
    delta_auc_se: float = float("nan")

    # LIME stability
    lime_rank_mean: float = float("nan")
    lime_rank_sd: float = float("nan")
    lime_rank_min: int = 0
    lime_rank_max: int = 0
    lime_ranks: List[int] = field(default_factory=list)
    lime_stable: bool = False

    # Logistic regression
    or_value: float = float("nan")
    or_ci_lower: float = float("nan")
    or_ci_upper: float = float("nan")
    p_value: float = float("nan")
    or_significant: bool = False

    # Classification
    threshold: float = 0.015
    classification: str = ""
    interpretation: str = ""

    def plot(self, title: Optional[str] = None, **kwargs):
        """Return a matplotlib Figure summarising the result."""
        return plot_result(self, title=title, **kwargs)

    def save_plot(self, path: str, **kwargs):
        """Save the summary figure to a file."""
        fig = self.plot(**kwargs)
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        fig.clf()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable summary dictionary."""
        return {
            "feature": self.feature,
            "outcome": self.outcome,
            "n_samples": self.n_samples,
            "n_events": self.n_events,
            "n_features": self.n_features,
            "auc_full": round(self.auc_full, 4),
            "auc_no_feat": round(self.auc_no_feat, 4),
            "shap_rank": self.shap_rank,
            "shap_value": round(self.shap_value, 4),
            "perm_rank": self.perm_rank,
            "perm_delta_auc": round(self.perm_delta_auc, 4),
            "delta_auc_ablation": round(self.delta_auc_ablation, 4),
            "delta_auc_se": round(self.delta_auc_se, 4),
            "lime_rank_mean": round(self.lime_rank_mean, 2) if self.lime_ranks else None,
            "lime_rank_sd": round(self.lime_rank_sd, 2) if self.lime_ranks else None,
            "or_value": round(self.or_value, 3),
            "or_ci_lower": round(self.or_ci_lower, 3),
            "or_ci_upper": round(self.or_ci_upper, 3),
            "p_value": round(self.p_value, 4),
            "or_significant": self.or_significant,
            "threshold": self.threshold,
            "classification": self.classification,
        }

    def summary(self):
        """Print a formatted summary to stdout."""
        n = self.n_features
        lime_str = (
            f"{self.lime_rank_mean:.1f}±{self.lime_rank_sd:.1f}/{n} "
            f"(range {self.lime_rank_min}–{self.lime_rank_max})"
            if self.lime_ranks else "not computed (run with run_lime=True)"
        )
        print(f"\n{'='*60}")
        print(f"Feature Diagnostic: '{self.feature}' → '{self.outcome}'")
        print(f"{'='*60}")
        print(f"  Dataset:       n={self.n_samples}, events={self.n_events} "
              f"({self.n_events/self.n_samples*100:.1f}%)")
        print(f"  Model AUC:     {self.auc_full:.3f}  (without {self.feature}: {self.auc_no_feat:.3f})")
        print(f"  SHAP rank:     {self.shap_rank}/{n}  (|SHAP|={self.shap_value:.3f})")
        print(f"  Perm. rank:    {self.perm_rank}/{n}  (ΔAUC={self.perm_delta_auc:+.4f})")
        print(f"  LIME rank:     {lime_str}")
        print(f"  Ablation ΔAUC: {self.delta_auc_ablation:+.4f}  (threshold={self.threshold})")
        print(f"  OR:            {self.or_value:.3f} [{self.or_ci_lower:.3f}–{self.or_ci_upper:.3f}], "
              f"p={self.p_value:.4f}")
        print(f"\n  Classification: *** {self.classification} ***")
        print(f"  {self.interpretation}")
        print(f"{'='*60}\n")


class DiagnosticFramework:
    """
    Three-metric diagnostic framework for clinical ML feature classification.

    Classifies a feature as INDEPENDENT, SUBSTITUTION, or REDUNDANT based on
    the joint verdict of SHAP, permutation-based ΔAUC, and logistic regression.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (numeric, complete-case).
    y : array-like
        Binary outcome vector (0/1).
    feature : str
        Name of the column in X to diagnose (must be in X.columns).
    outcome : str
        Label for the outcome (used in plots and summaries).
    model_class : class, optional
        Sklearn-compatible classifier. Defaults to XGBClassifier.
    model_params : dict, optional
        Parameters passed to model_class. Defaults to paper XGBoost settings.
    cv : int
        Number of cross-validation folds. Default 5.
    threshold : float
        ΔAUC threshold separating INDEPENDENT from SUBSTITUTION. Default 0.015.
    random_state : int
        Global random seed. Default 42.
    alpha : float
        Significance level for logistic regression OR. Default 0.05.

    Examples
    --------
    >>> from featdiag import DiagnosticFramework
    >>> fw = DiagnosticFramework(X, y, feature="pil", outcome="CVD")
    >>> result = fw.fit()
    >>> result.summary()
    >>> result.plot()
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y,
        feature: str,
        outcome: str = "outcome",
        model_class=None,
        model_params: Optional[dict] = None,
        cv: int = 5,
        threshold: float = 0.015,
        random_state: int = 42,
        alpha: float = 0.05,
    ):
        if model_class is None:
            try:
                from xgboost import XGBClassifier
                model_class = XGBClassifier
            except ImportError:
                raise ImportError("xgboost is required: pip install xgboost")

        if feature not in X.columns:
            raise ValueError(f"Feature '{feature}' not found in X.columns: {list(X.columns)}")

        self.X = X.copy().reset_index(drop=True)
        self.y = np.asarray(y)
        self.feature = feature
        self.outcome = outcome
        self.model_class = model_class
        self.random_state = random_state
        self.cv = cv
        self.threshold = threshold
        self.alpha = alpha

        # Auto-set scale_pos_weight for XGBoost if not provided
        if model_params is None:
            spw = (self.y == 0).sum() / max((self.y == 1).sum(), 1)
            self.model_params = {**_XGB_DEFAULTS, "scale_pos_weight": spw,
                                 "random_state": random_state}
        else:
            self.model_params = model_params

        self._clf_full = None

    def _fit_full_model(self):
        if self._clf_full is None:
            self._clf_full = self.model_class(**self.model_params)
            self._clf_full.fit(self.X, self.y)
        return self._clf_full

    def fit(
        self,
        run_lime: bool = False,
        lime_seeds: int = 10,
        lime_samples: int = 300,
        store_shap: bool = True,
        verbose: bool = True,
    ) -> DiagnosticResult:
        """
        Run the full three-metric diagnostic pipeline.

        Parameters
        ----------
        run_lime : bool
            Whether to run LIME stability analysis (slow, ~5 min). Default False.
        lime_seeds : int
            Number of random seeds for LIME stability. Default 10.
        lime_samples : int
            Number of instances per LIME seed. Default 300.
        store_shap : bool
            Whether to store SHAP values in the result (needed for beeswarm plot).
        verbose : bool
            Print progress. Default True.

        Returns
        -------
        DiagnosticResult
        """
        result = DiagnosticResult(
            feature=self.feature,
            outcome=self.outcome,
            n_samples=len(self.y),
            n_events=int(self.y.sum()),
            n_features=len(self.X.columns),
            threshold=self.threshold,
        )

        # Step 1: Full model CV AUC
        if verbose:
            print(f"[1/5] Computing 5-fold CV AUC for full model...")
        clf = self._fit_full_model()
        cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                      random_state=self.random_state)
        oof = cross_val_predict(self.model_class(**self.model_params),
                                self.X, self.y, cv=cv_splitter,
                                method="predict_proba")[:, 1]
        result.auc_full = float(roc_auc_score(self.y, oof))
        if verbose:
            print(f"    AUC (full) = {result.auc_full:.3f}")

        # Step 2: SHAP
        if verbose:
            print(f"[2/5] Computing SHAP attributions...")
        shap_out = compute_shap(clf, self.X, self.feature)
        result.shap_rank = shap_out["feature_rank"]
        result.shap_value = shap_out["feature_shap"]
        result.all_shap = shap_out["all_shap"]
        if store_shap:
            result.shap_values = shap_out["shap_values"]
            result.X = self.X
        if verbose:
            print(f"    {self.feature}: rank {result.shap_rank}/{result.n_features}, "
                  f"|SHAP|={result.shap_value:.3f}")

        # Step 3: Permutation importance
        if verbose:
            print(f"[3/5] Computing permutation importance...")
        perm_out = compute_permutation_rank(clf, self.X, self.y, self.feature,
                                            random_state=self.random_state)
        result.perm_rank = perm_out["feature_perm_rank"]
        result.perm_delta_auc = perm_out["feature_perm_delta"]
        result.all_perm = perm_out["all_perm"]
        if verbose:
            print(f"    {self.feature}: rank {result.perm_rank}/{result.n_features}, "
                  f"ΔAUC={result.perm_delta_auc:+.4f}")

        # Step 4: Ablation ΔAUC (used in diagnostic rule)
        if verbose:
            print(f"[4/5] Computing ablation ΔAUC (5-fold, retraining without {self.feature})...")
        abl_out = compute_ablation_auc(
            self.X, self.y, self.feature,
            self.model_class, self.model_params,
            cv=self.cv, random_state=self.random_state,
        )
        result.auc_no_feat = abl_out["auc_no_feat"]
        result.delta_auc_ablation = abl_out["delta_auc"]
        result.delta_auc_se = abl_out["delta_auc_se"]
        if verbose:
            print(f"    Ablation ΔAUC = {result.delta_auc_ablation:+.4f} "
                  f"(threshold={self.threshold})")

        # Step 5: Logistic regression OR
        if verbose:
            print(f"[5/5] Fitting logistic regression...")
        lr_out = compute_logistic_or(self.X, self.y, self.feature, alpha=self.alpha)
        result.or_value = lr_out["or_value"]
        result.or_ci_lower = lr_out["ci_lower"]
        result.or_ci_upper = lr_out["ci_upper"]
        result.p_value = lr_out["p_value"]
        result.or_significant = lr_out["significant"]
        if verbose:
            print(f"    OR = {result.or_value:.3f} [{result.or_ci_lower:.3f}–"
                  f"{result.or_ci_upper:.3f}], p={result.p_value:.4f}")

        # Optional: LIME
        if run_lime:
            if verbose:
                print(f"[+] Running LIME stability ({lime_seeds} seeds × {lime_samples} samples)...")
            lime_out = compute_lime_stability(
                clf, self.X, self.feature,
                n_seeds=lime_seeds, n_samples=lime_samples,
                verbose=verbose,
            )
            result.lime_rank_mean = lime_out["rank_mean"]
            result.lime_rank_sd = lime_out["rank_sd"]
            result.lime_rank_min = lime_out["rank_min"]
            result.lime_rank_max = lime_out["rank_max"]
            result.lime_ranks = lime_out["ranks"]
            result.lime_stable = lime_out["stable"]
            if verbose:
                print(f"    LIME rank: {result.lime_rank_mean:.1f}±{result.lime_rank_sd:.1f} "
                      f"(range {result.lime_rank_min}–{result.lime_rank_max})")

        # Apply diagnostic rule
        result.classification = classify(
            result.or_significant, result.delta_auc_ablation, self.threshold
        )
        result.interpretation = interpret(result.classification)

        if verbose:
            print(f"\n  *** Classification: {result.classification} ***")

        return result
