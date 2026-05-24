"""
featdiag — Three-Metric Diagnostic Framework for Clinical ML

Classifies features in clinical XGBoost models as:
  INDEPENDENT  — genuine signal, both SHAP and ΔAUC confirm importance
  SUBSTITUTION — genuine signal masked by correlated substitutes (SHAP high, ΔAUC near zero)
  REDUNDANT    — no independent signal above the feature baseline

Quick start
-----------
>>> from featdiag import DiagnosticFramework
>>> fw = DiagnosticFramework(X, y, feature="pil", outcome="CVD")
>>> result = fw.fit()
>>> result.summary()
>>> result.plot()

Reference
---------
Alotaibi SA. When SHAP, Incremental AUC, and Logistic Regression Diverge:
A Simulation-Validated Diagnostic Framework for Feature Substitution in
Clinical Machine Learning. npj Digital Medicine (submitted 2026).
"""

from .core import DiagnosticFramework, DiagnosticResult
from .rules import classify, interpret, CLASSIFICATIONS
from .metrics import (
    compute_shap,
    compute_ablation_auc,
    compute_permutation_rank,
    compute_logistic_or,
    compute_lime_stability,
)

__version__ = "0.1.0"
__author__ = "Sheikah Adel Alotaibi"
__email__ = "shekah.adel.b@gmail.com"

__all__ = [
    "DiagnosticFramework",
    "DiagnosticResult",
    "classify",
    "interpret",
    "CLASSIFICATIONS",
    "compute_shap",
    "compute_ablation_auc",
    "compute_permutation_rank",
    "compute_logistic_or",
    "compute_lime_stability",
]
