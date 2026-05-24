# featdiag

**Three-Metric Diagnostic Framework for Feature Substitution in Clinical ML**

`featdiag` classifies any feature in a clinical XGBoost model as one of three categories:

| Class | OR significant | Ablation ΔAUC ≥ threshold | Meaning |
|---|---|---|---|
| **INDEPENDENT** | ✓ | ✓ | Genuine signal; feature is irreplaceable |
| **SUBSTITUTION** | ✓ | ✗ | Genuine signal masked by correlated substitutes |
| **REDUNDANT** | ✗ | — | No detectable independent association |

The framework resolves the common divergence between high SHAP rank and near-zero incremental AUC — a phenomenon that has caused misclassification of feature importance in clinical AI pipelines.

## Installation

```bash
pip install featdiag
```

With LIME stability analysis (optional, adds ~5 min per run):

```bash
pip install "featdiag[lime]"
```

## Quick Start

```python
import pandas as pd
from featdiag import DiagnosticFramework

# X: pd.DataFrame of numeric features (complete case)
# y: binary outcome array (0/1)
fw = DiagnosticFramework(X, y, feature="pil", outcome="CVD")
result = fw.fit()

result.summary()   # print formatted summary
result.plot()      # return matplotlib Figure
result.to_dict()   # JSON-serialisable metrics dict
```

## Parameters

### `DiagnosticFramework(X, y, feature, ...)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | `pd.DataFrame` | — | Feature matrix (numeric, complete-case) |
| `y` | array-like | — | Binary outcome (0/1) |
| `feature` | `str` | — | Column in X to diagnose |
| `outcome` | `str` | `"outcome"` | Label for plots/summaries |
| `model_class` | class | `XGBClassifier` | Sklearn-compatible classifier |
| `model_params` | `dict` | paper defaults | Passed to `model_class` |
| `cv` | `int` | `5` | CV folds for ablation ΔAUC |
| `threshold` | `float` | `0.015` | ΔAUC cutoff (INDEPENDENT vs. SUBSTITUTION) |
| `random_state` | `int` | `42` | Global seed |
| `alpha` | `float` | `0.05` | Significance level for OR |

### `fw.fit(run_lime=False, lime_seeds=10, lime_samples=300, store_shap=True, verbose=True)`

Returns a `DiagnosticResult` dataclass containing all metrics and the final classification.

## Diagnostic Rule

```
OR significant (p < α)?
  ├── YES → ΔAUC ≥ threshold?
  │          ├── YES → INDEPENDENT
  │          └── NO  → SUBSTITUTION
  └── NO  → REDUNDANT
```

## Metrics

The framework computes five metrics:

1. **SHAP importance** — mean |SHAP| rank via TreeExplainer
2. **Permutation ΔAUC** — rank-order validation (30 repeats)
3. **Ablation ΔAUC** — retrain without feature each fold (used in rule)
4. **Logistic regression OR** — Wald inference via statsmodels, 95% CI
5. **LIME rank stability** — cross-seed rank SD (optional, `run_lime=True`)

## Example Output

```
============================================================
Feature Diagnostic: 'pil' → 'CVD'
============================================================
  Dataset:       n=3267, events=487 (14.9%)
  Model AUC:     0.741  (without pil: 0.737)
  SHAP rank:     3/9  (|SHAP|=0.231)
  Perm. rank:    3/9  (ΔAUC=+0.0317)
  LIME rank:     3.2±0.4/9 (range 3–4)
  Ablation ΔAUC: +0.004  (threshold=0.015)
  OR:            0.829 [0.741–0.926], p=0.0007

  Classification: *** SUBSTITUTION ***
  The feature has a genuine independent association but contributes
  negligible incremental AUC because correlated features absorb its
  predictive signal.
============================================================
```

## Discriminative Validity

A key validation: the framework produces different classifications for the same feature across outcomes with distinct biological mechanisms:

| Dataset | Feature | Outcome | Classification |
|---|---|---|---|
| MIDUS 2 | PIL | CVD | **SUBSTITUTION** |
| HRS (prospective) | PIL | incident CVD | **SUBSTITUTION** |
| MIDUS 2 Biomarker | PIL | CRP > 3 mg/L | **REDUNDANT** |

PIL is an inflammation-independent CVD risk factor — the framework correctly identifies no association with CRP, validating that it is not a default classifier.

## Reproducing the Paper

```bash
# Install with LIME support
pip install "featdiag[lime]"

# Run with your harmonised data files
python examples/midus_example.py midus2.csv biomarker.csv hrs.csv

# Or run the synthetic demo (no data needed)
python examples/midus_example.py
```

See `examples/midus_example.py` for data format and column naming conventions.

## API Reference

```python
from featdiag import (
    DiagnosticFramework,   # main class
    DiagnosticResult,      # result dataclass
    classify,              # classify(or_significant, delta_auc, threshold)
    interpret,             # interpret("SUBSTITUTION") → explanation string
    CLASSIFICATIONS,       # ["INDEPENDENT", "SUBSTITUTION", "REDUNDANT"]

    # individual metric functions (can be used standalone)
    compute_shap,
    compute_ablation_auc,
    compute_permutation_rank,
    compute_logistic_or,
    compute_lime_stability,
)
```

## Citation

If you use `featdiag`, please cite:

```bibtex
@article{alotaibi2026featdiag,
  title   = {When {SHAP}, Incremental {AUC}, and Logistic Regression Diverge:
             A Simulation-Validated Diagnostic Framework for Feature
             Substitution in Clinical Machine Learning},
  author  = {Alotaibi, Sheikah Adel},
  journal = {npj Digital Medicine},
  year    = {2026},
  note    = {Submitted},
  url     = {https://github.com/She0-bit/featdiag}
}
```

## License

MIT © 2026 Sheikah Adel Alotaibi
