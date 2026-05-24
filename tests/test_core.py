"""
Unit tests for featdiag.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=featdiag --cov-report=term-missing
"""

import numpy as np
import pandas as pd
import pytest

from featdiag.rules import classify, interpret, CLASSIFICATIONS


# ── Rules ─────────────────────────────────────────────────────────────────────

class TestClassify:
    def test_independent(self):
        assert classify(or_significant=True, delta_auc_ablation=0.02, threshold=0.015) == "INDEPENDENT"

    def test_substitution(self):
        assert classify(or_significant=True, delta_auc_ablation=0.004, threshold=0.015) == "SUBSTITUTION"

    def test_redundant(self):
        assert classify(or_significant=False, delta_auc_ablation=0.02, threshold=0.015) == "REDUNDANT"

    def test_redundant_negative_dauc(self):
        assert classify(or_significant=False, delta_auc_ablation=-0.01, threshold=0.015) == "REDUNDANT"

    def test_threshold_boundary_below(self):
        assert classify(or_significant=True, delta_auc_ablation=0.0149, threshold=0.015) == "SUBSTITUTION"

    def test_threshold_boundary_at(self):
        assert classify(or_significant=True, delta_auc_ablation=0.015, threshold=0.015) == "INDEPENDENT"

    def test_custom_threshold(self):
        assert classify(or_significant=True, delta_auc_ablation=0.01, threshold=0.005) == "INDEPENDENT"

    def test_all_classes_in_constants(self):
        assert set(CLASSIFICATIONS) == {"INDEPENDENT", "SUBSTITUTION", "REDUNDANT"}


class TestInterpret:
    def test_all_classifications_have_interpretation(self):
        for cls in CLASSIFICATIONS:
            text = interpret(cls)
            assert isinstance(text, str) and len(text) > 10

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            interpret("UNKNOWN")


# ── Metrics (lightweight, no real model) ──────────────────────────────────────

class TestComputeLogisticOR:
    """Test compute_logistic_or with synthetic data."""

    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.RandomState(42)
        n = 200
        x1 = rng.randn(n)
        x2 = rng.randn(n)
        logit = 0.5 * x1 - 0.3 * x2
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        X = pd.DataFrame({"x1": x1, "x2": x2})
        return X, y

    def test_returns_expected_keys(self, synthetic_data):
        from featdiag.metrics import compute_logistic_or
        X, y = synthetic_data
        out = compute_logistic_or(X, y, "x1")
        assert set(out) >= {"or_value", "ci_lower", "ci_upper", "p_value", "significant", "coef"}

    def test_or_positive(self, synthetic_data):
        from featdiag.metrics import compute_logistic_or
        X, y = synthetic_data
        out = compute_logistic_or(X, y, "x1")
        assert out["or_value"] > 0

    def test_significant_feature(self, synthetic_data):
        from featdiag.metrics import compute_logistic_or
        X, y = synthetic_data
        out = compute_logistic_or(X, y, "x1")
        assert out["significant"] is True

    def test_ci_contains_or(self, synthetic_data):
        from featdiag.metrics import compute_logistic_or
        X, y = synthetic_data
        out = compute_logistic_or(X, y, "x1")
        assert out["ci_lower"] < out["or_value"] < out["ci_upper"]

    def test_null_feature_not_significant(self):
        from featdiag.metrics import compute_logistic_or
        rng = np.random.RandomState(0)
        n = 300
        X = pd.DataFrame({"signal": rng.randn(n), "noise": rng.randn(n)})
        y = (rng.random(n) < 0.5).astype(int)
        out = compute_logistic_or(X, y, "noise")
        assert out["p_value"] > 0.05


# ── DiagnosticFramework (integration, uses real XGBoost) ─────────────────────

@pytest.fixture(scope="module")
def small_dataset():
    """300-sample synthetic dataset for integration tests."""
    rng = np.random.RandomState(99)
    n = 300
    x1 = rng.randn(n)
    x2 = x1 + 0.3 * rng.randn(n)   # correlated with x1 (substitute)
    x3 = rng.randn(n)               # independent noise
    logit = 0.8 * x1 + 0.1 * x3
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    return X, y


class TestDiagnosticFramework:
    def test_invalid_feature_raises(self, small_dataset):
        from featdiag import DiagnosticFramework
        X, y = small_dataset
        with pytest.raises(ValueError, match="not found"):
            DiagnosticFramework(X, y, feature="nonexistent")

    def test_fit_returns_result(self, small_dataset):
        from featdiag import DiagnosticFramework, DiagnosticResult
        X, y = small_dataset
        fw = DiagnosticFramework(X, y, feature="x1", outcome="test_outcome")
        result = fw.fit(run_lime=False, verbose=False)
        assert isinstance(result, DiagnosticResult)

    def test_result_classification_valid(self, small_dataset):
        from featdiag import DiagnosticFramework, CLASSIFICATIONS
        X, y = small_dataset
        fw = DiagnosticFramework(X, y, feature="x1")
        result = fw.fit(run_lime=False, verbose=False)
        assert result.classification in CLASSIFICATIONS

    def test_result_auc_reasonable(self, small_dataset):
        from featdiag import DiagnosticFramework
        X, y = small_dataset
        fw = DiagnosticFramework(X, y, feature="x1")
        result = fw.fit(run_lime=False, verbose=False)
        assert 0.5 <= result.auc_full <= 1.0

    def test_result_shap_rank_in_range(self, small_dataset):
        from featdiag import DiagnosticFramework
        X, y = small_dataset
        fw = DiagnosticFramework(X, y, feature="x1")
        result = fw.fit(run_lime=False, verbose=False)
        assert 1 <= result.shap_rank <= result.n_features

    def test_to_dict_serializable(self, small_dataset):
        import json
        from featdiag import DiagnosticFramework
        X, y = small_dataset
        fw = DiagnosticFramework(X, y, feature="x1")
        result = fw.fit(run_lime=False, verbose=False)
        d = result.to_dict()
        json.dumps(d)  # must not raise

    def test_noise_feature_lower_shap(self, small_dataset):
        from featdiag import DiagnosticFramework
        X, y = small_dataset
        fw_signal = DiagnosticFramework(X, y, feature="x1")
        fw_noise = DiagnosticFramework(X, y, feature="x3")
        r_signal = fw_signal.fit(run_lime=False, verbose=False)
        r_noise = fw_noise.fit(run_lime=False, verbose=False)
        assert r_signal.shap_value > r_noise.shap_value

    def test_substitute_feature_classification(self, small_dataset):
        """x2 is highly correlated with x1 — expect SUBSTITUTION or REDUNDANT,
        not INDEPENDENT (it has no independent signal above x1)."""
        from featdiag import DiagnosticFramework, CLASSIFICATIONS
        X, y = small_dataset
        fw = DiagnosticFramework(X, y, feature="x2", threshold=0.015)
        result = fw.fit(run_lime=False, verbose=False)
        assert result.classification in ("SUBSTITUTION", "REDUNDANT")


# ── DiagnosticResult helpers ──────────────────────────────────────────────────

class TestDiagnosticResult:
    def test_summary_runs(self, small_dataset, capsys):
        from featdiag import DiagnosticFramework
        X, y = small_dataset
        fw = DiagnosticFramework(X, y, feature="x1")
        result = fw.fit(run_lime=False, verbose=False)
        result.summary()
        captured = capsys.readouterr()
        assert "Classification" in captured.out

    def test_plot_returns_figure(self, small_dataset):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from featdiag import DiagnosticFramework
        X, y = small_dataset
        fw = DiagnosticFramework(X, y, feature="x1")
        result = fw.fit(run_lime=False, verbose=False)
        fig = result.plot()
        assert fig is not None
        plt.close("all")
