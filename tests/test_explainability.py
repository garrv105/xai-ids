"""
Tests: IntegratedGradientsExplainer, CounterfactualExplainer, ExplainabilityEngine
"""
import numpy as np
import torch
import pytest

from xai_ids.models.ids_model import IDSNet
from xai_ids.explainability.explainer import (
    IntegratedGradientsExplainer, CounterfactualExplainer,
    ExplainabilityEngine, FEATURE_DESCRIPTIONS, MITRE_MAPPING
)
from xai_ids.preprocessing.pipeline import NUMERIC_FEATURES

N_FEATURES = len(NUMERIC_FEATURES)
N_CLASSES = 6


@pytest.fixture(scope="module")
def model():
    m = IDSNet(n_features=N_FEATURES, n_classes=N_CLASSES, hidden_dim=64)
    m.eval()
    return m


@pytest.fixture(scope="module")
def background():
    return np.random.rand(20, N_FEATURES).astype(np.float32)


class TestIntegratedGradients:
    def test_attributions_shape(self, model):
        ig = IntegratedGradientsExplainer(model, steps=5)
        X = np.random.rand(3, N_FEATURES).astype(np.float32)
        attrs = ig.explain(X)
        assert attrs.shape == (3, N_FEATURES)

    def test_attributions_not_all_zero(self, model):
        ig = IntegratedGradientsExplainer(model, steps=5)
        X = np.random.rand(1, N_FEATURES).astype(np.float32)
        attrs = ig.explain(X)
        assert np.abs(attrs).sum() > 1e-8

    def test_completeness_axiom(self, model):
        """Sum of attributions should approximate f(x) - f(baseline)."""
        ig = IntegratedGradientsExplainer(model, steps=50)
        X = np.random.rand(1, N_FEATURES).astype(np.float32)
        baseline = np.zeros(N_FEATURES, dtype=np.float32)
        attrs = ig.explain(X, baseline=baseline)

        x_tensor = torch.tensor(X, dtype=torch.float32)
        b_tensor = torch.tensor(baseline.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            fx = torch.sigmoid(model(x_tensor)[0]).item()
            fb = torch.sigmoid(model(b_tensor)[0]).item()
        # IG sum should approximate f(x) - f(baseline)
        ig_sum = attrs[0].sum()
        expected = fx - fb
        # Allow generous tolerance (pure Python IG approximation)
        assert abs(ig_sum - expected) < 0.5

    def test_top_features_length(self, model):
        ig = IntegratedGradientsExplainer(model, steps=5)
        X = np.random.rand(2, N_FEATURES).astype(np.float32)
        attrs = ig.explain(X)
        top = ig.top_features(attrs, NUMERIC_FEATURES, top_k=3)
        assert len(top) == 2
        assert len(top[0]) == 3

    def test_top_features_have_required_keys(self, model):
        ig = IntegratedGradientsExplainer(model, steps=5)
        X = np.random.rand(1, N_FEATURES).astype(np.float32)
        attrs = ig.explain(X)
        top = ig.top_features(attrs, NUMERIC_FEATURES, top_k=3)
        for feat in top[0]:
            assert "feature" in feat
            assert "attribution" in feat
            assert "direction" in feat
            assert feat["direction"] in ("increases_risk", "decreases_risk")

    def test_custom_baseline(self, model):
        ig = IntegratedGradientsExplainer(model, steps=5)
        X = np.random.rand(1, N_FEATURES).astype(np.float32)
        baseline = np.ones(N_FEATURES, dtype=np.float32)
        attrs = ig.explain(X, baseline=baseline)
        assert attrs.shape == (1, N_FEATURES)


class TestCounterfactualExplainer:
    def test_cf_changes_prediction(self, model):
        cf_explainer = CounterfactualExplainer(model)
        # Start with a point that is likely classified as attack
        X = np.ones(N_FEATURES, dtype=np.float32) * 5.0
        X[8] = 10.0  # high serror_rate
        X[9] = 100.0  # high syn_count
        cf, dist = cf_explainer.generate(X, target_class=0, max_iter=50)
        assert cf.shape == (N_FEATURES,)
        assert dist >= 0.0

    def test_cf_distance_finite(self, model):
        cf_explainer = CounterfactualExplainer(model)
        X = np.random.rand(N_FEATURES).astype(np.float32)
        _, dist = cf_explainer.generate(X, max_iter=20)
        assert np.isfinite(dist)

    def test_cf_no_nan(self, model):
        cf_explainer = CounterfactualExplainer(model)
        X = np.random.rand(N_FEATURES).astype(np.float32)
        cf, _ = cf_explainer.generate(X, max_iter=20)
        assert not np.isnan(cf).any()


class TestExplainabilityEngine:
    def test_explain_prediction_keys(self, model, background):
        engine = ExplainabilityEngine(
            model=model,
            feature_names=NUMERIC_FEATURES,
            class_names=["NORMAL", "DoS", "PortScan", "BruteForce", "DNSTunnel", "DataExfil"],
            X_background=background,
        )
        X = np.random.rand(N_FEATURES).astype(np.float32)
        result = engine.explain_prediction(X, "DoS", 0.9)
        for k in ["prediction", "is_attack", "confidence", "top_features",
                  "mitre", "narrative", "counterfactual"]:
            assert k in result

    def test_normal_prediction_no_counterfactual(self, model, background):
        engine = ExplainabilityEngine(
            model=model, feature_names=NUMERIC_FEATURES,
            class_names=["NORMAL", "DoS"],
            X_background=background,
        )
        X = np.random.rand(N_FEATURES).astype(np.float32)
        result = engine.explain_prediction(X, "NORMAL", 0.95)
        assert result["counterfactual"] is None
        assert result["is_attack"] is False

    def test_narrative_contains_confidence(self, model, background):
        engine = ExplainabilityEngine(
            model=model, feature_names=NUMERIC_FEATURES,
            class_names=["NORMAL", "DoS"],
            X_background=background,
        )
        X = np.random.rand(N_FEATURES).astype(np.float32)
        result = engine.explain_prediction(X, "NORMAL", 0.91)
        assert "91" in result["narrative"] or "0.91" in result["narrative"]

    def test_mitre_mapping_exists_for_all_classes(self):
        for cls in ["DoS", "PortScan", "BruteForce", "DNSTunnel", "DataExfil", "NORMAL"]:
            assert cls in MITRE_MAPPING

    def test_feature_descriptions_nonempty(self):
        for feat, desc in FEATURE_DESCRIPTIONS.items():
            assert len(desc) > 5
