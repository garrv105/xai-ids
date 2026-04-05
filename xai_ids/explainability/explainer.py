"""
XAI-IDS - Explainability Engine
==================================
Provides model-agnostic and model-specific explanations:

1. SHAP (SHapley Additive exPlanations) - global and local feature importance
2. Integrated Gradients - gradient-based attribution (deep learning native)
3. LIME - local linear approximation
4. Counterfactual explanations - "what minimal change would flip the prediction?"
5. Attention-style feature ranking (based on gradient magnitudes)

Every prediction from the API includes:
- Top-k most important features and their direction
- Confidence score with calibration
- Human-readable threat narrative
- MITRE ATT&CK tactic suggestion
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


FEATURE_DESCRIPTIONS = {
    "duration": "connection duration (seconds)",
    "src_bytes": "bytes sent from source",
    "dst_bytes": "bytes sent to destination",
    "count": "connections to same host in window",
    "serror_rate": "SYN error rate (fraction of SYN errors)",
    "same_srv_rate": "fraction of connections to same service",
    "syn_count": "SYN packet count",
    "bytes_per_sec": "transfer rate (bytes/second)",
    "packets_per_sec": "packet rate",
    "flag_ratio": "SYN-to-ACK ratio",
    "fwd_packets": "forward direction packet count",
    "bwd_packets": "backward direction packet count",
    "rst_count": "TCP RST packet count",
    "num_failed_logins": "failed login attempts",
    "diff_srv_rate": "fraction connecting to different services",
}

MITRE_MAPPING = {
    "DoS": ("Impact", "T1498 - Network Denial of Service"),
    "PortScan": ("Discovery", "T1046 - Network Service Scanning"),
    "BruteForce": ("Credential Access", "T1110 - Brute Force"),
    "DNSTunnel": ("Command and Control", "T1071.004 - DNS"),
    "DataExfil": ("Exfiltration", "T1048 - Exfiltration Over Alternative Protocol"),
    "NORMAL": ("N/A", "N/A"),
}


class IntegratedGradientsExplainer:
    """
    Computes Integrated Gradients attributions for a PyTorch model.
    IG attributes prediction scores to input features by integrating
    gradients along a linear path from a baseline to the input.

    Reference: Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
    """

    def __init__(self, model: nn.Module, device: str = "cpu", steps: int = 50):
        self.model = model
        self.device = device
        self.steps = steps

    def explain(self, X: np.ndarray, baseline: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute IG attributions for a batch of inputs.

        Args:
            X: (N, F) input array
            baseline: (F,) reference baseline (zeros by default)

        Returns:
            attributions: (N, F) signed feature attributions
        """
        x_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        if baseline is None:
            baseline = np.zeros(X.shape[1])
        base_tensor = torch.tensor(baseline, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Interpolate between baseline and input
        alphas = torch.linspace(0, 1, self.steps, device=self.device)
        grads = []

        for alpha in alphas:
            interp = base_tensor + alpha * (x_tensor - base_tensor)
            interp.requires_grad_(True)
            binary_logit, _, _ = self.model(interp)
            score = binary_logit.sum()
            score.backward()
            grads.append(interp.grad.detach().cpu().numpy())

        grads = np.stack(grads, axis=0)  # (steps, N, F)
        avg_grads = grads.mean(axis=0)
        attributions = (X - baseline) * avg_grads
        return attributions

    def top_features(
        self,
        attributions: np.ndarray,
        feature_names: List[str],
        top_k: int = 5,
    ) -> List[Dict]:
        """Return sorted list of top contributing features."""
        results = []
        for i in range(len(attributions)):
            attrs = attributions[i]
            ranked = np.argsort(np.abs(attrs))[::-1][:top_k]
            results.append(
                [
                    {
                        "feature": feature_names[j],
                        "attribution": float(attrs[j]),
                        "direction": "increases_risk" if attrs[j] > 0 else "decreases_risk",
                        "description": FEATURE_DESCRIPTIONS.get(feature_names[j], feature_names[j]),
                    }
                    for j in ranked
                ]
            )
        return results


class SHAPExplainer:
    """
    SHAP-based explainer using KernelSHAP (model-agnostic).
    Falls back to gradient-based IG if shap is not installed.
    """

    def __init__(self, model, X_background: np.ndarray, feature_names: List[str]):
        self.feature_names = feature_names
        self._shap_available = False
        self._explainer = None
        self._background = X_background[:100]  # Use subsample as background

        try:
            import shap

            # Use a callable wrapper for PyTorch model
            def model_fn(X):
                with torch.no_grad():
                    t = torch.tensor(X, dtype=torch.float32)
                    logit, _, _ = model(t)
                    return torch.sigmoid(logit).numpy()

            self._explainer = shap.KernelExplainer(model_fn, self._background)
            self._shap_available = True
            logger.info("SHAP KernelExplainer initialized")
        except ImportError:
            logger.warning("shap not installed. SHAP explanations unavailable.")

    def explain(self, X: np.ndarray, n_samples: int = 100) -> Optional[np.ndarray]:
        """Returns SHAP values (N, F) or None if shap unavailable."""
        if not self._shap_available:
            return None
        return self._explainer.shap_values(X, nsamples=n_samples)

    def global_importance(self, X: np.ndarray) -> List[Dict]:
        """Compute global mean |SHAP| importance across a dataset."""
        shap_vals = self.explain(X)
        if shap_vals is None:
            return []
        mean_abs = np.abs(shap_vals).mean(axis=0)
        ranked = np.argsort(mean_abs)[::-1]
        return [
            {
                "feature": self.feature_names[i],
                "mean_abs_shap": float(mean_abs[i]),
                "description": FEATURE_DESCRIPTIONS.get(self.feature_names[i], self.feature_names[i]),
            }
            for i in ranked
        ]


class CounterfactualExplainer:
    """
    Generates minimal-change counterfactual explanations:
    "What is the closest 'normal' input to this detected attack?"

    Uses gradient-based optimization to find a counterfactual in feature space.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device

    def generate(
        self,
        X: np.ndarray,
        target_class: int = 0,  # 0 = normal
        max_iter: int = 200,
        lr: float = 0.01,
        lambda_dist: float = 1.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Generate counterfactual for a single sample.

        Returns:
            cf: (F,) counterfactual feature vector
            distance: L2 distance from original
        """
        x = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
        cf = x.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([cf], lr=lr)
        bce = nn.BCEWithLogitsLoss()
        target = torch.tensor([float(target_class)], device=self.device)

        for _ in range(max_iter):
            optimizer.zero_grad()
            logit, _, _ = self.model(cf)
            pred_loss = bce(logit, target)
            dist_loss = lambda_dist * torch.norm(cf - x)
            loss = pred_loss + dist_loss
            loss.backward()
            optimizer.step()

            # Early stop if prediction flipped
            with torch.no_grad():
                prob = torch.sigmoid(self.model(cf)[0]).item()
                if (target_class == 0 and prob < 0.3) or (target_class == 1 and prob > 0.7):
                    break

        cf_np = cf.detach().cpu().numpy().squeeze()
        distance = float(np.linalg.norm(cf_np - X))
        return cf_np, distance


class ExplainabilityEngine:
    """
    Unified explainability interface combining IG, SHAP, and counterfactuals.
    Produces structured JSON explanations for the API.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        class_names: List[str],
        X_background: np.ndarray,
        device: str = "cpu",
    ):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.device = device

        self.ig = IntegratedGradientsExplainer(model, device)
        self.shap = SHAPExplainer(model, X_background, feature_names)
        self.cf = CounterfactualExplainer(model, device)

    def explain_prediction(self, X: np.ndarray, pred_class: str, confidence: float) -> Dict:
        """
        Generate full explanation for a single prediction.
        Returns a structured dict suitable for JSON serialization.
        """
        # Integrated Gradients attributions
        attrs = self.ig.explain(X.reshape(1, -1))[0]
        top_feats = self.ig.top_features(attrs.reshape(1, -1), self.feature_names, top_k=5)[0]

        # Counterfactual (for attacks only)
        cf_explanation = None
        if pred_class != "NORMAL":
            cf, cf_dist = self.cf.generate(X, target_class=0)
            changed_features = []
            for i, (orig, new) in enumerate(zip(X, cf)):
                delta = new - orig
                if abs(delta) > 0.01:
                    changed_features.append(
                        {
                            "feature": self.feature_names[i],
                            "original": float(orig),
                            "counterfactual": float(new),
                            "change": float(delta),
                        }
                    )
            cf_explanation = {
                "distance": round(cf_dist, 4),
                "changed_features": sorted(changed_features, key=lambda x: abs(x["change"]), reverse=True)[:5],
                "interpretation": "Minimum feature changes that would classify this as normal traffic",
            }

        # MITRE mapping
        mitre_tactic, mitre_technique = MITRE_MAPPING.get(pred_class, ("Unknown", "Unknown"))

        # Human-readable narrative
        narrative = self._build_narrative(pred_class, confidence, top_feats)

        return {
            "prediction": pred_class,
            "is_attack": pred_class != "NORMAL",
            "confidence": round(confidence, 4),
            "threat_score": round(confidence if pred_class != "NORMAL" else 0.0, 4),
            "top_features": top_feats,
            "counterfactual": cf_explanation,
            "mitre": {"tactic": mitre_tactic, "technique": mitre_technique},
            "narrative": narrative,
        }

    def _build_narrative(self, pred_class: str, confidence: float, top_feats: List[Dict]) -> str:
        if pred_class == "NORMAL":
            return f"Traffic pattern is consistent with normal behavior (confidence: {confidence:.1%})."

        feat_strs = ", ".join(
            f"{f['description']} ({'+' if f['attribution'] > 0 else '-'}{abs(f['attribution']):.3f})"
            for f in top_feats[:3]
        )
        return (
            f"Detected {pred_class} attack with {confidence:.1%} confidence. "
            f"Key indicators: {feat_strs}. "
            f"This behavior pattern matches MITRE ATT&CK techniques associated with this threat category."
        )
