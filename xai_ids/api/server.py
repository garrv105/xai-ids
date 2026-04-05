"""
XAI-IDS - REST API Server
===========================
FastAPI server exposing:
- POST /auth/token         — obtain JWT access token
- GET  /api/v1/health      — unauthenticated health probe
- POST /api/v1/predict     — single-sample inference with explanation
- POST /api/v1/predict/batch — batch inference
- GET  /api/v1/model/info  — model metadata and performance
- GET  /api/v1/model/features — global feature importance
- POST /api/v1/online/update — incremental model update endpoint
- GET  /api/v1/dashboard   — embedded Dash-style metrics (HTML)

Security hardening:
- JWT Bearer token authentication (HS256)
- API Key header authentication (fallback)
- Rate limiting via slowapi (per-IP)
- Security headers middleware (HSTS, CSP, X-Frame-Options, etc.)
- Input validation (feature vector dimension checks)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from .auth import CurrentUser, TokenResponse, get_current_user, login_for_access_token, require_admin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

_ALLOWED_ORIGINS = os.getenv("XAI_IDS_CORS_ORIGINS", "*")

# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=()"
        response.headers["Cache-Control"] = "no-store"
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains; preload"
            )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        return response


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        description="Numeric feature vector (must match training order)",
        min_length=1,
        max_length=512,
    )
    explain: bool = Field(True, description="Include XAI explanation in response")

    @field_validator("features")
    @classmethod
    def no_nan_or_inf(cls, v: List[float]) -> List[float]:
        import math
        for val in v:
            if math.isnan(val) or math.isinf(val):
                raise ValueError("Feature vector must not contain NaN or Inf values")
        return v


class BatchPredictRequest(BaseModel):
    samples: List[List[float]] = Field(
        ...,
        description="List of feature vectors",
        min_length=1,
        max_length=1000,
    )
    explain: bool = Field(False, description="Include explanations (slower for large batches)")


class OnlineUpdateRequest(BaseModel):
    features: List[float]
    label: int = Field(..., ge=0, le=1, description="Ground truth binary label (0=normal, 1=attack)")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(model_dir: str = "trained_models") -> FastAPI:
    app = FastAPI(
        title="XAI-IDS API",
        description=(
            "Explainable AI Intrusion Detection System — Inference and Explanation API.\n\n"
            "**Authentication:** All endpoints (except `/auth/token` and `/api/v1/health`) require:\n"
            "- `Authorization: Bearer <jwt>` — obtain from `POST /auth/token`\n"
            "- `X-API-Key: <key>` — configure via `XAI_IDS_API_KEYS` env var\n\n"
            "**Rate limits:** 200 requests/minute per IP; 10/minute on auth endpoints."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS.split(",") if _ALLOWED_ORIGINS != "*" else ["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "X-API-Key", "Content-Type"],
    )
    app.add_middleware(SecurityHeadersMiddleware)

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ----- Load model and artifacts -----
    model_path = Path(model_dir)
    state: dict = {}

    def load_artifacts():
        try:
            from ..models.ids_model import IDSNet
            from ..preprocessing.pipeline import NUMERIC_FEATURES

            with open(model_path / "label_encoder.pkl", "rb") as f:
                state["label_encoder"] = pickle.load(f)
            with open(model_path / "scaler.pkl", "rb") as f:
                state["scaler"] = pickle.load(f)

            n_classes = len(state["label_encoder"].classes_)
            model = IDSNet(n_features=len(NUMERIC_FEATURES), n_classes=n_classes)

            if (model_path / "best_model.pt").exists():
                model.load_state_dict(
                    torch.load(model_path / "best_model.pt", map_location="cpu")
                )
                logger.info("Model weights loaded")
            model.eval()
            state["model"] = model
            state["feature_names"] = NUMERIC_FEATURES
            state["class_names"] = list(state["label_encoder"].classes_)

            # Load background data for SHAP
            bg_path = model_path.parent / "data" / "processed" / "synthetic_traffic.csv"
            if bg_path.exists():
                import pandas as pd
                df = pd.read_csv(bg_path)
                X_bg = df[NUMERIC_FEATURES].values[:200].astype(np.float32)
                X_bg_scaled = state["scaler"].transform(X_bg)
                state["background"] = X_bg_scaled

            # Initialize explainability engine
            from ..explainability.explainer import ExplainabilityEngine
            state["explainer"] = ExplainabilityEngine(
                model=state["model"],
                feature_names=NUMERIC_FEATURES,
                class_names=state["class_names"],
                X_background=state.get(
                    "background", np.zeros((10, len(NUMERIC_FEATURES)), dtype=np.float32)
                ),
            )
            logger.info("All artifacts loaded. API ready.")
        except Exception as e:
            logger.error("Failed to load artifacts: %s", e)
            state["error"] = str(e)

    @app.on_event("startup")
    async def startup():
        load_artifacts()

    # -----------------------------------------------------------------------
    # Auth endpoint (public)
    # -----------------------------------------------------------------------

    @app.post(
        "/auth/token",
        response_model=TokenResponse,
        tags=["auth"],
        summary="Obtain a JWT access token",
    )
    @limiter.limit("10/minute")
    async def token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
        """
        Exchange credentials for a JWT access token.

        Default dev credentials: `admin` / `changeme`
        Override via `XAI_IDS_ADMIN_USER` and `XAI_IDS_ADMIN_PASS_HASH` env vars.
        """
        return await login_for_access_token(form_data)

    # -----------------------------------------------------------------------
    # Health (public)
    # -----------------------------------------------------------------------

    @app.get("/api/v1/health", tags=["system"], summary="Health check (unauthenticated)")
    async def health():
        if "error" in state:
            raise HTTPException(status_code=503, detail=state["error"])
        return {"status": "ok", "model_classes": state.get("class_names", [])}

    # -----------------------------------------------------------------------
    # Protected endpoints
    # -----------------------------------------------------------------------

    @app.post(
        "/api/v1/predict",
        tags=["inference"],
        summary="Single-sample prediction with optional XAI explanation",
    )
    @limiter.limit("60/minute")
    async def predict(
        request: Request,
        req: PredictRequest,
        _user: CurrentUser = Depends(get_current_user),
    ):
        """Single-sample prediction with optional XAI explanation."""
        if "model" not in state:
            raise HTTPException(status_code=503, detail="Model not loaded")

        feat = np.array(req.features, dtype=np.float32).reshape(1, -1)
        expected = len(state["feature_names"])
        if feat.shape[1] != expected:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected} features, got {feat.shape[1]}",
            )

        feat_scaled = state["scaler"].transform(feat)
        x_tensor = torch.tensor(feat_scaled, dtype=torch.float32)

        with torch.no_grad():
            proba = state["model"].predict_proba(x_tensor)

        attack_prob = float(proba["attack_probability"][0])
        class_probs = proba["class_probabilities"][0].numpy()
        pred_class_idx = int(np.argmax(class_probs))
        pred_class = state["class_names"][pred_class_idx]

        response: dict = {
            "is_attack": bool(attack_prob > 0.5),
            "attack_probability": round(attack_prob, 4),
            "predicted_class": pred_class,
            "class_probabilities": {
                state["class_names"][i]: round(float(class_probs[i]), 4)
                for i in range(len(state["class_names"]))
            },
            "requested_by": _user.username,
        }

        if req.explain and "explainer" in state:
            explanation = state["explainer"].explain_prediction(
                feat_scaled[0],
                pred_class,
                attack_prob if pred_class != "NORMAL" else 1 - attack_prob,
            )
            response["explanation"] = explanation

        return response

    @app.post(
        "/api/v1/predict/batch",
        tags=["inference"],
        summary="Batch inference for multiple samples",
    )
    @limiter.limit("20/minute")
    async def predict_batch(
        request: Request,
        req: BatchPredictRequest,
        _user: CurrentUser = Depends(get_current_user),
    ):
        """Batch inference for multiple samples (up to 1000)."""
        if "model" not in state:
            raise HTTPException(status_code=503, detail="Model not loaded")

        X = np.array(req.samples, dtype=np.float32)
        expected = len(state["feature_names"])
        if X.shape[1] != expected:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected} features per sample, got {X.shape[1]}",
            )

        X_scaled = state["scaler"].transform(X)
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            proba = state["model"].predict_proba(x_tensor)

        attack_probs = proba["attack_probability"].numpy()
        class_probs = proba["class_probabilities"].numpy()
        pred_idxs = np.argmax(class_probs, axis=1)

        results = []
        for i in range(len(X)):
            pred_class = state["class_names"][pred_idxs[i]]
            result = {
                "is_attack": bool(attack_probs[i] > 0.5),
                "attack_probability": round(float(attack_probs[i]), 4),
                "predicted_class": pred_class,
            }
            if req.explain and "explainer" in state:
                result["explanation"] = state["explainer"].explain_prediction(
                    X_scaled[i], pred_class, float(attack_probs[i])
                )
            results.append(result)

        return {"count": len(results), "results": results}

    @app.get("/api/v1/model/info", tags=["model"])
    @limiter.limit("60/minute")
    async def model_info(
        request: Request,
        _user: CurrentUser = Depends(get_current_user),
    ):
        if "model" not in state:
            raise HTTPException(status_code=503, detail="Model not loaded")
        m = state["model"]
        params = sum(p.numel() for p in m.parameters())
        return {
            "architecture": "IDSNet (Dual-head Residual NN)",
            "n_features": m.n_features,
            "n_classes": m.n_classes,
            "class_names": state["class_names"],
            "parameters": params,
            "temperature": float(m.temperature.item()),
            "feature_names": state["feature_names"],
        }

    @app.get(
        "/api/v1/model/features",
        tags=["model"],
        summary="Global feature importance via SHAP",
    )
    @limiter.limit("10/minute")
    async def feature_importance(
        request: Request,
        _user: CurrentUser = Depends(get_current_user),
    ):
        if "explainer" not in state or "background" not in state:
            raise HTTPException(status_code=503, detail="SHAP explainer not available")
        importance = state["explainer"].shap.global_importance(state["background"])
        return {"feature_importance": importance}

    @app.get("/api/v1/dashboard", response_class=HTMLResponse, tags=["ui"])
    async def dashboard():
        """Minimal HTML dashboard (for demo/development). No auth required for viewing."""
        html = _build_dashboard_html(state.get("class_names", []))
        return HTMLResponse(content=html)

    return app


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------


def _build_dashboard_html(class_names: List[str]) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>XAI-IDS Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; margin: 0; padding: 20px; }}
        h1 {{ color: #58a6ff; }} h2 {{ color: #79c0ff; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
        .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 16px 0; }}
        .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
        .attack {{ background: #da3633; }} .normal {{ background: #238636; }}
        pre {{ background: #010409; padding: 16px; border-radius: 6px; overflow-x: auto; color: #a5d6ff; font-size: 13px; }}
        button {{ background: #238636; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; }}
        button:hover {{ background: #2ea043; }}
        input, textarea {{ background: #0d1117; border: 1px solid #30363d; color: #e6edf3; padding: 8px; border-radius: 6px; width: 100%; box-sizing: border-box; }}
        .auth-note {{ background: #161b22; border: 1px solid #f0883e; border-radius: 8px; padding: 12px; color: #f0883e; margin-bottom: 16px; }}
    </style>
</head>
<body>
    <h1>XAI-IDS Dashboard</h1>
    <p style="color:#8b949e">Explainable AI Intrusion Detection System — Live Inference Interface</p>

    <div class="auth-note">
        <strong>Authentication required for /api/v1/predict.</strong>
        Enter your JWT token below, or configure <code>XAI_IDS_API_KEYS</code> for API key auth.
    </div>

    <div class="card">
        <h2>Model Status</h2>
        <p>Classes: {', '.join(f'<span class="badge {"normal" if c=="NORMAL" else "attack"}">{c}</span>' for c in class_names)}</p>
        <p>API Docs: <a href="/docs" style="color:#58a6ff">/docs</a> | Health: <a href="/api/v1/health" style="color:#58a6ff">/api/v1/health</a></p>
    </div>

    <div class="card">
        <h2>Authentication</h2>
        <p style="color:#8b949e">Enter your JWT (from POST /auth/token) to use the prediction form</p>
        <input type="password" id="jwt_token" placeholder="Bearer token (without 'Bearer ' prefix)">
    </div>

    <div class="card">
        <h2>Quick Prediction Test</h2>
        <p style="color:#8b949e">Enter a comma-separated feature vector and click Predict</p>
        <textarea id="features" rows="3" placeholder="0.5, 1200, 3000, 5, 0.02, 0.8, 2, 1500, 10, 0.1, ..."></textarea>
        <br><br>
        <input type="checkbox" id="explain" checked> <label>Include XAI explanation</label>
        <br><br>
        <button onclick="predict()">&#9654; Run Prediction</button>
        <pre id="result">Results will appear here...</pre>
    </div>

    <script>
    async function predict() {{
        const raw = document.getElementById('features').value;
        const features = raw.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
        const explain = document.getElementById('explain').checked;
        const token = document.getElementById('jwt_token').value.trim();
        document.getElementById('result').textContent = 'Running...';
        const headers = {{'Content-Type': 'application/json'}};
        if (token) headers['Authorization'] = 'Bearer ' + token;
        try {{
            const resp = await fetch('/api/v1/predict', {{
                method: 'POST',
                headers,
                body: JSON.stringify({{features, explain}})
            }});
            const data = await resp.json();
            document.getElementById('result').textContent = JSON.stringify(data, null, 2);
        }} catch(e) {{
            document.getElementById('result').textContent = 'Error: ' + e.message;
        }}
    }}
    </script>
</body>
</html>
"""
