"""
XAI-IDS - Data Preprocessing Pipeline
=======================================
Handles:
- CICIDS2017/NSL-KDD/synthetic dataset ingestion
- Feature engineering (statistical flow features)
- Train/val/test stratified splitting
- Normalization and encoding
- Synthetic realistic dataset generation
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "duration",
    "protocol_type_enc",
    "src_bytes",
    "dst_bytes",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "num_compromised",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_serror_rate",
    "dst_host_rerror_rate",
    "fwd_packets",
    "bwd_packets",
    "fwd_bytes",
    "bwd_bytes",
    "bytes_per_sec",
    "packets_per_sec",
    "avg_fwd_iat",
    "avg_bwd_iat",
    "syn_count",
    "fin_count",
    "rst_count",
    "flag_ratio",
]

LABEL_COLUMN = "label"
BINARY_LABEL_COLUMN = "is_attack"


class SyntheticDataGenerator:
    """
    Generates realistic labeled network traffic dataset.
    Produces statistically grounded synthetic samples for each attack class.
    """

    ATTACK_PROFILES = {
        "NORMAL": {
            "duration": (30, 120),
            "src_bytes": (500, 8000),
            "dst_bytes": (1000, 15000),
            "count": (1, 20),
            "serror_rate": (0.0, 0.05),
            "same_srv_rate": (0.7, 1.0),
            "syn_count": (1, 3),
            "bytes_per_sec": (100, 5000),
            "packets_per_sec": (1, 30),
            "flag_ratio": (0.1, 0.5),
        },
        "DoS": {
            "duration": (0.01, 2),
            "src_bytes": (40, 100),
            "dst_bytes": (0, 50),
            "count": (100, 511),
            "serror_rate": (0.8, 1.0),
            "same_srv_rate": (0.9, 1.0),
            "syn_count": (100, 500),
            "bytes_per_sec": (10000, 500000),
            "packets_per_sec": (200, 5000),
            "flag_ratio": (5.0, 50.0),
        },
        "PortScan": {
            "duration": (0, 0.5),
            "src_bytes": (0, 40),
            "dst_bytes": (0, 40),
            "count": (20, 200),
            "serror_rate": (0.5, 1.0),
            "same_srv_rate": (0.0, 0.1),
            "syn_count": (20, 200),
            "bytes_per_sec": (100, 2000),
            "packets_per_sec": (10, 500),
            "flag_ratio": (0.9, 1.0),
        },
        "BruteForce": {
            "duration": (0.5, 5),
            "src_bytes": (100, 500),
            "dst_bytes": (100, 500),
            "count": (20, 200),
            "serror_rate": (0.0, 0.2),
            "same_srv_rate": (0.9, 1.0),
            "syn_count": (20, 200),
            "bytes_per_sec": (500, 10000),
            "packets_per_sec": (5, 100),
            "flag_ratio": (0.1, 1.0),
        },
        "DNSTunnel": {
            "duration": (1, 30),
            "src_bytes": (200, 1500),
            "dst_bytes": (500, 3000),
            "count": (30, 150),
            "serror_rate": (0.0, 0.05),
            "same_srv_rate": (0.8, 1.0),
            "syn_count": (1, 5),
            "bytes_per_sec": (1000, 20000),
            "packets_per_sec": (10, 200),
            "flag_ratio": (0.1, 0.5),
        },
        "DataExfil": {
            "duration": (60, 600),
            "src_bytes": (5000000, 50000000),
            "dst_bytes": (1000, 50000),
            "count": (5, 50),
            "serror_rate": (0.0, 0.02),
            "same_srv_rate": (0.5, 1.0),
            "syn_count": (1, 10),
            "bytes_per_sec": (10000, 200000),
            "packets_per_sec": (1, 20),
            "flag_ratio": (0.1, 0.5),
        },
    }

    def generate(self, n_per_class: int = 2000, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        records = []

        for label, profile in self.ATTACK_PROFILES.items():
            n = n_per_class
            for _ in range(n):
                row = {"label": label, "is_attack": int(label != "NORMAL")}
                for feat, (lo, hi) in profile.items():
                    val = rng.uniform(lo, hi)
                    # Add realistic noise
                    row[feat] = max(0.0, val + rng.normal(0, (hi - lo) * 0.05))
                # Fill remaining features with defaults
                row.setdefault("dst_bytes", row.get("src_bytes", 0) * rng.uniform(0.1, 3.0))
                row["wrong_fragment"] = int(rng.uniform(0, 1) > 0.95)
                row["urgent"] = int(rng.uniform(0, 1) > 0.99)
                row["hot"] = rng.integers(0, 10)
                row["num_failed_logins"] = rng.integers(0, 5 if label == "BruteForce" else 1)
                row["num_compromised"] = rng.integers(0, 3)
                row["srv_count"] = rng.integers(1, 200)
                row["srv_serror_rate"] = rng.uniform(0, 1) if label == "DoS" else rng.uniform(0, 0.1)
                row["rerror_rate"] = rng.uniform(0, 0.3)
                row["diff_srv_rate"] = rng.uniform(0.0, 0.1) if label == "NORMAL" else rng.uniform(0.3, 1.0)
                row["dst_host_count"] = rng.integers(1, 255)
                row["dst_host_srv_count"] = rng.integers(1, 255)
                row["dst_host_same_srv_rate"] = rng.uniform(0.5, 1.0)
                row["dst_host_diff_srv_rate"] = rng.uniform(0.0, 0.5)
                row["dst_host_serror_rate"] = rng.uniform(0, 1) if label == "DoS" else rng.uniform(0, 0.05)
                row["dst_host_rerror_rate"] = rng.uniform(0, 0.2)
                row["fwd_packets"] = rng.integers(1, 500)
                row["bwd_packets"] = rng.integers(1, 200)
                row["fwd_bytes"] = int(row.get("src_bytes", 100))
                row["bwd_bytes"] = int(row.get("dst_bytes", 100))
                row["avg_fwd_iat"] = rng.uniform(0, 1000)
                row["avg_bwd_iat"] = rng.uniform(0, 1000)
                row["fin_count"] = rng.integers(0, 5)
                row["rst_count"] = rng.integers(0, 3) if label != "DoS" else rng.integers(0, 50)
                row["protocol_type_enc"] = rng.integers(0, 3)
                records.append(row)

        df = pd.DataFrame(records)
        # Ensure all feature columns exist
        for feat in NUMERIC_FEATURES:
            if feat not in df.columns:
                df[feat] = 0.0
        df = df[NUMERIC_FEATURES + [LABEL_COLUMN, BINARY_LABEL_COLUMN]]
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)


class DataPipeline:
    """
    Complete preprocessing pipeline:
    1. Load data (CSV or generate synthetic)
    2. Feature selection and engineering
    3. Normalization
    4. Train/val/test splitting
    5. Artifact persistence (scaler, encoder)
    """

    def __init__(self, artifact_dir: str = "trained_models"):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._fitted = False

    def load_and_prepare(
        self,
        csv_path: Optional[str] = None,
        n_per_class: int = 2000,
        test_size: float = 0.15,
        val_size: float = 0.15,
    ) -> Dict[str, np.ndarray]:
        """
        Returns dict with keys:
            X_train, X_val, X_test,
            y_train, y_val, y_test,        # binary labels
            y_multi_train, ...,             # multiclass labels
            feature_names
        """
        if csv_path and Path(csv_path).exists():
            logger.info("Loading dataset from %s", csv_path)
            df = pd.read_csv(csv_path)
        else:
            logger.info("Generating synthetic dataset (%d samples/class)", n_per_class)
            gen = SyntheticDataGenerator()
            df = gen.generate(n_per_class=n_per_class)
            # Save synthetic dataset
            synthetic_path = self.artifact_dir.parent / "data" / "processed" / "synthetic_traffic.csv"
            synthetic_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(synthetic_path, index=False)
            logger.info("Synthetic dataset saved to %s", synthetic_path)

        # Ensure numeric features
        X = df[NUMERIC_FEATURES].astype(np.float32).values
        y_binary = df[BINARY_LABEL_COLUMN].values.astype(np.int64)
        y_multi = self.label_encoder.fit_transform(df[LABEL_COLUMN].values)

        # Train/val/test split
        X_tmp, X_test, y_tmp, y_test, ym_tmp, ym_test = train_test_split(
            X, y_binary, y_multi, test_size=test_size, stratify=y_binary, random_state=42
        )
        val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val, ym_train, ym_val = train_test_split(
            X_tmp, y_tmp, ym_tmp, test_size=val_ratio, stratify=y_tmp, random_state=42
        )

        # Fit scaler on training set only
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        self._fitted = True

        # Save artifacts
        self._save_artifacts()

        logger.info(
            "Split: train=%d val=%d test=%d | classes=%s",
            len(X_train),
            len(X_val),
            len(X_test),
            list(self.label_encoder.classes_),
        )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "y_multi_train": ym_train,
            "y_multi_val": ym_val,
            "y_multi_test": ym_test,
            "feature_names": NUMERIC_FEATURES,
            "class_names": list(self.label_encoder.classes_),
            "n_features": len(NUMERIC_FEATURES),
            "n_classes": len(self.label_encoder.classes_),
        }

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using the fitted scaler."""
        if not self._fitted:
            self._load_artifacts()
        return self.scaler.transform(X)

    def _save_artifacts(self):
        with open(self.artifact_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(self.artifact_dir / "label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        logger.info("Pipeline artifacts saved to %s", self.artifact_dir)

    def _load_artifacts(self):
        with open(self.artifact_dir / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(self.artifact_dir / "label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)
        self._fitted = True
