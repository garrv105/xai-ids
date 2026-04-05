"""
XAI-IDS - Real Dataset Loaders
================================
Production loaders for publicly available IDS benchmark datasets.

Supported datasets:
  - CICIDS2017     — Canadian Institute for Cybersecurity Intrusion Detection 2017
                     https://www.unb.ca/cic/datasets/ids-2017.html
  - NSL-KDD        — NSL-KDD (improved KDD Cup 1999)
                     https://www.unb.ca/cic/datasets/nsl.html
  - KDD Cup 1999   — Original KDD (for compatibility / comparison)

All loaders normalise column names and map labels to the XAI-IDS standard
label set used by SyntheticDataGenerator:
    NORMAL, DoS, PortScan, BruteForce, DNSTunnel, DataExfil

Usage
-----
    from xai_ids.preprocessing.dataset_loaders import (
        load_cicids2017,
        load_nslkdd,
        autoload_dataset,
    )

    # CICIDS2017 — pass directory containing the weekly CSV files
    df = load_cicids2017("/data/cicids2017/")

    # NSL-KDD — pass path to KDDTrain+.arff or KDDTrain+.txt
    df = load_nslkdd("/data/nslkdd/KDDTrain+.arff")

    # Auto-detect format and load
    df = autoload_dataset("/data/cicids2017/")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .pipeline import BINARY_LABEL_COLUMN, LABEL_COLUMN, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standard label mapping
# Maps dataset-specific attack labels → XAI-IDS canonical labels
# ---------------------------------------------------------------------------

CICIDS2017_LABEL_MAP: dict[str, str] = {
    # Normal
    "benign": "NORMAL",
    # DoS / DDoS
    "dos hulk": "DoS",
    "dos goldeneye": "DoS",
    "dos slowloris": "DoS",
    "dos slowhttptest": "DoS",
    "heartbleed": "DoS",
    "ddos": "DoS",
    # Port Scan
    "portscan": "PortScan",
    "ftp-patator": "BruteForce",
    "ssh-patator": "BruteForce",
    # Brute Force variants
    "brute force": "BruteForce",
    "web attacks \u2013 brute force": "BruteForce",
    "web attacks \u2013 xss": "BruteForce",
    "web attacks \u2013 sql injection": "BruteForce",
    "web attacks - brute force": "BruteForce",
    "web attacks - xss": "BruteForce",
    "web attacks - sql injection": "BruteForce",
    # Infiltration & Exfil
    "infiltration": "DataExfil",
    "botnet": "DataExfil",
    # Unknown → DataExfil (catch-all for exotic attacks)
    "unknown": "DataExfil",
}

NSLKDD_LABEL_MAP: dict[str, str] = {
    "normal": "NORMAL",
    # DoS
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "apache2": "DoS",
    "udpstorm": "DoS",
    "processtable": "DoS",
    "mailbomb": "DoS",
    # Probe / PortScan
    "ipsweep": "PortScan",
    "nmap": "PortScan",
    "portsweep": "PortScan",
    "satan": "PortScan",
    "mscan": "PortScan",
    "saint": "PortScan",
    # Remote-to-local (BruteForce)
    "ftp_write": "BruteForce",
    "guess_passwd": "BruteForce",
    "httptunnel": "BruteForce",
    "imap": "BruteForce",
    "multihop": "BruteForce",
    "named": "BruteForce",
    "phf": "BruteForce",
    "sendmail": "BruteForce",
    "snmpgetattack": "BruteForce",
    "snmpguess": "BruteForce",
    "spy": "BruteForce",
    "warezclient": "BruteForce",
    "warezmaster": "BruteForce",
    "worm": "BruteForce",
    "xlock": "BruteForce",
    "xsnoop": "BruteForce",
    "xterm": "BruteForce",
    # User-to-root (privilege escalation → DataExfil)
    "buffer_overflow": "DataExfil",
    "loadmodule": "DataExfil",
    "perl": "DataExfil",
    "ps": "DataExfil",
    "rootkit": "DataExfil",
    "sqlattack": "DataExfil",
}

# NSL-KDD feature column names (same order as in the .arff and .txt files)
_NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]

# Mapping from NSL-KDD raw column names → NUMERIC_FEATURES
_NSL_KDD_FEATURE_MAP: dict[str, str] = {
    "duration": "duration",
    "src_bytes": "src_bytes",
    "dst_bytes": "dst_bytes",
    "wrong_fragment": "wrong_fragment",
    "urgent": "urgent",
    "hot": "hot",
    "num_failed_logins": "num_failed_logins",
    "num_compromised": "num_compromised",
    "count": "count",
    "srv_count": "srv_count",
    "serror_rate": "serror_rate",
    "srv_serror_rate": "srv_serror_rate",
    "rerror_rate": "rerror_rate",
    "same_srv_rate": "same_srv_rate",
    "diff_srv_rate": "diff_srv_rate",
    "dst_host_count": "dst_host_count",
    "dst_host_srv_count": "dst_host_srv_count",
    "dst_host_same_srv_rate": "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate": "dst_host_diff_srv_rate",
    "dst_host_serror_rate": "dst_host_serror_rate",
    "dst_host_rerror_rate": "dst_host_rerror_rate",
}

# Mapping from CICIDS2017 CSV column names → NUMERIC_FEATURES
_CICIDS_FEATURE_MAP: dict[str, str] = {
    "flow duration": "duration",
    "total fwd packets": "fwd_packets",
    "total backward packets": "bwd_packets",
    "total length of fwd packets": "fwd_bytes",
    "total length of bwd packets": "bwd_bytes",
    "fwd packets/s": "packets_per_sec",
    "bwd packets/s": "packets_per_sec",  # merged below
    "flow bytes/s": "bytes_per_sec",
    "flow packets/s": "packets_per_sec",
    "fwd iat total": "avg_fwd_iat",
    "bwd iat total": "avg_bwd_iat",
    "fwd iat mean": "avg_fwd_iat",
    "bwd iat mean": "avg_bwd_iat",
    "syn flag count": "syn_count",
    "fin flag count": "fin_count",
    "rst flag count": "rst_count",
    "destination port": "dst_bytes",  # used as proxy for port feature
}


# ---------------------------------------------------------------------------
# CICIDS2017 Loader
# ---------------------------------------------------------------------------


def load_cicids2017(
    path: Union[str, Path],
    sample_frac: float = 1.0,
    seed: int = 42,
    drop_infinite: bool = True,
) -> pd.DataFrame:
    """
    Load CICIDS2017 dataset from a directory of CSV files or a single CSV file.

    The dataset consists of 8 weekly CSVs downloadable from:
    https://www.unb.ca/cic/datasets/ids-2017.html

    Parameters
    ----------
    path : str or Path
        Path to a directory containing CICIDS2017 CSVs, or a single CSV file.
    sample_frac : float
        Fraction of data to sample (for faster iteration). Default 1.0.
    seed : int
        Random seed for sampling.
    drop_infinite : bool
        Replace Inf/-Inf with NaN, then drop rows.

    Returns
    -------
    pd.DataFrame with columns matching NUMERIC_FEATURES + label + is_attack
    """
    path = Path(path)

    # Collect CSV files
    if path.is_dir():
        csv_files = sorted(path.glob("*.csv")) + sorted(path.glob("*.CSV"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {path}")
        logger.info("Loading CICIDS2017 from %d files in %s", len(csv_files), path)
        dfs = [pd.read_csv(f, low_memory=False) for f in csv_files]
        df_raw = pd.concat(dfs, ignore_index=True)
    elif path.is_file():
        logger.info("Loading CICIDS2017 from single file: %s", path)
        df_raw = pd.read_csv(path, low_memory=False)
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

    logger.info("Raw shape: %s", df_raw.shape)

    # Normalise column names
    df_raw.columns = [c.strip().lower() for c in df_raw.columns]

    # --- Label extraction ---
    label_col = None
    for candidate in [" label", "label", "class", "attack_type"]:
        if candidate in df_raw.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError(f"Could not find label column. Available: {list(df_raw.columns)}")

    df_raw[label_col] = df_raw[label_col].astype(str).str.strip().str.lower()
    df_raw[LABEL_COLUMN] = df_raw[label_col].map(lambda x: CICIDS2017_LABEL_MAP.get(x, "DataExfil"))
    df_raw[BINARY_LABEL_COLUMN] = (df_raw[LABEL_COLUMN] != "NORMAL").astype(int)

    # --- Feature extraction ---
    out = pd.DataFrame()
    for raw_col, feat_name in _CICIDS_FEATURE_MAP.items():
        if raw_col in df_raw.columns and feat_name not in out.columns:
            out[feat_name] = pd.to_numeric(df_raw[raw_col], errors="coerce")

    # Fill missing NUMERIC_FEATURES with zeros
    for feat in NUMERIC_FEATURES:
        if feat not in out.columns:
            out[feat] = 0.0

    out = out[NUMERIC_FEATURES]
    out[LABEL_COLUMN] = df_raw[LABEL_COLUMN].values
    out[BINARY_LABEL_COLUMN] = df_raw[BINARY_LABEL_COLUMN].values

    # --- Clean ---
    if drop_infinite:
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        before = len(out)
        out.dropna(subset=NUMERIC_FEATURES, inplace=True)
        logger.info(
            "Dropped %d rows with Inf/NaN (%.1f%%)", before - len(out), 100 * (before - len(out)) / max(before, 1)
        )

    # Clip extreme values (99.9th percentile) to handle outlier flows
    for feat in NUMERIC_FEATURES:
        upper = out[feat].quantile(0.999)
        out[feat] = out[feat].clip(upper=upper)

    # Sample if requested
    if sample_frac < 1.0:
        out = out.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)

    out = out.reset_index(drop=True)

    label_dist = out[LABEL_COLUMN].value_counts().to_dict()
    logger.info(
        "CICIDS2017 loaded: %d samples | labels: %s",
        len(out),
        label_dist,
    )
    return out


# ---------------------------------------------------------------------------
# NSL-KDD Loader
# ---------------------------------------------------------------------------


def load_nslkdd(
    path: Union[str, Path],
    sample_frac: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load the NSL-KDD dataset from an ARFF or plain-text CSV file.

    The dataset is available from:
    https://www.unb.ca/cic/datasets/nsl.html

    Supported file formats:
    - KDDTrain+.arff / KDDTest+.arff
    - KDDTrain+.txt  / KDDTest+.txt   (comma-separated, same column order)
    - KDDTrain+_20Percent.arff

    Parameters
    ----------
    path : str or Path
        Path to the NSL-KDD ARFF or TXT file.
    sample_frac : float
        Fraction of data to sample.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame with columns matching NUMERIC_FEATURES + label + is_attack
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NSL-KDD file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".arff":
        df_raw = _parse_arff(path)
    else:
        # Plain text — same column order as ARFF data section
        logger.info("Loading NSL-KDD as plain CSV: %s", path)
        try:
            df_raw = pd.read_csv(path, header=None, names=_NSL_KDD_COLUMNS)
        except Exception as exc:
            raise ValueError(f"Failed to parse NSL-KDD text file: {exc}") from exc

    logger.info("Raw NSL-KDD shape: %s", df_raw.shape)

    # --- Label extraction ---
    raw_labels = df_raw["label"].astype(str).str.strip().str.lower()
    # Strip trailing dot if present (some versions have "normal." etc.)
    raw_labels = raw_labels.str.rstrip(".")
    df_raw[LABEL_COLUMN] = raw_labels.map(lambda x: NSLKDD_LABEL_MAP.get(x, "DataExfil"))
    df_raw[BINARY_LABEL_COLUMN] = (df_raw[LABEL_COLUMN] != "NORMAL").astype(int)

    # --- Protocol type encoding ---
    proto_map = {"tcp": 0, "udp": 1, "icmp": 2}
    if "protocol_type" in df_raw.columns:
        df_raw["protocol_type_enc"] = df_raw["protocol_type"].astype(str).str.lower().map(proto_map).fillna(0)

    # --- Feature extraction ---
    out = pd.DataFrame()
    for raw_col, feat_name in _NSL_KDD_FEATURE_MAP.items():
        if raw_col in df_raw.columns:
            out[feat_name] = pd.to_numeric(df_raw[raw_col], errors="coerce").fillna(0.0)

    # Protocol type
    if "protocol_type_enc" in df_raw.columns:
        out["protocol_type_enc"] = df_raw["protocol_type_enc"]

    # Derive flow-level features not directly in NSL-KDD
    # (approximate from available fields)
    if "src_bytes" in out.columns and "duration" in out.columns:
        dur = out["duration"].clip(lower=0.001)
        out["bytes_per_sec"] = (out["src_bytes"] + out.get("dst_bytes", pd.Series(0, index=out.index))) / dur
        out["packets_per_sec"] = out.get("count", pd.Series(1, index=out.index)) / dur

    # Fill all remaining NUMERIC_FEATURES with zeros
    for feat in NUMERIC_FEATURES:
        if feat not in out.columns:
            out[feat] = 0.0

    out = out[NUMERIC_FEATURES]
    out[LABEL_COLUMN] = df_raw[LABEL_COLUMN].values
    out[BINARY_LABEL_COLUMN] = df_raw[BINARY_LABEL_COLUMN].values

    # Clean
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.fillna(0.0, inplace=True)

    if sample_frac < 1.0:
        out = out.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)

    out = out.reset_index(drop=True)

    label_dist = out[LABEL_COLUMN].value_counts().to_dict()
    logger.info(
        "NSL-KDD loaded: %d samples | labels: %s",
        len(out),
        label_dist,
    )
    return out


# ---------------------------------------------------------------------------
# ARFF parser (no scipy dependency — custom pure-Python)
# ---------------------------------------------------------------------------


def _parse_arff(path: Path) -> pd.DataFrame:
    """
    Minimal ARFF parser that handles NSL-KDD's format.
    Returns a DataFrame with column names from @ATTRIBUTE lines.
    """
    logger.info("Parsing ARFF: %s", path)
    columns: list[str] = []
    data_lines: list[str] = []
    in_data = False

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower().startswith("@attribute"):
                # e.g. @attribute duration numeric
                parts = line.split(None, 2)
                col_name = parts[1].strip("'\"").lower()
                columns.append(col_name)
            elif line.lower().startswith("@data"):
                in_data = True
            elif in_data:
                data_lines.append(line)

    from io import StringIO

    csv_content = "\n".join(data_lines)
    df = pd.read_csv(StringIO(csv_content), header=None)

    # Assign column names (handle mismatch gracefully)
    if len(columns) >= df.shape[1]:
        df.columns = columns[: df.shape[1]]
    else:
        logger.warning(
            "ARFF column count mismatch: %d attributes vs %d data columns. Using positional names.",
            len(columns),
            df.shape[1],
        )
        df.columns = [f"col_{i}" for i in range(df.shape[1])]

    logger.info("ARFF parsed: %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Auto-detect loader
# ---------------------------------------------------------------------------


def autoload_dataset(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Auto-detect dataset format from path and load accordingly.

    Heuristics:
    - If path is a directory or the filenames contain "cicids" / "cic" → CICIDS2017
    - If path is an .arff or .txt file with "kdd" in name → NSL-KDD
    - If path is a CSV file, inspect headers to determine format

    Parameters
    ----------
    path : str or Path
    **kwargs : passed through to the underlying loader

    Returns
    -------
    pd.DataFrame in XAI-IDS canonical format
    """
    path = Path(path)
    name_lower = path.name.lower()

    if path.is_dir():
        # Check for CICIDS markers
        csv_files = list(path.glob("*.csv")) + list(path.glob("*.CSV"))
        if csv_files:
            first_name = csv_files[0].name.lower()
            if any(
                kw in first_name
                for kw in ["cic", "ids", "traffic", "monday", "tuesday", "wednesday", "thursday", "friday"]
            ):
                logger.info("Auto-detected: CICIDS2017 directory")
                return load_cicids2017(path, **kwargs)
        # Fallback: try CICIDS loader regardless
        logger.info("Directory with CSVs — attempting CICIDS2017 loader")
        return load_cicids2017(path, **kwargs)

    if path.is_file():
        if path.suffix.lower() == ".arff":
            logger.info("Auto-detected: NSL-KDD ARFF")
            return load_nslkdd(path, **kwargs)

        if "kdd" in name_lower or "nsl" in name_lower:
            logger.info("Auto-detected: NSL-KDD text")
            return load_nslkdd(path, **kwargs)

        if "cic" in name_lower or "ids" in name_lower:
            logger.info("Auto-detected: CICIDS2017 CSV")
            return load_cicids2017(path, **kwargs)

        # Inspect headers
        try:
            sample = pd.read_csv(path, nrows=2)
            cols_lower = [c.strip().lower() for c in sample.columns]
            if "protocol_type" in cols_lower and "num_failed_logins" in cols_lower:
                logger.info("Header inspection: NSL-KDD")
                return load_nslkdd(path, **kwargs)
            if "flow duration" in cols_lower or "flow bytes/s" in cols_lower:
                logger.info("Header inspection: CICIDS2017")
                return load_cicids2017(path, **kwargs)
        except Exception:
            pass

    raise ValueError(
        f"Cannot auto-detect dataset format for: {path}\n" "Use load_cicids2017() or load_nslkdd() directly."
    )


# ---------------------------------------------------------------------------
# Integration with DataPipeline
# ---------------------------------------------------------------------------


def load_real_dataset_for_training(
    dataset_path: Union[str, Path],
    artifact_dir: str = "trained_models",
    test_size: float = 0.15,
    val_size: float = 0.15,
    sample_frac: float = 1.0,
) -> dict:
    """
    High-level helper: load a real dataset and run it through the DataPipeline.

    Returns the same dict as DataPipeline.load_and_prepare():
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        y_multi_train, y_multi_val, y_multi_test,
        feature_names, class_names, n_features, n_classes

    This function is called by scripts/train.py when --dataset is provided.
    """
    from .pipeline import DataPipeline

    logger.info("Loading real dataset from: %s", dataset_path)
    df = autoload_dataset(dataset_path, sample_frac=sample_frac)

    # Save to temp CSV for DataPipeline compatibility
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    df.to_csv(tmp_path, index=False)
    logger.info("Saved normalised dataset to temp file: %s", tmp_path)

    pipeline = DataPipeline(artifact_dir=artifact_dir)
    data = pipeline.load_and_prepare(
        csv_path=tmp_path,
        test_size=test_size,
        val_size=val_size,
    )

    import os

    os.unlink(tmp_path)
    return data
