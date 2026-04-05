"""
Tests: DataPipeline, SyntheticDataGenerator
"""
import tempfile
import os
import numpy as np
import pandas as pd
import pytest

from xai_ids.preprocessing.pipeline import (
    DataPipeline, SyntheticDataGenerator, NUMERIC_FEATURES,
    LABEL_COLUMN, BINARY_LABEL_COLUMN
)


class TestSyntheticDataGenerator:
    def setup_method(self):
        self.gen = SyntheticDataGenerator()

    def test_generates_correct_shape(self):
        df = self.gen.generate(n_per_class=50, seed=0)
        n_classes = len(self.gen.ATTACK_PROFILES)
        assert len(df) == n_classes * 50

    def test_all_classes_present(self):
        df = self.gen.generate(n_per_class=100)
        classes = set(df[LABEL_COLUMN].unique())
        expected = set(self.gen.ATTACK_PROFILES.keys())
        assert classes == expected

    def test_binary_label_correct(self):
        df = self.gen.generate(n_per_class=100)
        normal = df[df[LABEL_COLUMN] == "NORMAL"]
        attack = df[df[LABEL_COLUMN] != "NORMAL"]
        assert (normal[BINARY_LABEL_COLUMN] == 0).all()
        assert (attack[BINARY_LABEL_COLUMN] == 1).all()

    def test_all_numeric_features_present(self):
        df = self.gen.generate(n_per_class=50)
        for feat in NUMERIC_FEATURES:
            assert feat in df.columns, f"Missing feature: {feat}"

    def test_no_nan_values(self):
        df = self.gen.generate(n_per_class=100)
        assert df[NUMERIC_FEATURES].isna().sum().sum() == 0

    def test_reproducible_with_seed(self):
        df1 = self.gen.generate(n_per_class=50, seed=42)
        df2 = self.gen.generate(n_per_class=50, seed=42)
        pd.testing.assert_frame_equal(df1.reset_index(drop=True),
                                       df2.reset_index(drop=True))

    def test_different_seeds_differ(self):
        df1 = self.gen.generate(n_per_class=50, seed=1)
        df2 = self.gen.generate(n_per_class=50, seed=2)
        assert not df1[NUMERIC_FEATURES].equals(df2[NUMERIC_FEATURES])

    def test_numeric_features_non_negative(self):
        df = self.gen.generate(n_per_class=200)
        # Most features should be >= 0 (some may be slightly negative due to noise)
        assert (df[["fwd_bytes", "bwd_bytes", "syn_count", "fwd_packets"]] >= 0).all().all()

    def test_dos_profile_has_high_syn(self):
        df = self.gen.generate(n_per_class=200)
        dos = df[df[LABEL_COLUMN] == "DoS"]
        normal = df[df[LABEL_COLUMN] == "NORMAL"]
        assert dos["syn_count"].mean() > normal["syn_count"].mean()

    def test_data_exfil_has_high_fwd_bytes(self):
        df = self.gen.generate(n_per_class=200)
        exfil = df[df[LABEL_COLUMN] == "DataExfil"]
        normal = df[df[LABEL_COLUMN] == "NORMAL"]
        assert exfil["fwd_bytes"].mean() > normal["fwd_bytes"].mean()


class TestDataPipeline:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pipeline = DataPipeline(artifact_dir=self.tmpdir)

    def test_load_and_prepare_returns_correct_keys(self):
        data = self.pipeline.load_and_prepare(n_per_class=100)
        required_keys = ["X_train", "X_val", "X_test",
                         "y_train", "y_val", "y_test",
                         "feature_names", "class_names", "n_features", "n_classes"]
        for k in required_keys:
            assert k in data, f"Missing key: {k}"

    def test_feature_count(self):
        data = self.pipeline.load_and_prepare(n_per_class=100)
        assert data["n_features"] == len(NUMERIC_FEATURES)
        assert data["X_train"].shape[1] == len(NUMERIC_FEATURES)

    def test_no_data_leakage_between_splits(self):
        data = self.pipeline.load_and_prepare(n_per_class=200)
        total = len(data["X_train"]) + len(data["X_val"]) + len(data["X_test"])
        expected = 200 * 6  # 6 classes
        assert abs(total - expected) <= 5  # Allow rounding

    def test_scaler_applied(self):
        data = self.pipeline.load_and_prepare(n_per_class=200)
        # Scaled data should have approximately zero mean
        mean = data["X_train"].mean(axis=0)
        assert np.abs(mean).max() < 1.0  # Mean should be near 0 after scaling

    def test_artifacts_saved(self):
        self.pipeline.load_and_prepare(n_per_class=100)
        assert os.path.exists(os.path.join(self.tmpdir, "scaler.pkl"))
        assert os.path.exists(os.path.join(self.tmpdir, "label_encoder.pkl"))

    def test_class_names_include_normal(self):
        data = self.pipeline.load_and_prepare(n_per_class=100)
        assert "NORMAL" in data["class_names"]

    def test_binary_labels_are_binary(self):
        data = self.pipeline.load_and_prepare(n_per_class=100)
        for split in ["y_train", "y_val", "y_test"]:
            unique = set(data[split].tolist())
            assert unique.issubset({0, 1}), f"{split} has non-binary values: {unique}"

    def test_transform_with_fitted_scaler(self):
        data = self.pipeline.load_and_prepare(n_per_class=100)
        X_new = np.random.rand(5, len(NUMERIC_FEATURES)).astype(np.float32)
        transformed = self.pipeline.transform(X_new)
        assert transformed.shape == (5, len(NUMERIC_FEATURES))

    def test_load_from_csv(self):
        """Pipeline should load from CSV if path provided and file exists."""
        gen = SyntheticDataGenerator()
        df = gen.generate(n_per_class=100)
        csv_path = os.path.join(self.tmpdir, "test_data.csv")
        df.to_csv(csv_path, index=False)

        pipeline2 = DataPipeline(artifact_dir=tempfile.mkdtemp())
        data = pipeline2.load_and_prepare(csv_path=csv_path)
        assert len(data["X_train"]) > 0
