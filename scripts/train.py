"""
XAI-IDS Training Script
=========================
Full training pipeline:
1. Generate/load dataset (synthetic OR real CICIDS2017 / NSL-KDD)
2. Preprocess and split
3. Train IDSNet with adversarial hardening
4. Evaluate on test set
5. Save model and artifacts
6. Print classification report

Usage examples:
  # Synthetic dataset (default)
  python scripts/train.py --epochs 50

  # CICIDS2017 real dataset
  python scripts/train.py --dataset /data/cicids2017/ --epochs 50

  # NSL-KDD real dataset
  python scripts/train.py --dataset /data/nslkdd/KDDTrain+.arff --epochs 50

  # Smoke test (CI)
  python scripts/train.py --epochs 2 --samples-per-class 50 --batch-size 16 --no-adversarial
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from xai_ids.models.ids_model import IDSNet, IDSTrainer, make_data_loaders
from xai_ids.preprocessing.pipeline import DataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("xai_ids.train")


def evaluate(model, test_loader, class_names, device):
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    model.eval()
    all_probs, all_labels, all_multi_preds, all_multi_true = [], [], [], []

    with torch.no_grad():
        for X, y_bin, y_multi in test_loader:
            X = X.to(device)
            proba = model.predict_proba(X)
            attack_probs = proba["attack_probability"].cpu().numpy()
            multi_preds = proba["class_probabilities"].argmax(dim=1).cpu().numpy()
            all_probs.extend(attack_probs)
            all_labels.extend(y_bin.numpy())
            all_multi_preds.extend(multi_preds)
            all_multi_true.extend(y_multi.numpy())

    binary_preds = (np.array(all_probs) > 0.5).astype(int)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5

    print("\n" + "=" * 60)
    print("  XAI-IDS Test Set Evaluation")
    print("=" * 60)
    print("\n  Binary Classification (Attack vs Normal)")
    print(f"  AUC-ROC: {auc:.4f}")
    print(classification_report(all_labels, binary_preds, target_names=["Normal", "Attack"]))

    print("  Multi-class Attack Classification")
    print(
        classification_report(
            all_multi_true,
            all_multi_preds,
            target_names=class_names,
            zero_division=0,
        )
    )

    cm = confusion_matrix(all_multi_true, all_multi_preds)
    print("  Confusion Matrix (rows=true, cols=pred):")
    print("  Classes:", class_names)
    print(cm)
    print("=" * 60 + "\n")

    return {"auc": auc, "confusion_matrix": cm.tolist()}


def main():
    parser = argparse.ArgumentParser(
        description="XAI-IDS Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data",
        "--csv",
        default=None,
        help="Path to a preprocessed CSV file (legacy flag, use --dataset for auto-detection)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Path to a real IDS dataset for auto-detection and loading. "
            "Supports CICIDS2017 (directory or CSV) and NSL-KDD (.arff or .txt). "
            "If not provided, a synthetic dataset is generated."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--samples-per-class",
        "--n-per-class",
        type=int,
        default=3000,
        dest="n_per_class",
        help="Samples per class when using synthetic data (ignored for real datasets)",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Fraction of real dataset to sample (e.g. 0.1 for 10%% — useful for large datasets)",
    )
    parser.add_argument("--no-adversarial", action="store_true", help="Disable adversarial training")
    parser.add_argument(
        "--output-dir",
        "--save-dir",
        default="trained_models",
        dest="save_dir",
        help="Directory to save model, scaler, label encoder, and metrics",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # -------------------------------------------------------------------------
    # Data loading — real dataset takes priority over --data CSV
    # -------------------------------------------------------------------------
    if args.dataset:
        logger.info("Loading real dataset from: %s", args.dataset)
        from xai_ids.preprocessing.dataset_loaders import load_real_dataset_for_training

        data = load_real_dataset_for_training(
            dataset_path=args.dataset,
            artifact_dir=args.save_dir,
            sample_frac=args.sample_frac,
        )
    else:
        if args.data:
            logger.info("Loading CSV dataset from: %s", args.data)
        else:
            logger.info("No dataset provided — generating synthetic data (%d/class)", args.n_per_class)

        pipeline = DataPipeline(artifact_dir=args.save_dir)
        data = pipeline.load_and_prepare(csv_path=args.data, n_per_class=args.n_per_class)

    train_loader, val_loader, test_loader = make_data_loaders(data, batch_size=args.batch_size)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = IDSNet(
        n_features=data["n_features"],
        n_classes=data["n_classes"],
        hidden_dim=256,
        dropout=0.3,
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "IDSNet initialized | features=%d classes=%d params=%d",
        data["n_features"],
        data["n_classes"],
        total_params,
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    trainer = IDSTrainer(
        model=model,
        device=device,
        lr=args.lr,
        adversarial_training=not args.no_adversarial,
        save_dir=args.save_dir,
    )
    history = trainer.train(train_loader, val_loader, epochs=args.epochs, patience=10)

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    metrics = evaluate(model, test_loader, data["class_names"], device)

    # Save training metrics (for CI artifact verification)
    output_path = Path(args.save_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hist_path = output_path / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump({"history": history, "metrics": metrics}, f, indent=2)

    metrics_path = output_path / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "auc_roc": metrics["auc"],
                "n_classes": data["n_classes"],
                "class_names": data["class_names"],
                "n_features": data["n_features"],
                "epochs": args.epochs,
                "device": device,
            },
            f,
            indent=2,
        )

    logger.info("Training history saved to %s", hist_path)
    logger.info("Training complete. Model saved to %s/best_model.pt", args.save_dir)


if __name__ == "__main__":
    main()
