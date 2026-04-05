"""
XAI-IDS - Deep Learning IDS Model
====================================
Dual-head neural network architecture:
  - Binary head: attack vs normal classification
  - Multi-class head: attack type classification

Architecture:
  Input → BatchNorm → FC(256) → GELU → Dropout → 
          ResBlock(256) → ResBlock(128) →
          Shared Representation →
          ┌── Binary head: FC(64) → sigmoid
          └── Multi-class head: FC(64) → softmax

Supports:
- Standard batch training
- Online learning (incremental updates)
- Adversarial training (PGD attacks for robustness)
- Model export (ONNX, TorchScript)
- Confidence calibration via temperature scaling
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual block with pre-activation (He et al.)."""
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.block(x))


class IDSNet(nn.Module):
    """
    Dual-head IDS network.

    Args:
        n_features: Number of input features
        n_classes: Number of attack categories (including normal)
        hidden_dim: Width of hidden layers
        dropout: Dropout rate
        temperature: Temperature for confidence calibration
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.temperature = nn.Parameter(torch.ones(1) * temperature, requires_grad=False)

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            ResidualBlock(hidden_dim // 2, dropout),
        )
        shared_out = hidden_dim // 2

        # Binary classification head
        self.binary_head = nn.Sequential(
            nn.Linear(shared_out, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

        # Multi-class classification head
        self.multi_head = nn.Sequential(
            nn.Linear(shared_out, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            binary_logit: (B,) raw logit for attack probability
            multi_logit: (B, n_classes) raw logits for attack type
            embedding: (B, shared_dim) shared representation for XAI
        """
        emb = self.backbone(x)
        binary_logit = self.binary_head(emb).squeeze(-1)
        multi_logit = self.multi_head(emb)
        return binary_logit, multi_logit, emb

    def predict_proba(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calibrated probability predictions."""
        binary_logit, multi_logit, emb = self(x)
        binary_prob = torch.sigmoid(binary_logit / self.temperature)
        multi_prob = F.softmax(multi_logit / self.temperature, dim=-1)
        return {
            "attack_probability": binary_prob,
            "class_probabilities": multi_prob,
            "embedding": emb,
        }

    def calibrate_temperature(self, val_loader: DataLoader, device: str = "cpu"):
        """
        Post-hoc calibration via temperature scaling.
        Finds the temperature T that minimizes NLL on the validation set.
        """
        self.temperature.requires_grad_(True)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.BCEWithLogitsLoss()

        logits_list, labels_list = [], []
        with torch.no_grad():
            for X, y_bin, _ in val_loader:
                X, y_bin = X.to(device), y_bin.to(device)
                logit, _, _ = self(X)
                logits_list.append(logit)
                labels_list.append(y_bin.float())

        logits_all = torch.cat(logits_list)
        labels_all = torch.cat(labels_list)

        def eval_step():
            optimizer.zero_grad()
            scaled = logits_all / self.temperature
            loss = criterion(scaled, labels_all)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        self.temperature.requires_grad_(False)
        logger.info("Temperature calibrated to %.4f", self.temperature.item())


# ---------------------------------------------------------------------------
# Adversarial training utilities
# ---------------------------------------------------------------------------

class PGDAttack:
    """
    Projected Gradient Descent (Madry et al.) for adversarial robustness training.
    Applied to the input feature space.
    """

    def __init__(self, eps: float = 0.1, alpha: float = 0.01, steps: int = 10):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def perturb(self, model: IDSNet, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples via PGD."""
        X_adv = X.clone().detach() + torch.empty_like(X).uniform_(-self.eps, self.eps)
        X_adv = X_adv.requires_grad_(True)
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(self.steps):
            logit, _, _ = model(X_adv)
            loss = criterion(logit, y.float())
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                X_adv = X_adv + self.alpha * X_adv.grad.sign()
                delta = torch.clamp(X_adv - X, -self.eps, self.eps)
                X_adv = (X + delta).detach().requires_grad_(True)

        return X_adv.detach()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class IDSTrainer:
    """Full training pipeline with early stopping, mixed-loss, and adversarial hardening."""

    def __init__(
        self,
        model: IDSNet,
        device: str = "cpu",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        adversarial_training: bool = True,
        adv_ratio: float = 0.3,
        save_dir: str = "trained_models",
    ):
        self.model = model.to(device)
        self.device = device
        self.adv_training = adversarial_training
        self.adv_ratio = adv_ratio
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.binary_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))
        self.multi_criterion = nn.CrossEntropyLoss()
        self.pgd = PGDAttack()
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 8,
    ) -> Dict[str, List[float]]:
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_metrics = self._validate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_auc"].append(val_metrics["auc"])

            scheduler.step()

            logger.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_auc=%.4f",
                epoch, epochs, train_loss, val_metrics["loss"], val_metrics["accuracy"], val_metrics["auc"],
            )

            # Early stopping
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                self._save_checkpoint("best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # Load best model
        self._load_checkpoint("best_model.pt")
        # Calibrate temperature on val set
        self.model.calibrate_temperature(val_loader, self.device)
        return self.history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X, y_bin, y_multi in loader:
            X, y_bin, y_multi = X.to(self.device), y_bin.to(self.device), y_multi.to(self.device)

            # Adversarial examples (mixed batch)
            if self.adv_training:
                n_adv = int(len(X) * self.adv_ratio)
                X_adv = self.pgd.perturb(self.model, X[:n_adv], y_bin[:n_adv])
                X = torch.cat([X[n_adv:], X_adv], dim=0)
                y_bin = torch.cat([y_bin[n_adv:], y_bin[:n_adv]], dim=0)
                y_multi = torch.cat([y_multi[n_adv:], y_multi[:n_adv]], dim=0)

            self.optimizer.zero_grad()
            binary_logit, multi_logit, _ = self.model(X)
            loss_bin = self.binary_criterion(binary_logit, y_bin.float())
            loss_multi = self.multi_criterion(multi_logit, y_multi)
            loss = 0.6 * loss_bin + 0.4 * loss_multi
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        from sklearn.metrics import roc_auc_score, accuracy_score

        self.model.eval()
        total_loss, all_probs, all_labels, all_preds = 0.0, [], [], []

        with torch.no_grad():
            for X, y_bin, y_multi in loader:
                X, y_bin, y_multi = X.to(self.device), y_bin.to(self.device), y_multi.to(self.device)
                binary_logit, multi_logit, _ = self.model(X)
                loss = 0.6 * self.binary_criterion(binary_logit, y_bin.float()) + \
                       0.4 * self.multi_criterion(multi_logit, y_multi)
                total_loss += loss.item()
                probs = torch.sigmoid(binary_logit).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y_bin.cpu().numpy())
                all_preds.extend((probs > 0.5).astype(int))

        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
        return {
            "loss": total_loss / len(loader),
            "accuracy": accuracy_score(all_labels, all_preds),
            "auc": auc,
        }

    def _save_checkpoint(self, name: str):
        torch.save(self.model.state_dict(), self.save_dir / name)

    def _load_checkpoint(self, name: str):
        path = self.save_dir / name
        if path.exists():
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info("Loaded checkpoint from %s", path)


def make_data_loaders(data: Dict, batch_size: int = 512) -> Tuple[DataLoader, DataLoader, DataLoader]:
    def to_loader(X, y_bin, y_multi, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y_bin, dtype=torch.long),
            torch.tensor(y_multi, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)

    return (
        to_loader(data["X_train"], data["y_train"], data["y_multi_train"], shuffle=True),
        to_loader(data["X_val"], data["y_val"], data["y_multi_val"]),
        to_loader(data["X_test"], data["y_test"], data["y_multi_test"]),
    )
