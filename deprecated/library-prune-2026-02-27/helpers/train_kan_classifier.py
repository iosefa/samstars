#!/usr/bin/env python3
"""
Train a lightweight KAN-style classifier on per-crown features built from StarDist encoder pooling + descriptors.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class KANLayer(nn.Module):
    """Simple 1-D basis layer: project, apply RBF bases, sum with learnable coefficients."""

    def __init__(self, in_dim: int, out_dim: int, n_centers: int = 8):
        """Initialize the instance."""
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.centers = nn.Parameter(torch.linspace(-1, 1, n_centers).repeat(out_dim, 1))
        self.log_sigma = nn.Parameter(torch.zeros(out_dim, n_centers))
        self.coeff = nn.Parameter(torch.randn(out_dim, n_centers) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        z = self.proj(x)[:, :, None]  # (B, out_dim, 1)
        c = self.centers[None, :, :]
        s = torch.exp(self.log_sigma)[None, :, :]
        phi = torch.exp(-0.5 * ((z - c) / (s + 1e-6)) ** 2)  # (B, out_dim, n_centers)
        h = (phi * self.coeff[None]).sum(dim=2) + self.bias
        return h


class KANClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int, n_centers: int = 8):
        """Initialize the instance."""
        super().__init__()
        self.layer1 = KANLayer(in_dim, hidden, n_centers)
        self.layer2 = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        h = torch.relu(self.layer1(x))
        return self.layer2(h)


def standardize(X: np.ndarray):
    """Standardize."""
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    return (X - mean) / std, mean, std


def main():
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(description="Train KAN classifier on per-crown features.")
    pa.add_argument("--dataset", required=True, type=Path, help="npz from kan_feature_builder.py")
    pa.add_argument("--hidden", type=int, default=64)
    pa.add_argument("--centers", type=int, default=8)
    pa.add_argument("--epochs", type=int, default=50)
    pa.add_argument("--batch", type=int, default=128)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--val-frac", type=float, default=0.15)
    pa.add_argument("--out-dir", default="models/kan", type=Path)
    args = pa.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    X, y = data["X"], data["y"]
    meta = json.loads(str(data["meta"]))

    classes, y_idx = np.unique(y, return_inverse=True)
    Xs, mean, std = standardize(X)

    X_t = torch.tensor(Xs, dtype=torch.float32)
    y_t = torch.tensor(y_idx, dtype=torch.long)

    ds = TensorDataset(X_t, y_t)
    n_val = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    dl_train = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    dl_val = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KANClassifier(in_dim=X.shape[1], hidden=args.hidden, n_classes=len(classes), n_centers=args.centers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best_val = -1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        train_loss = tot / len(train_ds)

        model.eval()
        correct = 0
        totv = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                totv += len(xb)
        val_acc = correct / max(1, totv)
        print(f"epoch {ep:03d} train_loss={train_loss:.4f} val_acc={val_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "mean": mean,
                "std": std,
                "meta": meta,
                "config": vars(args),
            }, args.out_dir / "kan_weights.pt")

    print(f"✓ saved best model (val_acc={best_val:.3f}) to {args.out_dir/'kan_weights.pt'}")


if __name__ == "__main__":
    main()
