"""
train.py  —  Downy Brome SDM
Full training loop: BCE loss, Adam optimiser, early stopping, best checkpoint.

Usage
-----
python train.py \
    --shp  data/downy_brome_points.shp \
    --tif  data/environment_stack.tif  \
    --label_col label                  \
    --epochs 100                       \
    --batch  64                        \
    --lr     1e-3
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from sklearn.metrics import roc_auc_score

from data import make_dataloaders

# ---------------------------------------------------------------------------
# Training / validation steps
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(X)
        loss  = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device).unsqueeze(1)
        preds = model(X)
        total_loss += criterion(preds, y).item()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_preds)
    return total_loss / len(loader), auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  Downy Brome SDM  —  training on {device}")
    print(f"{'='*50}\n")

    # Data
    train_loader, val_loader, n_bands = make_dataloaders(
        shp_path    = args.shp,
        tif_path    = args.tif,
        label_col   = args.label_col,
        val_size    = 0.2,
        batch_size  = args.batch,
        scaler_path = args.scaler_out,
    )

    # Model
    model     = get_model(n_bands=n_bands).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    # Early stopping state
    best_auc        = 0.0
    patience_count  = 0
    best_ckpt_path  = Path(args.checkpoint)

    print(f"  Bands: {n_bands}  |  Epochs: {args.epochs}  |  "
          f"Batch: {args.batch}  |  LR: {args.lr}\n")
    print(f"  {'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val AUC':>8}  {'':>6}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss         = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, auc   = val_epoch(model, val_loader, criterion, device)
        scheduler.step(auc)

        flag = ""
        if auc > best_auc:
            best_auc = auc
            patience_count = 0
            torch.save({
                "epoch":    epoch,
                "n_bands":  n_bands,
                "state_dict": model.state_dict(),
                "best_auc": best_auc,
            }, best_ckpt_path)
            flag = "✓ saved"
        else:
            patience_count += 1

        elapsed = time.time() - t0
        print(f"  {epoch:>6}  {tr_loss:>11.4f}  {val_loss:>9.4f}  "
              f"{auc:>8.4f}  {flag:<8}  ({elapsed:.1f}s)")

        if patience_count >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}  "
                  f"(no improvement for {args.patience} epochs)")
            break

    print(f"\n  Best Val AUC: {best_auc:.4f}  →  checkpoint: {best_ckpt_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Downy Brome SDM")
    parser.add_argument("--shp",        required=True,  help="Path to presence/absence shapefile")
    parser.add_argument("--tif",        required=True,  help="Path to 64-band raster stack")
    parser.add_argument("--label_col",  default="label",help="Shapefile column with 0/1 labels")
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--batch",      type=int, default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int, default=20,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--checkpoint", default="best_model.pt",
                        help="Where to save the best model checkpoint")
    parser.add_argument("--scaler_out", default="scaler.joblib",
                        help="Where to save the fitted StandardScaler")
    args = parser.parse_args()
    main(args)
