"""
evaluate.py  —  Downy Brome SDM
5-fold stratified cross-validation to check for overfitting.
Reports mean ± std AUC across folds, plus train vs val AUC per fold.

Usage
-----
python evaluate.py \
    --shp  data/points_combined.shp \
    --tif  data/emb11Nclp.tif       \
    --label_col label                \
    --folds 5
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import matplotlib as mpl

from data import sample_raster_at_points, SpectraDataset
from model import get_model


# ---------------------------------------------------------------------------
# Single fold train + eval
# ---------------------------------------------------------------------------

def run_fold(X_tr, y_tr, X_val, y_val, n_bands, device,
             epochs=100, batch_size=64, patience=20, lr=1e-3):

    # Normalise within fold (fit on train only)
    scaler   = StandardScaler()
    X_tr     = scaler.fit_transform(X_tr).astype(np.float32)
    X_val    = scaler.transform(X_val).astype(np.float32)

    # Dataloaders
    train_ds = SpectraDataset(X_tr, y_tr)
    val_ds   = SpectraDataset(X_val, y_val)

    class_counts = np.bincount(y_tr)
    weights      = 1.0 / class_counts[y_tr]
    sampler      = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model     = get_model(n_bands=n_bands).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    best_val_auc   = 0.0
    best_state     = None
    patience_count = 0

    train_aucs = []
    val_aucs   = []

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        # --- Evaluate ---
        model.eval()
        with torch.no_grad():
            def get_preds(loader):
                all_p, all_y = [], []
                for X, y in loader:
                    all_p.append(model(X.to(device)).cpu().numpy())
                    all_y.append(y.numpy())
                return np.concatenate(all_p), np.concatenate(all_y)

            tr_preds, tr_labels = get_preds(train_loader)
            vl_preds, vl_labels = get_preds(val_loader)

        tr_auc = roc_auc_score(tr_labels, tr_preds)
        vl_auc = roc_auc_score(vl_labels, vl_preds)

        train_aucs.append(tr_auc)
        val_aucs.append(vl_auc)

        scheduler.step(vl_auc)

        if vl_auc > best_val_auc:
            best_val_auc = vl_auc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= patience:
            break

    return best_val_auc, train_aucs, val_aucs


# ---------------------------------------------------------------------------
# Plot: train vs val AUC per fold
# ---------------------------------------------------------------------------

def plot_fold_curves(all_train_aucs, all_val_aucs, out_path="cv_curves.png"):
    mpl.rcParams.update({"font.family": "serif"})

    fig, axes = plt.subplots(1, len(all_train_aucs),
                             figsize=(4 * len(all_train_aucs), 4),
                             sharey=True)
    fig.patch.set_facecolor("#1a1a1a")

    colors = {"train": "#f0a500", "val": "#00e5ff"}

    for i, (tr, vl) in enumerate(zip(all_train_aucs, all_val_aucs)):
        ax = axes[i]
        ax.set_facecolor("#1a1a1a")
        ax.plot(tr, color=colors["train"], lw=1.5, label="Train AUC")
        ax.plot(vl, color=colors["val"],   lw=1.5, label="Val AUC")
        ax.set_title(f"Fold {i+1}", color="white", fontsize=11)
        ax.set_xlabel("Epoch", color="#aaaaaa", fontsize=9)
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        if i == 0:
            ax.set_ylabel("AUC", color="#aaaaaa", fontsize=9)
            ax.legend(fontsize=8, labelcolor="white",
                      facecolor="#333333", edgecolor="#555555")

    fig.suptitle("Train vs Validation AUC — 5-Fold Cross Validation\nDowny Brome SDM",
                 color="white", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Curves saved → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  Downy Brome SDM  —  {args.folds}-fold CV on {device}")
    print(f"{'='*50}\n")

    X, y = sample_raster_at_points(args.shp, args.tif, args.label_col)
    n_bands = X.shape[1]

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    fold_aucs      = []
    all_train_aucs = []
    all_val_aucs   = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}/{args.folds} ", end="", flush=True)
        X_tr, X_val = X[tr_idx], X[vl_idx]
        y_tr, y_val = y[tr_idx], y[vl_idx]

        best_auc, tr_aucs, vl_aucs = run_fold(
            X_tr, y_tr, X_val, y_val, n_bands, device,
            epochs=args.epochs, patience=args.patience
        )

        fold_aucs.append(best_auc)
        all_train_aucs.append(tr_aucs)
        all_val_aucs.append(vl_aucs)

        gap = tr_aucs[np.argmax(vl_aucs)] - best_auc
        print(f"→  Val AUC: {best_auc:.4f}  |  Train-Val gap: {gap:+.4f}")

    mean_auc = np.mean(fold_aucs)
    std_auc  = np.std(fold_aucs)

    print(f"\n{'='*50}")
    print(f"  Mean Val AUC : {mean_auc:.4f}")
    print(f"  Std  Val AUC : {std_auc:.4f}")
    print(f"  Per-fold     : {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"{'='*50}\n")

    if std_auc < 0.05:
        print("  ✓ Low variance across folds — model is stable, not overfitting.")
    else:
        print("  ⚠ High variance across folds — possible overfitting or small dataset instability.")

    plot_fold_curves(all_train_aucs, all_val_aucs, out_path=args.out_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-fold CV for Downy Brome SDM")
    parser.add_argument("--shp",       required=True)
    parser.add_argument("--tif",       required=True)
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--folds",     type=int, default=5)
    parser.add_argument("--epochs",    type=int, default=100)
    parser.add_argument("--patience",  type=int, default=20)
    parser.add_argument("--out_png",   default="cv_curves.png")
    args = parser.parse_args()
    main(args)
