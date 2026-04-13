"""
evaluate.py  —  Downy Brome SDM
Spatial block cross-validation to check for overfitting.
Uses KMeans geographic blocking (matching MaxEnt evaluation approach) rather
than random stratified folds to account for spatial autocorrelation in presence data.
Reports mean ± std AUC and CBI across folds, plus train vs val AUC per fold.

Usage
-----
python evaluate.py \
    --shp  data/points_combined_culled.shp \
    --tif  data/emb11Nclp.tif             \
    --label_col label                      \
    --folds 5
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
import rasterio

from data import SpectraDataset
from model import get_model


# ---------------------------------------------------------------------------
# Spatial block fold builder
# Mirrors MaxEnt's _build_stratified_spatial_folds: cluster presence points
# geographically, assign absence/background points to nearest centroid.
# ---------------------------------------------------------------------------

def build_spatial_folds(coords, y, n_folds=5, random_state=42):
    """
    Build spatial CV folds stratified by presence point geography.

    Clusters presence points into n_folds geographic blocks via KMeans,
    then assigns absence points to the nearest block centroid. This prevents
    nearby points from appearing in both train and test sets, avoiding
    AUC inflation from spatial autocorrelation.

    Args:
        coords (np.ndarray): Shape (N, 2) — (x, y) coordinates per point.
        y (np.ndarray): Shape (N,) — binary labels (1=presence, 0=absence).
        n_folds (int): Number of spatial folds.
        random_state (int): Random seed.

    Yields:
        tuple[np.ndarray, np.ndarray]: (train_indices, val_indices) per fold.
    """
    presence_idx = np.where(y == 1)[0]
    absence_idx  = np.where(y == 0)[0]

    presence_coords = coords[presence_idx]
    absence_coords  = coords[absence_idx]

    # 1. Cluster presence points geographically
    km = KMeans(n_clusters=n_folds, random_state=random_state, n_init=10)
    presence_fold_labels = km.fit_predict(presence_coords)

    # 2. Assign absence points to nearest presence cluster centroid
    centroids = km.cluster_centers_
    dists = np.linalg.norm(absence_coords[:, None] - centroids[None], axis=2)
    absence_fold_labels = np.argmin(dists, axis=1)

    # 3. Build combined label array aligned to original indices
    fold_labels = np.empty(len(y), dtype=int)
    fold_labels[presence_idx] = presence_fold_labels
    fold_labels[absence_idx]  = absence_fold_labels

    for fold in range(n_folds):
        val_idx   = np.where(fold_labels == fold)[0]
        train_idx = np.where(fold_labels != fold)[0]
        yield train_idx, val_idx


# ---------------------------------------------------------------------------
# Continuous Boyce Index
# Ported from MaxEnt implementation (Hirzel et al. 2006).
# ---------------------------------------------------------------------------

def continuous_boyce_index(pred_presence, pred_absence, window_width=0.1, step=0.02):
    """
    Compute the Continuous Boyce Index (CBI) using a moving window.

    Slides an overlapping window across the predicted suitability range and
    computes the Spearman correlation between window centres and the
    predicted/expected frequency ratio. Positive values mean high suitability
    areas contain more presences than expected by chance.

    Args:
        pred_presence (np.ndarray): Model predictions at presence locations.
        pred_absence (np.ndarray): Model predictions at absence locations.
        window_width (float): Width of each moving window in suitability units.
        step (float): Step size between successive window centres.

    Returns:
        float: Spearman correlation (CBI), range [-1, 1]. NaN if insufficient data.
    """
    all_preds = np.concatenate([pred_presence, pred_absence])
    p_min, p_max = all_preds.min(), all_preds.max()

    centers   = np.arange(p_min + window_width / 2, p_max - window_width / 2 + step, step)
    pe_centers = []
    pe_ratios  = []

    for c in centers:
        lo, hi = c - window_width / 2, c + window_width / 2
        n_pres = np.sum((pred_presence >= lo) & (pred_presence < hi))
        n_abs  = np.sum((pred_absence  >= lo) & (pred_absence  < hi))
        if n_abs == 0:
            continue
        F_pres = n_pres / len(pred_presence)
        F_exp  = n_abs  / len(pred_absence)
        pe_centers.append(c)
        pe_ratios.append(F_pres / F_exp)

    if len(pe_ratios) < 3:
        return np.nan

    return spearmanr(pe_centers, pe_ratios).correlation


# ---------------------------------------------------------------------------
# Single fold train + eval
# ---------------------------------------------------------------------------

def run_fold(X_tr, y_tr, X_val, y_val, n_bands, device,
             epochs=100, batch_size=64, patience=20, lr=1e-3):

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr).astype(np.float32)
    X_val  = scaler.transform(X_val).astype(np.float32)

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
    best_preds_val = None
    patience_count = 0
    train_aucs     = []
    val_aucs       = []

    for epoch in range(1, epochs + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            def get_preds(loader):
                all_p, all_y = [], []
                for X, y in loader:
                    all_p.append(model(X.to(device)).cpu().numpy())
                    all_y.append(y.numpy())
                return np.concatenate(all_p).squeeze(), np.concatenate(all_y)

            tr_preds, tr_labels = get_preds(train_loader)
            vl_preds, vl_labels = get_preds(val_loader)

        tr_auc = roc_auc_score(tr_labels, tr_preds)
        vl_auc = roc_auc_score(vl_labels, vl_preds)
        train_aucs.append(tr_auc)
        val_aucs.append(vl_auc)

        scheduler.step(vl_auc)

        if vl_auc > best_val_auc:
            best_val_auc   = vl_auc
            best_preds_val = (vl_preds.copy(), vl_labels.copy())
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= patience:
            break

    return best_val_auc, best_preds_val, train_aucs, val_aucs


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

    fig.suptitle("Train vs Validation AUC — Spatial Block Cross-Validation\nDowny Brome SDM",
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
    print(f"\n{'='*60}")
    print(f"  Downy Brome SDM  —  {args.folds}-fold Spatial Block CV on {device}")
    print(f"{'='*60}\n")

    # Load points and extract raster values + coordinates
    gdf = gpd.read_file(args.shp)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=26911)

    with rasterio.open(args.tif) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        coords_xy = np.array([(geom.x, geom.y) for geom in gdf.geometry])
        samples   = list(src.sample([(x, y) for x, y in coords_xy]))
        nodata    = src.nodata if src.nodata is not None else -9999

    X = np.array(samples, dtype=np.float32)
    y = gdf[args.label_col].values.astype(np.int64)

    # Drop nodata rows, keeping coords aligned
    valid    = ~np.any(X == nodata, axis=1)
    X        = X[valid]
    y        = y[valid]
    coords   = coords_xy[valid]
    n_bands  = X.shape[1]

    print(f"  Sampled {len(X)} valid points  "
          f"({y.sum()} presence / {(y==0).sum()} absence)\n")

    fold_aucs      = []
    fold_cbis      = []
    all_train_aucs = []
    all_val_aucs   = []

    for fold, (tr_idx, vl_idx) in enumerate(
            build_spatial_folds(coords, y, n_folds=args.folds), 1):

        X_tr, X_val = X[tr_idx], X[vl_idx]
        y_tr, y_val = y[tr_idx], y[vl_idx]

        n_val_p = int((y_val == 1).sum())
        if n_val_p == 0:
            print(f"  Fold {fold}/{args.folds} → skipped (no presence points in val block)")
            continue
        if n_val_p < 5:
            print(f"  WARNING: Fold {fold} has only {n_val_p} val presence point(s) — metrics unreliable")

        print(f"  Fold {fold}/{args.folds}  "
              f"(train: {(y_tr==1).sum()}p/{(y_tr==0).sum()}a  "
              f"val: {(y_val==1).sum()}p/{(y_val==0).sum()}a) ", end="", flush=True)

        best_auc, best_preds, tr_aucs, vl_aucs = run_fold(
            X_tr, y_tr, X_val, y_val, n_bands, device,
            epochs=args.epochs, patience=args.patience
        )

        # CBI on best-epoch val predictions
        vl_preds, vl_labels = best_preds
        cbi = continuous_boyce_index(
            vl_preds[vl_labels == 1],
            vl_preds[vl_labels == 0],
        )

        fold_aucs.append(best_auc)
        fold_cbis.append(cbi)
        all_train_aucs.append(tr_aucs)
        all_val_aucs.append(vl_aucs)

        gap = tr_aucs[np.argmax(vl_aucs)] - best_auc
        cbi_str = f"{cbi:.4f}" if not np.isnan(cbi) else "  nan"
        print(f"→  Val AUC: {best_auc:.4f}  |  CBI: {cbi_str}  |  Train-Val gap: {gap:+.4f}")

    mean_auc = np.mean(fold_aucs)
    std_auc  = np.std(fold_aucs)
    mean_cbi = np.nanmean(fold_cbis)
    std_cbi  = np.nanstd(fold_cbis)

    print(f"\n{'='*60}")
    print(f"  Mean Val AUC : {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Mean CBI     : {mean_cbi:.4f} ± {std_cbi:.4f}")
    print(f"  Per-fold AUC : {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"  Per-fold CBI : {[f'{c:.4f}' if not np.isnan(c) else 'nan' for c in fold_cbis]}")
    print(f"{'='*60}\n")

    if std_auc < 0.05:
        print("  ✓ Low AUC variance across folds — model is stable.")
    else:
        print("  ⚠ High AUC variance across folds — performance varies across the study area.")

    plot_fold_curves(all_train_aucs, all_val_aucs, out_path=args.out_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial block CV for Downy Brome SDM")
    parser.add_argument("--shp",       required=True)
    parser.add_argument("--tif",       required=True)
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--folds",     type=int, default=5)
    parser.add_argument("--epochs",    type=int, default=100)
    parser.add_argument("--patience",  type=int, default=20)
    parser.add_argument("--out_png",   default="cv_curves.png")
    args = parser.parse_args()
    main(args)
