"""
data.py  —  Downy Brome SDM
Loads presence/absence shapefile, samples a multi-band raster at each point,
and returns train/val DataLoaders.

Expected shapefile schema:
  geometry : Point (CRS must be geographic, EPSG:4326, or matching the raster)
  label    : int  (1 = presence, 0 = absence)
              OR any column name you pass as `label_col`
"""

from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# 1.  Point sampling
# ---------------------------------------------------------------------------

def sample_raster_at_points(shp_path: str,
                             tif_path: str,
                             label_col: str = "label") -> tuple[np.ndarray, np.ndarray]:
    """
    Sample all bands of `tif_path` at every point in `shp_path`.

    Returns
    -------
    X : float32 array  (N, n_bands)
    y : int array      (N,)          1 = presence, 0 = absence
    """
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=3857)

    with rasterio.open(tif_path) as src:
        # Reproject points to raster CRS if needed
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        n_bands = src.count
        coords = [(geom.x, geom.y) for geom in gdf.geometry]

        # rasterio.sample returns an iterator of shape (n_bands,) per point
        samples = list(src.sample(coords))          # list of (n_bands,) arrays

    X = np.array(samples, dtype=np.float32)         # (N, n_bands)
    y = gdf[label_col].values.astype(np.int64)      # (N,)

    # Drop rows where any band is nodata (common at raster edges)
    with rasterio.open(tif_path) as src:
        nodata = src.nodata if src.nodata is not None else -9999
    valid = ~np.any(X == nodata, axis=1)
    X, y = X[valid], y[valid]

    print(f"  Sampled {len(X)} valid points  "
          f"({y.sum()} presence / {(y==0).sum()} absence)")
    return X, y


# ---------------------------------------------------------------------------
# 2.  Normalisation
# ---------------------------------------------------------------------------

def fit_and_save_scaler(X_train: np.ndarray, scaler_path: str = "scaler.joblib"):
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved → {scaler_path}")
    return scaler


def load_scaler(scaler_path: str = "scaler.joblib"):
    return joblib.load(scaler_path)


# ---------------------------------------------------------------------------
# 3.  PyTorch Dataset
# ---------------------------------------------------------------------------

class SpectraDataset(Dataset):
    """Each item is a (n_bands,) spectral vector + binary label."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)          # float32
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# 4.  DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(shp_path: str,
                     tif_path: str,
                     label_col: str = "label",
                     val_size: float = 0.2,
                     batch_size: int = 64,
                     scaler_path: str = "scaler.joblib",
                     random_state: int = 42):
    """
    Full pipeline: shapefile + raster → train/val DataLoaders.

    Returns
    -------
    train_loader, val_loader, n_bands
    """
    X, y = sample_raster_at_points(shp_path, tif_path, label_col)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=random_state
    )

    # Fit scaler on training set only
    scaler = fit_and_save_scaler(X_tr, scaler_path)
    X_tr  = scaler.transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    train_ds = SpectraDataset(X_tr, y_tr)
    val_ds   = SpectraDataset(X_val, y_val)

    # Weighted sampler to handle class imbalance
    class_counts = np.bincount(y_tr)
    weights = 1.0 / class_counts[y_tr]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    return train_loader, val_loader, X.shape[1]
