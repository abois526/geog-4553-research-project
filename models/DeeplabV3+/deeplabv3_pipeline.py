import os
import sys
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import rasterio
import rasterio.warp
import rasterio.features
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from rasterio.mask import mask as rasterio_mask
from scipy.ndimage import median_filter
import geopandas as gpd
import albumentations as A
import segmentation_models_pytorch as smp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.patches import Polygon
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

# ===================================================================
# USER CONFIG — Update these paths to match YOUR local data locations.
# All paths should be absolute or relative to this script's directory.
# ===================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Input data paths (REQUIRED — update before running) ---
EMBEDDING_PATH  = os.path.join(SCRIPT_DIR, "embeddings", "emb11Nclp.tif")   # Satellite embedding raster (multi-band GeoTIFF)
LANDCOVER_PATH  = os.path.join(SCRIPT_DIR, "masks", "ExLandcover.tif")       # Land cover classification raster
HABITAT_CLASSES = None  # List of land cover class values that represent habitat, e.g. [1, 2, 5]
                        # Set to None if landcover is already binary (0 = no habitat, 1 = habitat)
POINTS_PATH     = ""  # Path to presence/absence points shapefile (.shp)
STUDY_AREA_PATH = ""  # Path to study area boundary shapefile (.shp) — set to "" to skip clipping

# --- Point labeling config ---
PRESENCE_COLUMN = "PRESENT"  # Column name in the points shapefile that holds presence/absence
PRESENCE_VALUE  = "YES"      # Value in that column indicating species presence
ABSENCE_VALUE   = 0          # Value in that column indicating species absence
POINT_BUFFER_M  = 20         # Buffer radius in metres around each point (20m = 2 pixels at 10m resolution)

# --- Output directories ---
MASK_DIR      = os.path.join(SCRIPT_DIR, "masks")
OUTPUT_DIR    = os.path.join(SCRIPT_DIR, "output")
TILE_EMB_DIR  = os.path.join(SCRIPT_DIR, "tiles", "embedding_tiles")
TILE_MASK_DIR = os.path.join(SCRIPT_DIR, "tiles", "mask_tiles")

# --- Tiling config ---
TILE_SIZE = 256
MIN_VALID = 0.1  # Skip tiles where less than 10% of pixels are habitat

# --- Training config ---
NUM_CLASSES   = 2
BATCH_SIZE    = 8
EPOCHS        = 25
LEARNING_RATE = 1e-4
NODATA        = -32768.0  # Nodata value in the embedding raster
RANDOM_SEED   = 42        # Global seed for reproducibility

# --- Inference config ---
INFER_TILE = 256
OVERLAP    = 64

# ===================================================================
# DERIVED PATHS — computed from the config above, no need to edit
# ===================================================================
HABITAT_MASK_PATH        = os.path.join(MASK_DIR, "habitat_mask.tif")
HABITAT_MASK_POINTS_PATH = os.path.join(MASK_DIR, "habitat_mask_with_points.tif")
BEST_MODEL_PATH          = os.path.join(OUTPUT_DIR, "best_model.pth")
SUITABILITY_MAP_PATH     = os.path.join(OUTPUT_DIR, "suitability_map.tif")
SUITABILITY_SMOOTH_PATH  = os.path.join(OUTPUT_DIR, "suitability_map_smooth.tif")
SUITABILITY_MAP_PNG      = os.path.join(OUTPUT_DIR, "suitability_map.png")
SUITABILITY_MAP_SVG      = os.path.join(OUTPUT_DIR, "suitability_map.svg")


# ===================================================================
# REPRODUCIBILITY — seed all random number generators
# ===================================================================
def seed_everything(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===================================================================
# CACHING UTILITIES
# ===================================================================
def is_cache_fresh(output_path, source_paths):
    """Check if a cached output file exists and is newer than all source inputs."""
    if not os.path.exists(output_path):
        return False
    out_mtime = os.path.getmtime(output_path)
    for src_path in source_paths:
        if src_path and os.path.exists(src_path):
            if os.path.getmtime(src_path) > out_mtime:
                return False
    return True


def step_timer(step_name):
    """Context manager that prints elapsed time for a pipeline step."""
    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *_exc):
            elapsed = time.time() - self.start
            print(f"  [{step_name} completed in {elapsed:.1f}s]")
    return Timer()


# ===================================================================
# NODATA-AWARE NORMALIZATION (shared by dataset and inference)
# ===================================================================
def normalize_tile(tile, nodata=NODATA):
    """Z-normalize a (C, H, W) tile, excluding NODATA pixels from statistics.
    NODATA pixels are set to 0.0 after normalization."""
    if nodata is not None:
        valid_mask = np.all(tile != nodata, axis=0)
        for b in range(tile.shape[0]):
            band = tile[b]
            valid_vals = band[valid_mask]
            if valid_vals.size > 0:
                mu  = valid_vals.mean()
                std = valid_vals.std() + 1e-6
                tile[b] = (band - mu) / std
            tile[b][~valid_mask] = 0.0
    else:
        mu  = tile.mean(axis=(1, 2), keepdims=True)
        std = tile.std(axis=(1, 2),  keepdims=True) + 1e-6
        tile = (tile - mu) / std
    return tile


# ===================================================================
# MODEL DEFINITION
# ===================================================================
class HabitatModel(nn.Module):
    def __init__(self, in_channels=64, num_classes=2):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
        )
        self.segmenter = smp.DeepLabV3Plus(
            encoder_name    = "resnet34",
            encoder_weights = "imagenet",
            in_channels     = 64,
            classes         = num_classes,
        )

    def forward(self, x):
        return self.segmenter(self.projector(x))


# ===================================================================
# DATASET DEFINITION
# ===================================================================
class EmbeddingHabitatDataset(Dataset):
    def __init__(self, emb_paths, mask_paths, augment=False):
        self.emb_paths  = emb_paths
        self.mask_paths = mask_paths
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]) if augment else None

    def __getitem__(self, idx):
        with rasterio.open(self.emb_paths[idx]) as src:
            emb = src.read().astype(np.float32)

        emb = normalize_tile(emb)

        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1).astype(np.int64)

        if self.aug:
            out  = self.aug(image=emb.transpose(1, 2, 0), mask=mask)
            emb  = out['image'].transpose(2, 0, 1)
            mask = out['mask']

        return torch.from_numpy(emb.copy()), torch.from_numpy(mask.copy())

    def __len__(self):
        return len(self.emb_paths)


# ===================================================================
# STEP 1: Inspect input data
# ===================================================================
def inspect_data():
    print("=" * 60)
    print("STEP 1: Inspecting input data")
    print("=" * 60)

    with rasterio.open(EMBEDDING_PATH) as src:
        print(f"\nEmbedding raster: {EMBEDDING_PATH}")
        print(f"  Shape (bands, H, W): {src.count}, {src.height}, {src.width}")
        print(f"  CRS: {src.crs}")
        print(f"  Dtype: {src.dtypes[0]}")
        print(f"  Nodata: {src.nodata}")

    with rasterio.open(LANDCOVER_PATH) as src:
        print(f"\nLandcover raster: {LANDCOVER_PATH}")
        print(f"  CRS: {src.crs}")
        print(f"  Resolution: {src.res}")
        print(f"  Shape: {src.height} x {src.width}")
        print(f"  Nodata: {src.nodata}")
        data = src.read(1)
        print(f"  Unique class values: {np.unique(data)}")

    if not POINTS_PATH:
        print("\nPoints shapefile: NOT CONFIGURED — set POINTS_PATH at the top of this file")
        return
    points = gpd.read_file(POINTS_PATH)
    print(f"\nPoints shapefile: {POINTS_PATH}")
    print(f"  CRS: {points.crs}")
    print(f"  Number of points: {len(points)}")
    print(f"  Columns: {points.columns.tolist()}")
    print(f"\n  First 5 rows:")
    print(points.head())
    print(f"\n  Unique values per column:")
    for col in points.columns:
        if col != 'geometry':
            print(f"    {col}: {points[col].unique()}")


# ===================================================================
# STEP 2: Create habitat mask from landcover
# ===================================================================
def create_habitat_mask():
    print("\n" + "=" * 60)
    print("STEP 2: Creating habitat mask from landcover")
    print("=" * 60)

    if is_cache_fresh(HABITAT_MASK_PATH, [EMBEDDING_PATH, LANDCOVER_PATH]):
        print(f"  Cache hit: {HABITAT_MASK_PATH} is up to date. Skipping.")
        return

    with step_timer("Step 2"):
        os.makedirs(MASK_DIR, exist_ok=True)

        with rasterio.open(EMBEDDING_PATH) as emb_src:
            target_crs       = emb_src.crs
            target_transform = emb_src.transform
            target_height    = emb_src.height
            target_width     = emb_src.width
            target_profile   = emb_src.profile.copy()

        print(f"Target shape: {target_height} x {target_width}")

        with rasterio.open(LANDCOVER_PATH) as lc_src:
            habitat_mask = np.zeros((target_height, target_width), dtype=np.uint8)
            reproject(
                source        = rasterio.band(lc_src, 1),
                destination   = habitat_mask,
                src_transform = lc_src.transform,
                src_crs       = lc_src.crs,
                dst_transform = target_transform,
                dst_crs       = target_crs,
                resampling    = Resampling.nearest,
            )

        unique_vals = np.unique(habitat_mask)
        print(f"Unique values after reproject: {unique_vals}")

        if HABITAT_CLASSES is not None:
            print(f"Binarizing landcover: classes {HABITAT_CLASSES} -> habitat (1), all others -> no-habitat (0)")
            habitat_mask = np.isin(habitat_mask, HABITAT_CLASSES).astype(np.uint8)
        else:
            if not set(unique_vals).issubset({0, 1, 255}):
                print(f"WARNING: Landcover has values {unique_vals} but HABITAT_CLASSES is not set.")
                print("         Treating value 1 as habitat, all others as no-habitat.")
                print("         Set HABITAT_CLASSES for explicit control.")
                habitat_mask = (habitat_mask == 1).astype(np.uint8)

        present = np.sum(habitat_mask == 1)
        absent  = np.sum(habitat_mask == 0)
        print(f"Habitat present (1): {present:>10,} pixels ({present/habitat_mask.size*100:.1f}%)")
        print(f"Habitat absent  (0): {absent:>10,} pixels ({absent/habitat_mask.size*100:.1f}%)")

        target_profile.update(count=1, dtype='uint8', nodata=255, compress='lzw')
        with rasterio.open(HABITAT_MASK_PATH, 'w', **target_profile) as dst:
            dst.write(habitat_mask[None])

        print(f"Mask saved to {HABITAT_MASK_PATH}")


# ===================================================================
# STEP 3: Integrate presence/absence points into mask
# ===================================================================
def create_mask_with_points():
    print("\n" + "=" * 60)
    print("STEP 3: Integrating presence/absence points into mask")
    print("=" * 60)

    if is_cache_fresh(HABITAT_MASK_POINTS_PATH, [HABITAT_MASK_PATH, POINTS_PATH]):
        print(f"  Cache hit: {HABITAT_MASK_POINTS_PATH} is up to date. Skipping.")
        return

    with step_timer("Step 3"):
        with rasterio.open(EMBEDDING_PATH) as src:
            target_crs       = src.crs
            target_transform = src.transform
            target_height    = src.height
            target_width     = src.width
            target_profile   = src.profile.copy()

        if not POINTS_PATH or not os.path.exists(POINTS_PATH):
            raise FileNotFoundError(
                "POINTS_PATH is not configured or file not found. "
                "Set POINTS_PATH at the top of this file to your points shapefile."
            )
        points = gpd.read_file(POINTS_PATH)
        print(f"Loaded {len(points)} points in CRS: {points.crs}")

        points = points.to_crs(target_crs)
        print(f"Reprojected to: {target_crs}")

        # Filter presence/absence, casting values to match column dtype to avoid silent mismatches
        col_dtype = points[PRESENCE_COLUMN].dtype
        pres_val = type(points[PRESENCE_COLUMN].iloc[0])(PRESENCE_VALUE) if len(points) > 0 else PRESENCE_VALUE
        abs_val  = type(points[PRESENCE_COLUMN].iloc[0])(ABSENCE_VALUE)  if len(points) > 0 else ABSENCE_VALUE

        presence = points[points[PRESENCE_COLUMN] == pres_val].copy()
        absence  = points[points[PRESENCE_COLUMN] == abs_val].copy()
        unmatched = len(points) - len(presence) - len(absence)
        print(f"Presence points: {len(presence)}")
        print(f"Absence points:  {len(absence)}")
        if unmatched > 0:
            print(f"WARNING: {unmatched} points matched neither presence ({PRESENCE_VALUE}) nor absence ({ABSENCE_VALUE})")
            print(f"  Column dtype: {col_dtype}, unique values: {points[PRESENCE_COLUMN].unique()}")

        presence['geometry'] = presence.geometry.buffer(POINT_BUFFER_M)
        absence['geometry']  = absence.geometry.buffer(POINT_BUFFER_M)

        presence_raster = rasterio.features.rasterize(
            [(geom, 1) for geom in presence.geometry],
            out_shape   = (target_height, target_width),
            transform   = target_transform,
            fill        = 255,
            dtype       = np.uint8,
            all_touched = True,
        )

        absence_raster = rasterio.features.rasterize(
            [(geom, 0) for geom in absence.geometry],
            out_shape   = (target_height, target_width),
            transform   = target_transform,
            fill        = 255,
            dtype       = np.uint8,
            all_touched = True,
        )

        print(f"Presence pixels burned: {np.sum(presence_raster == 1):,}")
        print(f"Absence pixels burned:  {np.sum(absence_raster == 0):,}")

        with rasterio.open(HABITAT_MASK_PATH) as src:
            base_mask = src.read(1).astype(np.uint8)

        print(f"Base mask shape: {base_mask.shape}")

        combined_mask = base_mask.copy()
        combined_mask[absence_raster == 0] = 0
        combined_mask[presence_raster == 1] = 1

        present = np.sum(combined_mask == 1)
        absent  = np.sum(combined_mask == 0)
        total   = combined_mask.size
        print(f"\nFinal mask class distribution:")
        print(f"  Class 1 (habitat):    {present:>10,} pixels ({present/total*100:.1f}%)")
        print(f"  Class 0 (no habitat): {absent:>10,} pixels ({absent/total*100:.1f}%)")

        target_profile.update(count=1, dtype='uint8', nodata=255, compress='lzw')
        with rasterio.open(HABITAT_MASK_POINTS_PATH, 'w', **target_profile) as dst:
            dst.write(combined_mask[None])

        print(f"Saved combined mask to {HABITAT_MASK_POINTS_PATH}")


# ===================================================================
# STEP 4: Create tiles for training
# ===================================================================
def create_tiles():
    print("\n" + "=" * 60)
    print("STEP 4: Creating tiles for training")
    print("=" * 60)

    # Check if tiles already exist and are fresh
    existing_tiles = glob.glob(os.path.join(TILE_EMB_DIR, "*.tif"))
    if existing_tiles and is_cache_fresh(existing_tiles[0], [HABITAT_MASK_POINTS_PATH, EMBEDDING_PATH]):
        print(f"  Cache hit: {len(existing_tiles)} embedding tiles already exist and are up to date. Skipping.")
        return

    with step_timer("Step 4"):
        # Clear old tiles to prevent stale data from a previous run mixing
        # with fresh tiles (e.g., if config changed and fewer tiles qualify now)
        import shutil
        for d in [TILE_EMB_DIR, TILE_MASK_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)
        os.makedirs(TILE_EMB_DIR, exist_ok=True)
        os.makedirs(TILE_MASK_DIR, exist_ok=True)

        tile_count = 0
        skipped    = 0

        with rasterio.open(EMBEDDING_PATH) as emb_src, \
             rasterio.open(HABITAT_MASK_POINTS_PATH) as msk_src:

            H, W = emb_src.height, emb_src.width
            print(f"Raster size: {H} x {W}")
            print(f"Tiling into {TILE_SIZE}x{TILE_SIZE} patches...")

            for y in range(0, H - TILE_SIZE + 1, TILE_SIZE):
                for x in range(0, W - TILE_SIZE + 1, TILE_SIZE):
                    win = Window(x, y, TILE_SIZE, TILE_SIZE)
                    mask_tile = msk_src.read(1, window=win)

                    habitat_ratio = np.mean(mask_tile == 1)
                    if habitat_ratio < MIN_VALID:
                        skipped += 1
                        continue

                    emb_tile = emb_src.read(window=win)
                    tile_transform = rasterio.windows.transform(win, emb_src.transform)

                    emb_profile = emb_src.profile.copy()
                    emb_profile.update(
                        height    = TILE_SIZE,
                        width     = TILE_SIZE,
                        transform = tile_transform,
                        compress  = 'lzw',
                    )
                    emb_out = os.path.join(TILE_EMB_DIR, f"emb_tile_{y:05d}_{x:05d}.tif")
                    with rasterio.open(emb_out, 'w', **emb_profile) as dst:
                        dst.write(emb_tile)

                    msk_profile = msk_src.profile.copy()
                    msk_profile.update(
                        height    = TILE_SIZE,
                        width     = TILE_SIZE,
                        transform = tile_transform,
                        compress  = 'lzw',
                    )
                    msk_out = os.path.join(TILE_MASK_DIR, f"msk_tile_{y:05d}_{x:05d}.tif")
                    with rasterio.open(msk_out, 'w', **msk_profile) as dst:
                        dst.write(mask_tile[None])

                    tile_count += 1
                    if tile_count % 50 == 0:
                        print(f"  {tile_count} tiles saved so far...")

        print(f"\nTiles saved:   {tile_count}")
        print(f"Tiles skipped: {skipped}")


# ===================================================================
# STEP 5: Check class balance across tiles
# ===================================================================
def check_class_balance():
    print("\n" + "=" * 60)
    print("STEP 5: Checking class balance across tiles")
    print("=" * 60)

    mask_paths = sorted(glob.glob(os.path.join(TILE_MASK_DIR, "*.tif")))
    total_present = 0
    total_absent  = 0

    for path in mask_paths:
        with rasterio.open(path) as src:
            mask = src.read(1)
        total_present += np.sum(mask == 1)
        total_absent  += np.sum(mask == 0)

    total = total_present + total_absent
    if total == 0:
        print("No mask tiles found. Run the 'tile' step first.")
        return
    print(f"Habitat present (1): {total_present:>10,} pixels ({total_present/total*100:.1f}%)")
    print(f"Habitat absent  (0): {total_absent:>10,} pixels ({total_absent/total*100:.1f}%)")


# ===================================================================
# STEP 6: Train the model
# ===================================================================
def train():
    print("\n" + "=" * 60)
    print("STEP 6: Training the model")
    print("=" * 60)

    with step_timer("Step 6"):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        emb_paths  = sorted(glob.glob(os.path.join(TILE_EMB_DIR, "*.tif")))
        mask_paths = sorted(glob.glob(os.path.join(TILE_MASK_DIR, "*.tif")))

        print(f"Found {len(emb_paths)} embedding tiles")
        print(f"Found {len(mask_paths)} mask tiles")

        if len(emb_paths) < 2:
            print(f"ERROR: Need at least 2 tiles for train/val split, found {len(emb_paths)}.")
            print("  Check that the 'tile' step ran and that MIN_VALID isn't filtering out all tiles.")
            return
        if len(emb_paths) != len(mask_paths):
            raise RuntimeError(
                f"Tile count mismatch: {len(emb_paths)} embedding tiles vs "
                f"{len(mask_paths)} mask tiles. Clear the tiles/ directory and re-run the 'tile' step."
            )

        # Reproducible train/val split with separate datasets (no augmentation on val)
        n_total  = len(emb_paths)
        val_size = max(1, int(n_total * 0.2))
        indices  = torch.randperm(n_total, generator=torch.Generator().manual_seed(RANDOM_SEED)).tolist()
        val_idx   = indices[:val_size]
        train_idx = indices[val_size:]

        train_emb  = [emb_paths[i]  for i in train_idx]
        train_mask = [mask_paths[i] for i in train_idx]
        val_emb    = [emb_paths[i]  for i in val_idx]
        val_mask   = [mask_paths[i] for i in val_idx]

        train_ds = EmbeddingHabitatDataset(train_emb, train_mask, augment=True)
        val_ds   = EmbeddingHabitatDataset(val_emb, val_mask, augment=False)
        print(f"Split: {len(train_ds)} train / {len(val_ds)} val (seed={RANDOM_SEED})")

        # Use parallel data loading if not on Windows
        num_workers = 0 if sys.platform == "win32" else min(4, os.cpu_count() or 1)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {device} (data workers: {num_workers})")

        # Compute class weights from tile data to handle imbalanced classes
        print("Computing class weights from tiles...")
        total_present = 0
        total_absent  = 0
        for path in mask_paths:
            with rasterio.open(path) as src:
                m = src.read(1)
            total_present += np.sum(m == 1)
            total_absent  += np.sum(m == 0)
        total = total_present + total_absent
        weight_absent  = total / (2.0 * total_absent)  if total_absent  > 0 else 1.0
        weight_present = total / (2.0 * total_present) if total_present > 0 else 1.0
        class_weights = torch.tensor([weight_absent, weight_present], dtype=torch.float32).to(device)
        print(f"  Class weights: absent={weight_absent:.3f}, present={weight_present:.3f}")

        model     = HabitatModel(in_channels=64, num_classes=NUM_CLASSES).to(device)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # Combined loss: weighted cross-entropy + dice for better handling of class imbalance
        # ignore_index=255 ensures nodata pixels in the mask (stored as 255) are excluded from loss
        ce_loss_fn   = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
        dice_loss_fn = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)
        def loss_fn(preds, targets):
            return ce_loss_fn(preds, targets) + dice_loss_fn(preds, targets)

        best_val_loss = float('inf')
        patience = 7
        epochs_no_improve = 0

        for epoch in range(EPOCHS):
            epoch_start = time.time()
            model.train()
            train_loss = 0
            for embs, masks in train_loader:
                embs  = embs.to(device)
                masks = masks.to(device).long()
                preds = model(embs)
                loss  = loss_fn(preds, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            val_tp = val_fp = val_fn = 0
            with torch.no_grad():
                for embs, masks in val_loader:
                    embs  = embs.to(device)
                    masks = masks.to(device).long()
                    preds = model(embs)
                    val_loss += loss_fn(preds, masks).item()

                    pred_labels = preds.argmax(dim=1)
                    val_tp += ((pred_labels == 1) & (masks == 1)).sum().item()
                    val_fp += ((pred_labels == 1) & (masks == 0)).sum().item()
                    val_fn += ((pred_labels == 0) & (masks == 1)).sum().item()

            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss   / len(val_loader)
            iou       = val_tp / (val_tp + val_fp + val_fn + 1e-8)
            precision = val_tp / (val_tp + val_fp + 1e-8)
            recall    = val_tp / (val_tp + val_fn + 1e-8)
            f1        = 2 * precision * recall / (precision + recall + 1e-8)
            elapsed   = time.time() - epoch_start
            scheduler.step()
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {avg_train:.4f} | Val: {avg_val:.4f} "
                  f"| IoU: {iou:.4f} | F1: {f1:.4f} | {elapsed:.1f}s")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                epochs_no_improve = 0
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"  --> Saved best model (val loss {best_val_loss:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"  Early stopping: no improvement for {patience} epochs.")
                    break

        print("Training complete.")


# ===================================================================
# STEP 7: Run inference over full raster
# ===================================================================
def inference():
    print("\n" + "=" * 60)
    print("STEP 7: Running inference over full raster")
    print("=" * 60)

    with step_timer("Step 7"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = HabitatModel(in_channels=64, num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
        model.eval()

        with rasterio.open(EMBEDDING_PATH) as src:
            profile = src.profile.copy()
            H, W    = src.height, src.width
            pred_full = np.zeros((H, W), dtype=np.float32)
            stride = INFER_TILE - OVERLAP
            total_rows = len(range(0, H, stride))
            row_idx = 0

            for y in range(0, H, stride):
                for x in range(0, W, stride):
                    th  = min(INFER_TILE, H - y)
                    tw  = min(INFER_TILE, W - x)
                    win = Window(x, y, tw, th)

                    tile = src.read(window=win).astype(np.float32)
                    tile = normalize_tile(tile)

                    pad_h = INFER_TILE - th
                    pad_w = INFER_TILE - tw
                    if pad_h > 0 or pad_w > 0:
                        tile = np.pad(tile, ((0, 0), (0, pad_h), (0, pad_w)))

                    t = torch.from_numpy(tile[None].copy()).to(device)
                    with torch.no_grad():
                        logits = model(t)
                        probs  = torch.softmax(logits, dim=1)
                        prob_present = probs[0, 1].cpu().numpy()

                    core_y = OVERLAP // 2 if y > 0 else 0
                    core_x = OVERLAP // 2 if x > 0 else 0
                    pred_full[y+core_y:y+th, x+core_x:x+tw] = prob_present[core_y:th, core_x:tw]

                row_idx += 1
                if row_idx % 10 == 0 or row_idx == total_rows:
                    print(f"  Row {row_idx}/{total_rows} complete")

        profile.update(count=1, dtype='float32', nodata=-9999.0)
        with rasterio.open(SUITABILITY_MAP_PATH, 'w', **profile) as dst:
            dst.write(pred_full[None])

        print(f"Inference complete. Saved to {SUITABILITY_MAP_PATH}")


# ===================================================================
# STEP 8: Smooth suitability map
# ===================================================================
def smooth():
    print("\n" + "=" * 60)
    print("STEP 8: Smoothing suitability map")
    print("=" * 60)

    with step_timer("Step 8"):
        with rasterio.open(SUITABILITY_MAP_PATH) as src:
            profile = src.profile.copy()
            suit    = src.read(1)

        suit_smooth = median_filter(suit, size=5)

        with rasterio.open(SUITABILITY_SMOOTH_PATH, 'w', **profile) as dst:
            dst.write(suit_smooth[None])

        print(f"Smoothed map saved to {SUITABILITY_SMOOTH_PATH}")


# ===================================================================
# STEP 9: Clip output to study area boundary
# ===================================================================
def clip_to_study_area():
    print("\n" + "=" * 60)
    print("STEP 9: Clipping suitability map to study area boundary")
    print("=" * 60)

    if not STUDY_AREA_PATH or not os.path.exists(STUDY_AREA_PATH):
        print("WARNING: No study area shapefile configured or file not found.")
        print("Skipping clipping step. Output will cover full raster extent.")
        print("Set STUDY_AREA_PATH at the top of this file to enable clipping.")
        return

    with step_timer("Step 9"):
        boundary = gpd.read_file(STUDY_AREA_PATH)

        with rasterio.open(SUITABILITY_SMOOTH_PATH) as src:
            boundary = boundary.to_crs(src.crs)
            geometries = boundary.geometry.values

            clipped, clipped_transform = rasterio_mask(
                src, geometries, crop=True, nodata=-9999.0
            )
            profile = src.profile.copy()
            profile.update(
                height    = clipped.shape[1],
                width     = clipped.shape[2],
                transform = clipped_transform,
                nodata    = -9999.0,
            )

        clipped_path = os.path.join(OUTPUT_DIR, "suitability_map_clipped.tif")
        with rasterio.open(clipped_path, 'w', **profile) as dst:
            dst.write(clipped)

        print(f"Clipped map saved to {clipped_path}")
        print(f"  Output shape: {clipped.shape[1]} x {clipped.shape[2]}")


# ===================================================================
# STEP 10: Generate poster-quality suitability map
# ===================================================================
def make_suitability_map():
    print("\n" + "=" * 60)
    print("STEP 10: Generating suitability map visualization")
    print("=" * 60)

    with step_timer("Step 10"):
        # Use clipped map if available, otherwise smoothed, otherwise raw
        clipped_path = os.path.join(OUTPUT_DIR, "suitability_map_clipped.tif")
        if os.path.exists(clipped_path):
            tif_path = clipped_path
            print(f"  Using clipped map: {tif_path}")
        elif os.path.exists(SUITABILITY_SMOOTH_PATH):
            tif_path = SUITABILITY_SMOOTH_PATH
            print(f"  Using smoothed map: {tif_path}")
        else:
            tif_path = SUITABILITY_MAP_PATH
            print(f"  Using raw map: {tif_path}")

        mpl.rcParams.update({
            "font.family":       "sans-serif",
            "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "text.antialiased":  True,
            "savefig.dpi":       300,
        })

        with rasterio.open(tif_path) as src:
            suit = src.read(1).astype(np.float32)
            meta = src.meta.copy()
            nodata = src.nodata

        if nodata is not None:
            suit[suit == nodata] = np.nan

        # Continuous blue → yellow → orange colormap (matches MaxEnt visual style)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "deeplab_suit",
            [(0.0, "#2166ac"),
             (0.25, "#67a9cf"),
             (0.5, "#f7e76d"),
             (0.75, "#f5a623"),
             (1.0, "#e65100")],
        )
        cmap.set_bad(color="#1a1a1a")

        # Compute spatial extent from raster transform
        transform = meta["transform"]
        left   = transform.c
        top    = transform.f
        right  = left + transform.a * meta["width"]
        bottom = top  + transform.e * meta["height"]
        extent = [left, right, bottom, top]

        fig, ax = plt.subplots(figsize=(14, 11), dpi=200)
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#1a1a1a")

        im = ax.imshow(
            suit,
            cmap=cmap,
            vmin=0, vmax=1,
            extent=extent,
            interpolation="bilinear",
            aspect="equal",
        )

        # Overlay presence points if configured
        if POINTS_PATH and os.path.exists(POINTS_PATH):
            gdf = gpd.read_file(POINTS_PATH)
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=3857)
            raster_crs = meta.get("crs")
            if raster_crs:
                gdf = gdf.to_crs(raster_crs)

            # Filter to presence points only
            if PRESENCE_COLUMN in gdf.columns:
                pres_val = type(gdf[PRESENCE_COLUMN].iloc[0])(PRESENCE_VALUE) if len(gdf) > 0 else PRESENCE_VALUE
                gdf = gdf[gdf[PRESENCE_COLUMN] == pres_val]

            ax.scatter(gdf.geometry.x, gdf.geometry.y,
                       c="#ffe600", s=14, marker="o", linewidths=0.5,
                       edgecolors="black", alpha=0.9,
                       label=f"Presence ({len(gdf)} pts)", zorder=5)
            ax.legend(loc="lower left", framealpha=0.3,
                      labelcolor="white", fontsize=9,
                      facecolor="#333333", edgecolor="#555555")

        # Extend axis limits slightly beyond the raster for padding
        y_pad = (extent[3] - extent[2]) * 0.02
        x_pad = (extent[1] - extent[0]) * 0.02
        ax.set_xlim(extent[0] - x_pad, extent[1] + x_pad)
        ax.set_ylim(extent[2] - y_pad, extent[3] + y_pad)

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.12)
        cb  = fig.colorbar(im, cax=cax)
        cb.set_label("Habitat Suitability", color="white", fontsize=11, labelpad=10)
        cb.ax.yaxis.set_tick_params(color="white")
        cb.outline.set_edgecolor("white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=9)
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cb.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

        # Axis labels — Easting / Northing
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_xlabel("Easting (m)", color="white", fontsize=10)
        ax.set_ylabel("Northing (m)", color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        # Scalebar — CRS is EPSG:26911 (metres), fixed at 20 km
        scalebar = ScaleBar(
            dx=1, units="m", location="lower right",
            fixed_value=20, fixed_units="km",
            scale_formatter=lambda value, unit: "",
            color="white", box_alpha=0.3,
            font_properties={"size": 9},
            scale_loc="top",
            border_pad=0.8,
            sep=3,
        )
        ax.add_artist(scalebar)

        # Scale tick labels at 0, 5, 10, 20 km
        ax_xlim = ax.get_xlim()
        ax_width = ax_xlim[1] - ax_xlim[0]
        bar_20km_frac = 20000 / ax_width
        bar_right = 0.98
        bar_left  = bar_right - bar_20km_frac
        for km in [0, 5, 10, 20]:
            x_pos = bar_left + (km / 20) * bar_20km_frac
            ax.text(x_pos, 0.045, f"{km} km" if km == 20 else str(km),
                    transform=ax.transAxes, ha="center", va="bottom",
                    color="white", fontsize=8, zorder=10)

        # North arrow — classic two-tone chevron with "N" label
        cx, cy = 0.90, 0.14
        w, h   = 0.0125, 0.03

        left_tri = Polygon(
            [(cx, cy + h), (cx - w, cy - h * 0.3), (cx, cy)],
            closed=True, facecolor="white", edgecolor="white",
            linewidth=0.8, transform=ax.transAxes, zorder=10,
        )
        right_tri = Polygon(
            [(cx, cy + h), (cx + w, cy - h * 0.3), (cx, cy)],
            closed=True, facecolor="#888888", edgecolor="white",
            linewidth=0.8, transform=ax.transAxes, zorder=10,
        )
        ax.add_patch(left_tri)
        ax.add_patch(right_tri)
        ax.text(cx, cy - h * 0.3 - 0.015, "N",
                transform=ax.transAxes, ha="center", va="top",
                color="white", fontsize=12, fontweight="bold", zorder=10)

        # Title
        title = ("Habitat Suitability \u2014 Bromus tectorum\n"
                 "Municipal District of Ranchland No. 66, Alberta")
        ax.set_title(title, color="white", fontsize=16, fontweight="bold",
                     pad=18, loc="left")

        # Subtitle
        ax.text(0.02, 1.005,
                "Method: DeepLabV3+  \u00b7  CRS: NAD83 / UTM Zone 11N (EPSG:26911)  \u00b7  Resolution: 10m",
                transform=ax.transAxes, color="white", fontsize=9, style="italic",
                ha="left", va="bottom")

        # Inset locator map (Alberta with study area)
        # Wrapped in try/except because it downloads Natural Earth data from the
        # internet — if offline, the main map is still saved without the inset.
        if STUDY_AREA_PATH and os.path.exists(STUDY_AREA_PATH):
            try:
                inset_ax = ax.inset_axes([0.66, 0.05, 0.14, 0.24])
                inset_ax.set_facecolor("#1a1a1a")
                for sp in inset_ax.spines.values():
                    sp.set_visible(False)

                # Alberta province boundary from Natural Earth
                ne_url = ("https://naciscdn.org/naturalearth/10m/cultural/"
                          "ne_10m_admin_1_states_provinces.zip")
                provinces = gpd.read_file(ne_url)
                alberta = provinces[provinces["name"] == "Alberta"]
                alberta.plot(ax=inset_ax, facecolor="#3a3a3a", edgecolor="white",
                             linewidth=0.5)

                # Study area polygon
                md_bound = gpd.read_file(STUDY_AREA_PATH)
                md_bound_4326 = md_bound.to_crs(epsg=4326)
                md_bound_4326.plot(ax=inset_ax, facecolor="#ff4444", edgecolor="#ff4444",
                                   alpha=0.55, linewidth=0.8, zorder=5)

                inset_ax.set_xlim(-121, -109)
                inset_ax.set_ylim(48.5, 60.5)
                inset_ax.set_xticks([])
                inset_ax.set_yticks([])
                inset_ax.set_title("Alberta", color="white", fontsize=10, pad=3)
            except Exception as e:
                print(f"  WARNING: Could not create inset map: {e}")
                print("  (Requires internet to download Natural Earth province boundaries)")
                # Remove the empty inset axes if it was created before the error
                if 'inset_ax' in dir():
                    inset_ax.remove()

        plt.tight_layout(pad=1.5, rect=[0.03, 0.03, 0.97, 0.97])
        Path(SUITABILITY_MAP_PNG).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(SUITABILITY_MAP_PNG, dpi=300, bbox_inches="tight",
                    pad_inches=0.3, facecolor=fig.get_facecolor())
        print(f"  PNG saved → {SUITABILITY_MAP_PNG}")

        plt.savefig(SUITABILITY_MAP_SVG, bbox_inches="tight",
                    pad_inches=0.3, facecolor=fig.get_facecolor())
        print(f"  SVG saved → {SUITABILITY_MAP_SVG}")

        plt.close()


# ===================================================================
# MAIN — run the full pipeline
# ===================================================================
STEPS = {
    "inspect":  inspect_data,
    "mask":     create_habitat_mask,
    "points":   create_mask_with_points,
    "tile":     create_tiles,
    "balance":  check_class_balance,
    "train":    train,
    "infer":    inference,
    "smooth":   smooth,
    "clip":     clip_to_study_area,
    "map":      make_suitability_map,
}

ALL_STEPS = list(STEPS.keys())

def main(steps=None):
    """Run pipeline steps. If steps is None, run all steps."""
    seed_everything(RANDOM_SEED)
    pipeline_start = time.time()

    run_steps = steps if steps else ALL_STEPS
    for name in run_steps:
        if name not in STEPS:
            print(f"Unknown step: {name}. Available: {', '.join(ALL_STEPS)}")
            continue
        STEPS[name]()

    elapsed = time.time() - pipeline_start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE ({minutes}m {seconds:.0f}s total)")
    print("=" * 60)


if __name__ == "__main__":
    # Usage:
    #   python deeplabv3_pipeline.py              — run full pipeline
    #   python deeplabv3_pipeline.py train infer  — run only training + inference
    #   python deeplabv3_pipeline.py infer smooth clip  — re-run only post-training steps
    args = sys.argv[1:]
    if args:
        main(steps=args)
    else:
        main()
