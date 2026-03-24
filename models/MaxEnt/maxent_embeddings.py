"""
MaxEnt Habitat Suitability Model — Bromus tectorum
MD Ranchland No. 66, Alberta

Baseline model using Alpha Earth Foundation Model embeddings (64 bands).
Uses elapid for MaxEnt implementation and its built-in utilities for
background sampling, raster annotation, spatial cross-validation, and
raster prediction.

Preprocessing:
  - Pixel deduplication: removes presence points that fall in the same
    raster pixel (identical covariate values at 10m resolution).
  - PCA: reduces 64 correlated embedding bands to N_PCA_COMPONENTS
    uncorrelated principal components before modeling (Merow et al. 2013,
    Section III.B). The fitted PCA is embedded in the saved model so
    prediction can be applied directly to the original 64-band raster.

Evaluation: standard train/test split + spatial block cross-validation.
Metrics: AUC-ROC and continuous Boyce index (CBI).
Extracted raster values are cached to CACHE_DIR on first run to skip re-extraction.
"""

import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from pathlib import Path
import os
import argparse
import joblib
import elapid
import warnings

# Suppress known nuisance warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*palette.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
# 'unitialized' is a typo in the warning, this is intential
warnings.filterwarnings("ignore", category=UserWarning, message=".*where.*unitialized memory.*")


# Ensure convergence warnings are always visible
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("always", category=ConvergenceWarning)

try:
    import glmnet  # noqa: F401
except ImportError:
    print("WARNING: glmnet not installed — elapid will use slower sklearn fallback")


#-----------------------------------------------------------
# SECTION: Constants — fill in paths and adjust parameters as needed
#-----------------------------------------------------------

# Path to presence points shapefile (EPSG:26911)
PRESENCE_SHP = r"/Users/andrew/Workspace/github-repos/geog-4553-research-project/models/MaxEnt/data/inputs/db_points_11N_Clip.shp"  # e.g., r"C:\path\to\presence_points.shp"

# Path to stacked 64-band embedding raster (EPSG:26911, 10m)
RASTER_STACK = r"/Users/andrew/Workspace/github-repos/geog-4553-research-project/models/MaxEnt/data/inputs/emb11Nclp.tif"  # e.g., r"C:\path\to\MDR66_embeddings_stacked.tif"

# Path for output prediction raster (only needed for predict mode)
OUTPUT_RASTER = r"/Users/andrew/Workspace/github-repos/geog-4553-research-project/models/MaxEnt/data/outputs/maxent_suitability.tif"  # e.g., r"C:\path\to\maxent_suitability.tif"

# Number of background points to sample
N_BACKGROUND = 10000

# Random seed for reproducibility
RANDOM_SEED = 42

# Number of spatial CV folds
N_SPATIAL_FOLDS = 5

# Test size for standard split
TEST_SIZE = 0.2

# Directory to cache extracted raster values (set to "" to disable caching)
CACHE_DIR = r"/Users/andrew/Workspace/github-repos/geog-4553-research-project/models/MaxEnt/data/outputs/cache/"  # e.g., r"C:\path\to\cache"

# Path to save/load the fitted final model (required for predict mode)
MODEL_PATH = r"/Users/andrew/Workspace/github-repos/geog-4553-research-project/models/MaxEnt/data/outputs/maxent_model.joblib"  # e.g., r"C:\path\to\maxent_model.joblib"

# Regularization multiplier grid for tuning (Merow et al. 2013, Section III.C)
BETA_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0]

# Number of PCA components to reduce embedding bands to before modeling.
# Reduces correlated 64-band embeddings to uncorrelated components
# (Merow et al. 2013, Section III.B recommends PCA for correlated predictors).
N_PCA_COMPONENTS = 15

# Output paths for the poster-quality suitability map
OUTPUT_MAP_PNG = r"/Users/andrew/Workspace/github-repos/geog-4553-research-project/models/MaxEnt/data/outputs/maps/maxent_suitability_map.png"
OUTPUT_MAP_SVG = r"/Users/andrew/Workspace/github-repos/geog-4553-research-project/models/MaxEnt/data/outputs/maps/maxent_suitability_map.svg"


#-----------------------------------------------------------
# SECTION: Input Validation
#-----------------------------------------------------------

def validate_paths(**paths):
    """
    Check that all required path constants are non-empty and raise a clear error
    if any are missing.

    Args:
        **paths: Keyword arguments mapping parameter names to path strings.

    Raises:
        ValueError: If any path is empty or None.
    """
    missing = [name for name, path in paths.items() if not path]
    if missing:
        raise ValueError(
            f"Required path constant(s) not set: {', '.join(missing)}. "
            f"Fill in the constants at the top of the script before running."
        )


def validate_arrays(**arrays):
    """
    Check that extracted feature arrays contain no NaN or infinite values.

    Args:
        **arrays: Keyword arguments mapping array names to numpy arrays.

    Raises:
        ValueError: If any array contains NaN or infinite values.
    """
    for name, arr in arrays.items():
        if np.any(np.isnan(arr)):
            raise ValueError(f"NaN values found in {name} — check raster nodata handling")
        if np.any(np.isinf(arr)):
            raise ValueError(f"Infinite values found in {name} — check raster data")


#-----------------------------------------------------------
# SECTION: Load Presence Data
#-----------------------------------------------------------

def load_presence_points(shp_path, raster_path):
    """
    Load presence points from a shapefile and reproject to match the raster CRS if needed.

    Args:
        shp_path (str): Path to the presence points shapefile.
        raster_path (str): Path to the raster stack (used to read the target CRS).

    Returns:
        GeoDataFrame: Presence points in the raster's CRS.
    """
    gdf = gpd.read_file(shp_path)

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
    if gdf.crs != raster_crs:
        print(f"Reprojecting presence points from {gdf.crs} to {raster_crs}")
        gdf = gdf.to_crs(raster_crs)

    print(f"Loaded {len(gdf)} presence points")
    return gdf


#-----------------------------------------------------------
# SECTION: Pixel Deduplication
#-----------------------------------------------------------

def deduplicate_to_unique_pixels(gdf, raster_path):
    """
    Remove presence points that fall in the same raster pixel.

    At 10m resolution, multiple GPS waypoints often land in the same pixel
    and would extract identical covariate values — effectively
    pseudoreplicating that environment in the model. This keeps one point
    per unique pixel.

    Args:
        gdf (GeoDataFrame): Presence points.
        raster_path (str): Path to the raster (used to get the pixel grid).

    Returns:
        GeoDataFrame: Deduplicated presence points (one per unique pixel).
    """
    n_before = len(gdf)

    with rasterio.open(raster_path) as src:
        coords = np.array([(g.x, g.y) for g in gdf.geometry])
        rows, cols = rasterio.transform.rowcol(src.transform, coords[:, 0], coords[:, 1])

    # Build a pixel identifier and keep first occurrence
    pixel_ids = list(zip(rows, cols))
    gdf = gdf.copy()
    gdf["_pixel_id"] = pixel_ids
    gdf = gdf.drop_duplicates(subset="_pixel_id", keep="first").drop(columns="_pixel_id")

    n_after = len(gdf)
    n_removed = n_before - n_after
    print(f"  Pixel deduplication: {n_before} points → {n_after} unique pixels "
          f"({n_removed} duplicates removed)")

    return gdf


#-----------------------------------------------------------
# SECTION: Extract Band Values from Annotated GeoDataFrame
#-----------------------------------------------------------

def get_band_values(annotated_gdf):
    """
    Extract the numeric band-value columns from an elapid.annotate() result
    as a numpy array, excluding geometry and internal metadata columns.

    Args:
        annotated_gdf (GeoDataFrame): Output from elapid.annotate().

    Returns:
        numpy.ndarray: Shape (n_points, n_bands).
    """
    band_cols = annotated_gdf.select_dtypes(include=[np.number]).columns
    band_cols = [c for c in band_cols if c != "valid"]
    values = annotated_gdf[band_cols].values
    if values.shape[1] != 64:
        print(f"  WARNING: Expected 64 bands, got {values.shape[1]}. "
              f"Columns: {list(band_cols[:5])}... Check for extra numeric columns.")
    return values


#-----------------------------------------------------------
# SECTION: Fit MaxEnt Model
#-----------------------------------------------------------

def fit_maxent(X_presence, X_background, beta_multiplier=1.5, seed=RANDOM_SEED,
               preprocessor=None):
    """
    Fit a MaxEnt model using elapid.

    Uses linear, quadratic, and hinge features. Product features are disabled
    to avoid O(n²) feature expansion with high-dimensional embedding input
    (64 bands would create 2,016 pairwise product features). Quadratic adds
    only 64 squared terms and captures nonlinear single-band responses.

    Args:
        X_presence (numpy.ndarray): Shape (n_presence, n_features).
        X_background (numpy.ndarray): Shape (n_background, n_features).
        beta_multiplier (float): Regularization multiplier (higher = simpler model).
        seed (int): Random seed for model reproducibility.
        preprocessor: Optional sklearn transformer (e.g. fitted PCA) to embed
            in the model. elapid applies it automatically during predict().

    Returns:
        elapid.MaxentModel: Fitted model (with preprocessor attached if provided).
    """
    model = elapid.MaxentModel(
        feature_types=["linear", "quadratic", "hinge"],
        beta_multiplier=beta_multiplier,
        random_state=seed,
        # pbr=100: presence weight=1, background weight=100
        # (maxnet default for Poisson point process approximation)
        class_weights=100.0,
    )

    # elapid expects: x = feature array, y = labels (1=presence, 0=background)
    X = np.vstack([X_presence, X_background])
    y = np.concatenate([
        np.ones(len(X_presence)),
        np.zeros(len(X_background)),
    ])

    model.fit(X, y, preprocessor=preprocessor)
    print(f"MaxEnt model fitted (beta={beta_multiplier})")
    return model


#-----------------------------------------------------------
# SECTION: Continuous Boyce Index
#-----------------------------------------------------------

def continuous_boyce_index(pred_presence, pred_background, window_width=0.1,
                           step=0.02):
    """
    Compute the continuous Boyce index (CBI) using a moving window
    (Hirzel et al. 2006).

    Slides an overlapping window across the predicted suitability gradient
    and computes the Spearman rank correlation between window centres and
    the predicted/expected (P/E) frequency ratio.  The moving-window
    approach is more robust than fixed histogram bins when test presence
    counts are small, because each point contributes to multiple windows.

    The suitability range is derived from the data (not fixed to 0–1) so
    bins span the actual prediction distribution.

    Args:
        pred_presence (numpy.ndarray): Model predictions at presence locations.
        pred_background (numpy.ndarray): Model predictions at background locations.
        window_width (float): Width of each moving window in suitability units.
        step (float): Step size between successive window centres.

    Returns:
        float: Spearman correlation coefficient (CBI), range [-1, 1].
    """
    all_preds = np.concatenate([pred_presence, pred_background])
    p_min, p_max = all_preds.min(), all_preds.max()

    # Slide overlapping windows across the suitability gradient
    centers = np.arange(p_min + window_width / 2,
                        p_max - window_width / 2 + step, step)
    pe_centers = []
    pe_ratios = []

    for c in centers:
        lo, hi = c - window_width / 2, c + window_width / 2
        n_pres = np.sum((pred_presence >= lo) & (pred_presence < hi))
        n_bg = np.sum((pred_background >= lo) & (pred_background < hi))
        if n_bg == 0:
            continue
        F_pres = n_pres / len(pred_presence)
        F_exp = n_bg / len(pred_background)
        pe_centers.append(c)
        pe_ratios.append(F_pres / F_exp)

    if len(pe_ratios) < 3:
        return np.nan

    return spearmanr(pe_centers, pe_ratios).correlation


#-----------------------------------------------------------
# SECTION: Evaluation — Standard Train/Test Split
#-----------------------------------------------------------

def evaluate_standard_split(X_presence, X_background, test_size=0.2, seed=42,
                            beta_multiplier=1.5):
    """
    Evaluate the model using a standard random train/test split.

    Reports AUC-ROC and continuous Boyce index. Note: likely overestimates
    performance due to spatial autocorrelation in clustered presence data.
    Use alongside spatial CV for comparison.

    Args:
        X_presence (numpy.ndarray): Shape (n_presence, n_features).
        X_background (numpy.ndarray): Shape (n_background, n_features).
        test_size (float): Fraction held out for testing.
        seed (int): Random seed.
        beta_multiplier (float): Regularization multiplier for MaxEnt.

    Returns:
        tuple:
            - elapid.MaxentModel: Model fitted on training data (not the final model).
            - float: AUC-ROC on the test set.
            - float: Continuous Boyce index on the test set.
    """
    print("\n" + "=" * 60)
    print("EVALUATION: Standard Train/Test Split")
    print("=" * 60)

    Xp_train, Xp_test = train_test_split(
        X_presence, test_size=test_size, random_state=seed
    )
    Xb_train, Xb_test = train_test_split(
        X_background, test_size=test_size, random_state=seed
    )

    model = fit_maxent(Xp_train, Xb_train, beta_multiplier=beta_multiplier, seed=seed)

    X_test = np.vstack([Xp_test, Xb_test])
    y_test = np.concatenate([np.ones(len(Xp_test)), np.zeros(len(Xb_test))])

    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_pred)
    cbi = continuous_boyce_index(model.predict(Xp_test), model.predict(Xb_test))

    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  Boyce index: {cbi:.4f}")

    return model, auc, cbi


def _build_stratified_spatial_folds(presence_gdf, background_gdf, n_presence,
                                    n_background, n_folds, seed):
    """
    Build spatial CV folds stratified by presence points.

    Clusters presence points geographically into *n_folds* groups using
    elapid.GeographicKFold, then assigns each background point to the fold
    whose presence-cluster centroid is nearest.  This guarantees every fold
    contains presence points, avoiding the empty-fold problem that occurs
    when geographic blocking is driven by the much larger background set.

    Args:
        presence_gdf (GeoDataFrame): Presence point geometries.
        background_gdf (GeoDataFrame): Background point geometries.
        n_presence (int): Number of presence samples.
        n_background (int): Number of background samples.
        n_folds (int): Number of spatial folds.
        seed (int): Random seed.

    Returns:
        list[tuple[ndarray, ndarray]]: (train_indices, test_indices) per fold,
            indexed into the concatenated [presence | background] array.
    """
    from scipy.spatial import cKDTree

    # 1. Assign presence points to folds via geographic clustering
    gkf = elapid.GeographicKFold(n_splits=n_folds, random_state=seed)
    presence_fold_labels = np.full(n_presence, -1, dtype=int)
    for fold, (_, test_idx) in enumerate(gkf.split(presence_gdf)):
        presence_fold_labels[test_idx] = fold

    # 2. Compute centroid of each fold's presence cluster
    presence_coords = np.array(
        [(g.x, g.y) for g in presence_gdf.geometry]
    )
    bg_coords = np.array(
        [(g.x, g.y) for g in background_gdf.geometry]
    )

    centroids = np.array([
        presence_coords[presence_fold_labels == f].mean(axis=0)
        for f in range(n_folds)
    ])

    # 3. Assign each background point to nearest fold centroid
    tree = cKDTree(centroids)
    _, bg_fold_labels = tree.query(bg_coords)

    # 4. Build combined indices (presence first, then background offset)
    all_fold_labels = np.concatenate([presence_fold_labels, bg_fold_labels])
    folds = []
    for f in range(n_folds):
        test_mask = all_fold_labels == f
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        folds.append((train_idx, test_idx))

    return folds


#-----------------------------------------------------------
# SECTION: Evaluation — Spatial Block Cross-Validation
#-----------------------------------------------------------

def evaluate_spatial_cv(X_presence, X_background, presence_gdf, background_gdf,
                        n_folds=5, seed=42, beta_multiplier=1.5):
    """
    Evaluate the model using spatial block cross-validation.

    Uses elapid.GeographicKFold (KMeans-based) to hold out geographically
    coherent blocks, forcing the model to predict into unseen areas. Reports
    AUC-ROC, continuous Boyce index, and permutation importance per fold.
    Importance is computed on held-out test data to measure predictive
    contribution rather than training-data fit.

    Args:
        X_presence (numpy.ndarray): Shape (n_presence, n_features).
        X_background (numpy.ndarray): Shape (n_background, n_features).
        presence_gdf (GeoDataFrame): Presence points (geometry used for blocking).
        background_gdf (GeoDataFrame): Background points (geometry used for blocking).
        n_folds (int): Number of spatial folds.
        seed (int): Random seed.
        beta_multiplier (float): Regularization multiplier for MaxEnt.

    Returns:
        tuple:
            - list[float]: AUC-ROC per fold.
            - list[float]: CBI per fold.
            - numpy.ndarray: Mean permutation importance across folds, shape (n_features,).
    """
    print("\n" + "=" * 60)
    print(f"EVALUATION: Spatial Block Cross-Validation ({n_folds} folds)")
    print("=" * 60)

    # Build folds stratified by presence geography to ensure every fold
    # contains presence points (avoids background-density-dominated clustering)
    folds = _build_stratified_spatial_folds(
        presence_gdf, background_gdf,
        n_presence=len(X_presence), n_background=len(X_background),
        n_folds=n_folds, seed=seed,
    )

    X_all = np.vstack([X_presence, X_background])
    y_all = np.concatenate([np.ones(len(X_presence)), np.zeros(len(X_background))])
    n_bands = X_all.shape[1]

    fold_aucs = []
    fold_cbis = []
    fold_importances = []

    for fold, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        n_test_presence = int((y_test == 1).sum())
        n_test_bg = int((y_test == 0).sum())

        if n_test_presence == 0:
            print(f"  Fold {fold + 1}: skipped (no test presence points)")
            continue
        if n_test_presence < 5:
            print(f"  WARNING: Fold {fold + 1} has only {n_test_presence} test presence point(s) — metrics unreliable")

        # Separate presence/background in training set for MaxEnt fitting
        Xp_train = X_train[y_train == 1]
        Xb_train = X_train[y_train == 0]

        model = fit_maxent(Xp_train, Xb_train, beta_multiplier=beta_multiplier, seed=seed)

        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        cbi = continuous_boyce_index(y_pred[y_test == 1], y_pred[y_test == 0])

        # Permutation importance on held-out test fold
        fold_imp = permutation_importance(model, X_test, y_test, seed=seed)
        fold_importances.append(fold_imp)

        fold_aucs.append(auc)
        fold_cbis.append(cbi)

        n_train_p = int((y_train == 1).sum())
        n_train_bg = int((y_train == 0).sum())
        print(f"  Fold {fold + 1}: AUC = {auc:.4f}, CBI = {cbi:.4f}  "
              f"(train: {n_train_p}p/{n_train_bg}bg, "
              f"test: {n_test_presence}p/{n_test_bg}bg)")

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    mean_cbi = np.nanmean(fold_cbis)
    std_cbi = np.nanstd(fold_cbis)
    print(f"\n  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Mean CBI: {mean_cbi:.4f} ± {std_cbi:.4f}")

    # Average importance across folds
    if fold_importances:
        mean_importances = np.mean(fold_importances, axis=0)
    else:
        mean_importances = np.zeros(n_bands)

    return fold_aucs, fold_cbis, mean_importances


#-----------------------------------------------------------
# SECTION: Regularization Tuning
#-----------------------------------------------------------

def tune_beta(X_presence, X_background, presence_gdf, background_gdf,
              beta_grid, n_folds=5, seed=42):
    """
    Select the best beta_multiplier via spatial block cross-validation.

    For each candidate beta value, runs spatial CV and records mean AUC.
    Returns the beta with the highest mean AUC across folds.

    Args:
        X_presence (numpy.ndarray): Shape (n_presence, n_features).
        X_background (numpy.ndarray): Shape (n_background, n_features).
        presence_gdf (GeoDataFrame): Presence points (geometry used for blocking).
        background_gdf (GeoDataFrame): Background points (geometry used for blocking).
        beta_grid (list[float]): Candidate beta_multiplier values to evaluate.
        n_folds (int): Number of spatial folds.
        seed (int): Random seed.

    Returns:
        float: Best beta_multiplier value.
    """
    print("\n" + "=" * 60)
    print("REGULARIZATION TUNING (spatial CV)")
    print("=" * 60)

    # Pre-compute geographic folds once so all betas use the same splits
    # Folds are stratified by presence geography (see _build_stratified_spatial_folds)
    folds = _build_stratified_spatial_folds(
        presence_gdf, background_gdf,
        n_presence=len(X_presence), n_background=len(X_background),
        n_folds=n_folds, seed=seed,
    )

    X_all = np.vstack([X_presence, X_background])
    y_all = np.concatenate([np.ones(len(X_presence)), np.zeros(len(X_background))])

    best_beta = beta_grid[0]
    best_mean_auc = -1.0

    for beta in beta_grid:
        fold_aucs = []

        for train_idx, test_idx in folds:
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]

            n_test_p = int((y_test == 1).sum())
            if n_test_p == 0:
                continue
            if n_test_p < 5:
                print(f"  WARNING: Fold has only {n_test_p} test presence point(s) — AUC unreliable")

            Xp_train = X_train[y_train == 1]
            Xb_train = X_train[y_train == 0]

            model = fit_maxent(Xp_train, Xb_train, beta_multiplier=beta, seed=seed)
            auc = roc_auc_score(y_test, model.predict(X_test))
            fold_aucs.append(auc)

        mean_auc = np.mean(fold_aucs) if fold_aucs else 0.0
        print(f"  beta = {beta:.1f}: mean AUC = {mean_auc:.4f} ({len(fold_aucs)} folds)")

        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_beta = beta

    print(f"\n  Selected beta_multiplier = {best_beta} (AUC = {best_mean_auc:.4f})")
    return best_beta


#-----------------------------------------------------------
# SECTION: Permutation Importance
#-----------------------------------------------------------

def permutation_importance(model, X_test, y_test, n_repeats=10, seed=42):
    """
    Compute permutation importance for each feature band on held-out data.

    Shuffles each band independently, measures the drop in AUC-ROC relative
    to the unshuffled baseline, and reports mean importance across repeats.
    Must be called on data the model was NOT trained on to measure predictive
    (not descriptive) importance.

    Args:
        model (elapid.MaxentModel): Fitted model.
        X_test (numpy.ndarray): Held-out feature array, shape (n_samples, n_features).
        y_test (numpy.ndarray): Held-out labels (1=presence, 0=background), shape (n_samples,).
        n_repeats (int): Number of shuffle repetitions per band.
        seed (int): Random seed.

    Returns:
        numpy.ndarray: Mean importance score per band, shape (n_features,).
    """
    baseline_auc = roc_auc_score(y_test, model.predict(X_test))

    rng = np.random.default_rng(seed)
    n_bands = X_test.shape[1]
    importances = np.zeros(n_bands)

    for band in range(n_bands):
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X_test.copy()
            rng.shuffle(X_shuffled[:, band])
            shuffled_auc = roc_auc_score(y_test, model.predict(X_shuffled))
            drops.append(baseline_auc - shuffled_auc)
        importances[band] = np.mean(drops)

    return importances


def print_importance(importances, top_n=10):
    """
    Print the top-N most important features by mean AUC drop.

    Args:
        importances (numpy.ndarray): Mean importance per feature, shape (n_features,).
        top_n (int): Number of top features to display.
    """
    print("\n  Permutation Importance (top features by AUC drop, held-out data):")
    ranked = np.argsort(importances)[::-1]
    for rank, idx in enumerate(ranked[:top_n]):
        print(f"    {rank + 1}. PC{idx + 1:02d}: {importances[idx]:.4f}")


#-----------------------------------------------------------
# SECTION: Extraction Cache
#-----------------------------------------------------------

def save_extraction_cache(cache_dir, presence_annotated, background_annotated):
    """
    Save annotated GeoDataFrames (geometry + band values) to disk for reuse.

    Writes two .gpkg files. Skips silently if cache_dir is falsy.

    Args:
        cache_dir (str): Directory to write cache files into (created if absent).
        presence_annotated (GeoDataFrame): Annotated presence points.
        background_annotated (GeoDataFrame): Annotated background points.
    """
    if not cache_dir:
        return

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    presence_annotated.to_file(cache_path / "presence_annotated.gpkg", driver="GPKG")
    background_annotated.to_file(cache_path / "background_annotated.gpkg", driver="GPKG")

    print(f"  Cache saved to: {cache_path}")


def load_extraction_cache(cache_dir, source_paths=None):
    """
    Load annotated GeoDataFrames from disk if available and fresh.

    Returns None if cache_dir is falsy, any expected file is missing, or the
    cache is older than the source inputs (stale).

    Args:
        cache_dir (str): Directory to look for cache files.
        source_paths (list[str] | None): Paths to source files (e.g. shapefile,
            raster) used to generate the cache.  If provided, the cache is
            invalidated when any source file is newer than the cache.

    Returns:
        tuple or None: (presence_annotated, background_annotated)
            if all cache files exist and are fresh, otherwise None.
    """
    if not cache_dir:
        return None

    cache_path = Path(cache_dir)
    expected_files = [
        cache_path / "presence_annotated.gpkg",
        cache_path / "background_annotated.gpkg",
    ]

    if not all(f.exists() for f in expected_files):
        return None

    # Invalidate cache if source inputs have been modified since cache creation
    if source_paths:
        cache_mtime = min(f.stat().st_mtime for f in expected_files)
        src_mtime = max(
            Path(p).stat().st_mtime for p in source_paths if Path(p).exists()
        )
        if cache_mtime < src_mtime:
            print("  Cache is stale (inputs modified after cache creation). Re-extracting...")
            return None

    print(f"  Loading cached extraction from: {cache_path}")
    presence = gpd.read_file(expected_files[0])
    background = gpd.read_file(expected_files[1])

    return presence, background


#-----------------------------------------------------------
# SECTION: Model Persistence
#-----------------------------------------------------------

def save_model(model, model_path):
    """
    Save a fitted MaxEnt model to disk using joblib.

    Skips silently if model_path is falsy.

    Args:
        model (elapid.MaxentModel): Fitted model to save.
        model_path (str): Destination file path (e.g. "maxent_model.joblib").
    """
    if not model_path:
        return

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"  Model saved to: {model_path}")


def load_model(model_path):
    """
    Load a fitted MaxEnt model from disk.

    Args:
        model_path (str): Path to a saved joblib model file.

    Returns:
        elapid.MaxentModel: The loaded model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(
            f"No saved model found at: '{model_path}'. Run 'fit' mode first."
        )

    model = joblib.load(model_path)
    print(f"  Model loaded from: {model_path}")
    return model


#-----------------------------------------------------------
# SECTION: CLI Workflows
#-----------------------------------------------------------

def run_fit():
    """
    Run the full fit workflow: data extraction (cached), evaluation, final model
    fit, permutation importance, and model save.

    Reads all configuration from module-level constants.
    """
    validate_paths(PRESENCE_SHP=PRESENCE_SHP, RASTER_STACK=RASTER_STACK)

    # --- Load from cache or run full extraction ---
    cached = load_extraction_cache(CACHE_DIR, source_paths=[PRESENCE_SHP, RASTER_STACK])

    if cached is not None:
        presence_annotated, background_annotated = cached
        X_presence_raw = get_band_values(presence_annotated)
        X_background_raw = get_band_values(background_annotated)
        print(f"  Presence samples:   {X_presence_raw.shape}")
        print(f"  Background samples: {X_background_raw.shape}")
    else:
        print("Loading presence points...")
        presence_gdf = load_presence_points(PRESENCE_SHP, RASTER_STACK)

        # Deduplicate: keep one point per unique raster pixel
        print("\nDeduplicating presence points to unique raster pixels...")
        presence_gdf = deduplicate_to_unique_pixels(presence_gdf, RASTER_STACK)

        print("\nGenerating background points...")
        np.random.seed(RANDOM_SEED)  # seed before elapid call for reproducibility
        bg_points = elapid.sample_raster(RASTER_STACK, N_BACKGROUND)
        print(f"Generated {len(bg_points)} background points")

        print("\nExtracting raster values at presence locations...")
        presence_annotated = elapid.annotate(
            presence_gdf.geometry, [RASTER_STACK], drop_na=True
        )
        print(f"  Presence samples after annotation: {len(presence_annotated)}")

        print("Extracting raster values at background locations...")
        background_annotated = elapid.annotate(
            bg_points, [RASTER_STACK], drop_na=True
        )
        print(f"  Background samples: {len(background_annotated)}")

        save_extraction_cache(CACHE_DIR, presence_annotated, background_annotated)

        X_presence_raw = get_band_values(presence_annotated)
        X_background_raw = get_band_values(background_annotated)

    validate_arrays(X_presence=X_presence_raw, X_background=X_background_raw)

    # --- PCA dimensionality reduction ---
    # Reduces correlated 64-band embeddings to uncorrelated principal components.
    # Merow et al. (2013, Section III.B) recommends PCA for correlated predictors.
    print("\n" + "=" * 60)
    print(f"PCA: Reducing {X_presence_raw.shape[1]} bands to {N_PCA_COMPONENTS} components")
    print("=" * 60)

    # Fit PCA on the full dataset (presence + background) so it captures the
    # landscape's environmental variation. The fitted PCA is then passed into
    # elapid as a preprocessor so it is automatically applied during predict()
    # and apply_model_to_rasters().
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_SEED)
    X_all_raw = np.vstack([X_presence_raw, X_background_raw])
    pca.fit(X_all_raw)

    X_presence = pca.transform(X_presence_raw)
    X_background = pca.transform(X_background_raw)

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  Variance explained: {explained:.1f}%")
    print(f"  Presence shape:     {X_presence.shape}")
    print(f"  Background shape:   {X_background.shape}")

    # --- Regularization tuning ---
    best_beta = tune_beta(
        X_presence, X_background, presence_annotated, background_annotated,
        beta_grid=BETA_GRID, n_folds=N_SPATIAL_FOLDS, seed=RANDOM_SEED,
    )

    # --- Evaluation: Standard split ---
    _, std_auc, std_cbi = evaluate_standard_split(
        X_presence, X_background, test_size=TEST_SIZE, seed=RANDOM_SEED,
        beta_multiplier=best_beta,
    )

    # --- Evaluation: Spatial CV (also computes held-out permutation importance) ---
    spatial_aucs, spatial_cbis, cv_importances = evaluate_spatial_cv(
        X_presence, X_background, presence_annotated, background_annotated,
        n_folds=N_SPATIAL_FOLDS, seed=RANDOM_SEED, beta_multiplier=best_beta,
    )

    # --- Fit final model on ALL data ---
    # Fit on PCA-transformed data (no preprocessor), then attach PCA manually.
    # We can't pass preprocessor to fit() because elapid's fit() internally
    # calls predict() on already-transformed data, which double-applies the
    # preprocessor and crashes. Attaching afterward means predict() and
    # apply_model_to_rasters() will apply PCA to raw 64-band input correctly.
    print("\n" + "=" * 60)
    print("Fitting final model on all data...")
    print("=" * 60)
    final_model = fit_maxent(
        X_presence, X_background,
        beta_multiplier=best_beta, seed=RANDOM_SEED,
    )
    final_model.preprocessor = pca
    save_model(final_model, MODEL_PATH)

    # --- Variable importance (from spatial CV held-out folds) ---
    print("\n" + "=" * 60)
    print("VARIABLE IMPORTANCE")
    print("=" * 60)
    print_importance(cv_importances)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Presence points used:  {len(X_presence)}")
    print(f"  Background points:     {len(X_background)}")
    print(f"  PCA components:        {N_PCA_COMPONENTS} ({explained:.1f}% variance explained)")
    print(f"  Beta multiplier:       {best_beta}")
    print(f"  Standard split AUC:    {std_auc:.4f}")
    print(f"  Standard split CBI:    {std_cbi:.4f}")
    print(f"  Spatial CV mean AUC:   {np.mean(spatial_aucs):.4f} ± {np.std(spatial_aucs):.4f}")
    print(f"  Spatial CV mean CBI:   {np.nanmean(spatial_cbis):.4f} ± {np.nanstd(spatial_cbis):.4f}")

    if std_auc - np.mean(spatial_aucs) > 0.05:
        print("\n  NOTE: Standard split AUC is notably higher than spatial CV AUC.")
        print("  This suggests spatial autocorrelation is inflating the standard metric.")
        print("  Report the spatial CV result as the more honest estimate.")


def run_predict():
    """
    Run the predict workflow: load the saved model and write the suitability raster.

    Uses elapid.apply_model_to_rasters for block-by-block prediction with
    automatic nodata handling and deflate compression.

    Requires MODEL_PATH and OUTPUT_RASTER to be set in constants.
    Run 'fit' mode first to generate the saved model.
    """
    validate_paths(
        OUTPUT_RASTER=OUTPUT_RASTER, MODEL_PATH=MODEL_PATH, RASTER_STACK=RASTER_STACK
    )

    model = load_model(MODEL_PATH)

    print("\nGenerating suitability raster...")
    Path(OUTPUT_RASTER).parent.mkdir(parents=True, exist_ok=True)
    elapid.apply_model_to_rasters(model, [RASTER_STACK], OUTPUT_RASTER)
    print(f"Suitability raster written to: {OUTPUT_RASTER}")


def run_export_folds():
    """
    Export spatial CV fold assignments as shapefiles for visualization.

    Loads presence and background points from cache, recomputes fold
    assignments using the same seed and parameters as run_fit(), and
    writes one shapefile per fold plus a combined shapefile with all
    points labeled by fold number.

    Output directory: data/outputs/folds/
    """
    cached = load_extraction_cache(CACHE_DIR, source_paths=[PRESENCE_SHP, RASTER_STACK])
    if cached is None:
        raise RuntimeError(
            "No cached extraction found. Run 'fit' mode first to generate the cache."
        )

    presence_annotated, background_annotated = cached
    print(f"  Presence samples:   {len(presence_annotated)}")
    print(f"  Background samples: {len(background_annotated)}")

    folds = _build_stratified_spatial_folds(
        presence_annotated, background_annotated,
        n_presence=len(presence_annotated),
        n_background=len(background_annotated),
        n_folds=N_SPATIAL_FOLDS, seed=RANDOM_SEED,
    )

    n_presence = len(presence_annotated)
    out_dir = Path(CACHE_DIR).parent / "folds"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a combined GeoDataFrame with fold and type labels
    all_gdf = gpd.GeoDataFrame(
        {"geometry": list(presence_annotated.geometry) + list(background_annotated.geometry)},
        crs=presence_annotated.crs,
    )
    all_gdf["fold"] = -1
    all_gdf["point_type"] = ["presence"] * n_presence + ["background"] * len(background_annotated)

    for fold_idx, (_, test_idx) in enumerate(folds):
        all_gdf.loc[test_idx, "fold"] = fold_idx + 1  # 1-indexed for readability

    # Export per-fold shapefiles (presence points only for cleaner visualization)
    for fold_num in range(1, N_SPATIAL_FOLDS + 1):
        fold_presence = all_gdf[
            (all_gdf["fold"] == fold_num) & (all_gdf["point_type"] == "presence")
        ][["geometry", "fold"]].copy()

        fold_path = out_dir / f"fold_{fold_num}_presence.shp"
        fold_presence.to_file(fold_path)
        print(f"  Fold {fold_num}: {len(fold_presence)} presence points → {fold_path}")

    # Export combined file with all points
    combined_path = out_dir / "all_folds_presence.shp"
    all_presence = all_gdf[all_gdf["point_type"] == "presence"][["geometry", "fold"]].copy()
    all_presence.to_file(combined_path)
    print(f"\n  Combined presence folds → {combined_path}")

    # Also export background folds for reference
    combined_bg_path = out_dir / "all_folds_background.shp"
    all_bg = all_gdf[all_gdf["point_type"] == "background"][["geometry", "fold"]].copy()
    all_bg.to_file(combined_bg_path)
    print(f"  Combined background folds → {combined_bg_path}")

    print(f"\nAll fold shapefiles written to: {out_dir}")


# ---------------------------------------------------------------------------
# Poster-quality suitability map
# ---------------------------------------------------------------------------

def make_maxent_map(tif_path: str,
                    out_png: str,
                    out_svg: str = None,
                    title: str = "Habitat Suitability — Bromus tectorum\nMunicipal District of Ranchland No. 66, Alberta",
                    shp_path: str = None,
                    raster_path: str = None):
    """
    Renders a poster-quality MaxEnt suitability heatmap from the prediction
    GeoTIFF with standard cartographic elements (title, scalebar, north arrow,
    easting/northing axes, colorbar).

    Optionally overlays deduplicated presence points from the shapefile.
    Deduplication uses the raster pixel grid to remove points that fall in
    the same 10m cell.

    Args:
        tif_path:    Path to the suitability GeoTIFF (single-band, values 0–1).
        out_png:     Output path for the PNG map.
        out_svg:     Output path for the SVG map (vector, optional).
        title:       Map title (supports newlines).
        shp_path:    Path to the presence points shapefile. If provided,
                     deduplicated presence points are overlaid on the map.
        raster_path: Path to the 64-band raster stack (used for pixel
                     deduplication of presence points). Required if shp_path
                     is provided.
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top":  False,
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

    # Continuous blue → yellow → orange colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "maxent_suit",
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

    # Overlay deduplicated presence points
    if shp_path:
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=3857)
        gdf = gdf.to_crs(epsg=26911)

        # Deduplicate to unique raster pixels (same logic as training pipeline)
        if raster_path:
            gdf = deduplicate_to_unique_pixels(gdf, raster_path)

        ax.scatter(gdf.geometry.x, gdf.geometry.y,
                   c="#ffe600", s=14, marker="o", linewidths=0.5,
                   edgecolors="black", alpha=0.9,
                   label=f"Presence ({len(gdf)} pts, deduplicated)", zorder=5)
        ax.legend(loc="lower left", framealpha=0.3,
                  labelcolor="white", fontsize=9,
                  facecolor="#333333", edgecolor="#555555")

    # Extend axis limits slightly beyond the raster to add padding
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

    # Axis labels — Easting / Northing with full metre values
    import matplotlib.ticker as mticker
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Easting (m)", color="white", fontsize=10)
    ax.set_ylabel("Northing (m)", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    # Scalebar — CRS is EPSG:26911 (metres), fixed at 20 km with ticks at 0, 5, 10, 20
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

    # Add tick labels at 0, 5, 10, 20 km along the scalebar
    # Use actual axis width (accounts for padding) to position labels
    ax_xlim = ax.get_xlim()
    ax_width = ax_xlim[1] - ax_xlim[0]
    bar_20km_frac = 20000 / ax_width  # fraction of axes width that 20 km spans
    # ScaleBar "lower right" anchors right edge ~2% from right
    bar_right = 0.98
    bar_left  = bar_right - bar_20km_frac
    for km in [0, 5, 10, 20]:
        x_pos = bar_left + (km / 20) * bar_20km_frac
        ax.text(x_pos, 0.045, f"{km} km" if km == 20 else str(km),
                transform=ax.transAxes, ha="center", va="bottom",
                color="white", fontsize=8, zorder=10)

    # North arrow — classic two-tone chevron with "N" label
    from matplotlib.patches import Polygon
    # Centre and size in axes-fraction coordinates
    cx, cy = 0.90, 0.14   # centre of the arrow base
    w, h   = 0.0125, 0.03  # half-width and height of the chevron

    # Left half (white fill)
    left_tri = Polygon(
        [(cx, cy + h), (cx - w, cy - h * 0.3), (cx, cy)],
        closed=True, facecolor="white", edgecolor="white",
        linewidth=0.8, transform=ax.transAxes, zorder=10,
    )
    # Right half (dark fill)
    right_tri = Polygon(
        [(cx, cy + h), (cx + w, cy - h * 0.3), (cx, cy)],
        closed=True, facecolor="#888888", edgecolor="white",
        linewidth=0.8, transform=ax.transAxes, zorder=10,
    )
    ax.add_patch(left_tri)
    ax.add_patch(right_tri)

    # "N" label beneath the chevron
    ax.text(cx, cy - h * 0.3 - 0.015, "N",
            transform=ax.transAxes, ha="center", va="top",
            color="white", fontsize=12, fontweight="bold", zorder=10)

    # Title
    ax.set_title(title, color="white", fontsize=16, fontweight="bold",
                 pad=18, loc="left")

    # Subtitle — aligned with title, nudged slightly right
    ax.text(0.02, 1.005, "Method: MaxEnt  ·  CRS: NAD83 / UTM Zone 11N (EPSG:26911)  ·  Resolution: 10m",
            transform=ax.transAxes, color="white", fontsize=9, style="italic",
            ha="left", va="bottom")

    # ── Inset locator map (Alberta with study area) ──────────────────────
    from pyproj import Transformer
    from matplotlib.patches import Rectangle

    # Place inset in the lower-right blank area, near the north arrow
    inset_ax = ax.inset_axes([0.66, 0.05, 0.14, 0.24])  # [x, y, w, h] in axes fraction
    inset_ax.set_facecolor("#1a1a1a")
    for sp in inset_ax.spines.values():
        sp.set_visible(False)

    # Load Alberta province boundary from Natural Earth (provinces)
    ne_url = ("https://naciscdn.org/naturalearth/10m/cultural/"
              "ne_10m_admin_1_states_provinces.zip")
    provinces = gpd.read_file(ne_url)
    alberta = provinces[provinces["name"] == "Alberta"]
    alberta.plot(ax=inset_ax, facecolor="#3a3a3a", edgecolor="white",
                 linewidth=0.5)

    # Study area polygon from the MD boundary shapefile
    md_bound = gpd.read_file(
        os.path.join(os.path.dirname(__file__),
                     "data", "inputs", "extras", "MD_bound_zipped_11N.shp"))
    md_bound_4326 = md_bound.to_crs(epsg=4326)
    md_bound_4326.plot(ax=inset_ax, facecolor="#ff4444", edgecolor="#ff4444",
                       alpha=0.55, linewidth=0.8, zorder=5)

    inset_ax.set_xlim(-121, -109)
    inset_ax.set_ylim(48.5, 60.5)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_title("Alberta", color="white", fontsize=10, pad=3)

    plt.tight_layout(pad=1.5, rect=[0.03, 0.03, 0.97, 0.97])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight",
                pad_inches=0.3, facecolor=fig.get_facecolor())
    print(f"  PNG saved → {out_png}")

    if out_svg:
        Path(out_svg).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_svg, bbox_inches="tight",
                    pad_inches=0.3, facecolor=fig.get_facecolor())
        print(f"  SVG saved → {out_svg}")

    plt.close()


def run_create_map():
    """Load the suitability raster and render a poster-quality map with
    deduplicated presence points overlaid."""
    validate_paths(OUTPUT_RASTER=OUTPUT_RASTER, PRESENCE_SHP=PRESENCE_SHP,
                   RASTER_STACK=RASTER_STACK)
    print("\n── Creating suitability map ──")
    make_maxent_map(
        tif_path=OUTPUT_RASTER,
        out_png=OUTPUT_MAP_PNG,
        out_svg=OUTPUT_MAP_SVG,
        shp_path=PRESENCE_SHP,
        raster_path=RASTER_STACK,
    )
    print("  Done.\n")


def main():
    """
    Entry point. Parses the mode argument and dispatches to the appropriate workflow.

    Modes:
        fit         — extract features (cached), evaluate, fit final model, save to disk
        predict     — load saved model, write suitability raster
        all         — run fit then predict
        exportfolds — export spatial CV fold assignments as shapefiles
        createmap   — render a poster-quality suitability map from the prediction raster
    """
    parser = argparse.ArgumentParser(
        description="MaxEnt habitat suitability model — Bromus tectorum, MD Ranchland No. 66"
    )
    parser.add_argument(
        "mode",
        choices=["fit", "predict", "all", "exportfolds", "createmap"],
        help=(
            "fit: evaluate and fit the model; "
            "predict: generate suitability raster from saved model; "
            "all: fit then predict; "
            "exportfolds: export spatial CV fold assignments as shapefiles; "
            "createmap: render poster-quality suitability map from prediction raster"
        ),
    )
    args = parser.parse_args()

    if args.mode == "fit":
        run_fit()
    elif args.mode == "predict":
        run_predict()
    elif args.mode == "exportfolds":
        run_export_folds()
    elif args.mode == "createmap":
        run_create_map()
    else:  # all
        run_fit()
        run_predict()


if __name__ == "__main__":
    main()
