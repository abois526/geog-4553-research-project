"""
MaxEnt Habitat Suitability Model — Bromus tectorum
MD Ranchland No. 66, Alberta

Baseline model using Alpha Earth Foundation Model embeddings (64 bands).
Uses elapid for MaxEnt implementation and its built-in utilities for
background sampling, raster annotation, spatial cross-validation, and
raster prediction.

Evaluation: standard train/test split + spatial block cross-validation.
Metrics: AUC-ROC and continuous Boyce index (CBI).
Extracted raster values are cached to CACHE_DIR on first run to skip re-extraction.
"""

import numpy as np
import geopandas as gpd
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from pathlib import Path
import argparse
import joblib
import elapid
import warnings

# Suppress known nuisance warnings; keep convergence/CRS warnings visible
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*palette.*")

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
N_SPATIAL_FOLDS = 4

# Test size for standard split
TEST_SIZE = 0.2

# Directory to cache extracted raster values (set to "" to disable caching)
CACHE_DIR = r"/Users/andrew/Workspace/github-repos/geog-4553-research-project/models/MaxEnt/data/outputs/cache/"  # e.g., r"C:\path\to\cache"

# Path to save/load the fitted final model (required for predict mode)
MODEL_PATH = r"/Users/andrew/Workspace/github-repos/geog-4553-research-project/models/MaxEnt/data/outputs/maxent_model.joblib"  # e.g., r"C:\path\to\maxent_model.joblib"

# Regularization multiplier grid for tuning (Merow et al. 2013)
BETA_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


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

def fit_maxent(X_presence, X_background, beta_multiplier=1.5):
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

    Returns:
        elapid.MaxentModel: Fitted model.
    """
    model = elapid.MaxentModel(
        feature_types=["linear", "quadratic", "hinge"],
        beta_multiplier=beta_multiplier,
    )

    # elapid expects: x = feature array, y = labels (1=presence, 0=background)
    X = np.vstack([X_presence, X_background])
    y = np.concatenate([
        np.ones(len(X_presence)),
        np.zeros(len(X_background)),
    ])

    model.fit(X, y)
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
    pe_ratios = []

    for c in centers:
        lo, hi = c - window_width / 2, c + window_width / 2
        n_pres = np.sum((pred_presence >= lo) & (pred_presence < hi))
        n_bg = np.sum((pred_background >= lo) & (pred_background < hi))
        if n_bg == 0:
            continue
        F_pres = n_pres / len(pred_presence)
        F_exp = n_bg / len(pred_background)
        pe_ratios.append(F_pres / F_exp)

    if len(pe_ratios) < 3:
        return np.nan

    return spearmanr(np.arange(len(pe_ratios)), pe_ratios).correlation


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

    model = fit_maxent(Xp_train, Xb_train, beta_multiplier=beta_multiplier)

    X_test = np.vstack([Xp_test, Xb_test])
    y_test = np.concatenate([np.ones(len(Xp_test)), np.zeros(len(Xb_test))])

    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_pred)
    cbi = continuous_boyce_index(model.predict(Xp_test), model.predict(Xb_test))

    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  Boyce index: {cbi:.4f}")

    return model, auc, cbi


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

    # Combine presence and background for geographic splitting
    all_geom = presence_gdf.geometry.tolist() + background_gdf.geometry.tolist()
    all_gdf = gpd.GeoDataFrame(geometry=all_geom, crs=presence_gdf.crs)

    X_all = np.vstack([X_presence, X_background])
    y_all = np.concatenate([np.ones(len(X_presence)), np.zeros(len(X_background))])
    n_bands = X_all.shape[1]

    gkf = elapid.GeographicKFold(n_splits=n_folds, random_state=seed)

    fold_aucs = []
    fold_cbis = []
    fold_importances = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(all_gdf)):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        n_test_presence = int((y_test == 1).sum())
        n_test_bg = int((y_test == 0).sum())

        if n_test_presence == 0:
            print(f"  Fold {fold + 1}: skipped (no test presence points)")
            continue

        # Separate presence/background in training set for MaxEnt fitting
        Xp_train = X_train[y_train == 1]
        Xb_train = X_train[y_train == 0]

        model = fit_maxent(Xp_train, Xb_train, beta_multiplier=beta_multiplier)

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
    all_geom = presence_gdf.geometry.tolist() + background_gdf.geometry.tolist()
    all_gdf = gpd.GeoDataFrame(geometry=all_geom, crs=presence_gdf.crs)

    X_all = np.vstack([X_presence, X_background])
    y_all = np.concatenate([np.ones(len(X_presence)), np.zeros(len(X_background))])

    gkf = elapid.GeographicKFold(n_splits=n_folds, random_state=seed)
    folds = list(gkf.split(all_gdf))

    best_beta = beta_grid[0]
    best_mean_auc = -1.0

    for beta in beta_grid:
        fold_aucs = []

        for train_idx, test_idx in folds:
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]

            if (y_test == 1).sum() == 0:
                continue

            Xp_train = X_train[y_train == 1]
            Xb_train = X_train[y_train == 0]

            model = fit_maxent(Xp_train, Xb_train, beta_multiplier=beta)
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
    Print the top-N most important bands by mean AUC drop.

    Args:
        importances (numpy.ndarray): Mean importance per band, shape (n_features,).
        top_n (int): Number of top bands to display.
    """
    print("\n  Permutation Importance (top bands by AUC drop, held-out data):")
    ranked = np.argsort(importances)[::-1]
    for rank, band_idx in enumerate(ranked[:top_n]):
        print(f"    {rank + 1}. Band {band_idx}: {importances[band_idx]:.4f}")


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


def load_extraction_cache(cache_dir):
    """
    Load annotated GeoDataFrames from disk if available.

    Returns None if cache_dir is falsy or any expected file is missing.

    Args:
        cache_dir (str): Directory to look for cache files.

    Returns:
        tuple or None: (presence_annotated, background_annotated)
            if all cache files exist, otherwise None.
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
    cached = load_extraction_cache(CACHE_DIR)

    if cached is not None:
        presence_annotated, background_annotated = cached
        X_presence = get_band_values(presence_annotated)
        X_background = get_band_values(background_annotated)
        print(f"  Presence samples:   {X_presence.shape}")
        print(f"  Background samples: {X_background.shape}")
    else:
        print("Loading presence points...")
        presence_gdf = load_presence_points(PRESENCE_SHP, RASTER_STACK)

        print("\nGenerating background points...")
        np.random.seed(RANDOM_SEED)  # seed before elapid call for reproducibility
        bg_points = elapid.sample_raster(RASTER_STACK, N_BACKGROUND)
        print(f"Generated {len(bg_points)} background points")

        print("\nExtracting raster values at presence locations...")
        presence_annotated = elapid.annotate(
            presence_gdf.geometry, [RASTER_STACK], drop_na=True
        )
        print(f"  Presence samples: {len(presence_annotated)}")

        print("Extracting raster values at background locations...")
        background_annotated = elapid.annotate(
            bg_points, [RASTER_STACK], drop_na=True
        )
        print(f"  Background samples: {len(background_annotated)}")

        save_extraction_cache(CACHE_DIR, presence_annotated, background_annotated)

        X_presence = get_band_values(presence_annotated)
        X_background = get_band_values(background_annotated)

    validate_arrays(X_presence=X_presence, X_background=X_background)

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
    print("\n" + "=" * 60)
    print("Fitting final model on all data...")
    print("=" * 60)
    final_model = fit_maxent(X_presence, X_background, beta_multiplier=best_beta)
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
    print(f"  Predictor bands:       {X_presence.shape[1]}")
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
    elapid.apply_model_to_rasters(model, [RASTER_STACK], OUTPUT_RASTER)
    print(f"Suitability raster written to: {OUTPUT_RASTER}")


def main():
    """
    Entry point. Parses the mode argument and dispatches to the appropriate workflow.

    Modes:
        fit     — extract features (cached), evaluate, fit final model, save to disk
        predict — load saved model, write suitability raster
        all     — run fit then predict
    """
    parser = argparse.ArgumentParser(
        description="MaxEnt habitat suitability model — Bromus tectorum, MD Ranchland No. 66"
    )
    parser.add_argument(
        "mode",
        choices=["fit", "predict", "all"],
        help=(
            "fit: evaluate and fit the model; "
            "predict: generate suitability raster from saved model; "
            "all: fit then predict"
        ),
    )
    args = parser.parse_args()

    if args.mode == "fit":
        run_fit()
    elif args.mode == "predict":
        run_predict()
    else:  # all
        run_fit()
        run_predict()


if __name__ == "__main__":
    main()
