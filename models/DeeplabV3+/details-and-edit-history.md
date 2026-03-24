# DeepLabV3+ Edit History

## Preface

This document logs every change made to Devon's original 12-script DeepLabV3+ pipeline during consolidation into a single file (`deeplabv3_pipeline.py`). Review the changes marked **Critical** before running the pipeline — those fix bugs that would produce incorrect results or crash at runtime.

### Dependencies

Install the following before running. All are available via pip:

```
pip install torch rasterio geopandas numpy scipy albumentations segmentation-models-pytorch matplotlib matplotlib-scalebar
```

- **PyTorch**: Install the CUDA variant if you have a GPU — see https://pytorch.org/get-started/locally/. CPU works but training will be significantly slower.
- **GDAL/rasterio**: On some systems, rasterio requires GDAL to be installed at the OS level first. If `pip install rasterio` fails, try `conda install -c conda-forge rasterio` instead.

### How to run

```bash
# Run the full pipeline (all 10 steps in order)
python deeplabv3_pipeline.py

# Run only specific steps
python deeplabv3_pipeline.py train infer

# Re-run only post-training steps
python deeplabv3_pipeline.py infer smooth clip map
```

Available step names: `inspect`, `mask`, `points`, `tile`, `balance`, `train`, `infer`, `smooth`, `clip`, `map`

Steps must be run in order at least once. After the first full run, you can re-run individual steps selectively. The pipeline caches intermediate results (masks, tiles) and skips them if they're up to date.

### Where your original scripts ended up

| Original script | Pipeline function | Step |
|----------------|-------------------|------|
| `habitat_inspect.py`, `inspect_points.py`, `inspect_shapefile.py` | `inspect_data()` | Step 1 |
| `create_masks.py` | `create_habitat_mask()` | Step 2 |
| `create_masks_with_points.py` | `create_mask_with_points()` | Step 3 |
| `create_tiles.py` | `create_tiles()` | Step 4 |
| `calssBalanceTest.py` | `check_class_balance()` | Step 5 |
| `step3_dataset.py` | `EmbeddingHabitatDataset` class (inline) | — |
| `step4_model.py` | `HabitatModel` class (inline) | — |
| `step6_train.py` | `train()` | Step 6 |
| `step7_inference.py` | `inference()` | Step 7 |
| `step8_suitability.py` | `smooth()` | Step 8 |
| *(new)* | `clip_to_study_area()` | Step 9 |
| *(new)* | `make_suitability_map()` | Step 10 |

### How the input data flows

The pipeline expects a **single multi-band GeoTIFF** as its predictor input (`EMBEDDING_PATH`). Currently this is a 64-band satellite embedding raster (`emb11Nclp.tif`). Every band in this raster becomes one input channel to the model. The pipeline reads this raster in three places:

1. **Step 2 (mask creation)** — uses it as the spatial reference (CRS, resolution, extent) to reproject the landcover raster
2. **Step 4 (tiling)** — slices it into 256x256 training tiles, reading all bands per tile
3. **Step 7 (inference)** — slides a window across it to produce the suitability map

The model processes all bands simultaneously — they are **not** handled individually. The 1x1 Conv projector maps the input bands down to 64 channels, which then feed into the DeepLabV3+ encoder.

### Adding additional predictor variables

If you want to include additional rasters (e.g., elevation, slope, climate layers), you must **stack them into the embedding raster as additional bands before running the pipeline**. The pipeline reads exactly one raster file for predictor data. Here's what to do:

1. **Pre-stack your data externally.** Use `rasterio`, GDAL, or QGIS to merge the new rasters as additional bands into a single GeoTIFF. All bands must share the same CRS, resolution, extent, and pixel alignment. If they don't, reproject/resample them to match the embedding raster first.

2. **Update `EMBEDDING_PATH`** to point to your new stacked raster.

3. **Update `in_channels` in two places:**
   - `HabitatModel.__init__()` — line ~149: `def __init__(self, in_channels=64, ...)` — change `64` to the new band count
   - `train()` — line ~565: `model = HabitatModel(in_channels=64, ...)` — change `64` to match

   For example, if you stack the current 64 embedding bands + 3 new bands (elevation, slope, aspect), set `in_channels=67`.

4. **Update `NODATA`** if the new bands use a different nodata value than `-32768.0`. The `normalize_tile()` function uses this value to exclude invalid pixels from normalization statistics across **all** bands — pixels where any band equals NODATA are zeroed out. If your new bands use a different nodata convention, you'll need to reconcile them (e.g., set all nodata to the same sentinel value when stacking).

5. **Delete the `tiles/` directory** to force re-tiling. The tile cache will be stale since the input changed.

6. **Delete `output/best_model.pth`** — the saved model weights expect the old channel count and can't be loaded with a different `in_channels`.

### Config variable reference

| Variable | What it does | When to change |
|----------|-------------|----------------|
| `EMBEDDING_PATH` | Path to the multi-band predictor raster | When changing input data |
| `LANDCOVER_PATH` | Path to the land cover classification raster (used to build training masks) | When changing mask source |
| `HABITAT_CLASSES` | Which land cover class values count as habitat | When the landcover raster has multiple classes |
| `NODATA` | Sentinel value for invalid pixels in the embedding raster | When switching to a raster with a different nodata value |
| `in_channels` (in model code) | Number of input bands the model expects | **Must match the band count of `EMBEDDING_PATH`** |
| `TILE_SIZE` | Training tile dimensions (must be divisible by 16) | Rarely — 256 is a good default |
| `MIN_VALID` | Minimum fraction of habitat pixels to keep a tile | Lower it if too many tiles are being skipped |

### Common pitfalls

- **`TILE_SIZE` must be divisible by 16.** The DeepLabV3+ encoder (ResNet34) downsamples by a factor of 16. Non-divisible sizes will crash. 256 is a safe default.
- **Changing input data or config requires clearing caches.** Delete `tiles/` and `output/best_model.pth` whenever you change `EMBEDDING_PATH`, `LANDCOVER_PATH`, `POINTS_PATH`, `TILE_SIZE`, `MIN_VALID`, or `in_channels`. The pipeline's stale detection catches some cases but not all (e.g., changing `MIN_VALID` doesn't change file mtimes).
- **Too few tiles after filtering.** If Step 4 reports many skipped tiles and Step 6 says "Need at least 2 tiles", try lowering `MIN_VALID` (e.g., from `0.1` to `0.01`). This threshold controls what fraction of a tile must be habitat to keep it.
- **Training on CPU is slow but works.** The pipeline auto-detects CUDA. If no GPU is available, training will run on CPU. For 25 epochs this could take hours instead of minutes. Consider reducing `EPOCHS` or `TILE_SIZE` for faster iteration during debugging.
- **The map step needs internet for the inset.** Step 10's Alberta inset map downloads province boundaries from Natural Earth. If you're offline, the inset is skipped but the main map still saves.

---

## Table of Contents

| # | Change | Severity |
|---|--------|----------|
| [1](#change-1-combined-all-scripts-into-a-single-pipeline-file) | Combined all scripts into a single pipeline file | Structural |
| [2](#change-2-fixed-hardcoded-windows-paths--relative-paths) | Fixed hardcoded Windows paths → relative paths | Critical |
| [3](#change-3-fixed-inference-tile-size-mismatch-512--256) | Fixed inference tile size mismatch (512 → 256) | Critical |
| [4](#change-4-enabled-imagenet-pretrained-encoder-weights) | Enabled ImageNet pretrained encoder weights | Medium |
| [5](#change-5-fixed-nodata-contamination-in-normalization) | Fixed NODATA contamination in normalization | Critical |
| [6](#change-6-added-study-area-clipping-step-9) | Added study area clipping (Step 9) | Feature |
| [7](#change-7-added-class-weighted-loss-function) | Added class-weighted loss function | Medium |
| [8](#change-8-increased-batch-size-from-4-to-8) | Increased batch size from 4 to 8 | Minor |
| [9](#change-9-added-validation-metrics-iou-f1-precision-recall) | Added validation metrics (IoU, F1, Precision, Recall) | Feature |
| [10](#change-10-reorganized-configuration-into-a-user-editable-section) | Reorganized configuration into a user-editable section | Structural |
| [11](#change-11-fixed-missing-land-cover-binarization) | Fixed missing land cover binarization | Critical |
| [12](#change-12-fixed-type-mismatch-in-presenceabsence-point-filtering) | Fixed type mismatch in presence/absence point filtering | Critical |
| [13](#change-13-fixed-augmentation-leaking-into-validation-data) | Fixed augmentation leaking into validation data | Critical |
| [14](#change-14-fixed-nodata-mask-to-use-union-across-all-bands) | Fixed NODATA mask to use union across all bands | Medium |
| [15](#change-15-extracted-shared-normalize_tile-function) | Extracted shared `normalize_tile()` function | Structural |
| [16](#change-16-global-random-seed-via-seed_everything) | Global random seed via `seed_everything()` | Medium |
| [17](#change-17-step-caching-with-stale-detection) | Step caching with stale detection | Feature |
| [18](#change-18-per-step-timing) | Per-step timing | Minor |
| [19](#change-19-parallel-data-loading-with-num_workers) | Parallel data loading with `num_workers` | Minor |
| [20](#change-20-early-stopping) | Early stopping | Feature |
| [21](#change-21-cli-step-selection) | CLI step selection | Feature |
| [22](#change-22-replaced-torchtensor-with-torchfrom_numpy) | Replaced `torch.tensor()` with `torch.from_numpy()` | Minor |
| [23](#change-23-added-step-10--poster-quality-suitability-map-generation-make_suitability_map) | Added Step 10 — poster-quality suitability map | Feature |
| [24](#change-24-added-ignore_index255-to-both-loss-functions) | Added `ignore_index=255` to both loss functions | Critical |
| [25](#change-25-force-binarize-multi-class-landcover-when-habitat_classes-is-not-set) | Force-binarize multi-class landcover when `HABITAT_CLASSES` not set | Critical |
| [26](#change-26-added-weights_onlytrue-to-torchload) | Added `weights_only=True` to `torch.load` | Minor |
| [27](#change-27-guard-against-division-by-zero-in-step-5-class-balance-check) | Guard against division by zero in Step 5 | Minor |
| [28](#change-28-clear-stale-tiles-before-regenerating-step-4) | Clear stale tiles before regenerating (Step 4) | Critical |
| [29](#change-29-guard-against-empty-tile-list-and-mismatched-tile-counts-in-training-step-6) | Guard against mismatched tile counts in training | Medium |
| [30](#change-30-minimum-tile-count-raised-from-0-to-2) | Minimum tile count raised from 0 to 2 | Medium |
| [31](#change-31-wrap-inset-locator-map-in-tryexcept-to-handle-offline-environments) | Wrap inset locator map in try/except for offline use | Medium |

---

## Change 1: Combined all scripts into a single pipeline file

**Files affected:** Created `deeplabv3_pipeline.py`, replaces all 12 original scripts.

**What changed:**
- Merged `create_masks.py`, `create_masks_with_points.py`, `create_tiles.py`, `step3_dataset.py`, `step4_model.py`, `step6_train.py`, `step7_inference.py`, `step8_suitability.py`, `calssBalanceTest.py`, `habitat_inspect.py`, `inspect_points.py`, and `inspect_shapefile.py` into one file.
- Each original script became a function: `inspect_data()`, `create_habitat_mask()`, `create_mask_with_points()`, `create_tiles()`, `check_class_balance()`, `train()`, `inference()`, `smooth()`.
- The `HabitatModel` and `EmbeddingHabitatDataset` classes are defined inline.
- A `__main__` block runs the full pipeline in order.

**Why:**
The original codebase was split across 12 files with no clear reason for modularization. Several scripts were one-off inspection utilities, and the training scripts imported from each other using relative imports that only worked on Devon's machine. A single file is easier to understand, run, and debug.

**Additional sub-changes:**
- All hardcoded Windows paths (`C:\Users\devon\...`) were replaced with relative paths computed from `PROJECT_ROOT`.
- All configuration constants were moved to the top of the file.
- Inference tile size was changed from 512 to 256 to match the training tile size (see Change 3 for details).

---

## Change 2: Fixed hardcoded Windows paths → relative paths

**What changed:**
Every file path in the original scripts was an absolute Windows path pointing to Devon's machine, e.g.:
```python
r"C:\Users\devon\Documents\School-Work\geog-4553-research-project\models\DeeplabV3+\embeddings\emb11Nclp.tif"
```
These were all replaced with paths computed relative to the project root:
```python
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EMBEDDING_PATH = os.path.join(MODEL_DIR, "embeddings", "emb11Nclp.tif")
```

**Why:**
The hardcoded paths made the code completely non-portable. It could only run on Devon's Windows machine at that exact directory. Using relative paths means it works for any team member on any OS, as long as the project directory structure is preserved.

---

## Change 3: Fixed inference tile size mismatch (512 → 256)

**What changed:**
The original `step7_inference.py` used `TILE = 512` for the sliding window, while training in `step6_train.py` used `TILE_SIZE = 256` tiles. Changed `INFER_TILE` to 256 to match.

**Why:**
When a model trains on 256×256 patches but sees 512×512 patches at inference time, the internal feature statistics (batch norm running means/variances) and learned spatial patterns don't transfer correctly. This is one of the primary reasons the prediction map "made no sense" — the model was being fed inputs at a scale it had never seen during training.

---

## Change 4: Enabled ImageNet pretrained encoder weights

**What changed:**
In the `HabitatModel` class, changed `encoder_weights = None` to `encoder_weights = "imagenet"`.

**Why:**
Training a ResNet34 backbone from scratch requires a very large dataset (hundreds of thousands of samples). With the small number of training tiles available from a single study area, the model almost certainly underfits — the encoder never learns useful low-level features like edges and textures. ImageNet pretraining provides a strong initialization for these general visual features, so the model only needs to learn the habitat-specific higher-level patterns. This is standard practice for any segmentation task with limited training data.

---

## Change 5: Fixed NODATA contamination in normalization

**What changed:**
Replaced the naive normalization in both the dataset (`__getitem__`) and inference that did:
```python
emb[emb == NODATA] = 0.0  # replace NODATA with 0 BEFORE computing stats
mu = emb.mean(...)         # mean is now biased by the 0-filled NODATA pixels
```
With NODATA-aware normalization that:
1. Creates a valid pixel mask from the first band
2. Computes mean/std only from valid pixels per band
3. Normalizes all pixels using valid-only statistics
4. Sets NODATA pixels to 0.0 only after normalization

**Why:**
The original code replaced NODATA values (-32768) with 0.0 *before* computing the mean and standard deviation. If a tile had significant NODATA regions (e.g. edge tiles, areas outside the study area), the 0-filled pixels would drag the mean down and inflate the variance, corrupting the normalization for all valid pixels in that tile. This caused the model to see different statistical distributions depending on how much NODATA was in each tile, making training unstable and inference inconsistent.

The fix also ensures that the same normalization logic is used in both training and inference, eliminating another source of train/inference distribution mismatch.

---

## Change 6: Added study area clipping (Step 9)

**What changed:**
Added a new pipeline step `clip_to_study_area()` that:
1. Loads the MD Ranchland No. 66 boundary shapefile (`MD_bound_zipped_11N.shp`, already present in the MaxEnt model's data directory)
2. Reprojects the boundary to match the suitability raster's CRS
3. Uses `rasterio.mask` to clip and crop the smoothed suitability map to the study area boundary
4. Saves the result as `suitability_map_clipped.tif`

**Why:**
The original pipeline had **no clipping step at any point**. The embedding raster may extend beyond the MD Ranchland No. 66 boundary, and the model was predicting habitat suitability for regions outside the study area — which is both scientifically invalid and visually confusing. This was one of the two primary issues reported ("the TIFF file wasn't clipped to the study bounds"). The clipped output ensures the final map only covers the jurisdiction of interest.

---

## Change 7: Added class-weighted loss function

**What changed:**
Replaced the single `DiceLoss(mode='multiclass')` with a combined loss:
1. **Weighted CrossEntropyLoss** — class weights are computed dynamically from the tile data using inverse frequency weighting: `weight = total / (2 * class_count)`
2. **DiceLoss** — kept as a complement for spatial overlap optimization
3. Final loss = CE + Dice

**Why:**
In habitat suitability modeling, the habitat class is typically much rarer than the background class. With unweighted loss, the model can achieve low loss simply by predicting "no habitat" everywhere (since that's correct for 90%+ of pixels). The weighted cross-entropy penalizes misclassification of the rare class more heavily, forcing the model to actually learn the habitat distribution. DiceLoss is kept because it directly optimizes spatial overlap (IoU-like), complementing the per-pixel CE loss.

---

## Change 8: Increased batch size from 4 to 8

**What changed:**
Changed `BATCH_SIZE` from 4 to 8.

**Why:**
A batch size of 4 produces very noisy gradient estimates, especially with augmentation and class imbalance. Each batch may have a very different class distribution, causing the loss to fluctuate wildly. Increasing to 8 provides more stable gradients and better batch normalization statistics (since batch norm layers in the ResNet34 encoder compute running stats from each batch). This is still conservative enough to fit in GPU memory for 256×256 tiles with 64 channels.

---

## Change 9: Added validation metrics (IoU, F1, Precision, Recall)

**What changed:**
The training loop now computes and prints per-epoch validation metrics for the habitat class:
- **IoU** (Intersection over Union) — the standard segmentation metric
- **F1** — harmonic mean of precision and recall
- These are computed from true positives, false positives, and false negatives accumulated across validation batches

**Why:**
The original training loop only printed loss values, which are poor indicators of actual model performance for imbalanced segmentation tasks. A model predicting all-background can have low Dice loss but 0.0 IoU for the habitat class. By tracking IoU and F1, you can immediately see whether the model is actually learning to detect habitat or just predicting the majority class. This is essential for diagnosing training issues and knowing when to stop.

---

## Change 10: Reorganized configuration into a user-editable section

**What changed:**
Replaced the previous configuration block (which computed paths from `PROJECT_ROOT` and pointed to files in other team members' model directories) with a clear `USER CONFIG` section at the top of the file. The new layout:

1. **USER CONFIG** — paths the user *must* set before running:
   - `EMBEDDING_PATH` — satellite embedding raster (defaults to local `embeddings/` subfolder)
   - `LANDCOVER_PATH` — land cover raster (defaults to local `masks/` subfolder)
   - `POINTS_PATH` — presence/absence shapefile (blank by default, must be configured)
   - `STUDY_AREA_PATH` — study area boundary shapefile (blank by default, clipping skipped if not set)
   - Point labeling config (`PRESENCE_COLUMN`, `PRESENCE_VALUE`, `ABSENCE_VALUE`, `POINT_BUFFER_M`)
   - Output directories, tiling config, training config, inference config

2. **DERIVED PATHS** — automatically computed from the user config (intermediate files, model checkpoint, output maps)

Additionally:
- All paths are now relative to `SCRIPT_DIR` (the directory containing this file) instead of `PROJECT_ROOT`, so the DeepLabV3+ folder is self-contained.
- Tile output directories moved from `data/embedding_tiles` (shared project data folder) to `tiles/` within the DeepLabV3+ directory.
- `POINTS_PATH` and `STUDY_AREA_PATH` default to empty strings. The pipeline raises a clear error if `POINTS_PATH` is not set when needed, and gracefully skips clipping if `STUDY_AREA_PATH` is not set.
- Removed the hardcoded reference to the MaxEnt model's boundary shapefile (`models/MaxEnt/data/inputs/extras/MD_bound_zipped_11N.shp`), since that file is gitignored and won't exist for other team members.

**Why:**
The previous config assumed access to data files scattered across the project (including another teammate's gitignored MaxEnt data directory). This made the script non-portable between team members. The new layout puts all configurable values in one clearly marked section at the top, with inline comments explaining what each path should point to. A teammate can clone the repo, fill in their local data paths, and run the pipeline without needing to understand the rest of the code first.

---

## Change 11: Fixed missing land cover binarization

**What changed:**
Added explicit binarization of the land cover raster after reprojection in `create_habitat_mask()`. Added a new config option `HABITAT_CLASSES` (list of class IDs that represent habitat, e.g. `[1, 2, 5]`).

- If `HABITAT_CLASSES` is set: pixels matching any listed class become 1, all others become 0
- If `HABITAT_CLASSES` is `None`: the raster is assumed to already be binary (0/1)
- If `None` but the raster contains values other than 0/1/255: a warning is printed

**Why:**
The original code reprojected ExLandcover.tif and immediately used the raw values, only checking for `== 1` (habitat) and `== 0` (no habitat). If the land cover raster contains standard multi-class values (e.g. 0-15 for different land cover types), classes 2+ would be silently treated as no-habitat — even if some of those classes (e.g. shrubland, grassland) represent suitable Bromus tectorum habitat. This is a semantic error that corrupts the training labels.

---

## Change 12: Fixed type mismatch in presence/absence point filtering

**What changed:**
The point filtering now casts `PRESENCE_VALUE` and `ABSENCE_VALUE` to match the actual dtype of the shapefile column before comparison. Also added a warning if any points don't match either value.

Before:
```python
presence = points[points["PRESENT"] == "YES"]   # string comparison
absence  = points[points["PRESENT"] == 0]       # int comparison against string column → matches nothing
```

After:
```python
# Cast to match actual column dtype (e.g. if column is string, 0 → "0")
pres_val = type(points[col].iloc[0])(PRESENCE_VALUE)
abs_val  = type(points[col].iloc[0])(ABSENCE_VALUE)
```

**Why:**
`PRESENCE_VALUE = "YES"` (string) but `ABSENCE_VALUE = 0` (integer). If the column contains strings, `== 0` matches nothing in pandas — producing zero absence points silently. The model then trains on a mask with no absence labels from points, relying entirely on the (potentially incorrect) land cover mask for negative examples. The cast ensures both values are compared in the column's native type, and the unmatched-points warning catches any remaining mismatches.

---

## Change 13: Fixed augmentation leaking into validation data

**What changed:**
Replaced the `random_split` approach (which split a single augmented dataset) with a manual index-based split that creates two separate `EmbeddingHabitatDataset` instances:
- Training dataset: `augment=True`
- Validation dataset: `augment=False`

Also added a fixed random seed (`seed=42`) via `torch.Generator().manual_seed(42)` for reproducible splits.

**Why:**
Two problems:
1. **Augmentation on validation**: The original code created one dataset with `augment=True`, then used `random_split` to split it. Since augmentation is applied in `__getitem__`, validation samples were randomly flipped/rotated every time they were loaded. This makes validation metrics non-deterministic between epochs and inflates them (the model gets evaluated on easier/different views of the same tile each time).
2. **Non-reproducible split**: `random_split` without a seed produces a different train/val partition every run, making experiments non-comparable and debugging impossible.

---

## Change 14: Fixed NODATA mask to use union across all bands

**What changed:**
In both the dataset `__getitem__` and the inference normalization, changed the valid pixel mask from checking only band 0 to checking all bands:

Before:
```python
valid_mask = emb[0] != NODATA  # only band 0
```

After:
```python
valid_mask = np.all(emb != NODATA, axis=0)  # union: valid only if ALL bands are valid
```

**Why:**
If different bands have NODATA at different pixel locations (which can happen with multi-source embeddings or edge effects during resampling), using only band 0 would include pixels where other bands contain NODATA values. Those NODATA values (-32768) would be included in the normalization statistics and passed to the model as real data, corrupting both the per-band mean/std and the model's predictions. The union mask ensures a pixel is only considered valid if every band has real data.

---

## Change 15: Extracted shared `normalize_tile()` function

**What changed:**
The NODATA-aware normalization logic was duplicated in two places: `EmbeddingHabitatDataset.__getitem__()` and the `inference()` function. Extracted it into a single shared function `normalize_tile(tile, nodata)` used by both. Additionally, the inference version was still using the old band-0-only NODATA mask (`tile[0] != NODATA`) — the shared function uses the correct all-bands union mask (`np.all(tile != nodata, axis=0)`).

**Why:**
1. **Bug fix**: The inference NODATA mask was inconsistent with the training dataset — inference used band 0 only while training used the union mask (from Change 14). This meant train/inference saw differently normalized data for tiles with per-band NODATA variation.
2. **Maintainability**: Having normalization logic in one place means future changes (e.g., switching to global statistics) only need to happen once, eliminating the risk of train/inference drift.

---

## Change 16: Global random seed via `seed_everything()`

**What changed:**
Added a `seed_everything()` function called at the start of the pipeline that seeds all random number generators:
```python
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
Added `RANDOM_SEED = 42` as a configurable constant. The train/val split generator also uses this constant instead of a hardcoded `42`.

**Why:**
Previously only the train/val split was seeded. Model weight initialization (`nn.Conv2d`, `nn.BatchNorm2d`), DataLoader shuffle order, and augmentation random flips/rotations were all non-deterministic. This meant consecutive runs with identical data and config could produce different models, making debugging and comparison impossible. Setting `cudnn.deterministic = True` and `benchmark = False` sacrifices a small amount of GPU performance for exact reproducibility. This matches the pattern used in the MaxEnt model where `RANDOM_SEED` is propagated to every stochastic operation.

---

## Change 17: Step caching with stale detection

**What changed:**
Added `is_cache_fresh(output_path, source_paths)` — a utility that checks whether a cached output file exists and is newer than all its source inputs (by comparing file modification timestamps). Steps 2, 3, and 4 now check their caches before running:

- **Step 2** (`create_habitat_mask`): Skips if `habitat_mask.tif` is newer than embedding + landcover rasters
- **Step 3** (`create_mask_with_points`): Skips if `habitat_mask_with_points.tif` is newer than base mask + points shapefile
- **Step 4** (`create_tiles`): Skips if tiles exist and are newer than the combined mask + embedding

If any source input has been modified since the cache was created, the cache is automatically invalidated and the step re-runs. This matches the MaxEnt model's cache pattern where `cache_mtime < src_mtime` triggers re-extraction.

**Why:**
Steps 2-4 are pure data transformations — given the same inputs they always produce the same outputs. On re-runs (e.g., after tweaking training hyperparameters), these steps were wasting time recreating identical tiles. With caching, a typical re-run skips straight to training, cutting the 30-minute runtime significantly when only hyperparameters changed. The stale detection ensures that if the user updates their points shapefile or landcover raster, the pipeline correctly re-processes instead of serving outdated cached data.

---

## Change 18: Per-step timing

**What changed:**
Added a `step_timer` context manager that prints elapsed wall-clock time after each major step completes:
```
  [Step 2 completed in 4.3s]
```
Also added total pipeline elapsed time to the final summary:
```
PIPELINE COMPLETE (12m 34s total)
```
Per-epoch timing was also added to the training loop output.

**Why:**
With 30-minute runtimes, knowing where time is spent is critical for optimization. Is it tiling? Training? Inference? The timers answer this immediately. The MaxEnt model uses similar print-based progress reporting at each stage.

---

## Change 19: Parallel data loading with `num_workers`

**What changed:**
Changed DataLoader `num_workers` from `0` (single-threaded) to `min(4, os.cpu_count())` on non-Windows platforms. Added `pin_memory=True` for faster CPU-to-GPU transfer. On Windows, `num_workers` stays at `0` because multiprocessing with rasterio and forked workers can deadlock on that platform.

**Why:**
With `num_workers=0`, the training loop blocks on disk I/O for every batch — it reads tiles from disk, normalizes them, then feeds them to the GPU. While the GPU is computing the forward/backward pass, the CPU sits idle instead of prefetching the next batch. With multiple workers, the next batch is loaded in parallel with GPU computation, which can significantly reduce per-epoch time. `pin_memory=True` pre-allocates page-locked memory for faster `.to(device)` transfers.

---

## Change 20: Early stopping

**What changed:**
Added early stopping with a patience of 7 epochs. If validation loss does not improve for 7 consecutive epochs, training terminates early instead of running all 25 epochs.

**Why:**
Without early stopping, the model always trains for the full 25 epochs even if it converged at epoch 10 and is now overfitting. This wastes time and can actually degrade the saved model if a brief val loss dip at epoch 24 happens to beat the epoch 10 best (overfitting to validation noise). Early stopping saves time and reduces overfitting risk.

---

## Change 21: CLI step selection

**What changed:**
Added command-line argument support so individual steps can be run selectively:
```bash
python deeplabv3_pipeline.py                    # full pipeline
python deeplabv3_pipeline.py train infer        # only training + inference
python deeplabv3_pipeline.py infer smooth clip  # only post-training steps
```

Steps are identified by short names: `inspect`, `mask`, `points`, `tile`, `balance`, `train`, `infer`, `smooth`, `clip`.

**Why:**
The most common re-run scenario is tweaking training hyperparameters and retraining — but the old pipeline forced a full re-run of all 9 steps. Combined with caching (Change 17), this gives two levels of skip control: caching auto-skips data prep when inputs haven't changed, and CLI args let the user explicitly run only the steps they care about.

---

## Change 22: Replaced `torch.tensor()` with `torch.from_numpy()`

**What changed:**
In both the dataset `__getitem__` and inference, replaced:
```python
torch.tensor(emb)              # copies data unconditionally
torch.tensor(tile[None])
```
With:
```python
torch.from_numpy(emb.copy())   # shares memory when possible
torch.from_numpy(tile[None].copy())
```

**Why:**
`torch.tensor()` always copies data and can produce warnings about non-writable tensors from numpy arrays. `torch.from_numpy()` is the idiomatic way to convert numpy arrays to PyTorch tensors. The `.copy()` is needed here because rasterio returns read-only arrays, but `from_numpy` is still preferred as it makes the intent explicit and avoids the deprecation path.

## Change 23: Added Step 10 — poster-quality suitability map generation (`make_suitability_map`)

**What changed:**
Added a new pipeline step `make_suitability_map()` that renders the suitability prediction as a publication-ready map with standard cartographic elements. The visual design is based directly on the MaxEnt model's `make_maxent_map()` function but adapted for the DeepLabV3+ pipeline.

**Map elements (matching MaxEnt layout):**
- Dark background (`#1a1a1a`) with blue→yellow→orange continuous colormap
- Colorbar with percentage labels (0%–100%)
- Easting/Northing axis labels with comma-formatted metre values
- 20 km scalebar with tick labels at 0, 5, 10, 20 km
- Two-tone north arrow chevron with "N" label
- Title: "Habitat Suitability — Bromus tectorum / Municipal District of Ranchland No. 66, Alberta"
- Subtitle: "Method: DeepLabV3+ · CRS: NAD83 / UTM Zone 11N (EPSG:26911) · Resolution: 10m"
- Inset locator map showing Alberta with study area highlighted (requires `STUDY_AREA_PATH`)
- Optional presence point overlay (uses `POINTS_PATH` if configured)

**Adaptations from MaxEnt version:**
- Automatically selects best available input: clipped > smoothed > raw suitability map
- Subtitle identifies method as "DeepLabV3+" instead of "MaxEnt"
- Point overlay filters to presence-only using the pipeline's `PRESENCE_COLUMN`/`PRESENCE_VALUE` config
- Inset map uses `STUDY_AREA_PATH` (pipeline config) instead of hardcoded MaxEnt data path
- Outputs both PNG (300 dpi) and SVG to `output/suitability_map.{png,svg}`

**New derived paths added:**
```python
SUITABILITY_MAP_PNG = os.path.join(OUTPUT_DIR, "suitability_map.png")
SUITABILITY_MAP_SVG = os.path.join(OUTPUT_DIR, "suitability_map.svg")
```

**New imports added:**
```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.patches import Polygon
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
```

**Registered in STEPS dict as `"map"`**, so it can be run standalone:
```bash
python deeplabv3_pipeline.py map
```

**Why:**
The MaxEnt model already had a polished map output that could go directly into a report or presentation. The DeepLabV3+ pipeline was missing this — it only produced raw GeoTIFFs that require a GIS tool to visualize. This step gives Devon the same poster-quality output with consistent visual branding across both models, making side-by-side comparison straightforward.

## Change 24: Added `ignore_index=255` to both loss functions

**What changed:**
```python
# Before
ce_loss_fn   = nn.CrossEntropyLoss(weight=class_weights)
dice_loss_fn = smp.losses.DiceLoss(mode='multiclass')

# After
ce_loss_fn   = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
dice_loss_fn = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)
```

**Why:**
The habitat mask uses nodata=255. If any mask tile contains pixels with value 255 (nodata), those flow into the loss function as target class 255 — but the model only has 2 output classes (0 and 1). `CrossEntropyLoss` would crash with an index-out-of-bounds error, or worse, silently produce garbage gradients depending on the PyTorch version. Setting `ignore_index=255` is the standard approach in semantic segmentation: it tells both loss functions to skip those pixels entirely during backpropagation, so nodata regions have zero influence on the model's learned weights.

## Change 25: Force-binarize multi-class landcover when `HABITAT_CLASSES` is not set

**What changed:**
```python
# Before — warns but leaves multi-class values in the mask
if not set(unique_vals).issubset({0, 1, 255}):
    print(f"WARNING: Landcover has values {unique_vals} but HABITAT_CLASSES is not set.")
    print("         Non-0/1 values will be treated as no-habitat. Set HABITAT_CLASSES to fix this.")

# After — warns and force-binarizes to prevent downstream crash
if not set(unique_vals).issubset({0, 1, 255}):
    print(f"WARNING: Landcover has values {unique_vals} but HABITAT_CLASSES is not set.")
    print("         Treating value 1 as habitat, all others as no-habitat.")
    print("         Set HABITAT_CLASSES for explicit control.")
    habitat_mask = (habitat_mask == 1).astype(np.uint8)
```

**Why:**
Without this fix, if the landcover raster has multi-class values (e.g., 0, 1, 2, 3, ...) and `HABITAT_CLASSES` is not configured, values like 2, 3, etc. would persist in the mask and flow through to training. During loss computation, `CrossEntropyLoss` with `NUM_CLASSES=2` receiving target values >1 (and not equal to `ignore_index`) would crash. The warning alone didn't prevent the issue — now the code both warns and safely binarizes by treating only class 1 as habitat.

## Change 26: Added `weights_only=True` to `torch.load`

**What changed:**
```python
# Before
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

# After
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
```

**Why:**
PyTorch 2.6+ emits a `FutureWarning` when calling `torch.load` without the `weights_only` parameter, and plans to change the default to `True` in a future release. Since we're loading a state dict (just tensor weights, no arbitrary Python objects), `weights_only=True` is the correct and safe option. This suppresses the warning and future-proofs against the upcoming default change.

## Change 27: Guard against division by zero in Step 5 (class balance check)

**What changed:**
```python
# Before
total = total_present + total_absent
print(f"Habitat present (1): {total_present:>10,} pixels ({total_present/total*100:.1f}%)")

# After
total = total_present + total_absent
if total == 0:
    print("No mask tiles found. Run the 'tile' step first.")
    return
print(f"Habitat present (1): {total_present:>10,} pixels ({total_present/total*100:.1f}%)")
```

**Why:**
If the `tile` step hasn't been run yet (or produced zero tiles), `mask_paths` would be empty, `total` would be 0, and the percentage calculation would crash with `ZeroDivisionError`. This adds a guard with a helpful message pointing to the fix.

## Change 28: Clear stale tiles before regenerating (Step 4)

**What changed:**
```python
# Before — old tiles from previous runs persist on disk
os.makedirs(TILE_EMB_DIR, exist_ok=True)
os.makedirs(TILE_MASK_DIR, exist_ok=True)

# After — wipe and recreate tile directories when cache is stale
import shutil
for d in [TILE_EMB_DIR, TILE_MASK_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)
os.makedirs(TILE_EMB_DIR, exist_ok=True)
os.makedirs(TILE_MASK_DIR, exist_ok=True)
```

**Why:**
When the tile cache is stale (e.g., the mask or embedding was updated), Step 4 regenerates tiles. But the old tile files were never deleted — they stayed on disk alongside the new ones. If the new config produces fewer qualifying tiles (different `MIN_VALID`, different mask), old tiles from the previous run persist. Step 6 uses `sorted(glob.glob("*.tif"))` to discover tiles, so it picks up ALL tiles — stale and fresh mixed together. This means the model trains on mask data from a completely different configuration run, silently corrupting training. Wiping the directories ensures only the current run's tiles exist.

## Change 29: Guard against empty tile list and mismatched tile counts in training (Step 6)

**What changed:**
```python
# Added after discovering tile paths
if len(emb_paths) == 0:
    print("ERROR: No tiles found. Run the 'tile' step first.")
    return
if len(emb_paths) != len(mask_paths):
    raise RuntimeError(
        f"Tile count mismatch: {len(emb_paths)} embedding tiles vs "
        f"{len(mask_paths)} mask tiles. Clear the tiles/ directory and re-run the 'tile' step."
    )
```

**Why:**
Two failure modes:
1. If no tiles exist (Step 4 wasn't run or produced zero qualifying tiles), the DataLoader would have 0 batches. After the empty epoch loop, `train_loss / len(train_loader)` would crash with `ZeroDivisionError`.
2. The training step pairs embedding and mask tiles by sorted index position (`emb_paths[i]` ↔ `mask_paths[i]`). This pairing is correct ONLY if both directories have the same number of files with matching (y, x) coordinates. If the directories are out of sync (manual edits, partial runs), sorted-index pairing silently feeds wrong masks to wrong embeddings. The assertion catches this immediately with a clear error message.

## Change 30: Minimum tile count raised from 0 to 2

**What changed:**
```python
# Before
if len(emb_paths) == 0:
    print("ERROR: No tiles found. Run the 'tile' step first.")
    return

# After
if len(emb_paths) < 2:
    print(f"ERROR: Need at least 2 tiles for train/val split, found {len(emb_paths)}.")
    print("  Check that the 'tile' step ran and that MIN_VALID isn't filtering out all tiles.")
    return
```

**Why:**
The train/val split uses `val_size = max(1, int(n_total * 0.2))`, which always reserves at least 1 tile for validation. With exactly 1 tile: `val_size=1`, `val_idx=[0]`, `train_idx=[]`. The training DataLoader would have 0 batches, and the epoch loop would produce `train_loss = 0`. Then `avg_train = train_loss / len(train_loader)` divides by 0 → `ZeroDivisionError`. Requiring at least 2 tiles ensures there is always at least 1 tile for training and 1 for validation.

---

## Change 31: Wrap inset locator map in try/except to handle offline environments

**Files affected:** `deeplabv3_pipeline.py` — `make_suitability_map()` (Step 10)

**What changed:**
The inset locator map code (which downloads Alberta province boundaries from Natural Earth via `gpd.read_file(ne_url)`) is now wrapped in a `try/except` block. If the download fails (no internet, DNS error, timeout), the main suitability map is still saved — only the small Alberta inset is skipped.

```python
# Before — network failure crashes the entire map step
if STUDY_AREA_PATH and os.path.exists(STUDY_AREA_PATH):
    inset_ax = ax.inset_axes(...)
    provinces = gpd.read_file(ne_url)  # <- crashes here if offline
    ...

# After — graceful fallback, main map always saved
if STUDY_AREA_PATH and os.path.exists(STUDY_AREA_PATH):
    try:
        inset_ax = ax.inset_axes(...)
        provinces = gpd.read_file(ne_url)
        ...
    except Exception as e:
        print(f"  WARNING: Could not create inset map: {e}")
        print("  (Requires internet to download Natural Earth province boundaries)")
        if 'inset_ax' in dir():
            inset_ax.remove()
```

**Why:**
The inset map downloads province boundary data from `naciscdn.org` at runtime. Devon is likely running this pipeline on a local machine for ML training, possibly without reliable internet. Without this guard, any network issue (offline, firewall, DNS failure, server down) would crash `make_suitability_map()` entirely — discarding the already-computed main suitability visualization. The inset is a cosmetic addition; losing it should never prevent the primary deliverable from being saved.
