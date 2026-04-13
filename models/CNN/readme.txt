# Downy Brome SDM
1D CNN presence probability model for *Bromus tectorum*, MD Ranchland No. 66, AB.
 
## Setup
```bash
pip install torch geopandas rasterio scikit-learn matplotlib matplotlib-scalebar joblib tqdm
```
 
## Workflow
 
```bash
# 1. Prepare points (dedup, reproject, combine)
python combine_presence_absence.py
 
# 2. Train
python train.py --shp data/points_combined_culled.shp --tif data/emb11Nclp.tif
 
# 3. Cross-validate
python evaluate.py --shp data/points_combined_culled.shp --tif data/emb11Nclp.tif
 
# 4. Predict
python predict.py predict --tif data/emb11Nclp.tif --checkpoint best_model.pt --scaler scaler.joblib
```
 
## Outputs
| File | Description |
|---|---|
| `best_model.pt` | Best model checkpoint |
| `scaler.joblib` | Fitted StandardScaler |
| `cv_curves.png` | Per-fold AUC curves |
| `probability_map.tif` | Suitability raster |
| `probability_map.png` | Poster heatmap |