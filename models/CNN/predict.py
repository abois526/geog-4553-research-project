"""
predict.py  —  Downy Brome SDM
Runs inference across the full raster extent and produces a
poster-quality probability heatmap.

Usage
-----
python predict.py \
    --tif       data/environment_stack.tif \
    --checkpoint best_model.pt             \
    --scaler    scaler.joblib              \
    --out_tif   probability_map.tif        \
    --out_png   probability_map.png        \
    --title     "Downy Brome — Ranchland County, AB"
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.patches import Patch
# from matplotlib_scalebar.scalebar import ScaleBar
from model import get_model
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.transform import from_bounds
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Raster inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_raster(tif_path: str,
                   checkpoint_path: str,
                   scaler_path: str,
                   batch_size: int = 4096,
                   device: torch.device = None) -> tuple[np.ndarray, dict]:
    """
    Read every pixel from the raster, run SDM inference, return probability map.

    Returns
    -------
    prob_map : float32 (H, W)   values in [0, 1], nodata pixels = NaN
    meta     : rasterio profile for writing output GeoTIFF
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint & model
    ckpt    = torch.load(checkpoint_path, map_location=device)
    n_bands = ckpt["n_bands"]
    model   = get_model(n_bands=n_bands).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  Loaded model  (n_bands={n_bands}, best AUC={ckpt['best_auc']:.4f})")

    # Load scaler
    scaler = joblib.load(scaler_path)

    with rasterio.open(tif_path) as src:
        meta   = src.meta.copy()
        nodata = src.nodata if src.nodata is not None else -9999
        H, W   = src.height, src.width

        print(f"  Raster: {W} × {H} pixels  ({n_bands} bands)")
        print(f"  Total pixels: {H * W:,}")

        # Read all bands at once — shape (n_bands, H, W)
        data = src.read().astype(np.float32)

    # Reshape to (H*W, n_bands)
    flat = data.reshape(n_bands, -1).T          # (N, n_bands)

    # Mask nodata pixels
    nodata_mask = np.any(flat == nodata, axis=1) | np.any(~np.isfinite(flat), axis=1)

    # Normalise valid pixels
    valid_idx = np.where(~nodata_mask)[0]
    flat_valid = scaler.transform(flat[valid_idx]).astype(np.float32)

    # Batch inference
    probs = np.full(H * W, np.nan, dtype=np.float32)
    n_valid = len(valid_idx)

    print(f"  Running inference on {n_valid:,} valid pixels …")
    for start in tqdm(range(0, n_valid, batch_size)):
        end   = min(start + batch_size, n_valid)
        batch = torch.from_numpy(flat_valid[start:end]).to(device)
        out   = model(batch).cpu().numpy().squeeze()
        probs[valid_idx[start:end]] = out

    prob_map = probs.reshape(H, W)

    # Update metadata for single-band float output
    meta.update(dtype="float32", count=1, nodata=np.nan)
    return prob_map, meta


# ---------------------------------------------------------------------------
# GeoTIFF export
# ---------------------------------------------------------------------------

def save_geotiff(prob_map: np.ndarray, meta: dict, out_path: str):
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(prob_map[np.newaxis, :, :])
    print(f"  GeoTIFF saved → {out_path}")


# ---------------------------------------------------------------------------
# Poster heatmap
# ---------------------------------------------------------------------------

def make_poster_heatmap(prob_map: np.ndarray,
                        meta: dict,
                        out_path: str,
                        title: str = "Downy Brome Presence Probability\nRanchland County, Alberta",
                        shp_path: str = None):
    """
    Renders a poster-quality probability heatmap.
    Optionally overlays the presence/absence points from the shapefile.
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Georgia", "Times New Roman", "DejaVu Serif"],
        "axes.spines.top":  False,
        "axes.spines.right":False,
    })

    # Custom colormap: white → straw → orange → deep red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "brome",
        [(0.0, "#f7f4ef"),
         (0.3, "#e8d5a3"),
         (0.6, "#c97b2f"),
         (0.8, "#8b2500"),
         (1.0, "#550000")],
    )

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
        prob_map,
        cmap=cmap,
        vmin=0, vmax=1,
        extent=extent,
        interpolation="bilinear",
        aspect="equal",
    )

    # Overlay presence / absence points if shapefile provided
    if shp_path:
        import geopandas as gpd
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=3857)
        gdf = gdf.to_crs(epsg=26911)   # reproject to match raster
        presence = gdf[gdf["label"] == 1]
        absence  = gdf[gdf["label"] == 0]
        ax.scatter(presence.geometry.x, presence.geometry.y,
                   c="#E300E3", s=12, marker="^", linewidths=0.4,
                   edgecolors="white", alpha=0.85, label="Presence", zorder=5)
        ax.scatter(absence.geometry.x,  absence.geometry.y,
                   c="#002ae8", s=8,  marker="o", linewidths=0.3,
                   edgecolors="#aaaaaa", alpha=0.5, label="Absence",  zorder=5)
        ax.legend(loc="lower left", framealpha=0.3,
                  labelcolor="white", fontsize=9,
                  facecolor="#333333", edgecolor="#555555")

    # Lock axes to raster extent only
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.12)
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label("Presence Probability", color="white", fontsize=11, labelpad=10)
    cb.ax.yaxis.set_tick_params(color="white")
    cb.outline.set_edgecolor("white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=9)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    # Axis labels
    ax.set_xlabel("Easting", color="#cccccc", fontsize=10)
    ax.set_ylabel("Northing", color="#cccccc", fontsize=10)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")

    # Title
    ax.set_title(title, color="white", fontsize=16, fontweight="bold",
                 pad=18, loc="left")

    # Subtitle / model note
    fig.text(0.13, 0.91,
             "1D CNN  ·  64-band multispectral stack  ·  Binary presence / absence",
             color="#aaaaaa", fontsize=9, style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Heatmap saved → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  Downy Brome SDM  —  inference on {device}")
    print(f"{'='*50}\n")

    prob_map, meta = predict_raster(
        tif_path        = args.tif,
        checkpoint_path = args.checkpoint,
        scaler_path     = args.scaler,
        batch_size      = args.batch_size,
        device          = device,
    )

    if args.out_tif:
        save_geotiff(prob_map, meta, args.out_tif)

    make_poster_heatmap(
        prob_map  = prob_map,
        meta      = meta,
        out_path  = args.out_png,
        title     = args.title,
        shp_path  = args.shp if args.overlay_points else None,
    )

    print("\n  Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Downy Brome probability map")
    parser.add_argument("--tif",            required=True,  help="64-band raster stack")
    parser.add_argument("--checkpoint",     default="best_model.pt")
    parser.add_argument("--scaler",         default="scaler.joblib")
    parser.add_argument("--out_tif",        default="probability_map.tif",
                        help="Output GeoTIFF (set to '' to skip)")
    parser.add_argument("--out_png",        default="probability_map.png")
    parser.add_argument("--title",
                        default="Downy Brome Presence Probability\nRanchland County, Alberta")
    parser.add_argument("--shp",            default=None,
                        help="Shapefile to overlay points (optional)")
    parser.add_argument("--overlay_points", action="store_true",
                        help="Overlay presence/absence points on heatmap")
    parser.add_argument("--batch_size",     type=int, default=4096)
    args = parser.parse_args()
    main(args)
