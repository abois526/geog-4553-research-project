"""
predict.py  —  Downy Brome SDM
Runs inference across the full raster extent and produces a
poster-quality probability heatmap.

Usage
-----
# Run inference and generate heatmap:
python predict.py predict \
    --tif       data/environment_stack.tif \
    --checkpoint best_model.pt             \
    --scaler    scaler.joblib              \
    --out_tif   probability_map.tif        \
    --out_png   probability_map.png        \
    --title     "Downy Brome — Ranchland County, AB"

# Generate only the suitability map (no model inference):
python predict.py createmap
"""

import argparse
import os
from pathlib import Path

import geopandas as gpd
import joblib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.patches import Patch, Polygon
from matplotlib_scalebar.scalebar import ScaleBar
from model import get_model
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.transform import from_bounds
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants — paths for createmap mode
# ---------------------------------------------------------------------------

PROBABILITY_TIF = os.path.join(os.path.dirname(__file__), "probability_map.tif")
POINTS_SHP      = os.path.join(os.path.dirname(__file__), "points_combined.shp")
RASTER_STACK    = r"/Users/andrew/Workspace/github-repos/geog-4553-research-project/models/MaxEnt/data/inputs/emb11Nclp.tif"
OUTPUT_MAP_PNG  = os.path.join(os.path.dirname(__file__), "cnn_suitability_map.png")
OUTPUT_MAP_SVG  = os.path.join(os.path.dirname(__file__), "cnn_suitability_map.svg")

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
# Poster heatmap (legacy — kept for backwards compatibility)
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
# Poster-quality suitability map (matches MaxEnt style)
# ---------------------------------------------------------------------------

def make_cnn_map(tif_path: str,
                 out_png: str,
                 out_svg: str = None,
                 title: str = "Habitat Suitability — Bromus tectorum\nMunicipal District of Ranchland No. 66, Alberta",
                 shp_path: str = None,
                 raster_path: str = None):
    """
    Renders a poster-quality CNN suitability heatmap from the prediction
    GeoTIFF with standard cartographic elements (title, scalebar, north arrow,
    easting/northing axes, colorbar).

    Optionally overlays presence/absence points from the shapefile.

    Args:
        tif_path:    Path to the suitability GeoTIFF (single-band, values 0–1).
        out_png:     Output path for the PNG map.
        out_svg:     Output path for the SVG map (vector, optional).
        title:       Map title (supports newlines).
        shp_path:    Path to the points_combined shapefile. If provided,
                     presence/absence points are overlaid on the map.
        raster_path: Path to the 64-band raster stack (unused, kept for
                     interface consistency with MaxEnt).
    """
    import matplotlib as mpl
    import matplotlib.ticker as mticker
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
        "cnn_suit",
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

    # Overlay presence/absence points
    if shp_path:
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=3857)
        gdf = gdf.to_crs(epsg=26911)

        presence = gdf[gdf["label"] == 1]
        absence  = gdf[gdf["label"] == 0]
        ax.scatter(presence.geometry.x, presence.geometry.y,
                   c="#ffe600", s=14, marker="^", linewidths=0.5,
                   edgecolors="black", alpha=0.9,
                   label=f"Presence ({len(presence)} pts)", zorder=5)
        ax.scatter(absence.geometry.x, absence.geometry.y,
                   c="#303234", s=8, marker="o", linewidths=0.3,
                   edgecolors="#000000", 
                   label=f"Absence ({len(absence)} pts)", zorder=5)
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
    ax.set_title(title, color="white", fontsize=16, fontweight="bold",
                 pad=18, loc="left")

    # Subtitle
    ax.text(0.02, 1.005, "Method: 1D CNN  ·  CRS: NAD83 / UTM Zone 11N (EPSG:26911)  ·  Resolution: 10m",
            transform=ax.transAxes, color="white", fontsize=9, style="italic",
            ha="left", va="bottom")

    # ── Inset locator map (Alberta with study area) ──────────────────────
    from matplotlib.patches import Rectangle

    inset_ax = ax.inset_axes([0.66, 0.05, 0.14, 0.24])
    inset_ax.set_facecolor("#1a1a1a")
    for sp in inset_ax.spines.values():
        sp.set_visible(False)

    # Load Alberta province boundary from Natural Earth
    ne_url = ("https://naciscdn.org/naturalearth/10m/cultural/"
              "ne_10m_admin_1_states_provinces.zip")
    provinces = gpd.read_file(ne_url)
    alberta = provinces[provinces["name"] == "Alberta"]
    alberta.plot(ax=inset_ax, facecolor="#3a3a3a", edgecolor="white",
                 linewidth=0.5)

    # Study area polygon from the MD boundary shapefile
    md_bound = gpd.read_file(
        os.path.join(os.path.dirname(__file__),
                     "..", "MaxEnt", "data", "inputs", "extras", "MD_bound_zipped_11N.shp"))
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


# ---------------------------------------------------------------------------
# CLI — mode dispatchers
# ---------------------------------------------------------------------------

def run_predict(args):
    """Run full inference pipeline."""
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


def run_create_map():
    """Load the suitability raster and render a poster-quality map with
    presence/absence points overlaid."""
    for name, path in [("PROBABILITY_TIF", PROBABILITY_TIF),
                       ("POINTS_SHP", POINTS_SHP)]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} not found: {path}\n"
                f"Run inference first (python predict.py predict ...) or check paths."
            )

    print("\n── Creating CNN suitability map ──")
    make_cnn_map(
        tif_path=PROBABILITY_TIF,
        out_png=OUTPUT_MAP_PNG,
        out_svg=OUTPUT_MAP_SVG,
        shp_path=POINTS_SHP,
        raster_path=RASTER_STACK,
    )
    print("  Done.\n")


def main():
    """
    Entry point. Parses the mode argument and dispatches to the appropriate workflow.

    Modes:
        predict   — load model, run inference on full raster, generate heatmap
        createmap — render a poster-quality suitability map from the prediction raster
    """
    parser = argparse.ArgumentParser(
        description="1D CNN habitat suitability model — Bromus tectorum, MD Ranchland No. 66"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- predict subcommand ---
    pred_parser = subparsers.add_parser("predict", help="Run inference and generate heatmap")
    pred_parser.add_argument("--tif",            required=True,  help="64-band raster stack")
    pred_parser.add_argument("--checkpoint",     default="best_model.pt")
    pred_parser.add_argument("--scaler",         default="scaler.joblib")
    pred_parser.add_argument("--out_tif",        default="probability_map.tif",
                             help="Output GeoTIFF (set to '' to skip)")
    pred_parser.add_argument("--out_png",        default="probability_map.png")
    pred_parser.add_argument("--title",
                             default="Downy Brome Presence Probability\nRanchland County, Alberta")
    pred_parser.add_argument("--shp",            default=None,
                             help="Shapefile to overlay points (optional)")
    pred_parser.add_argument("--overlay_points", action="store_true",
                             help="Overlay presence/absence points on heatmap")
    pred_parser.add_argument("--batch_size",     type=int, default=4096)

    # --- createmap subcommand ---
    subparsers.add_parser("createmap", help="Render poster-quality suitability map from existing prediction raster")

    args = parser.parse_args()

    if args.mode == "predict":
        run_predict(args)
    elif args.mode == "createmap":
        run_create_map()


if __name__ == "__main__":
    main()
