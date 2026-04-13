import geopandas as gpd
import numpy as np
import rasterio


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

presence = gpd.read_file("data/db_points_11N_Clip.shp")
absence  = gpd.read_file("data/LC_ABS_PNTS.shp")

presence["label"] = 1
absence["label"]  = 0

# Presence is already 26911, reproject absence from 3857 to match
if presence.crs is None:
    presence = presence.set_crs(epsg=26911)
    
absence = absence.set_crs(epsg=3857).to_crs(epsg=26911)

# Deduplicate presence points to one per raster pixel
presence = deduplicate_to_unique_pixels(presence, "data/emb11Nclp.tif")


combined = gpd.pd.concat([presence, absence], ignore_index=True)
print("Combined CRS:", combined.crs)
print(combined["label"].value_counts())
combined.to_file("data/points_combined_culled.shp")