import rasterio
import glob

# want to run raster-validation.py before this

input_files = "embeddings/MDR66_A*.tif"
output_path = "/embeddings/embeddings_stacked.tif"

# List all embedding band files
files = sorted(glob.glob(input_files))

# Guard against empty file list
if not files:
    raise FileNotFoundError(
        "No .tif files found in 'embeddings/' — check that your path is correct "
        "and that the script is being run from the right directory."
    )

print(f"Found {len(files)} files. Opening...")

# Open files one at a time so we can catch individual failures
src_files = []
try:
    for f in files:
        try:
            src_files.append(rasterio.open(f))
        except rasterio.errors.RasterioIOError as e:
            raise RuntimeError(f"Failed to open file: {f}\n{e}")

    # Build output profile from the first file
    profile = src_files[0].profile
    profile.update(count=len(src_files))

    print(f"Stacking {len(src_files)} bands into embeddings_stacked.tif...")

    with rasterio.open(output_path, "w", **profile) as dst:
        for i, src in enumerate(src_files, 1):
            try:
                dst.write(src.read(1), i)
            except Exception as e:
                raise RuntimeError(f"Failed to write band {i} from file: {files[i-1]}\n{e}")

    print("Done. Output written to embeddings_stacked.tif")

finally:
    # Always close open files, even if something went wrong above
    for src in src_files:
        src.close()