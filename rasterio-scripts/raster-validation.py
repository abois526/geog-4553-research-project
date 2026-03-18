# validates that raster data is alright to stack
# need to pip install the dependencies (rasterio)

import rasterio
import glob
import math

input_path = "/some/directory" # update as needed

files = sorted(glob.glob(input_path))

def transforms_close(t1, t2, tol=1e-9):
    return all(math.isclose(a, b, abs_tol=tol) for a, b in zip(t1, t2))

# Use first file as the reference
with rasterio.open(files[0]) as ref:
    ref_crs = ref.crs
    ref_transform = ref.transform
    ref_shape = ref.shape
    ref_count = ref.count
    ref_dtypes = ref.dtypes

print(f"Reference — CRS: {ref_crs}, Shape: {ref_shape}, Bands: {ref_count}, Dtypes: {ref_dtypes}")
print(f"Transform: {ref_transform}\n")

# Check all others against the reference
all_match = True
for f in files[1:]:
    with rasterio.open(f) as src:
        issues = []
        if src.crs != ref_crs:
            issues.append(f"CRS mismatch: {src.crs}")
        if not transforms_close(src.transform, ref_transform):
            issues.append(f"Transform mismatch: {src.transform}")
        if src.shape != ref_shape:
            issues.append(f"Shape mismatch: {src.shape}")
        if src.count != ref_count:
            issues.append(f"Band count mismatch: {src.count}")
        if src.dtypes != ref_dtypes:
            issues.append(f"Dtype mismatch: {src.dtypes}")

        if issues:
            print(f"PROBLEM — {f}: {', '.join(issues)}")
            all_match = False

if all_match:
    print(f"All {len(files)} files match — safe to stack.")