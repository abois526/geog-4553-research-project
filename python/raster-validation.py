import rasterio
import glob

files = sorted(glob.glob("embeddings/MDR66_A*.tif"))

# Use first file as the reference
with rasterio.open(files[0]) as ref:
    ref_crs = ref.crs
    ref_transform = ref.transform
    ref_shape = ref.shape

print(f"Reference — CRS: {ref_crs}, Shape: {ref_shape}")
print(f"Transform: {ref_transform}\n")

# Check all others against the reference
all_match = True
for f in files[1:]:
    with rasterio.open(f) as src:
        issues = []
        if src.crs != ref_crs:
            issues.append(f"CRS mismatch: {src.crs}")
        if src.transform != ref_transform:
            issues.append(f"Transform mismatch: {src.transform}")
        if src.shape != ref_shape:
            issues.append(f"Shape mismatch: {src.shape}")
        
        if issues:
            print(f"PROBLEM — {f}: {', '.join(issues)}")
            all_match = False

if all_match:
    print("All 64 files match — safe to stack.")