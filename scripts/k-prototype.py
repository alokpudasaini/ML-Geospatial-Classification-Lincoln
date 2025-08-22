import numpy as np
import rasterio
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt

# File paths
slope_path = "data/processed/cleaned_slope.tif"
hydgrpdcd_path = "data/processed/cleaned_hydgrpdcd.tif"
drnclassdcd_path = "data/processed/cleaned_drclassdcd.tif"
landcover_path = "data/processed/cleaned_landcover.tif"
output_tif = "results/clusters.tif"

# Read raster function
def read_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile
    return data, profile

# Read rasters
slope, profile = read_raster(slope_path)
hydgrpdcd, _ = read_raster(hydgrpdcd_path)
drnclassdcd, _ = read_raster(drnclassdcd_path)
landcover, _ = read_raster(landcover_path)

# Identify invalid pixels
invalid_mask = (
    (slope < 0.0307084117) | (slope > 26.5048675537) |
    (hydgrpdcd < 1) | (hydgrpdcd > 6) |
    (drnclassdcd < 1) | (drnclassdcd > 7) |
    (landcover < 11) | (landcover > 95) |
    np.isnan(slope) | np.isnan(hydgrpdcd) |
    np.isnan(drnclassdcd) | np.isnan(landcover) |
    (slope == -9999) | (hydgrpdcd == -9999) |
    (drnclassdcd == -9999) | (landcover == -9999)
)

# Normalize slope
slope_min = slope[~invalid_mask].min()
slope_max = slope[~invalid_mask].max()
slope_norm = np.zeros_like(slope, dtype=np.float32)
slope_norm[~invalid_mask] = (slope[~invalid_mask] - slope_min) / (slope_max - slope_min)

# Prepare valid pixels
valid_indices = np.where(~invalid_mask)
n_valid = valid_indices[0].size

# Sample a fraction of valid pixels for clustering
sample_fraction = 0.05  # 5% sample, adjust for speed/memory
sample_size = max(1, int(n_valid * sample_fraction))
sample_indices = np.random.choice(n_valid, sample_size, replace=False)

# Extract sampled features
features_valid = np.zeros((n_valid, 4), dtype=np.float32)
features_valid[:, 0] = slope_norm[valid_indices]
features_valid[:, 1] = hydgrpdcd[valid_indices]
features_valid[:, 2] = drnclassdcd[valid_indices]
features_valid[:, 3] = landcover[valid_indices]

# Convert categorical columns to integer
cat_cols = [1, 2, 3]
features_valid[:, cat_cols] = features_valid[:, cat_cols].astype(int)

# Fit K-Prototypes on sampled data
sample_features = features_valid[sample_indices]
k = 5
kproto = KPrototypes(n_clusters=k, init='Huang', n_init=5, verbose=1)
clusters_sample = kproto.fit_predict(sample_features, categorical=cat_cols)

# Assign clusters to full valid dataset using trained centroids
full_clusters = kproto.predict(features_valid, categorical=cat_cols)

# Map clusters back to full raster
cluster_raster = np.full(slope.shape, -1, dtype=np.int32)
cluster_raster[valid_indices] = full_clusters

# Save cluster raster
profile.update(dtype=rasterio.int32, count=1, compress='lzw')
with rasterio.open(output_tif, 'w', **profile) as dst:
    dst.write(cluster_raster, 1)

# Plot clusters
plt.figure(figsize=(10, 8))
masked_clusters = np.ma.masked_equal(cluster_raster, -1)
plt.imshow(masked_clusters, cmap='tab20')
plt.colorbar(label='Cluster')
plt.title('K-Prototypes Clustering (Sampled Fit)')
plt.show()
