import rasterio
import numpy as np
import pandas as pd

# Paths to data
clusters_path = "results/clusters.tif"
slope_path = "data/processed/cleaned_slope.tif"
hydgrpdcd_path = "data/processed/cleaned_hydgrpdcd.tif"
drnclassdcd_path = "data/processed/cleaned_drclassdcd.tif"
landcover_path = "data/processed/cleaned_landcover.tif"

# Read raster function
def read_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1)
    return data

# Load rasters
clusters = read_raster(clusters_path)
slope = read_raster(slope_path)
hydro = read_raster(hydgrpdcd_path)
drainage = read_raster(drnclassdcd_path)
landcover = read_raster(landcover_path)

# Map numeric codes to readable labels
hydro_labels = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'A/D', 6: 'C/D'}
drainage_labels = {
    1: 'Excessively drained',
    2: 'Somewhat excessively drained',
    3: 'Well drained',
    4: 'Moderately well drained',
    5: 'Somewhat poorly drained',
    6: 'Poorly drained',
    7: 'Very poorly drained'
}
landcover_labels = {
    11: 'Open Water', 12: 'Perennial Ice/Snow', 21: 'Developed, Open Space', 22: 'Developed, Low Intensity',
    23: 'Developed, Medium Intensity', 24: 'Developed, High Intensity',
    31: 'Barren Land', 41: 'Deciduous Forest', 42: 'Evergreen Forest',
    43: 'Mixed Forest', 51: 'Dwarf Scrub', 52: 'Shrub/Scrub', 71: 'Grassland/Herbaceous',
    72: 'Sedge/Herbaceous', 73:'Lichens', 74: 'Moss', 81: 'Pasture/Hay', 82: 'Cultivated Crops', 
    90: 'Woody Wetlands', 95: 'Emergent Herbaceous Wetlands'
}

# Filter valid pixels (clusters != -1)
valid_mask = clusters != -1

# Prepare DataFrame with labels
df = pd.DataFrame({
    'cluster': clusters[valid_mask],
    'slope': slope[valid_mask],
    'hydro_group': [hydro_labels[x] for x in hydro[valid_mask]],
    'drainage_class': [drainage_labels[x] for x in drainage[valid_mask]],
    'landcover': [landcover_labels.get(x, f"Other({x})") for x in landcover[valid_mask]]
})

# Function to find dominant feature and percentage per cluster
def dominant_with_percentage(df, feature):
    summary = {}
    for cluster, group in df.groupby('cluster'):
        counts = group[feature].value_counts()
        dominant = counts.idxmax()
        percent = (counts.max() / counts.sum()) * 100
        summary[cluster] = f"{dominant} ({percent:.1f}%)"
    return pd.Series(summary)

# Slope statistics per cluster
slope_stats = df.groupby('cluster')['slope'].agg(['mean', 'median', 'min', 'max'])

# Dominant features with percentages
dominant_hydro = dominant_with_percentage(df, 'hydro_group')
dominant_drainage = dominant_with_percentage(df, 'drainage_class')
dominant_landcover = dominant_with_percentage(df, 'landcover')

# Combine into one summary table
cluster_summary = pd.concat([
    dominant_hydro.rename('Dominant Hydrological Group'),
    dominant_drainage.rename('Dominant Drainage Class'),
    dominant_landcover.rename('Dominant Landcover'),
    slope_stats
], axis=1)

print("Cluster Summary Table:")
print(cluster_summary)

# Save to CSV
cluster_summary.to_csv("results/cluster_summary_with_percent.csv")
