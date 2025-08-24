# Geospatial Classification of Lincoln County, Oregon Using K-Prototypes Clustering  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)  
![License](https://img.shields.io/badge/License-MIT-green.svg)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)  
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-K--Prototypes-orange)  
![Made with](https://img.shields.io/badge/Made%20with-QGIS%20%7C%20Rasterio%20%7C%20Pandas-lightgrey)  

## Overview  
This project applies **unsupervised machine learning (K-Prototypes clustering)** to geospatial datasets in **Lincoln County, Oregon**. By combining terrain features such as **slope, land cover, hydrological group, and drainage class**, the study identifies distinct environmental zones. The project demonstrates how clustering can uncover hidden spatial patterns in heterogeneous landscapes and highlights the potential of geospatial ML for land classification and environmental planning.  

## Motivation  
- **Why Lincoln County?**  
  Lincoln County provides a diverse testbed with coastal lowlands, upland forests, and river valleys, making it ideal for exploring unsupervised classification.  

- **Why K-Prototypes?**  
  K-Prototypes supports **mixed-type data** (numeric + categorical), making it well-suited for terrain metrics and land/soil attributes.  

## Objective  
To classify the landscape of Lincoln County into **environmental zones** using K-Prototypes clustering on multi-layered geospatial datasets.  

## Data Sources  
- **Slope** (derived from DEM – Oregon Dept. of Forestry)  
- **Land Cover** (NLCD – USGS)  
- **Hydrological Soil Group** (SSURGO – USDA NRCS)  
- **Drainage Class** (SSURGO – USDA NRCS)  

## Methodology  
1. **Preprocessing:**  
   - Resampling and alignment of rasters  
   - Normalization of slope values  
   - Encoding categorical features  
   - Handling NoData values  

2. **Clustering:**  
   - Random sampling (5%) for computation efficiency  
   - K-Prototypes applied with *k = 5* (based on elbow method)  
   - Cluster labels mapped back to raster format  

3. **Evaluation:**  
   - Internal validation using cost function  
   - Interpretation of environmental zones based on terrain + land attributes  

## Key Findings  
- Clusters revealed distinct zones such as **mixed forests on moderate slopes, evergreen forests on gentle slopes, and shrub-dominated lowlands**.  
- Slope and land cover played the strongest roles in differentiating clusters.  
- Hydrological and drainage factors were relatively stable across the region.  

## Applications  
- Exploratory zoning and land-use planning  
- Environmental monitoring and management  
- Foundation for hazard zoning and sustainability studies  

## Limitations  
- Sensitive to categorical encoding  
- Limited variability in soil/drainage attributes  
- No ground-truth validation yet  

## Future Work  
- Extend workflow to other regions (e.g., Nepal)  
- Incorporate higher-resolution datasets  
- Validate clusters against hazard records  
- Compare with other clustering methods (DBSCAN, hierarchical)  

## Repository Structure  
- ├── data/ # Input DEM, landcover, soil datasets
- ├── scripts/ # Python preprocessing + clustering scripts
- ├── reports/ # Project's detail report
- ├── results/ # Clustered raster outputs and maps
- ├── README.md # Project summary

## References  
- Huang, Z. (1997). *Clustering large data sets with mixed numeric and categorical values.*  
- USGS, USDA NRCS, Oregon DEM datasets, Lincoln County GIS  
