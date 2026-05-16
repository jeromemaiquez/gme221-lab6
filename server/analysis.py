import geopandas as gpd

# Load the dataset for parcels, the prediction units
parcels = gpd.read_file('data/parcel.geojson')
# print(parcels.head())
# print(parcels.crs)

# Load the supporting datasets
roads = gpd.read_file("data/roads.geojson") 
water = gpd.read_file("data/water_network.geojson") 
landuse = gpd.read_file("data/landuse.geojson") 
schools = gpd.read_file("data/schools.geojson") 
tourism = gpd.read_file("data/tourism.geojson") 

# Ensure CRS of supporting datasets match that of parcels'
roads = roads.to_crs(parcels.crs) 
water = water.to_crs(parcels.crs) 
landuse = landuse.to_crs(parcels.crs) 
schools = schools.to_crs(parcels.crs) 
tourism = tourism.to_crs(parcels.crs) 

# Calculate geometry-based features
# Features: predictors for a machine learning model
# Geometry-based features allow the model to understand
# ...traits of the geometries (and not the geometries themselves)
parcels["area"] = parcels.geometry.area
parcels["perimeter"] = parcels.geometry.length
parcels["compactness"] = (
    parcels["area"] / (parcels["perimeter"] ** 2)
)

# Create parcel centroids for distance calculations
parcels["centroid"] = parcels.geometry.centroid

# Calculate distance to other spatial feature types
for feat_type, gdf in zip(
    ["road", "water", "school", "tourism"],
    [roads, water, schools, tourism]
):
    parcels[f"dist_to{feat_type}"] = parcels["centroid"].apply(
        lambda p: gdf.distance(p).min()
    )

# Spatially join parcels with land use
parcels_landuse = gpd.sjoin(
    parcels,
    landuse[["Name", "geometry"]],
    how="left",
    predicate="intersects"
)

# Encode land use categories
parcels_landuse["landuse_code"] = (
    parcels_landuse["Name"]
    .astype("category")
    .cat.codes
)

# Check land use categories and their codes
print(
    parcels_landuse[["Name", "landuse_code"]]
    .drop_duplicates()
    .sort_values("landuse_code")
)