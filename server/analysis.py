import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
    parcels[f"dist_to_{feat_type}"] = parcels["centroid"].apply(
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

# Encode target variable (parcel class)
parcels_landuse["target_code"] = (
    parcels_landuse["ASS_CLASSI"]
    .astype("category")
    .cat.codes
)

# Define feature matrix
features = [
    "area",
    "perimeter",
    "compactness",
    "dist_to_road",
    "dist_to_water",
    "dist_to_school",
    "dist_to_tourism",
    "landuse_code"
]

# Remove missing values in any feature/target column
# ML models typically cannot train with missing values
data = parcels_landuse.dropna(
    subset=features + ["target_code"]
)

# Convert dataset into ML data structure
X = data[features]          # Features/predictors
y = data["target_code"]     # Target/response

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# Train random forest (RF) classifier model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Generate predictions
y_pred = model.predict(X_test)

# Evaluate prediction accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)