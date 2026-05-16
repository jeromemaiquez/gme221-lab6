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