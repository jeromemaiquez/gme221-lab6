# GmE 221 - Laboratory 6

## Overview
This laboratory exercise performs geospatial machine learning on parcel objects using the scikit-learn Python package.

## Environment Setup
- Python 3.x
- PostgreSQL with PostGIS
- GeoPandas, Scikit-learn, QGIS, Github

## How to Run
1. Activate the virtual environment
2. Run `analysis.py` to run the full spatial statistical analysis pipeline

## Reflections - Part B
1. Parcels were selected as the prediction unit mainly due to the objective of the exercise, which is to predict parcel classification based on its geometry and spatial context. It is possible to come up with other research questions or goals using other objects as the prediciton unit.
2. Since the main purpose of roads in a spatial context is to convey movement and provide access, they have an influence on the proximity and accessibility of spatial objects with each other.
3. Tourist points of interest are trip generators, pulling in an amount of visitors disproportionate to other POI types. As such, their presence may have an influence on the classification of nearby parcels, i.e., commercial or high-density classes may be more pervalent in the area.
4. No machine learning has occurred at this stage yet, since all that has been done is to load the necessary datasets into the analysis.