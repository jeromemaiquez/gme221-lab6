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

## Reflections - Part C
5. Machine learning models cannot directly use geometry as input data since machine learning models are numerical in nature, while geometries are complex data types made up of sequences of numbers in specific configurations. Instead, we can extract important quantitative properties of geometries and use those as ML inputs.
6. Distances are a measure of spatial proximity, which is a proxy for spatial interaction (following Tobler's First law that nearer things are more related). This allows us to account for spatial context in the machine learning model.
7. I predict that distance to roads will be the most influential factor, since higher access to roads leads to the prevalence of higher-density parcel classes. This is followed by parcel area.

## Reflections - Part D
8. The accuracy score can be interpreted as a measure of how much geometric properties and spatial context (which together comprise the feature set) determine a parcel's classification.
9. Yes. Some models reach high accuracy scores because they use correlated feature sets, which give a bonus to accuracy scores while not adding new information into the model. Other "accurate" models are trained and evaluated using datasets that had biased spatial distributions of their training and testing portions.
10. If there were any properties of buildings available in the data (e.g., density, height, use, etc.), that could also be a powerful predictor of parcel classification. Buildings and parcels are closely related spatial objects.

## Reflections - Part E & F

11. Wrong predictions can be found all throughout the area of interest, but some clustering can be observed in the north and western portions of the AOI. Parcel area might be one factor driving errors, as several small non-residential parcels were misclassified as residential due to their small area. 

## Final Reflections

11. Traditional GIS analysis often makes use of manually defined rules or thresholds. Meanwhile, GeoAI extracts patterns from the data itself using statistical and machine learning methods. This allows GeoAI to uncover hidden trends and patterns in the data at a faster and larger scale.
12. It is hard to tell without quantitatively measuring feature imoprtance, but as I've observed from the errors, it seems that parcel area is an important feature driving predicted parcel classes.
13. Many other factors could also be considered as features/predictors, such as building density, other measures of accessibility, etc. Moreover, the splitting of the input data into training and testing sets might not have been minimized for bias. 
14. Machine learning can quickly provide categorical predictions or numerical estimates of various variables given different driving factors. This is a useful tool for any spatial decision-making involved with forecasting or planning for the future, as well as for generating inventories of resources at a large scale and low cost (e.g., mapping farm plots, houses, etc.).
15. Many machine learning models work as a black box, where it isn't clear and intuitive how exactly the inputs were transformed into the predicted class or value. In such cases, if the researchers are not careful, biases inherent in the input data may be carried over into the results without the researchers' explicit knowledge, inadvertently tainting decision-making.