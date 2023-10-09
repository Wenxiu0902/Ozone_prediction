# Ozone_prediction

## Prepare input data
Data preparation involves ozone station monitoring data, multiple meteorological variables data from ERA5_land, OMI remote sensing data (total column concentration and near-surface), socioeconomic data (gdp and population distribution), and land use data.
We constructed a 0.1° × 0.1° grid over China and averaged all the concurrent surface ozone measurements of monitoring sites within each grid cell to obtain grid-level surface ozone concentrations. Correspondingly, all predictor variables were aggregated or resampled to the targeted grid resolution of 0.1° × 0.1°.
#### ERA5_land_processing.py file 
Handles ERA5 meteorological data;
#### The OMI-SFO3_data_processing.py
Handles rremote sensing near-surface concentration data processing;
#### OMI_TO3_data_processing.R
handles rremote sensing total column concentration data processing;

## Model train&test
Transform the prepared time series data (Input_data file) into sequences suitable for the LSTM model, and proceed with model training and test.
#### LSTM_predict.py
Train and test the LSTM model to obtain the best model.

## Model predict
Use the trained model to predict the ozone concentration for the entire China and visualize the results.
#### Spatial_Visualization.py
Spatial visualization of model predictions.








