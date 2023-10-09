# Ozone_prediction

## Prepare input data
Data preparation involves ozone station monitoring data (sahsja), multiple meteorological variables data from ERA5_land, OMI remote sensing data (total column concentration and near-surface), socioeconomic data (gdp and population distribution), and land use data.
### ERA5_land_processing.py file 
handles ERA5 meteorological data;
### The OMI-SFO3_data_processing.py and OMI_TO3_data_processing.R
handles rremote sensing data processing;

## Model train&test

### LSTM_predict.py
Train and test the LSTM model to obtain the best model.

## Model predict
### Spatial_Visualization.py
Spatial visualization of model predictions.








