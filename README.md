# Forsea 
This repo contains the code for the training of the Forsea model along with tasks associated with training such as preparing the dataset, regridding geospatial data, and evaluating the models.

## Project Structure
The project is devided into modules corresponding to specific tasks.
* `interpolation` contains code related to resampling geospatial data. 
* `model` contains the code where the models are defined.  
* `scheduled` contains the code for scheduled downloads and uploads to the Azure Blob Storage container.
* `training` contains the code to train and tune the models. 

## Scheduled Pipeline
The code in this repo runs on our local machine as part of the Forsea data-pipeline.

## Deployment
The code currently runs on our local GPU-machine which has two RTX 4080 GPUs, each with 16 GBs of VRAM.