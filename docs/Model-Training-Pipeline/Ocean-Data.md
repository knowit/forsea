![multivar.gif](/.attachments/multivar-f8512223-b72c-4ab3-a77b-f1dd7d79cba4.gif)

The ocean data we use to train the models contains daily values for physical and biological oceanographic variables. The variables are defined over a three-dimensional grid with the dimensions: time, longitude and latitude.

## Data Sources
The ocean data is collected from [Copernicus Marine](https://data.marine.copernicus.eu/products) who provide free and open marine data. The data is collected from the following four datasets:

**Physics:**

* [Global Ocean Physics Analysis and Forecast](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description)

* [Global Ocean Physics Reanalysis](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description)

**Biogeochemistry:**

* [Global Ocean Biogeochemistry Analysis and Forecast](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSIS_FORECAST_BIO_001_028/description)

* [Global Ocean Biogeochemistry Hindcast](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_BGC_001_029/description)

**Physics** and **Biogeochemistry** refer to the types of variables they contain. The **Physics** datasets contain variables such as currents, temperature and salinity. The **Biogeochemistry** datasets contain variables such as chlorophyll concentration, dissolved oxygen and sea water PH.

**Analysis/Forecast** datasets contain data from `2020-11-01` onwards and are updated either daily or weekly. **Reanalysis/Hindcast** datasets contain data from `1993-01-01` to `2020-12-31`.

## How to download the ocean data
Copernicus Marine provide multiple ways to download data. The recommended method depends on if you want to download more than ~1 GB.

* **Downloads smaller than 1 GB** - Follow either this guide: [What is OPeNDAP and how to access Copernicus Marine data?](https://help.marine.copernicus.eu/en/articles/6522070-what-is-opendap-and-how-to-access-copernicus-marine-data)

* **Downloads larger than 1 GB** - Follow this guide: [How to download a large volume of data in NetCDF or CSV file format?](https://help.marine.copernicus.eu/en/articles/4808073-how-to-download-a-large-volume-of-data-in-netcdf-or-csv-file-format)



