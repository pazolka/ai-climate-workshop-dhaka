# 1. Data Preprocessing

**From raw observations to model-ready data**

**Instructor: Anna Pazola**

Machine Learning models are only as good as the data they are trained on. This folder works through three typical data-curation tasks: detecting outliers, filling gaps, and aligning and validating datasets. It relies on example data from the Bangladesh Water Development Board - BWDB (groundwater level) and Bangladesh Meteorological Department - BMD (precipitation) that appear throughout the workshop.

Each notebook is **run and interpret**: run the cells in order with **Shift + Enter**, look at the figures, and discuss the questions. No coding experience is needed. Cells marked 🔧 **Try this** contain settings you can change and run again.

| Session | Notebook | Open |
|---|---|---|
| **Anomalies** | `1 Anomalies.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pazolka/ai-climate-workshop-dhaka/blob/main/1.%20Data%20Preprocessing/1%20Anomalies.ipynb) |
| **Gap filling** | `2 Gap Filling.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pazolka/ai-climate-workshop-dhaka/blob/main/1.%20Data%20Preprocessing/2%20Gap%20Filling.ipynb) |
| **Resampling and validation** | `3 Resampling and validation.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pazolka/ai-climate-workshop-dhaka/blob/main/1.%20Data%20Preprocessing/3%20Resampling%20and%20validation.ipynb) |

**1 · Anomaly detection.** Distinguish isolated outliers from systematic shifts in monthly groundwater levels, using rolling statistics and change-point detection.

**2 · Gap filling.** Compare a Kalman filter, random-forest imputation, and random forest with rainfall as an extra covariate for filling missing months in groundwater level time series.

**3 · Resampling and validation.** Resample weekly groundwater readings to monthly and daily series, then validate gridded rainfall products (ERA5, TerraClimate, BDGR) against BMD station observations using correlation, RMSE and anomaly correlation.

## How to run

**In Colab (recommended):** click a badge above, then choose **File → Save a copy in Drive** so you can make changes. The first code cell clones this repository to get the data and installs the extra packages (for example `ruptures`), which takes about a minute.

**On your own computer:** clone or download this repository, install Python 3 with `numpy pandas matplotlib scipy scikit-learn ruptures jupyter`, and open the notebooks from this folder. They read the shared data from the repository's `data/` folder.

`validation_helpers.py` contains the metric function imported by notebook 3.

## Data used

From this repository's `data/` folder:

- BWDB example groundwater data `point_observations/BWDB_gwl_ts_monthly.csv`, `MY050_gwl_ts_weekly.csv`, `BWDB_gwl_station_info.csv`
- BMD example rainfall data `point_observations/BMD_precip_ts_monthly.csv`, `BMD_precip_station_info.csv`
- Gridded rainfall products `gridded/ERA5_precip_M_acc_1972_2022_BGD_025.nc`, `TerraClimate_precip_aet_M_1972_2022_BGD_005.nc`, `BDGR_precip_M_acc_BGD_2000_2018_001.nc`
- Bangladesh - country area `BGD_shp/Bangladesh.shp`
