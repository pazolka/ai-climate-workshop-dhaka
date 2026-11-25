# 3. Modelling groundwater storage changes

**Mapping groundwater storage anomalies with convolutional neural networks**

**Instructor: Anna Pazola**

Bangladesh has one of the most extensive groundwater monitoring networks in the world. This folder demonstrates a grid-to-grid deep-learning workflow: predict monthly groundwater storage anomaly (GWSA) maps from climate and satellite covariates, first with a 2D U-Net that treats each month independently, then with a 3D U-Net that sees a short sequence of months.

The notebook is **run and interpret**: run the cells in order with **Shift + Enter**, look at the maps and training curves, and discuss the evaluation with your neighbour. Cells marked 🔧 **Try this** contain settings you can change and run again.

| Session | Notebook | Open |
|---|---|---|
| **GWSA with 2D and 3D U-Nets** | `GWSA_CNN_Bangladesh.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pazolka/ai-climate-workshop-dhaka/blob/main/3.%20Modelling%20groundwater%20storage%20changes/GWSA_CNN_Bangladesh.ipynb) |

**1 · Prepare the data.** Load the preprocessed GWSA grid and covariates (NDVI, precipitation, evapotranspiration, GRACE terrestrial water storage, elevation), scale the covariates, and build the fixed observation mask.

**2 · 2D U-Net.** One map in, one map out: train a U-Net on each month independently with a masked loss, inspect the learning curves, and map the predicted GWSA.

**3 · 3D U-Net.** A sequence of maps in, one map out: add temporal context with a short lag window and a loss that also rewards the seasonal cycle and domain mean.

**4 · Evaluation across space and time.** Score the models separately on unseen test months at training locations (temporal extrapolation) and on training-period months at held-out wells (spatial generalisation).

## How to run

**In Colab (recommended):** click the badge above, then choose **File → Save a copy in Drive**. The first code cell clones this repository to get the data, which takes up to a minute.

**On your own computer:** clone or download this repository, install Python 3 with `numpy pandas matplotlib xarray geopandas rioxarray tensorflow keras jupyter`, and open the notebook from this folder. It reads the shared data from the repository's `data/` folder.

The folder also contains `models.py` (2D/3D U-Net architectures and masked losses), `helpers.py` (covariate scaling, stacking, map plotting and evaluation helpers) and the figures used in the notebook (`task_diagram.png`, `gwsa_definition.png`, `time_split.png`).

## Data used

From this repository's `data/` folder:

- Preprocessed GWSA data derived from BWDB GWL data `gridded/GWSA_M_2002_2017_BGD_025.nc`
- Gridded satellite covariate products: NDVI `gridded/NDVI_M_2002_2017_BGD_025.nc4`, Precipitation and evapotranspiration `gridded/PR_AET_M_2002_2017_BGD_025.nc4`, GRACE terrestrial water storage anomalies `gridded/TWSA_M_2002_2017_BGD_025.nc4`, elevation `gridded/DEM_BGD_025.nc4`
- Sample BWDB dataset used for evaluation `point_observations/BWDB_gwl_ts_monthly_filled.csv`, `point_observations/BWDB_gwl_station_info.csv`
- Bangladesh - country area `BGD_shp/Bangladesh.shp`