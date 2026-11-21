# Workshop on AI for Climate -Resilient Water Management in Bangladesh

Hands-on Jupyter notebooks on Machine Learning and environmental applications in Bangladesh. Each numbered folder is a self-contained session; participants run the notebooks, interpret the figures and discuss the results.


## Sessions

|  |  |
|---|---|
| [1. Data Preprocessing](1.%20Data%20Preprocessing/README.md) | Introductory session on getting observations ready for modelling: anomaly detection, gap filling, and temporal resampling plus validation of gridded rainfall against station data. |
| [2. ENSO, IOD and Bangladesh floods](2.%20ENSO%2C%20IOD%20and%20Bangladesh%20floods/README.md) | Two sessions on the Indo-Pacific ocean–monsoon system: building Niño 3.4 and Dipole Mode indices and discovering ocean patterns with Machine Learning, then linking ENSO/IOD to Brahmaputra–Ganges flood years and testing whether the ocean state improves a flood forecast. |
| [3. Modelling groundwater storage changes](3.%20Modelling%20groundwater%20storage%20changes/GWSA_CNN_Bangladesh.ipynb) | A practical session on predicting monthly groundwater storage anomaly (GWSA) maps from NDVI, precipitation, evapotranspiration, GRACE terrestrial water storage and elevation with Deep Learning (convolutional neural networks); covers masked-loss training, validation, and separate spatial and temporal testing. |
| [4. Forecasting groundwater levels](4.%20Forecasting%20groundwater%20levels/Groundwater_Forecasting.ipynb) | Forecast monthly BWDB well levels in Darts with Chronos-2 Small or TiRex-2, compare original and gap-filled well histories with optional GRACE satellite input, and optionally fine-tune on local records. |
| [5. Water quality group work](5.%20Water%20quality%20group%20work/Salinity.ipynb) | Three group exercises on river- and soil-salinity observations: gap filling, a rainfall product comparison, and a test of whether temperature, soil moisture or evaporation add skill beyond rainfall. Participants choose subsets, lags and one of introduced Machine Learning techniques. |
| `data/` | Shared, preprocessed datasets read by the notebooks: gridded rainfall and covariates, point observations and shapefiles. |

## Instructions

- **Open a notebook** from its folder, or use the **Open In Colab** badge at the top of the notebook or in the folder README.
- **Run the cells in order with `Shift + Enter`** and look at the figures; the sessions are written to be run and interpreted, not coded from scratch. Cells marked 🔧 **Try this** contain settings you can change and run again.
- **In Colab:** after opening a notebook, choose **File → Save a copy in Drive** so you can make changes. The first code cell clones this repository (and installs any extra packages, such as `ruptures`, `xgboost` or `rioxarray`), which takes about a minute.
- **Locally:** clone or download this repository and open the notebooks from their folders. Install Python 3 with `numpy pandas matplotlib scipy scikit-learn jupyter`; the ENSO sessions also use `shap`, the groundwater modelling notebook uses `tensorflow`, `keras` and `rioxarray`, and the water-quality exercises use `xgboost`.
- **Session 4** loads small exercise-specific ZIPs from its data folder or from compact copies inside the standalone notebook. In Colab choose **Runtime → Run all**.
- **Data** for the other sessions are read from the shared `data/` folder with relative paths. Each folder README or notebook lists the exact files and their sources.

## Data sources

The datasets in `data/` are processed subsets of open products:

- **Groundwater and water quality:** Bangladesh Water Development Board (BWDB) groundwater levels; SRDI river- and soil-salinity observations; BMD rainfall stations.
- **Gridded climate reanalysis:** ERA5 and ERA5-Land (Copernicus Climate Change Service), GloFAS discharge (Copernicus Emergency Management Service), TerraClimate (University of California Merced), BDGR gridded rainfall.
- **Satellite:** GRACE terrestrial water storage, MODIS NDVI, and elevation.
- **Ocean indices:** NOAA ERSSTv5 sea surface temperature, the Oceanic Niño Index and the Dipole Mode Index.

