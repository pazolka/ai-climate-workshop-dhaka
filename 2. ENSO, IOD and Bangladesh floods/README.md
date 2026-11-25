# 2. ENSO, IOD and Bangladesh floods

**Two Oceans, One Delta: Interactions between ENSO and the Indian Ocean Dipole and Their Impact on Bangladesh Floods**

**Instructor: Dr. Yusuf Aydogdu**

Two hands-on sessions. Each notebook is **run and interpret**: run the cells in order with **Shift + Enter**, look at the figures, and discuss the *What to look for* questions with your neighbour. No coding experience is needed. Cells marked 🔧 **Try this** have settings you can change and run again.

| Session | Notebook | Open |
|---|---|---|
| **Two Oceans** | `Session1_ENSO_IOD_interaction.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pazolka/ai-climate-workshop-dhaka/blob/main/2.%20ENSO%2C%20IOD%20and%20Bangladesh%20floods/Session1_ENSO_IOD_interaction.ipynb) |
| **One Delta** | `Session2_ENSO_IOD_Bangladesh_floods.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pazolka/ai-climate-workshop-dhaka/blob/main/2.%20ENSO%2C%20IOD%20and%20Bangladesh%20floods/Session2_ENSO_IOD_Bangladesh_floods.ipynb) |

**Session 1 · Two Oceans.** Participants build the Niño 3.4 and Dipole Mode indices from NOAA's Extended Reconstructed Sea Surface Temperature dataset (ERSSTv5, 1979–2026) and explore how El Niño–Southern Oscillation and the Indian Ocean Dipole interact across the seasons. They then use machine learning to let the data identify recurring Indo-Pacific ocean patterns on its own, and compare these with the conventional ENSO and IOD classifications.

**Session 2 · One Delta.** Participants turn daily river discharge from the Copernicus GloFAS reanalysis (1979–2025) into a record of flood years on the Brahmaputra and Ganges, and use ERA5 rainfall data to map how the El Niño–Southern Oscillation and the Indian Ocean Dipole shape the South Asian monsoon. They then use machine learning to test whether the state of the two oceans can help predict Bangladesh's flood years, and learn how to check whether a forecast is better than chance.

## How to run

**In Colab (recommended):** click a badge above, then choose **File → Save a copy in Drive** so you can make changes. The first code cell copies this repository into Colab to get the data, which takes up to a minute.

**On your own computer:** download or clone this repository, install Python 3 with `numpy pandas matplotlib scipy scikit-learn shap jupyter`, and open the notebooks from this folder. They read the data from the repository's `data/` folder.

## Data used

From this repository's `data/` folder: `gridded/sst_indopacific.npz`, `gridded/era5_precip_gbm.npz`, `point_observations/glofas_discharge.csv`, `oni_official.csv` and `dmi_official.csv`. These are processed subsets of open datasets:

- **Sea surface temperature:** NOAA Extended Reconstructed Sea Surface Temperature (ERSST) version 5, NOAA Physical Sciences Laboratory.
- **Oceanic Niño Index:** NOAA Climate Prediction Center. **Dipole Mode Index:** NOAA Physical Sciences Laboratory (HadISST-based).
- **Rainfall:** ERA5 monthly averaged data on single levels. Contains modified Copernicus Climate Change Service information (2026).
- **River discharge:** GloFAS v4 historical reanalysis. Contains modified Copernicus Emergency Management Service information (2026).

Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.
