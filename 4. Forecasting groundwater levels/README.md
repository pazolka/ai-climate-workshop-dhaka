# 4. Forecasting groundwater levels

**Advanced Deep Learning for Time Series Forecasting of Groundwater Levels**

**13:30–14:45 · Instructor: Dr. Mojtaba Valipour**

Imagine it is January 2013. How deep will the water table at Haluaghat be over the next six months? Use Darts to run Chronos-2 Small, a pretrained forecasting model, on monthly well readings. Compare its forecasts with the same month last year, then test whether filling gaps, adding satellite observations, or using a longer well record improves accuracy.

| Session | Notebook | Open |
|---|---|---|
| **Advanced Deep Learning for Time Series Forecasting of Groundwater Levels** | [Groundwater_Forecasting_Workshop.ipynb](Groundwater_Forecasting_Workshop.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pazolka/ai-climate-workshop-dhaka/blob/session-4-groundwater-forecasting/4.%20Forecasting%20groundwater%20levels/Groundwater_Forecasting_Workshop.ipynb) |

## How to run

**In Colab**, open the **Live workshop** notebook and choose **File → Save a copy in Drive**. Keep the default CPU runtime. Run the cells in order with **Shift + Enter** and pause at the questions before revealing the observations. The setup cell installs the packages, and the pretrained model downloads on first use. The notebook includes the small data files it needs.

**On your own computer**, open the notebook in Jupyter with a Python kernel and run the same cells in order. The first run needs internet access for the packages and model weights.

## What to look for

- **Forecast accuracy.** Which method gives the lowest mean absolute error in metres? Check whether the answer changes across the three forecast dates.
- **Additional inputs.** Compare the original and gap-filled well histories first. Then add GRACE to the gap-filled history. What does each comparison tell you?
- **Length of the well record.** Give the same model either ten years of readings or all available history back to 1980. At which forecast dates do the extra years help?

If time permits, try a neighbouring well or known-future calendar features as covariates, compare forecasts across wells, or switch to a GPU for the optional fine-tuning exercise.

## Data used

- **Monthly groundwater levels.** Bangladesh Water Development Board well readings, including MY050 in Haluaghat. The exercise uses signed levels relative to ground, so a more negative reading means a deeper water table.
- **Gap-filled well records.** Published estimates from the preprocessing session illustrate where gaps were filled. For forecasting, the notebook estimates missing values using only the history preceding each forecast.
- **Regional water storage.** The GRACE terrestrial water-storage anomaly from the grid cell nearest MY050 connects this exercise to the groundwater-storage session. The exercise assumes a two-month reporting delay.

The [data folder](data/) contains separate, compact ZIP files for each exercise. A standalone copy of the notebook loads the same data automatically.
