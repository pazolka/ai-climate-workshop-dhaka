"""Helpers for the salinity group-work notebook."""

import pandas as pd
import xarray as xr


def build_panel(salinity_ts, stations, era5_path, tc_path):
    """Sample ERA5 and TerraClimate rainfall at the salinity stations.

    Parameters
    ----------
    salinity_ts : DataFrame
        Monthly salinity, index = PeriodIndex, columns = station names.
    stations : DataFrame
        Station table with ``Station``, ``Latitude``, ``Longitude`` and ``Type``.
    era5_path, tc_path : str
        NetCDF files with monthly precipitation in a ``pr`` variable.
    Returns
    -------
    DataFrame
        One row per station-month with columns
        ``date, Station, salinity, ERA5, TerraClimate, type``.
    """
    lat = xr.DataArray(stations.Latitude.values, dims="Station")
    lon = xr.DataArray(stations.Longitude.values, dims="Station")
    ids = xr.DataArray(stations.Station.values, dims="Station")

    era5 = (xr.open_dataset(era5_path)["pr"]
            .sel(lat=lat, lon=lon, method="nearest")
            .assign_coords(Station=ids)
            .rename(time="date")
            .to_dataframe(name="ERA5")["ERA5"].unstack("Station"))
    era5.index = pd.to_datetime(era5.index).to_period("M")
    era5 = era5.reindex(salinity_ts.index)

    tc = (xr.open_dataset(tc_path)["pr"]
          .sel(lat=lat, lon=lon, method="nearest")
          .assign_coords(Station=ids)
          .rename(time="date")
          .to_dataframe(name="TerraClimate")["TerraClimate"].unstack("Station"))
    tc.index = pd.to_datetime(tc.index).to_period("M")
    tc = tc.reindex(salinity_ts.index)

    sal_long = (salinity_ts.rename_axis(index="date", columns="Station")
                .stack().rename("salinity").reset_index())
    era5_long = (era5.rename_axis(index="date", columns="Station")
                 .stack().rename("ERA5").reset_index())
    tc_long = (tc.rename_axis(index="date", columns="Station")
               .stack().rename("TerraClimate").reset_index())

    panel = (sal_long
             .merge(era5_long, on=["date", "Station"])
             .merge(tc_long, on=["date", "Station"]))
    panel["type"] = panel.Station.map(stations.set_index("Station")["Type"])

    return panel.reset_index(drop=True)


def add_lags(panel, max_lag=3):
    """Add rainfall lag columns 0..``max_lag`` and return the feature names.

    Returns
    -------
    panel : DataFrame
        The input panel with ``ERA5_lag0..k`` and ``TC_lag0..k`` columns.
    era5_features, tc_features : list of str
        The lag column names for each product.
    """
    panel = panel.sort_values(["Station", "date"]).copy()

    for k in range(max_lag + 1):
        panel[f"ERA5_lag{k}"] = panel.groupby("Station")["ERA5"].shift(k)
        panel[f"TC_lag{k}"] = panel.groupby("Station")["TerraClimate"].shift(k)

    era5_features = [f"ERA5_lag{k}" for k in range(max_lag + 1)]
    tc_features = [f"TC_lag{k}" for k in range(max_lag + 1)]

    return panel, era5_features, tc_features
