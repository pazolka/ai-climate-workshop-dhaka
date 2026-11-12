"""Helpers for validating gridded climate products against station observations."""

import numpy as np
import pandas as pd


def validation_metrics(pair):
    """Summarise agreement between observations and one sampled product.

    ``pair`` must have ``station``, ``date``, ``obs`` and ``grid`` columns.
    Returns one row of metrics pooled over all station-months:

    - r, r_anom: Pearson correlation, raw and after removing each station's
      mean seasonal cycle;
    - rmse, mae: error in the units of the data.
    """
    d = pair.dropna(subset=["obs", "grid"]).copy()
    d["month"] = d.date.dt.month

    # remove each station's climatological monthly mean from both series
    clim = d.groupby(["station", "month"])[["obs", "grid"]].transform("mean")
    d["obs_anom"] = d.obs - clim.obs
    d["grid_anom"] = d.grid - clim.grid

    return pd.Series({
        "n_pairs": len(d),
        "stations": d.station.nunique(),
        "r": d.obs.corr(d.grid),
        "r_anom": d.obs_anom.corr(d.grid_anom),
        "rmse": np.sqrt(((d.grid - d.obs) ** 2).mean()),
        "mae": (d.grid - d.obs).abs().mean(),
    })
