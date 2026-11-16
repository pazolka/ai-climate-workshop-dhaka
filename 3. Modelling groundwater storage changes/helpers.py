import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scaling method per covariate
# mean_std -> (x - mean) / std; min_max -> (x - mid) / half-range -> [-1, 1];
# positive_semidefinite -> (x - min) / std.
SCALING_METHODS = {
    "dem": "positive_semidefinite",
    "ndvi": "min_max",
    "pr": "positive_semidefinite",
    "aet": "min_max",
    "twsa": "mean_std",
}


def scale_covariates(das_cov, train_times):
    """Scale each covariate, fit on the training months only."""
    scaled = []
    for da in das_cov:
        method = SCALING_METHODS.get(str(da.name).lower(), "min_max")
        fit = da.sel(time=train_times) if "time" in da.dims else da
        values = fit.values
        if method == "mean_std":
            offset, scale = np.nanmean(values), np.nanstd(values)
        elif method == "min_max":
            lo, hi = np.nanmin(values), np.nanmax(values)
            offset, scale = (hi + lo) / 2.0, (hi - lo) / 2.0
        else:  # positive_semidefinite
            offset, scale = np.nanmin(values), np.nanstd(values)
        scaled.append((da - offset) / scale)
    return scaled


def stack_covariates(das_cov, months):
    """Returns xr.DataArray with dims ("time","lat","lon","cov") -> [T,H,W,K]"""
    base = das_cov[0]
    lat = base["lat"]; lon = base["lon"]
    T = len(months)

    chans = []
    cov_names = []
    for da in das_cov:
        name = getattr(da, "name", None) or "cov"
        cov_names.append(name)
        if "time" in da.dims:
            # select our timeline (assumes contains these times)
            sel = da.sel(time=months)
            arr = sel.to_numpy().astype("float32")  # [T,H,W]
        else:
            # static -> tile across time
            arr0 = da.to_numpy().astype("float32")  # [H,W]
            arr = np.tile(arr0[None, ...], (T, 1, 1))  # [T,H,W]
        chans.append(arr[..., None])  # [T,H,W,1]

    cov = np.concatenate(chans, axis=-1)  # [T,H,W,K]
    return xr.DataArray(
        cov,
        dims=("time","lat","lon","cov"),
        coords={"time": pd.to_datetime(months), "lat": lat, "lon": lon, "cov": cov_names},
        name="cov_stack"
    )

def masked_scores(obs, pred):
    """RMSE and R² over the cells where the target is observed."""
    m = np.isfinite(obs)
    o, p = obs[m], pred[m]
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - o.mean()) ** 2)
    return pd.Series({'n': int(m.sum()), 'rmse': np.sqrt(np.mean((o - p) ** 2)),
                      'r2': 1 - ss_res / ss_tot})

def clip_to_country(da, bgd_shp):
    """Clip a (lat, lon) map to the Bangladesh boundary, as the sister scripts do."""
    da = da.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    da = da.rio.write_crs('EPSG:4326')
    return da.rio.clip(bgd_shp.geometry.values, bgd_shp.crs, all_touched=True)



def to_da(arr, gwsa_incomplete, bgd_shp):
    da = xr.DataArray(arr, coords={'lat': gwsa_incomplete.lat, 'lon': gwsa_incomplete.lon},
                      dims=('lat', 'lon'))
    return clip_to_country(da, bgd_shp)



# Storage coefficients (specific yield) per well
STORAGE_COEFFICIENTS = {
    'MY050': 0.03215715,
    'DH071': 0.04737767,
    'JA001': 0.04618642,
    'MY055': 0.02314187,
    'RJ007': 0.08298258,
    'BO034': 0.07957942,
    'SY105': 0.02784715,
    'RJ105': 0.08990290,
    'KT010': 0.07160585,
    'CT038': 0.01555133,
    'DI074': 0.10514053,
}

# Wells that never contributed to the gridded target
HOLDOUT_WELLS = ['JA001', 'DH071', 'RJ105', 'CT038']

BASE_PERIOD = ('2004-01', '2009-12')


def load_gwl_observations(gwl_path, station_info_path, base_period=BASE_PERIOD):
    """Convert filled GWL to GWSA anomalies for the held-out wells.

    GWSA = (GWL - mean over ``base_period``) * storage coefficient
    """
    gwl = pd.read_csv(gwl_path, parse_dates=['Date'], index_col='Date')
    gwl.index = gwl.index.to_period('M')
    info = pd.read_csv(station_info_path).set_index('WellID')
    start, end = (pd.Period(base_period[0], 'M'), pd.Period(base_period[1], 'M'))

    frames = []
    for well in HOLDOUT_WELLS:
        if well not in gwl.columns:
            continue
        s = gwl[well].dropna()
        gwsa = (s - s.loc[start:end].mean()) * STORAGE_COEFFICIENTS[well]
        frames.append(pd.DataFrame({
            'time': s.index.to_timestamp(),
            'well_id': well,
            'lat': float(info.loc[well, 'Latitude']),
            'lon': float(info.loc[well, 'Longitude']),
            'gwsa': gwsa.values,
        }))
    return pd.concat(frames, ignore_index=True)


def predictions_at_wells(pred, pred_times, obs, lat, lon):
    """Sample a ``(time, lat, lon)`` prediction at every observed well and join the observations."""
    wells = obs[['well_id', 'lat', 'lon']].drop_duplicates('well_id')
    well_lat = xr.DataArray(wells.lat.values, dims='well')
    well_lon = xr.DataArray(wells.lon.values, dims='well')

    da = xr.DataArray(pred, coords={'time': pred_times, 'lat': lat, 'lon': lon},
                      dims=('time', 'lat', 'lon'))
    sampled = (da.sel(lat=well_lat, lon=well_lon, method='nearest')
                 .assign_coords(well=wells.well_id.values))
    long = sampled.to_dataframe(name='gwsa_pred').reset_index()[['time', 'well', 'gwsa_pred']]
    return obs.merge(long.rename(columns={'well': 'well_id'}), on=['time', 'well_id'])


def cell_scores(pred, pred_times, target, months):
    """RMSE/R² against the gridded target over ``months`` at cells with an observation.

    The target in each cell is the average of the wells in that cell, so this is a
    cell-level score.
    """
    pred_times = pd.DatetimeIndex(pred_times)
    target = np.asarray(target)
    if target.shape[0] != len(pred_times):
        raise ValueError('target and prediction time axes do not match')
    mask = pred_times.isin(pd.DatetimeIndex(months))
    return masked_scores(target[mask], np.asarray(pred)[mask])


def point_scores(pred, pred_times, obs, months, lat, lon):
    """RMSE/R² of sampled predictions against individual held-out well observations."""
    merged = predictions_at_wells(pred, pred_times, obs, lat, lon)
    d = merged[merged.time.isin(pd.DatetimeIndex(months))
               & merged.gwsa.notna() & merged.gwsa_pred.notna()]
    ss_res = ((d.gwsa - d.gwsa_pred) ** 2).sum()
    ss_tot = ((d.gwsa - d.gwsa.mean()) ** 2).sum()
    return pd.Series({'n': len(d), 'rmse': np.sqrt(((d.gwsa - d.gwsa_pred) ** 2).mean()),
                      'r2': 1 - ss_res / ss_tot})


def cross_split_table(models, obs, lat, lon):
    """Score test-time grid cells and train-time held-out wells for several models."""
    rows = {}
    for name, (pred, pred_times, target, train_months, test_months) in models.items():
        cell = cell_scores(pred, pred_times, target, test_months)[['n', 'rmse', 'r2']]
        point = point_scores(pred, pred_times, obs, train_months, lat, lon)[['n', 'rmse', 'r2']]
        row = pd.concat({'test time (grid cells)': cell,
                         'train time @ held-out wells': point})
        row.index.names = ['task', 'metric']
        rows[name] = row
    table = pd.DataFrame(rows).T
    table.index.name = 'model'
    table.columns.names = ['task', 'metric']
    return table


def plot_maps(obs, gwsa_incomplete, pred, bgd_shp, times, idx, title):
    fig, axes = plt.subplots(len(idx), 2, figsize=(6, 4 * len(idx)), squeeze=False)
    for row, i in enumerate(idx):
        o, p = to_da(obs[i], gwsa_incomplete, bgd_shp), to_da(pred[i], gwsa_incomplete, bgd_shp)
        values = np.concatenate([o.values[np.isfinite(o.values)],
                                 p.values[np.isfinite(p.values)]])
        vmin, vmax = np.percentile(values, [2, 98])
        im = o.plot(ax=axes[row, 0], cmap='RdBu_r', vmin=vmin, vmax=vmax, add_colorbar=False)
        p.plot(ax=axes[row, 1], cmap='RdBu_r', vmin=vmin, vmax=vmax, add_colorbar=False)
        axes[row, 0].set_title(f'observed - {times[i]:%Y-%m}')
        axes[row, 1].set_title(f'predicted - {times[i]:%Y-%m}')
        for ax in axes[row]:
            bgd_shp.boundary.plot(color='gray', linewidth=0.5, ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='GWSA', shrink=0.6)
    fig.suptitle(title)
    plt.show()
