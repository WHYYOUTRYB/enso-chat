"""Precompute monthly climatologies (1991-2020) for the realtime sources.

Run once before using ``forecast_cnn_lstm(mode="realtime")``::

    python scripts/build_climatology.py

Downloads historical fields for each realtime source (OISST/GODAS/NCEP),
resamples onto the SODA grid, and computes per-month 30-year means. Cached as
``data/processed/{sst,t300,uwnd,vwnd}_climatology.nc``. This is offline and
network-heavy (30 years × 4 sources); it is never imported by the Streamlit app.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR
from src.data.climatology import CLIMATOLOGY_YEARS, compute_monthly_climatology, save_climatology
from src.data.realtime_fetch import (
    GODAS_POTTMP_OPENDAP,
    NCEP_WIND_OPENDAP,
    OISST_DAILY_URL,
    TARGET_LAT,
    TARGET_LON,
    _download_to,
    _resample_to_target,
)


OISST_MONTHLY_OPENDAP = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2/sst.mnmean.nc"


def _build_sst_climatology(cache_dir: Path, years: range) -> np.ndarray:
    """OISST monthly mean (PSL OPeNDAP) → 30-yr per-month climatology on SODA grid.

    Uses the PSL monthly-mean product (covers 1981-2023) for the climatology
    history; the realtime path uses NCEI daily OISST for the freshest months.
    """
    import xarray as xr

    ds = xr.open_dataset(OISST_MONTHLY_OPENDAP)
    try:
        times = xr.decode_cf(ds).time
        years_arr = times.dt.year.values
        da = ds["sst"]
        idx = np.array([i for i, y in enumerate(years_arr) if y in years])
        sub = da.isel(time=idx)
        # Resample slice-by-slice (3D OPeNDAP interp returns zeros).
        slices = []
        for i in range(sub.sizes["time"]):
            sl = _resample_to_target(sub.isel(time=i))
            slices.append(np.asarray(sl.values, dtype=np.float32))
        vals = np.stack(slices)
        ttimes = pd.to_datetime(times.values)
    finally:
        ds.close()
    months = np.array([ttimes[i].month for i in idx], dtype=int)
    print(f"  oisst monthly: {len(months)} months in climatology window")
    return compute_monthly_climatology(vals, months)


def _build_t300_climatology(cache_dir: Path, years: range) -> np.ndarray:
    """GODAS pottmp → 300m °C → 30-yr per-month climatology on SODA grid (OPeNDAP slice)."""
    import xarray as xr

    fields = []
    months = []
    for year in years:
        url = GODAS_POTTMP_OPENDAP.format(year=year)
        try:
            ds = xr.open_dataset(url)
        except Exception as exc:
            print(f"  godas {year} skip: {exc}")
            continue
        try:
            da = ds["pottmp"].sel(level=303, method="nearest")
            # Resample slice-by-slice (3D OPeNDAP interp returns zeros).
            slices = []
            for i in range(da.sizes["time"]):
                sl = _resample_to_target(da.isel(time=i))
                slices.append(np.asarray(sl.values, dtype=np.float32))
            vals = np.stack(slices) - 273.15
            ttimes = pd.to_datetime(ds["time"].values)
        except Exception as exc:
            print(f"  godas {year} slice fail: {exc}")
            ds.close()
            continue
        ds.close()
        for i, t in enumerate(ttimes):
            if t.year in years:
                fields.append(vals[i])
                months.append(t.month)
        print(f"  godas {year} ({len(ttimes)} months)")
    if not fields:
        raise RuntimeError("no GODAS history downloaded")
    return compute_monthly_climatology(np.stack(fields), np.array(months, dtype=int))


def _build_wind_climatology(comp: str, cache_dir: Path, years: range) -> np.ndarray:
    """NCEP/NCAR R1 monthly wind → 850 hPa → 30-yr per-month climatology (OPeNDAP slice)."""
    import xarray as xr

    var = {"ua": "uwnd", "va": "vwnd"}[comp]
    url = NCEP_WIND_OPENDAP.format(var=var)
    ds = xr.open_dataset(url)
    try:
        da = ds[var].sel(level=850, method="nearest")
        # Resample slice-by-slice (3D OPeNDAP interp returns zeros).
        slices = []
        for i in range(da.sizes["time"]):
            sl = _resample_to_target(da.isel(time=i))
            slices.append(np.asarray(sl.values, dtype=np.float32))
        vals = np.stack(slices)
        ttimes = pd.to_datetime(ds["time"].values)
    finally:
        ds.close()
    idx = np.array([i for i, t in enumerate(ttimes) if t.year in years])
    fields = vals[idx]
    months = np.array([ttimes[i].month for i in idx], dtype=int)
    print(f"  ncep {var}: {len(months)} months in climatology window")
    return compute_monthly_climatology(fields, months)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build realtime-source climatologies.")
    parser.add_argument("--cache-dir", default=str(PROCESSED_DATA_DIR / "realtime_cache"))
    parser.add_argument("--out-dir", default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--start-year", type=int, default=min(CLIMATOLOGY_YEARS))
    parser.add_argument("--end-year", type=int, default=max(CLIMATOLOGY_YEARS))
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    out = Path(args.out_dir)
    years = range(args.start_year, args.end_year + 1)

    print(f"Building climatologies for {args.start_year}-{args.end_year}...")
    print("[1/4] SST (OISST daily → monthly)...")
    save_climatology(out / "sst_climatology.nc", _build_sst_climatology(cache, years), "sst")
    print("[2/4] T300 (GODAS pottmp 300m)...")
    save_climatology(out / "t300_climatology.nc", _build_t300_climatology(cache, years), "t300")
    print("[3/4] U-wind (NCEP R1 850hPa)...")
    save_climatology(out / "uwnd_climatology.nc", _build_wind_climatology("ua", cache, years), "uwnd")
    print("[4/4] V-wind (NCEP R1 850hPa)...")
    save_climatology(out / "vwnd_climatology.nc", _build_wind_climatology("va", cache, years), "vwnd")
    print(f"\nDone. Climatologies in {out} (sst/t300/uwnd/vwnd_climatology.nc).")


if __name__ == "__main__":
    main()
