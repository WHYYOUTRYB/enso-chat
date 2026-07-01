"""Run the realtime-domain hindcast and report cross-domain CNN-LSTM skill.

Two-step: (1) build a leakage-free climatology (1991-2015, NOT overlapping the
2020-2021 eval window) if not already present; (2) run the hindcast.

Usage::

    python scripts/run_realtime_hindcast.py
    python scripts/run_realtime_hindcast.py --eval-start 2020-01-01 --eval-end 2021-12-01
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
from src.data.climatology import compute_monthly_climatology, save_climatology
from src.data.realtime_fetch import (
    GODAS_POTTMP_OPENDAP,
    NCEP_WIND_OPENDAP,
    _resample_to_target,
)
from src.models.realtime_hindcast import (
    realtime_hindcast_report_text,
    run_realtime_hindcast,
    save_realtime_hindcast,
)

# A separate climatology dir so the leakage-free 1991-2015 baseline doesn't
# clobber the operational 2020-2023 mini climatology used for live forecasts.
HINDCAST_CLIM_DIR = PROCESSED_DATA_DIR / "hindcast_clim"


def _build_hindcast_climatology(years: range) -> None:
    """Build the 1991-2015 (leakage-free) climatology for sst/t300/uwnd/vwnd."""
    import xarray as xr

    HINDCAST_CLIM_DIR.mkdir(parents=True, exist_ok=True)
    # SST (PSL monthly OPeNDAP).
    ds = xr.open_dataset("https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2/sst.mnmean.nc")
    times = xr.decode_cf(ds).time
    idx = np.array([i for i, y in enumerate(times.dt.year.values) if y in years])
    da = ds["sst"].isel(time=idx)
    slices = [np.nan_to_num(np.asarray(_resample_to_target(da.isel(time=i)).values, dtype=np.float32))
              for i in range(da.sizes["time"])]
    fields = np.stack(slices)
    months = np.array([pd.Timestamp(times.values[idx[i]]).month for i in range(len(idx))], dtype=int)
    ds.close()
    save_climatology(HINDCAST_CLIM_DIR / "sst_climatology.nc", compute_monthly_climatology(fields, months), "sst")
    print(f"  sst climatology: {len(months)} months")

    # T300 (GODAS per-year).
    t300_fields = []
    t300_months = []
    for year in years:
        try:
            ds = xr.open_dataset(GODAS_POTTMP_OPENDAP.format(year=year))
        except Exception:
            continue
        da = ds["pottmp"].sel(level=303, method="nearest")
        ttimes = pd.to_datetime(ds["time"].values)
        for i in range(da.sizes["time"]):
            sl = _resample_to_target(da.isel(time=i))
            t300_fields.append(np.nan_to_num(np.asarray(sl.values, dtype=np.float32)))
            t300_months.append(ttimes[i].month)
        ds.close()
    save_climatology(HINDCAST_CLIM_DIR / "t300_climatology.nc",
                     compute_monthly_climatology(np.stack(t300_fields), np.array(t300_months, dtype=int)), "t300")
    print(f"  t300 climatology: {len(t300_months)} months")

    # Winds (NCEP).
    for var, name in [("uwnd", "uwnd"), ("vwnd", "vwnd")]:
        ds = xr.open_dataset(NCEP_WIND_OPENDAP.format(var=var))
        times = xr.decode_cf(ds).time
        idx = np.array([i for i, y in enumerate(times.dt.year.values) if y in years])
        da = ds[var].sel(level=850, method="nearest").isel(time=idx)
        slices = [np.nan_to_num(np.asarray(_resample_to_target(da.isel(time=i)).values, dtype=np.float32))
                  for i in range(da.sizes["time"])]
        fields = np.stack(slices)
        months = np.array([pd.Timestamp(times.values[idx[i]]).month for i in range(len(idx))], dtype=int)
        ds.close()
        save_climatology(HINDCAST_CLIM_DIR / f"{name}_climatology.nc",
                         compute_monthly_climatology(fields, months), name)
        print(f"  {name} climatology: {len(months)} months")


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime-domain CNN-LSTM hindcast.")
    parser.add_argument("--weights-path", default=str(_PROJECT_ROOT / "weights" / "cnn_lstm_soda.pth"))
    parser.add_argument("--eval-start", default="2020-01-01")
    parser.add_argument("--eval-end", default="2021-12-01")
    parser.add_argument("--clim-start-year", type=int, default=1991)
    parser.add_argument("--clim-end-year", type=int, default=2015)
    parser.add_argument("--rebuild-clim", action="store_true", help="Force rebuild the leakage-free climatology.")
    parser.add_argument("--json-path", default=str(_PROJECT_ROOT / "reports" / "outputs" / "cnn_lstm_realtime_hindcast.json"))
    args = parser.parse_args()

    clim_ok = all((HINDCAST_CLIM_DIR / f"{n}_climatology.npz").exists()
                  for n in ["sst", "t300", "uwnd", "vwnd"])
    if args.rebuild_clim or not clim_ok:
        print(f"Building leakage-free climatology {args.clim_start_year}-{args.clim_end_year}...")
        _build_hindcast_climatology(range(args.clim_start_year, args.clim_end_year + 1))

    print(f"\nRunning realtime hindcast, eval {args.eval_start} to {args.eval_end}...")
    res = run_realtime_hindcast(
        Path(args.weights_path),
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        clim_dir=HINDCAST_CLIM_DIR,
    )
    print(realtime_hindcast_report_text(res))
    save_realtime_hindcast(res, Path(args.json_path))
    print(f"\nSaved → {args.json_path}")


if __name__ == "__main__":
    main()
