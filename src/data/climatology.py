"""Monthly climatology baselines for the realtime spatial-field track.

The CNN-LSTM was trained on SODA, which stores **anomalies** (sst mean≈0). The
realtime sources (OISST/GODAS/NCEP) store **absolute** values, so feeding them
raw would be a fatal domain shift. This module computes per-month climatology
baselines (1991-2020 standard climatology) and converts absolute fields to
anomalies — the mandatory alignment step, not an optional nicety.

Climatologies are precomputed once by ``scripts/build_climatology.py`` and
cached as NetCDF under ``data/processed/``. The realtime fetcher reads them
back to anomalize on each fetch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLIMATOLOGY_YEARS = range(1991, 2021)  # standard WMO 30-year climate normal


def compute_monthly_climatology(
    fields: np.ndarray,
    months: np.ndarray,
    *,
    years: range = CLIMATOLOGY_YEARS,
) -> np.ndarray:
    """Per-month mean over the climatology period → shape ``(12, *field_shape)``.

    Args:
        fields: ``(n_time, lat, lon)`` absolute-value monthly fields.
        months: ``(n_time,)`` int month-of-year (1..12) aligned with ``fields``.
        years: climatology years (rows outside this range, if tagged, are
            excluded by the caller — here we just group by month).

    Returns:
        ``(12, lat, lon)`` — entry ``i`` is the mean of all Januaries (i=0),
        Februaries (i=1), … over the supplied samples.
    """
    fields = np.asarray(fields, dtype=np.float32)
    months = np.asarray(months, dtype=int)
    if fields.ndim != 3:
        raise ValueError(f"fields must be 3D (n_time,lat,lon), got {fields.shape}")
    if months.shape[0] != fields.shape[0]:
        raise ValueError("months length must match fields time dim")
    lat, lon = fields.shape[1], fields.shape[2]
    clim = np.zeros((12, lat, lon), dtype=np.float32)
    for m in range(1, 13):
        mask = months == m
        if not mask.any():
            continue
        clim[m - 1] = np.nanmean(fields[mask], axis=0)
    return clim


def anomalize(
    fields: np.ndarray,
    climatology: np.ndarray,
    months: np.ndarray,
) -> np.ndarray:
    """Subtract the per-month climatology to convert absolute → anomaly.

    Args:
        fields: ``(n_time, lat, lon)`` absolute fields.
        climatology: ``(12, lat, lon)`` from :func:`compute_monthly_climatology`.
        months: ``(n_time,)`` int month-of-year (1..12).

    Returns:
        ``(n_time, lat, lon)`` anomalies. NaNs preserved.
    """
    fields = np.asarray(fields, dtype=np.float32)
    months = np.asarray(months, dtype=int)
    if climatology.shape[0] != 12:
        raise ValueError(f"climatology must have 12 months, got {climatology.shape[0]}")
    out = fields.copy()
    for m in range(1, 13):
        mask = months == m
        if not mask.any():
            continue
        out[mask] = fields[mask] - climatology[m - 1]
    return out


def save_climatology(path: Path, climatology: np.ndarray, source: str) -> None:
    """Cache a climatology as a .npz (plain numpy, avoids netCDF4's write issues
    on non-ASCII Windows paths and lingering file handles).

    ``path`` should end in ``.npz``; if a ``.nc`` path is passed it is rewritten.
    """
    if path.suffix == ".nc":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, climatology=climatology.astype(np.float32), source=np.array(source))


def load_climatology(path: Path) -> np.ndarray:
    """Read a cached climatology back as ``(12, lat, lon)`` numpy array.

    Accepts either ``.npz`` (preferred) or a legacy ``.nc`` NetCDF file.
    """
    import xarray as xr

    if path.suffix == ".nc":
        path = path.with_suffix(".npz")
    if not path.exists():
        raise FileNotFoundError(f"Climatology not found: {path}. Run scripts/build_climatology.py first.")
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        arr = np.asarray(data["climatology"], dtype=np.float32)
    else:
        da = xr.open_dataarray(path)
        arr = np.asarray(da.values, dtype=np.float32)
        da.close()
    if arr.shape[0] != 12:
        raise ValueError(f"Climatology {path} has {arr.shape[0]} months, expected 12")
    return arr
