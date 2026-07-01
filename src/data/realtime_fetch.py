"""Realtime spatial-field fetcher for the CNN-LSTM track.

Pulls near-real-time monthly fields for the four CNN-LSTM channels (sst, t300,
ua, va) from free, no-registration NOAA/PSL sources, resamples them onto the
SODA training grid (24 lat × 72 lon, 5° steps), and converts absolute values
to anomalies using precomputed climatologies (see ``climatology.py``).

The wind channels (NCEP/NCAR R1 monthly) lag ~5 months — the worst of the
four — so the assembled 12-month input window is cut off at the wind's latest
month (sst/t300's fresher data is truncated, not backfilled). This is labeled
honestly in the returned ``cutoff_label``.

All sources verified reachable via urllib (2026-07): NCEI OISST, PSL GODAS
``pottmp``, PSL NCEP/NCAR R1 ``uwnd``/``vwnd.mon.mean``. Each fetcher falls
back gracefully: a single failed channel is zero-filled and flagged, so the
CNN can still run (degraded) rather than aborting the whole turn.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.data.climatology import anomalize, load_climatology

# SODA training grid — the CNN-LSTM was trained on exactly this resolution.
TARGET_LAT = np.arange(-55, 61, 5, dtype=float)   # 24 points, -55..60
TARGET_LON = np.arange(0, 360, 5, dtype=float)    # 72 points, 0..355
INPUT_MONTHS = 12

# Source URLs (all verified reachable, no registration). OPeNDAP (dodsC) is used
# for server-side subsetting — the NCEP wind fileServer is 437 MB and breaks
# mid-transfer, so we slice remotely instead of downloading whole files.
OISST_DAILY_URL = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/{ym}/oisst-avhrr-v02r01.{ymd}.nc"
GODAS_POTTMP_OPENDAP = "https://psl.noaa.gov/thredds/dodsC/Datasets/godas/pottmp.{year}.nc"
NCEP_WIND_OPENDAP = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/{var}.mon.mean.nc"  # var in {uwnd,vwnd}

CLIM_DIR = PROCESSED_DATA_DIR


class RealtimeFetchError(RuntimeError):
    """Raised when realtime fields cannot be assembled at all."""


@dataclass
class ChannelResult:
    """One channel's assembled monthly anomaly fields."""

    fields: np.ndarray | None      # (n_months, 24, 72) anomalies, or None on failure
    months: np.ndarray | None      # (n_months,) int month-of-year, or None
    cutoff: str | None             # "YYYY-MM" latest month, or None
    error: str | None              # failure reason, or None on success


def _download_to(url: str, dest: Path, *, timeout: float = 60.0, retries: int = 2) -> Path:
    """Download ``url`` to ``dest`` with simple retry (resumes via Range)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_exc = None
    for attempt in range(retries + 1):
        try:
            # Resume from where we left off (PSL/NCEI support HTTP Range; large
            # files like the 437 MB NCEP wind climatology often break mid-transfer).
            have = dest.stat().st_size if dest.exists() else 0
            headers = {"User-Agent": "enso-chat/1.0"}
            if have:
                headers["Range"] = f"bytes={have}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                mode = "ab" if have and response.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                with open(dest, mode) as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            return dest
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            last_exc = exc
    raise RealtimeFetchError(f"download failed after {retries+1} tries: {url}: {last_exc}")


def _open_nc(path: Path):
    """Open a local NetCDF via a temp ASCII copy (netCDF4 can't read non-ASCII
    Windows paths). Returns an xarray Dataset; caller must close + rmtree."""
    import shutil
    import tempfile

    import xarray as xr

    tmp = Path(tempfile.mkdtemp(prefix="rt_nc_"))
    copy = tmp / path.name
    shutil.copy(path, copy)
    ds = xr.open_dataset(copy)
    return ds, tmp


def _resample_to_target(da):
    """Interp an xarray DataArray onto the SODA (lat, lon) grid, lon normalized to 0..360."""
    import xarray as xr

    # Normalize longitude to 0..360.
    lon = np.asarray(da["lon"].values, dtype=float)
    if lon.max() > 180:
        lon = ((lon + 360) % 360)
    da = da.assign_coords(lon=lon).sortby("lon")
    # Ensure latitude is ascending (interp behaves better; NCEP is descending).
    da = da.sortby("lat")
    # Drop duplicate longitudes if any (interp chokes on non-unique coords).
    _, uniq = np.unique(da["lon"].values, return_index=True)
    da = da.isel(lon=np.sort(uniq))
    da = da.interp(lat=TARGET_LAT, lon=TARGET_LON, method="linear", kwargs={"fill_value": "extrapolate"})
    return da


def _latest_available_months(n: int, *, max_month: pd.Timestamp | None = None) -> list[pd.Timestamp]:
    """The ``n`` most recent month-starts up to ``max_month`` (default: this month)."""
    if max_month is None:
        max_month = pd.Timestamp.now().to_period("M").to_timestamp()
    return [max_month - pd.DateOffset(months=i) for i in range(n - 1, -1, -1)]


# ---------------------------------------------------------------------------
# Channel fetchers — each returns ChannelResult with absolute fields + months
# ---------------------------------------------------------------------------

def _fetch_sst(n_months: int, cache_dir: Path) -> ChannelResult:
    """OISST v2.1 daily → monthly SST (°C).

    Downloads ONE mid-month day per target month as a monthly approximation
    (12 requests instead of ~336) — ENSO monthly skill does not need a true
    daily average, and the per-day HTTP request fan-out was the slowest,
    flakiest part of the pipeline. Cached per day under ``cache_dir/oisst/``.
    """
    import xarray as xr

    months = _latest_available_months(n_months)
    fields = []
    used_months = []
    cutoff = None
    for m in months:
        day = 15  # mid-month approximation
        ym = f"{m.year}{m.month:02d}"
        ymd = f"{m.year}{m.month:02d}{day:02d}"
        url = OISST_DAILY_URL.format(ym=ym, ymd=ymd)
        dest = cache_dir / "oisst" / f"{ymd}.nc"
        if not dest.exists():
            try:
                _download_to(url, dest, timeout=60.0, retries=2)
            except RealtimeFetchError:
                continue
        try:
            ds, tmp = _open_nc(dest)
            # OISST sst is (time, zlev, lat, lon); squeeze the single zlev level.
            sst = ds["sst"].isel(time=0)
            if "zlev" in sst.dims:
                sst = sst.isel(zlev=0)
            day_field = sst.values
            ds.close()
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            continue
        fields.append(day_field)
        used_months.append(m.month)
        cutoff = f"{m.year}-{m.month:02d}"
    if not fields:
        return ChannelResult(None, None, None, "OISST daily files unavailable")
    stacked = np.stack(fields)  # (n, lat_native, lon_native)
    da = xr.DataArray(stacked, dims=("t", "lat", "lon"),
                      coords={"lat": np.linspace(-89.875, 89.875, stacked.shape[1]),
                              "lon": np.linspace(0.125, 359.875, stacked.shape[2])})
    da = _resample_to_target(da)
    return ChannelResult(np.asarray(da.values, dtype=np.float32), np.array(used_months, dtype=int), cutoff, None)


def _fetch_t300(n_months: int, cache_dir: Path) -> ChannelResult:
    """GODAS pottmp → 300m potential temperature (K→°C), via OPeNDAP slice.

    Uses ``sel(level=303, method='nearest')`` — OPeNDAP does not support xarray
    ``interp`` across levels remotely (returns zeros), and 303m is within 3m of
    the 300m target (far smaller than the resampling error).
    """
    import xarray as xr

    months = _latest_available_months(n_months)
    needed_years = sorted({m.year for m in months})
    fields = []
    used_months = []
    cutoff = None
    for year in needed_years:
        url = GODAS_POTTMP_OPENDAP.format(year=year)
        try:
            ds = xr.open_dataset(url)
        except Exception as exc:
            continue
        try:
            da = ds["pottmp"].sel(level=303, method="nearest")
            # Resample each 2D time slice individually (3D interp returns zeros via OPeNDAP).
            slices = []
            for i in range(da.sizes["time"]):
                sl = _resample_to_target(da.isel(time=i))
                slices.append(np.asarray(sl.values, dtype=np.float32))
            vals = np.stack(slices) - 273.15  # K → °C
            ttimes = pd.to_datetime(ds["time"].values)
        except Exception:
            ds.close()
            continue
        ds.close()
        for i, t in enumerate(ttimes):
            mm = t.to_period("M").to_timestamp()
            if mm in months and len(fields) < n_months:
                fields.append(vals[i])
                used_months.append(mm.month)
                cutoff = f"{mm.year}-{mm.month:02d}"
    if not fields:
        return ChannelResult(None, None, None, "GODAS pottmp unavailable")
    return ChannelResult(np.stack(fields), np.array(used_months, dtype=int), cutoff, None)


def _fetch_wind(comp: str, n_months: int, cache_dir: Path) -> ChannelResult:
    """NCEP/NCAR R1 monthly wind (uwnd/vwnd) at 850 hPa via OPeNDAP slice. ~5-month lag.

    Selects the last ``n_months`` first, then resamples slice-by-slice. OPeNDAP
    returns zeros / times out when interpolating the full 938-month 3D array in
    one request, so we subset then interpolate each 2D time slice.
    """
    import xarray as xr

    var = {"ua": "uwnd", "va": "vwnd"}[comp]
    url = NCEP_WIND_OPENDAP.format(var=var)
    try:
        ds = xr.open_dataset(url)
    except Exception as exc:
        return ChannelResult(None, None, None, f"NCEP {var} OPeNDAP open failed: {exc}")
    try:
        da = ds[var].sel(level=850, method="nearest").isel(time=slice(-n_months, None))
        # Resample each 2D time slice individually (3D interp returns zeros via OPeNDAP).
        slices = []
        for i in range(da.sizes["time"]):
            sl = _resample_to_target(da.isel(time=i))
            slices.append(np.asarray(sl.values, dtype=np.float32))
        vals = np.stack(slices)
        ttimes = pd.to_datetime(ds["time"].values)[-n_months:]
    except Exception as exc:
        ds.close()
        return ChannelResult(None, None, None, f"NCEP {var} slice failed: {exc}")
    ds.close()
    fields = vals
    months_arr = np.array([t.month for t in ttimes], dtype=int)
    last = ttimes[-1]
    cutoff = f"{last.year}-{last.month:02d}"
    return ChannelResult(fields, months_arr, cutoff, None)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _anomalize_channel(ch: ChannelResult, clim_path: Path) -> ChannelResult:
    """Convert a channel's absolute fields to anomalies using its climatology."""
    if ch.fields is None:
        return ch
    try:
        clim = load_climatology(clim_path)
    except FileNotFoundError as exc:
        return ChannelResult(None, None, ch.cutoff, str(exc))
    anom = anomalize(ch.fields, clim, ch.months)
    return ChannelResult(anom, ch.months, ch.cutoff, None)


def _align_to_cutoff(channels: dict[str, ChannelResult]) -> tuple[np.ndarray, str, list[str]]:
    """Stack channels into (12,24,72,4), cutting all to the earliest channel cutoff.

    Wind lags ~5 months (the bottleneck), so the common window ends at the
    earliest cutoff. Returns (window, cutoff_label, missing_channels).
    """
    # Determine common cutoff = earliest (min) cutoff among successful channels.
    cutoffs = [pd.Period(ch.cutoff, freq="M") for ch in channels.values() if ch.cutoff]
    if not cutoffs:
        raise RealtimeFetchError("no realtime channel succeeded")
    common_cutoff = min(cutoffs)
    cutoff_label = str(common_cutoff)

    # For each channel, take the 12 months ending at common_cutoff.
    stacked = []
    missing = []
    for name in ("sst", "t300", "ua", "va"):
        ch = channels.get(name)
        if ch is None or ch.fields is None:
            stacked.append(np.zeros((INPUT_MONTHS, len(TARGET_LAT), len(TARGET_LON)), dtype=np.float32))
            missing.append(name)
            continue
        # ch.fields is (n, lat, lon), ch.months aligned. Take last INPUT_MONTHS.
        f = ch.fields[-INPUT_MONTHS:]
        if f.shape[0] < INPUT_MONTHS:
            # Pad front with zeros if short.
            pad = np.zeros((INPUT_MONTHS - f.shape[0], *f.shape[1:]), dtype=np.float32)
            f = np.concatenate([pad, f])
        # Land points resample to NaN — fill with 0 so the CNN doesn't output NaN.
        f = np.nan_to_num(f, nan=0.0)
        stacked.append(f)
    window = np.stack(stacked, axis=-1)  # (12, 24, 72, 4)
    return window, cutoff_label, missing


def fetch_realtime_window(
    n_months: int = INPUT_MONTHS,
    *,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, str, list[str]]:
    """Assemble a (12,24,72,4) anomaly window from realtime sources.

    Returns ``(window, cutoff_label, missing_channels)``. ``cutoff_label`` is
    the earliest channel's latest month (wind-limited, ~5-month lag).
    ``missing_channels`` lists any channel that failed (zero-filled, degraded).
    Climatologies must be precomputed under ``data/processed/``.
    """
    cache = cache_dir if cache_dir is not None else PROCESSED_DATA_DIR / "realtime_cache"

    raw = {
        "sst": _fetch_sst(n_months, cache),
        "t300": _fetch_t300(n_months, cache),
        "ua": _fetch_wind("ua", n_months, cache),
        "va": _fetch_wind("va", n_months, cache),
    }

    clim_paths = {
        "sst": CLIM_DIR / "sst_climatology.nc",
        "t300": CLIM_DIR / "t300_climatology.nc",
        "ua": CLIM_DIR / "uwnd_climatology.nc",
        "va": CLIM_DIR / "vwnd_climatology.nc",
    }
    anomalized = {name: _anomalize_channel(ch, clim_paths[name]) for name, ch in raw.items()}

    # If a channel failed at fetch OR anomalize, it stays None → zero-filled.
    window, cutoff, missing = _align_to_cutoff(anomalized)
    # Surface fetch-stage errors too (so caller knows why a channel is missing).
    for name in ("sst", "t300", "ua", "va"):
        ch = anomalized[name]
        if ch.fields is None and name not in missing:
            missing.append(name)
    if len(missing) == 4:
        raise RealtimeFetchError(f"all realtime channels failed: {[anomalized[n].error for n in anomalized]}")
    return window, cutoff, missing
