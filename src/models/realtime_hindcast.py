"""Realtime-domain hindcast: evaluate the CNN-LSTM on historical realtime fields.

The SODA hindcast (``hindcast.py``) answers "is the model skillful on the
domain it was trained on?". This module answers the question that actually
matters for real-time use: "is it skillful on the OISST/GODAS/NCEP fields it
sees at inference time?" — i.e. the cross-domain skill.

Protocol:
  1. Pull realtime sst/t300/ua/va for an evaluation window (e.g. 2020-2021),
     anomalized against a climatology built from a NON-overlapping period
     (1991-2015) to avoid leakage.
  2. For each 12-month input window, run the CNN-LSTM forward → 24 lead forecasts.
  3. Compare against the true Niño3.4 from NOAA/PSL (online) for the 24 months
     after each window.
  4. Report all-season ACC per lead, plus a Persistence baseline (forecast =
     last observed Niño3.4) — same convention as the SODA hindcast and Ham et
     al. 2019.

This is the only metric that legitimately judges realtime predictions. The
SODA-hindcast ACC does NOT transfer across domains.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.data.climatology import anomalize
from src.data.realtime_fetch import (
    GODAS_POTTMP_OPENDAP,
    NCEP_WIND_OPENDAP,
    TARGET_LAT,
    TARGET_LON,
    _resample_to_target,
)
from src.models.cnn_lstm import CHANNELS, INPUT_MONTHS, LEAD_MONTHS, _build_model
from src.models.evaluation import calculate_acc


@dataclass
class RealtimeHindcastResult:
    leads: list[int]
    cnn_acc: list[float]
    persistence_acc: list[float]
    skill_gap: list[float]
    n_windows: int
    eval_period: str
    climatology_period: str


def _fetch_realtime_history(channel: str, start: pd.Timestamp, end: pd.Timestamp, clim: np.ndarray) -> np.ndarray:
    """Fetch one realtime channel's monthly anomaly fields over [start, end].

    Returns ``(n_months, 24, 72)`` anomalies. Uses a caller-supplied climatology
    (already built from a non-overlapping period).
    """
    import xarray as xr

    months_needed = pd.date_range(start, end, freq="MS")
    if channel == "sst":
        ds = xr.open_dataset("https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2/sst.mnmean.nc")
        times = xr.decode_cf(ds).time
        # Select months in range.
        idx = np.array([i for i, t in enumerate(times.values)
                        if start <= pd.Timestamp(t) <= end])
        da = ds["sst"].isel(time=idx)
        slices = []
        for i in range(da.sizes["time"]):
            sl = _resample_to_target(da.isel(time=i))
            slices.append(np.nan_to_num(np.asarray(sl.values, dtype=np.float32)))
        fields = np.stack(slices)
        months = np.array([pd.Timestamp(times.values[idx[i]]).month for i in range(len(idx))], dtype=int)
        ds.close()
    elif channel == "t300":
        fields_list = []
        months_list = []
        for year in range(start.year, end.year + 1):
            ds = xr.open_dataset(GODAS_POTTMP_OPENDAP.format(year=year))
            da = ds["pottmp"].sel(level=303, method="nearest")
            ttimes = pd.to_datetime(ds["time"].values)
            for i in range(da.sizes["time"]):
                t = ttimes[i]
                if start <= t <= end:
                    sl = _resample_to_target(da.isel(time=i))
                    fields_list.append(np.nan_to_num(np.asarray(sl.values, dtype=np.float32)))
                    months_list.append(t.month)
            ds.close()
        fields = np.stack(fields_list)
        months = np.array(months_list, dtype=int)
    else:
        var = {"ua": "uwnd", "va": "vwnd"}[channel]
        ds = xr.open_dataset(NCEP_WIND_OPENDAP.format(var=var))
        times = xr.decode_cf(ds).time
        idx = np.array([i for i, t in enumerate(times.values)
                        if start <= pd.Timestamp(t) <= end])
        da = ds[var].sel(level=850, method="nearest").isel(time=idx)
        slices = []
        for i in range(da.sizes["time"]):
            sl = _resample_to_target(da.isel(time=i))
            slices.append(np.nan_to_num(np.asarray(sl.values, dtype=np.float32)))
        fields = np.stack(slices)
        months = np.array([pd.Timestamp(times.values[idx[i]]).month for i in range(len(idx))], dtype=int)
        ds.close()
    return np.nan_to_num(anomalize(fields, clim, months), nan=0.0)


def _fetch_true_nino34(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """True Niño3.4 monthly series from NOAA/PSL over [start, end]."""
    from src.data.source_registry import load_index

    df = load_index("nino34")
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].sort_values("date").reset_index(drop=True)


def run_realtime_hindcast(
    weights_path: Path,
    *,
    eval_start: str = "2020-01-01",
    eval_end: str = "2021-12-01",
    clim_dir: Path | None = None,
) -> RealtimeHindcastResult:
    """Run the CNN-LSTM over historical realtime windows and score vs truth.

    Args:
        weights_path: trained CNN-LSTM checkpoint.
        eval_start/eval_end: evaluation window (input windows start here; their
            24-month horizons extend past eval_end, so true Niño3.4 must be
            available through eval_end + 24 months).
        clim_dir: directory with ``{sst,t300,uwnd,vwnd}_climatology.npz`` built
            from a period NOT overlapping the evaluation window (leakage guard).
    """
    import torch

    clim_dir = clim_dir if clim_dir is not None else PROCESSED_DATA_DIR
    eval_start_ts = pd.Timestamp(eval_start)
    eval_end_ts = pd.Timestamp(eval_end)

    # Load climatologies (must be prebuilt for a non-overlapping period).
    from src.data.climatology import load_climatology

    clims = {ch: load_climatology(clim_dir / f"{clim_name}_climatology.nc")
             for ch, clim_name in [("sst", "sst"), ("t300", "t300"), ("ua", "uwnd"), ("va", "vwnd")]}

    # Fetch realtime anomaly fields over the evaluation window.
    fields = {ch: _fetch_realtime_history(ch, eval_start_ts, eval_end_ts, clims[ch]) for ch in CHANNELS}

    # Align channels to the common length (wind may lag / differ slightly).
    n_common = min(len(f) for f in fields.values())
    for ch in CHANNELS:
        fields[ch] = fields[ch][-n_common:]

    # Build input windows: each starts at month i, input = months i..i+11.
    # The 24-month horizon extends past the input-source window into the truth
    # series, so input length only needs to cover INPUT_MONTHS per window.
    n_windows = max(0, n_common - INPUT_MONTHS + 1)
    if n_windows == 0:
        raise RuntimeError("evaluation window too short for a 12-month input window")

    # True Niño3.4 over the full span (windows + horizons).
    truth_start = eval_start_ts
    truth_end = eval_end_ts + pd.DateOffset(months=LEAD_MONTHS)
    truth_df = _fetch_true_nino34(truth_start, truth_end)
    truth_series = truth_df.set_index("date")["nino34"]

    # Assemble windows + truth targets.
    window_starts = pd.date_range(eval_start_ts, periods=n_common, freq="MS")
    X = []
    y_true = []
    valid_starts = []
    for i in range(n_windows):
        ws = window_starts[i]
        we = window_starts[i + INPUT_MONTHS - 1]  # last input month
        # Horizon: months ws+12 .. ws+35
        horizon_dates = pd.date_range(ws + pd.DateOffset(months=INPUT_MONTHS),
                                      periods=LEAD_MONTHS, freq="MS")
        # All horizon months must be in truth_series.
        if not all(d in truth_series.index for d in horizon_dates):
            continue
        win = np.stack([fields[ch][i : i + INPUT_MONTHS] for ch in CHANNELS], axis=-1)  # (12,24,72,4)
        X.append(win)
        y_true.append([float(truth_series[d]) for d in horizon_dates])
        valid_starts.append(ws)
    if not X:
        raise RuntimeError("no complete windows with full truth coverage")
    X = np.stack(X).astype(np.float32)  # (n,12,24,72,4)
    y_true = np.asarray(y_true, dtype=np.float32)  # (n,24)

    # CNN forward.
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.where(np.asarray(ckpt["x_std"], dtype=np.float32) == 0, 1.0, np.asarray(ckpt["x_std"], dtype=np.float32))
    model = _build_model()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    cnn_pred = np.empty((len(X), LEAD_MONTHS), dtype=np.float32)
    bs = 16
    with torch.no_grad():
        for i in range(0, len(X), bs):
            chunk = torch.tensor((X[i : i + bs] - x_mean) / x_std, dtype=torch.float32)
            cnn_pred[i : i + bs] = model(chunk)[:, -1, :].cpu().numpy()

    # Persistence baseline: forecast = last input Niño3.4 (truth at window end).
    pers_pred = np.empty((len(X), LEAD_MONTHS), dtype=np.float32)
    for i, ws in enumerate(valid_starts):
        we = ws + pd.DateOffset(months=INPUT_MONTHS - 1)
        last_val = float(truth_series.get(we, 0.0))
        pers_pred[i, :] = last_val

    cnn_acc, pers_acc, gap = [], [], []
    for lead in range(LEAD_MONTHS):
        c = calculate_acc(y_true[:, lead], cnn_pred[:, lead])
        p = calculate_acc(y_true[:, lead], pers_pred[:, lead])
        cnn_acc.append(round(c, 4))
        pers_acc.append(round(p, 4))
        gap.append(round(c - p, 4))

    return RealtimeHindcastResult(
        leads=list(range(1, LEAD_MONTHS + 1)),
        cnn_acc=cnn_acc,
        persistence_acc=pers_acc,
        skill_gap=gap,
        n_windows=len(X),
        eval_period=f"{eval_start}_to_{eval_end}",
        climatology_period="(see clim_dir, must not overlap eval window)",
    )


def realtime_hindcast_report_text(res: RealtimeHindcastResult) -> str:
    lines = [
        f"Realtime-domain hindcast skill (n={res.n_windows} windows, eval={res.eval_period}).",
        f"All-season ACC on OISST/GODAS/NCEP fields — the cross-domain metric. "
        f"This is the ONLY ACC that judges realtime predictions (SODA hindcast does not transfer).",
        f"{'lead':>4} {'CNN-ACC':>8} {'Persist':>8} {'gap':>7}",
    ]
    for i, lead in enumerate(res.leads):
        lines.append(f"{lead:>4} {res.cnn_acc[i]:>8.3f} {res.persistence_acc[i]:>8.3f} {res.skill_gap[i]:>+7.3f}")
    above_pers = [res.leads[i] for i in range(len(res.leads)) if res.skill_gap[i] > 0]
    above_05 = [res.leads[i] for i in range(len(res.leads)) if res.cnn_acc[i] >= 0.5]
    lines.append(f"CNN beats Persistence at leads={above_pers}.")
    lines.append(f"CNN ACC>=0.5 at leads={above_05}.")
    return "\n".join(lines)


def save_realtime_hindcast(res: RealtimeHindcastResult, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "eval_period": res.eval_period,
        "n_windows": res.n_windows,
        "leads": res.leads,
        "cnn_acc": res.cnn_acc,
        "persistence_acc": res.persistence_acc,
        "skill_gap": res.skill_gap,
        "metric": "all-season ACC on realtime (OISST/GODAS/NCEP) fields — cross-domain",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
