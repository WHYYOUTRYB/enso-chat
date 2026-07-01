"""CNN-LSTM ENSO forecaster on SODA spatial fields.

A second prediction track alongside the scalar Ridge/RF models in
``enso_ml.py``. Inputs are 12-month windows of 4 spatial channels
(sst, t300, ua, va) on a 24x72 grid; the model emits 24 monthly Niño3.4
leads in one pass. Architecture follows ``参考/enso_cmip_soda.ipynb``:
per-timestep CNN feature extractor → 2-layer LSTM → FC(24).

Design notes (see plan ``drifting-sparking-tiger.md``):
- **Training is offline** (``scripts/train_cnn_lstm.py``); the Streamlit
  process only ever calls :func:`predict_cnn_lstm` for a forward pass on CPU.
- **Train/val/test split leaves a buffer** (years 82-84 unused) so the test
  windows (start year ≥85) cannot leak training-tail information.
- **Standardization uses train-set statistics only** (stored in the checkpoint)
  — the Rain_CN convention to avoid test-set leakage.
- SODA is a reanalysis with a fixed end month; the online tool therefore
  reports forecasts as ``source=SODA末端窗口`` (not real-time spatial fields).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# torch / xarray are imported lazily inside the functions that need them so
# that importing this module (for the ToolRegistry / tests) does not hard-require
# the heavy training stack. ``predict_cnn_lstm`` raises a clear error if torch
# is missing at call time.

CHANNELS = ("sst", "t300", "ua", "va")
LEAD_MONTHS = 24
INPUT_MONTHS = 12
LAT, LON = 24, 72

# SODA stores 36 months per "year" block (really a 3-year block); 100 blocks
# give a 3600-month continuous series. We slide a true 12-month input window
# over it and predict the next 24 months. Splits are by continuous month index
# with a buffer so test windows cannot overlap the train tail.
_MONTHS_PER_BLOCK = 36
SPLIT_MONTH_RANGES: dict[str, tuple[int, int]] = {
    "train": (0, 70 * _MONTHS_PER_BLOCK),         # months 0..2519
    "val": (70 * _MONTHS_PER_BLOCK, 82 * _MONTHS_PER_BLOCK),     # 2520..2951
    # buffer 2952..3059 (years 82-84) deliberately unused
    "test": (85 * _MONTHS_PER_BLOCK, 100 * _MONTHS_PER_BLOCK),   # 3060..3599
}


@dataclass(frozen=True)
class CnnLstmCheckpoint:
    """Metadata + weights bundle written to ``weights/cnn_lstm_soda.pth``."""

    state_dict: dict
    x_mean: np.ndarray  # shape (4,) per-channel train mean
    x_std: np.ndarray  # shape (4,) per-channel train std
    best_val_loss: float
    lead_months: int = LEAD_MONTHS


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _open_soda(train_path: Path, label_path: Path):
    """Open the two SODA NetCDF files (handles the Chinese-path quirk).

    netCDF4 fails on non-ASCII paths under Windows; copying to a temp ASCII
    path sidesteps it. Returns ``(train_ds, label_ds, tmp_dir)``.
    """
    import shutil
    import tempfile

    import xarray as xr

    tmp = Path(tempfile.mkdtemp(prefix="soda_"))
    try:
        train_copy = tmp / "train.nc"
        label_copy = tmp / "label.nc"
        shutil.copy(train_path, train_copy)
        shutil.copy(label_path, label_copy)
        train_ds = xr.open_dataset(train_copy)
        label_ds = xr.open_dataset(label_copy)
        return train_ds, label_ds, tmp
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _stack_soda_arrays(train_ds, label_ds) -> tuple[np.ndarray, np.ndarray]:
    """Stack SODA into a continuous monthly series.

    Returns X ``(total_months, lat, lon, 4)`` and y ``(total_months,)`` — the
    "year" block dimension is flattened so a true sliding window can be applied.
    """
    blocks = [np.asarray(train_ds[ch].values, dtype=np.float32) for ch in CHANNELS]
    # each block: (year_block, month, lat, lon) → flatten year_block*month
    blocks = [b.reshape(-1, LAT, LON) for b in blocks]
    x = np.stack(blocks, axis=-1)  # (total_months, lat, lon, 4)
    y = np.asarray(label_ds["nino"].values, dtype=np.float32).reshape(-1)  # (total_months,)
    x = np.nan_to_num(x, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    return x, y


def _sliding_windows(x: np.ndarray, y: np.ndarray, m_start: int, m_end: int):
    """Slide 12-month input windows whose full horizon (in+out=36 months)
    stays within [m_start, m_end).

    A window starting at month ``i`` covers input [i, i+11] and targets
    [i+12, i+35]; require ``i+35 < m_end`` and ``i >= m_start``.
    """
    horizon = INPUT_MONTHS + LEAD_MONTHS  # 36
    X_list, y_list = [], []
    i = m_start
    while i + horizon <= m_end and i + horizon <= len(y):
        X_list.append(x[i : i + INPUT_MONTHS])
        y_list.append(y[i + INPUT_MONTHS : i + INPUT_MONTHS + LEAD_MONTHS])
        i += 1
    if not X_list:
        return np.empty((0, INPUT_MONTHS, LAT, LON, len(CHANNELS)), dtype=np.float32), np.empty((0, LEAD_MONTHS), dtype=np.float32)
    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


def make_cnn_lstm_dataset(
    train_path: Path,
    label_path: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(X, y)`` for one split.

    Args:
        train_path, label_path: SODA NetCDF paths.
        split: one of ``"train"`` / ``"val"`` / ``"test"``.

    Returns:
        X ``(n, 12, 24, 72, 4)`` float32, y ``(n, 24)`` float32.
    """
    if split not in SPLIT_MONTH_RANGES:
        raise ValueError(f"split must be one of {sorted(SPLIT_MONTH_RANGES)}, got {split!r}")
    train_ds, label_ds, tmp = _open_soda(train_path, label_path)
    try:
        x, y = _stack_soda_arrays(train_ds, label_ds)
    finally:
        import shutil

        train_ds.close()
        label_ds.close()
        shutil.rmtree(tmp, ignore_errors=True)
    m_start, m_end = SPLIT_MONTH_RANGES[split]
    return _sliding_windows(x, y, m_start, m_end)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _build_model():
    """CNN-LSTM matching ``参考/enso_cmip_soda.ipynb`` Model."""
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(len(CHANNELS), 16, kernel_size=7, stride=2, padding=3)
            self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
            self.bn = nn.BatchNorm2d(16)
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            # After conv1(stride2)+avgpool(2): 24->12->6, 72->36->18 → 6*18*16 = 1728
            self.lstm1 = nn.LSTM(1728, 1024, batch_first=True)
            self.lstm2 = nn.LSTM(1024, 256, batch_first=True)
            self.fc = nn.Linear(256, LEAD_MONTHS)
            self.dropout = nn.Dropout(0.7)

        def forward(self, x):
            n, t, h, w, c = x.shape
            x = x.permute(0, 1, 4, 2, 3).contiguous()
            x = x.view(n * t, c, h, w)
            x = self.conv1(x)
            x = F.relu(self.bn(x))
            x = self.dropout(self.conv2(x))
            x = F.relu(self.bn(x))
            x = self.avgpool(x)
            x = self.flatten(x)
            _, c_new = x.shape
            x = x.view(n, t, c_new)
            x, _ = self.lstm1(x)
            x, _ = self.lstm2(x)
            x = self.fc(x)  # (n, t, 24)
            return x

    return Model()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_cnn_lstm(
    train_path: Path,
    label_path: Path,
    *,
    weights_path: Path,
    metrics_path: Path | None = None,
    epochs: int = 80,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.001,
    batch_size: int = 8,
    seed: int = 42,
) -> dict:
    """Train the CNN-LSTM on SODA and write the checkpoint + per-lead metrics.

    Returns the per-lead test metrics dict (also written to ``metrics_path``).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.evaluation import per_lead_metrics

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_tr, y_tr = make_cnn_lstm_dataset(train_path, label_path, "train")
    x_va, y_va = make_cnn_lstm_dataset(train_path, label_path, "val")
    x_te, y_te = make_cnn_lstm_dataset(train_path, label_path, "test")

    # Per-channel standardization using TRAIN statistics only.
    x_mean = x_tr.reshape(-1, len(CHANNELS)).mean(axis=0)
    x_std = x_tr.reshape(-1, len(CHANNELS)).std(axis=0)
    x_std = np.where(x_std == 0, 1.0, x_std)

    def norm(x):
        return (x - x_mean) / x_std

    def ds(x, y):
        return TensorDataset(torch.tensor(norm(x), dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

    train_loader = DataLoader(ds(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds(x_va, y_va), batch_size=batch_size, shuffle=False)

    model = _build_model().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_val = float("inf")
    best_state = None
    counter = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)[:, -1, :]  # last timestep → 24 leads
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        vloss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)[:, -1, :]
                vloss += criterion(pred, yb).item()
                n += 1
        vloss = vloss / max(n, 1)
        scheduler.step(vloss)
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test-set per-lead metrics.
    model.eval()
    with torch.no_grad():
        x_te_t = torch.tensor(norm(x_te), dtype=torch.float32).to(device)
        pred_te = model(x_te_t)[:, -1, :].cpu().numpy()
    metrics = per_lead_metrics(y_te, pred_te, leads=range(1, LEAD_MONTHS + 1))

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state or model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
            "best_val_loss": best_val,
            "lead_months": LEAD_MONTHS,
        },
        weights_path,
    )
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": "cnn_lstm",
            "data_source": "SODA",
            "split_month_ranges": {k: list(v) for k, v in SPLIT_MONTH_RANGES.items()},
            "n_samples": {"train": int(len(x_tr)), "val": int(len(x_va)), "test": int(len(x_te))},
            "best_val_loss": best_val,
            "per_lead_metrics": metrics,
        }
        metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_cnn_lstm(window: np.ndarray, weights_path: Path) -> list[dict]:
    """Run a forward pass on one 12-month spatial window.

    Args:
        window: ``(12, 24, 72, 4)`` float array (sst/t300/ua/va).
        weights_path: checkpoint written by :func:`train_cnn_lstm`.

    Returns:
        list of 24 ``{lead, value, phase}`` dicts (lead 1..24).
    """
    from src.analysis.enso_phase import classify_enso_phase

    import torch

    if not weights_path.exists():
        raise FileNotFoundError(f"CNN-LSTM weights not found: {weights_path}. Run scripts/train_cnn_lstm.py first.")

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    x_std = np.where(x_std == 0, 1.0, x_std)

    model = _build_model()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x = np.asarray(window, dtype=np.float32)
    if x.shape != (INPUT_MONTHS, LAT, LON, len(CHANNELS)):
        raise ValueError(f"window must be shape (12,24,72,4), got {x.shape}")
    x = (x - x_mean) / x_std
    with torch.no_grad():
        inp = torch.tensor(x[None], dtype=torch.float32)  # (1,12,24,72,4)
        pred = model(inp)[0, -1, :].cpu().numpy()  # (24,)

    return [
        {"lead": i + 1, "value": round(float(v), 4), "phase": classify_enso_phase(float(v))}
        for i, v in enumerate(pred)
    ]


def load_soda_tail_window(train_path: Path, label_path: Path | None = None, n_months: int = INPUT_MONTHS) -> tuple[np.ndarray, str]:
    """Return the last ``n_months`` SODA months as a spatial window + its end label.

    Used by the online ``forecast_cnn_lstm`` tool. SODA's final month is a
    fixed reanalysis point, so the returned label reflects the data, not "now".
    ``label_path`` is accepted for signature symmetry but not required.
    """
    import shutil
    import tempfile

    import xarray as xr

    tmp = Path(tempfile.mkdtemp(prefix="soda_"))
    try:
        train_copy = tmp / "train.nc"
        shutil.copy(train_path, train_copy)
        train_ds = xr.open_dataset(train_copy)
        # Stack only the spatial channels (y not needed for inference).
        blocks = [np.asarray(train_ds[ch].values, dtype=np.float32) for ch in CHANNELS]
        blocks = [b.reshape(-1, LAT, LON) for b in blocks]
        x = np.nan_to_num(np.stack(blocks, axis=-1), nan=0.0)  # (total_months, lat, lon, 4)
        train_ds.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    window = x[-n_months:]
    n_total = x.shape[0]
    block = (n_total - 1) // _MONTHS_PER_BLOCK + 1
    within = ((n_total - 1) % _MONTHS_PER_BLOCK) + 1
    return window, f"SODA_block{block}_month{within}"


def predict_cnn_lstm_realtime(window: np.ndarray, weights_path: Path) -> list[dict]:
    """Run a forward pass on a realtime (cross-domain) spatial window.

    Identical forward pass to :func:`predict_cnn_lstm`, but the caller is
    responsible for having anomalized+resampled the realtime window onto the
    SODA grid first (see ``src/data/realtime_fetch.py``). The checkpoint's
    train-set x_mean/x_std are still applied for SODA-distribution standardization.

    The caller MUST label results as cross-domain — realtime fields come from
    OISST/GODAS/NCEP, not SODA, so SODA-hindcast ACC does not transfer. This
    function returns only the 24 lead {value, phase} dicts; the labeling is the
    tool layer's responsibility.
    """
    # Same mechanics as predict_cnn_lstm; kept separate so the call site reads
    # as "realtime" and the cross-domain caveat is documented at the boundary.
    from src.analysis.enso_phase import classify_enso_phase

    import torch

    if not weights_path.exists():
        raise FileNotFoundError(f"CNN-LSTM weights not found: {weights_path}. Run scripts/train_cnn_lstm.py first.")

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    x_std = np.where(x_std == 0, 1.0, x_std)

    model = _build_model()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x = np.asarray(window, dtype=np.float32)
    if x.shape != (INPUT_MONTHS, LAT, LON, len(CHANNELS)):
        raise ValueError(f"window must be shape (12,24,72,4), got {x.shape}")
    x = (x - x_mean) / x_std
    with torch.no_grad():
        inp = torch.tensor(x[None], dtype=torch.float32)
        pred = model(inp)[0, -1, :].cpu().numpy()

    return [
        {"lead": i + 1, "value": round(float(v), 4), "phase": classify_enso_phase(float(v))}
        for i, v in enumerate(pred)
    ]
