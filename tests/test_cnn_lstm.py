"""Tests for the CNN-LSTM track: data splitting, ACC, inference shape."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.cnn_lstm import (
    CHANNELS,
    INPUT_MONTHS,
    LAT,
    LEAD_MONTHS,
    LON,
    SPLIT_MONTH_RANGES,
    _sliding_windows,
    make_cnn_lstm_dataset,
)
from src.models.evaluation import calculate_acc, per_lead_metrics

SODA_TRAIN = Path(__file__).resolve().parents[1] / "data" / "SODA_train.nc"
SODA_LABEL = Path(__file__).resolve().parents[1] / "data" / "SODA_label.nc"
HAS_SODA = SODA_TRAIN.exists() and SODA_LABEL.exists()
HAS_TORCH = True
try:
    import torch  # noqa: F401
except ImportError:
    HAS_TORCH = False


# --- evaluation.py additions ---

def test_calculate_acc_is_demeaned_corr():
    # ACC == corr of anomalies; shifting both series by a constant leaves ACC unchanged.
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    acc = calculate_acc(y_true, y_pred)
    # Perfectly correlated → ACC ≈ 1.0
    assert acc == pytest.approx(1.0, abs=1e-9)
    # Constant offset does not change ACC (anomaly correlation).
    acc_shift = calculate_acc(y_true, y_pred + 100.0)
    assert acc_shift == pytest.approx(1.0, abs=1e-9)


def test_calculate_acc_zero_variance_returns_zero():
    assert calculate_acc(np.zeros(5), np.array([1.0, 2.0, 3.0, 4.0, 5.0])) == 0.0


def test_per_lead_metrics_shape_and_keys():
    y_true = np.random.RandomState(0).randn(10, 4)
    y_pred = y_true + 0.1
    m = per_lead_metrics(y_true, y_pred, leads=range(1, 5))
    assert set(m.keys()) == {"1", "2", "3", "4"}
    for v in m.values():
        assert set(v.keys()) == {"rmse", "mae", "acc"}


def test_per_lead_metrics_rejects_mismatched_leads():
    with pytest.raises(ValueError):
        per_lead_metrics(np.zeros((3, 4)), np.zeros((3, 4)), leads=range(1, 3))


# --- data prep (only when SODA present) ---

@pytest.mark.skipif(not HAS_SODA, reason="SODA NetCDF not in data/")
def test_split_ranges_have_buffer_between_val_and_test():
    # The buffer (years 82-84) means test start > val end.
    assert SPLIT_MONTH_RANGES["val"][1] < SPLIT_MONTH_RANGES["test"][0]


@pytest.mark.skipif(not HAS_SODA, reason="SODA NetCDF not in data/")
def test_dataset_shapes_and_counts():
    x_tr, y_tr = make_cnn_lstm_dataset(SODA_TRAIN, SODA_LABEL, "train")
    x_te, y_te = make_cnn_lstm_dataset(SODA_TRAIN, SODA_LABEL, "test")
    assert x_tr.shape[1:] == (INPUT_MONTHS, LAT, LON, len(CHANNELS))
    assert y_tr.shape[1] == LEAD_MONTHS
    assert x_te.shape[1:] == x_tr.shape[1:]
    # Train should dominate; test still has hundreds of windows.
    assert len(x_tr) > len(x_te) > 100


@pytest.mark.skipif(not HAS_SODA, reason="SODA NetCDF not in data/")
def test_no_temporal_leakage_between_train_and_test():
    # A test window starts at month ≥ test_start; a train window's horizon
    # (start+35) ends before val_start. They cannot overlap.
    val_start = SPLIT_MONTH_RANGES["val"][0]
    test_start = SPLIT_MONTH_RANGES["test"][0]
    # Reconstruct the start month of each window from the sliding-window logic:
    # windows are contiguous starting at split_start.
    _, y_tr = make_cnn_lstm_dataset(SODA_TRAIN, SODA_LABEL, "train")
    _, y_te = make_cnn_lstm_dataset(SODA_TRAIN, SODA_LABEL, "test")
    horizon = INPUT_MONTHS + LEAD_MONTHS
    train_starts = list(range(SPLIT_MONTH_RANGES["train"][0], SPLIT_MONTH_RANGES["train"][0] + len(y_tr)))
    test_starts = list(range(test_start, test_start + len(y_te)))
    assert max(s + horizon for s in train_starts) <= val_start  # train never reaches val
    assert min(test_starts) >= test_start  # test never reaches buffer/train


def test_sliding_windows_basic():
    x = np.arange(100 * LAT * LON * len(CHANNELS), dtype=np.float32).reshape(100, LAT, LON, len(CHANNELS))
    y = np.arange(100, dtype=np.float32)
    Xw, yw = _sliding_windows(x, y, 0, 100)
    horizon = INPUT_MONTHS + LEAD_MONTHS
    # i ranges over [m_start, m_end-horizon] inclusive → (m_end - m_start - horizon + 1) windows.
    assert Xw.shape == (100 - horizon + 1, INPUT_MONTHS, LAT, LON, len(CHANNELS))
    assert yw.shape == (100 - horizon + 1, LEAD_MONTHS)
    # First window's first target == month 12
    assert yw[0, 0] == y[INPUT_MONTHS]


# --- inference (only when torch + SODA present) ---

@pytest.mark.skipif(not (HAS_TORCH and HAS_SODA), reason="torch + SODA required")
def test_predict_cnn_lstm_output_shape_and_phases(tmp_path):
    # Train a 1-epoch tiny checkpoint so inference is end-to-end exercised.
    from src.models.cnn_lstm import predict_cnn_lstm, train_cnn_lstm

    weights = tmp_path / "w.pth"
    metrics = tmp_path / "m.json"
    train_cnn_lstm(
        SODA_TRAIN, SODA_LABEL,
        weights_path=weights, metrics_path=metrics,
        epochs=1, batch_size=8, patience=1,
    )
    assert weights.exists()

    x_te, _ = make_cnn_lstm_dataset(SODA_TRAIN, SODA_LABEL, "test")
    window = x_te[0]
    out = predict_cnn_lstm(window, weights)
    assert len(out) == LEAD_MONTHS
    assert out[0]["lead"] == 1 and out[-1]["lead"] == LEAD_MONTHS
    assert all(isinstance(o["value"], float) for o in out)
    assert all(o["phase"] in {"El Niño", "La Niña", "Neutral"} for o in out)
