from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def temporal_train_test_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if len(df) < 5:
        raise ValueError("at least 5 rows are required for a temporal split")

    split_index = int(round(len(df) * (1.0 - test_fraction)))
    split_index = min(max(split_index, 1), len(df) - 1)
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
    return train, test


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = mean_absolute_error(y_true, y_pred)
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])

    return {"rmse": float(rmse), "mae": float(mae), "corr": corr}


def calculate_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Anomaly Correlation Coefficient (ACC).

    The ENSO-forecast standard skill score: Pearson correlation of the
    *anomalies* (each series minus its own mean). Equivalent to ``corr`` for a
    single lead when the means are non-zero; named separately to match the
    operational-forecast literature and to make reports unambiguous.

    Returns 0.0 when either series has zero variance.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(np.corrcoef(y_true - y_true.mean(), y_pred - y_pred.mean())[0, 1])


def per_lead_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, leads: range | tuple[int, ...] | None = None
) -> dict[str, dict[str, float]]:
    """RMSE/MAE/ACC for each lead column independently.

    Args:
        y_true, y_pred: shape ``(n_samples, n_leads)`` — one column per lead.
        leads: lead numbers labeling each column. Defaults to ``1..n_leads``.

    Returns:
        ``{"1": {"rmse":..., "mae":..., "acc":...}, "2": {...}, ...}`` keyed by
        lead (as string, for JSON compatibility with ``enso_results.json``).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 2 or y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must share a 2D (n_samples, n_leads) shape")
    n_leads = y_true.shape[1]
    if leads is None:
        leads = range(1, n_leads + 1)
    leads = tuple(leads)
    if len(leads) != n_leads:
        raise ValueError(f"leads has {len(leads)} entries but data has {n_leads} lead columns")

    out: dict[str, dict[str, float]] = {}
    for i, lead in enumerate(leads):
        m = calculate_regression_metrics(y_true[:, i], y_pred[:, i])
        out[str(int(lead))] = {"rmse": m["rmse"], "mae": m["mae"], "acc": calculate_acc(y_true[:, i], y_pred[:, i])}
    return out
