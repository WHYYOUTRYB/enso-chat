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
