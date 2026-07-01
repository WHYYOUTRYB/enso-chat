from __future__ import annotations

import numpy as np
import pandas as pd


def make_enso_supervised_table(
    df: pd.DataFrame,
    leads: tuple[int, ...] = (1, 3, 6),
    max_lag: int = 12,
    target_col: str = "nino34",
    exog_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a supervised feature table, optionally with exogenous-variable lags.

    Args:
        df: must contain ``date`` and ``target_col``; if ``exog_cols`` is given,
            those columns must also be present (caller merges them in first).
        exog_cols: extra scalar index columns (e.g. ``["soi", "nino12"]``) to
            add as lag features ``{col}_lag_0..max_lag``. ``None`` keeps the
            original Niño3.4-only behavior (backward compatible).
    """
    required = {"date", target_col}
    if exog_cols:
        required = required.union(exog_cols)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"ENSO dataframe missing required columns: {sorted(missing)}")
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1")
    if not leads:
        raise ValueError("at least one lead time is required")

    keep_cols = ["date", target_col] + (exog_cols or [])
    data = df[keep_cols].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    feature_cols: list[str] = []
    for lag in range(max_lag + 1):
        col = f"{target_col}_lag_{lag}"
        data[col] = data[target_col].shift(lag)
        feature_cols.append(col)

    data[f"{target_col}_roll_mean_3"] = data[target_col].rolling(window=3, min_periods=1).mean()
    data[f"{target_col}_roll_mean_6"] = data[target_col].rolling(window=6, min_periods=1).mean()
    feature_cols.extend([f"{target_col}_roll_mean_3", f"{target_col}_roll_mean_6"])

    # Exogenous index lags — these are the precursors (SOI atmosphere, Niño1+2
    # eastern-Pacific ocean) that let the model see beyond Niño3.4's own memory.
    if exog_cols:
        for exog in exog_cols:
            for lag in range(max_lag + 1):
                col = f"{exog}_lag_{lag}"
                data[col] = data[exog].shift(lag)
                feature_cols.append(col)

    month = data["date"].dt.month.astype(float)
    data["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    feature_cols.extend(["month_sin", "month_cos"])

    target_cols: list[str] = []
    for lead in leads:
        if lead < 1:
            raise ValueError("lead times must be positive integers")
        target_name = f"target_lead_{lead}"
        data[target_name] = data[target_col].shift(-lead)
        target_cols.append(target_name)

    clean = data.dropna(subset=feature_cols + target_cols).reset_index(drop=True)
    return clean, feature_cols
