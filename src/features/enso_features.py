from __future__ import annotations

import numpy as np
import pandas as pd


def make_enso_supervised_table(
    df: pd.DataFrame,
    leads: tuple[int, ...] = (1, 3, 6),
    max_lag: int = 12,
    target_col: str = "nino34",
) -> tuple[pd.DataFrame, list[str]]:
    required = {"date", target_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"ENSO dataframe missing required columns: {sorted(missing)}")
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1")
    if not leads:
        raise ValueError("at least one lead time is required")

    data = df[["date", target_col]].copy()
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
