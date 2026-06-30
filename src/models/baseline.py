from __future__ import annotations

import numpy as np
import pandas as pd


def persistence_predict(df: pd.DataFrame, current_col: str = "nino34_lag_0") -> np.ndarray:
    if current_col not in df.columns:
        raise ValueError(f"Persistence column not found: {current_col}")
    return df[current_col].to_numpy(dtype=float)
