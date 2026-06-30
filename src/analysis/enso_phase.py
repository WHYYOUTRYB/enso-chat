from __future__ import annotations

import pandas as pd


def classify_enso_phase(value: float) -> str:
    if value >= 0.5:
        return "El Niño"
    if value <= -0.5:
        return "La Niña"
    return "Neutral"


def add_enso_phase(df: pd.DataFrame, value_col: str = "nino34") -> pd.DataFrame:
    if value_col not in df.columns:
        raise ValueError(f"Column not found for ENSO phase classification: {value_col}")
    result = df.copy()
    result["enso_phase"] = result[value_col].map(classify_enso_phase)
    return result
