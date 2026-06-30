from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_enso_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "nino34"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"ENSO CSV missing required columns: {sorted(missing)}")
    return df.sort_values("date").reset_index(drop=True)


def load_precipitation_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "precip_anomaly"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Precipitation CSV missing required columns: {sorted(missing)}")
    return df.sort_values("date").reset_index(drop=True)


def load_tide_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "water_level_m"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Tide CSV missing required columns: {sorted(missing)}")
    return df.sort_values("timestamp").reset_index(drop=True)
