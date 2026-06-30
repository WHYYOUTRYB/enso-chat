from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_enso(start: str = "1980-01-01", periods: int = 540) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(start=start, periods=periods, freq="MS")
    t = np.arange(periods, dtype=float)

    low_frequency = 0.9 * np.sin(2 * np.pi * t / 48.0)
    decadal_component = 0.25 * np.sin(2 * np.pi * t / 132.0)
    seasonal_component = 0.12 * np.sin(2 * np.pi * t / 12.0)
    noise = rng.normal(loc=0.0, scale=0.18, size=periods)
    nino34 = low_frequency + decadal_component + seasonal_component + noise

    return pd.DataFrame({"date": dates, "nino34": np.round(nino34, 3)})


def generate_sample_precipitation(enso: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(43)
    nino = enso["nino34"].to_numpy(dtype=float)
    lagged_nino = pd.Series(nino).shift(2).fillna(0.0).to_numpy(dtype=float)
    noise = rng.normal(loc=0.0, scale=0.35, size=len(enso))
    precip_anomaly = 0.45 * lagged_nino + noise

    return pd.DataFrame(
        {
            "date": pd.to_datetime(enso["date"]),
            "precip_anomaly": np.round(precip_anomaly, 3),
        }
    )


def generate_sample_tide(start: str = "2024-01-01", periods: int = 14 * 24) -> pd.DataFrame:
    rng = np.random.default_rng(44)
    timestamps = pd.date_range(start=start, periods=periods, freq="h")
    hours = np.arange(periods, dtype=float)

    semidiurnal = 0.85 * np.sin(2 * np.pi * hours / 12.42)
    diurnal = 0.25 * np.sin(2 * np.pi * hours / 24.0)
    meteorological_noise = rng.normal(loc=0.0, scale=0.05, size=periods)
    water_level = semidiurnal + diurnal + meteorological_noise

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "water_level_m": np.round(water_level, 3),
        }
    )


def write_sample_datasets(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    enso = generate_sample_enso()
    precipitation = generate_sample_precipitation(enso)
    tide = generate_sample_tide()

    paths = {
        "enso": output_dir / "sample_enso.csv",
        "precipitation": output_dir / "sample_precipitation.csv",
        "tide": output_dir / "sample_tide.csv",
    }

    enso.to_csv(paths["enso"], index=False)
    precipitation.to_csv(paths["precipitation"], index=False)
    tide.to_csv(paths["tide"], index=False)

    return paths
