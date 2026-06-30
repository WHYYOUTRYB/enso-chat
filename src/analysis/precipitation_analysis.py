from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.enso_phase import add_enso_phase


@dataclass(frozen=True)
class PrecipitationAnalysisResult:
    summary: dict
    figure_path: Path


def analyze_precipitation_by_enso_phase(
    enso: pd.DataFrame,
    precipitation: pd.DataFrame,
    output_dir: Path,
) -> PrecipitationAnalysisResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    enso_phase = add_enso_phase(enso, value_col="nino34")[["date", "enso_phase"]]
    merged = precipitation.merge(enso_phase, on="date", how="inner")
    if merged.empty:
        raise ValueError("No overlapping dates between ENSO and precipitation data")

    stats = (
        merged.groupby("enso_phase")["precip_anomaly"]
        .agg(["mean", "std", "count"])
        .round(4)
        .to_dict(orient="index")
    )

    figure_path = output_dir / "precipitation_by_phase.png"
    ordered = [phase for phase in ["La Niña", "Neutral", "El Niño"] if phase in merged["enso_phase"].unique()]
    data = [merged.loc[merged["enso_phase"] == phase, "precip_anomaly"] for phase in ordered]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=ordered, patch_artist=True)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Precipitation anomaly by ENSO phase")
    ax.set_xlabel("ENSO phase")
    ax.set_ylabel("Precipitation anomaly")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)

    return PrecipitationAnalysisResult(
        summary={
            "phase_statistics": stats,
            "interpretation": "The sample precipitation anomaly is summarized by ENSO phase for course demonstration.",
        },
        figure_path=figure_path,
    )
