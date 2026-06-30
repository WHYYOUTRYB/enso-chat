from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.enso_phase import add_enso_phase


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_enso_timeseries(enso: pd.DataFrame, output_dir: Path) -> Path:
    _prepare_output_dir(output_dir)
    path = output_dir / "enso_timeseries.png"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(enso["date"]), enso["nino34"], color="#1f77b4", linewidth=1.4)
    ax.axhline(0.5, color="#d62728", linestyle="--", linewidth=1.0, label="El Niño threshold")
    ax.axhline(-0.5, color="#2ca02c", linestyle="--", linewidth=1.0, label="La Niña threshold")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Sample Niño3.4 Index")
    ax.set_xlabel("Date")
    ax.set_ylabel("Niño3.4 anomaly")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_observed_vs_predicted(
    predictions: pd.DataFrame,
    output_dir: Path,
    lead: int,
    model: str,
) -> Path:
    _prepare_output_dir(output_dir)
    path = output_dir / "enso_observed_vs_predicted.png"
    subset = predictions[(predictions["lead"] == lead) & (predictions["model"] == model)].copy()
    subset["date"] = pd.to_datetime(subset["date"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(subset["date"], subset["observed"], label="Observed", linewidth=1.6)
    ax.plot(subset["date"], subset["predicted"], label="Predicted", linewidth=1.6)
    ax.set_title(f"Observed vs Predicted Niño3.4, lead={lead}, model={model}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Niño3.4 anomaly")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_enso_rmse_by_model(results: dict, output_dir: Path) -> Path:
    _prepare_output_dir(output_dir)
    path = output_dir / "enso_rmse_by_model.png"

    rows = []
    for lead, models in results["leads"].items():
        for model_name, metrics in models.items():
            rows.append({"lead": str(lead), "model": model_name, "rmse": metrics["rmse"]})
    frame = pd.DataFrame(rows)

    labels = [f"L{row.lead}-{row.model}" for row in frame.itertuples()]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(labels, frame["rmse"], color="#4c78a8")
    ax.set_title("RMSE by model and lead time")
    ax.set_xlabel("Lead and model")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_enso_phase_timeline(enso: pd.DataFrame, output_dir: Path) -> Path:
    _prepare_output_dir(output_dir)
    path = output_dir / "enso_phase_timeline.png"
    phased = add_enso_phase(enso, value_col="nino34")
    colors = {"El Niño": "#d62728", "La Niña": "#2ca02c", "Neutral": "#7f7f7f"}

    fig, ax = plt.subplots(figsize=(10, 2.6))
    for phase, group in phased.groupby("enso_phase"):
        ax.scatter(
            pd.to_datetime(group["date"]),
            group["nino34"],
            label=phase,
            s=12,
            color=colors[phase],
        )
    ax.axhline(0.5, color="#d62728", linestyle="--", linewidth=0.8)
    ax.axhline(-0.5, color="#2ca02c", linestyle="--", linewidth=0.8)
    ax.set_title("ENSO phase classification")
    ax.set_xlabel("Date")
    ax.set_ylabel("Niño3.4")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
