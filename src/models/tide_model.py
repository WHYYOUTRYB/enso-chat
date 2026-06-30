from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.models.evaluation import calculate_regression_metrics


@dataclass(frozen=True)
class TidePredictionResult:
    metrics: dict[str, float]
    predictions: pd.DataFrame
    figure_path: Path


def _make_tide_features(tide: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = tide.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    elapsed_hours = (data["timestamp"] - data["timestamp"].min()).dt.total_seconds() / 3600.0

    data["sin_12h"] = np.sin(2 * np.pi * elapsed_hours / 12.42)
    data["cos_12h"] = np.cos(2 * np.pi * elapsed_hours / 12.42)
    data["sin_24h"] = np.sin(2 * np.pi * elapsed_hours / 24.0)
    data["cos_24h"] = np.cos(2 * np.pi * elapsed_hours / 24.0)

    return data, ["sin_12h", "cos_12h", "sin_24h", "cos_24h"]


def run_tide_demo_prediction(tide: pd.DataFrame, output_dir: Path) -> TidePredictionResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    data, feature_cols = _make_tide_features(tide)
    split_index = int(round(len(data) * 0.75))
    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()

    model = Ridge(alpha=0.1)
    model.fit(train[feature_cols], train["water_level_m"])
    y_pred = model.predict(test[feature_cols])
    metrics = calculate_regression_metrics(test["water_level_m"].to_numpy(), y_pred)
    metrics = {key: round(value, 4) for key, value in metrics.items()}

    predictions = test[["timestamp", "water_level_m"]].copy()
    predictions["predicted_water_level_m"] = y_pred

    figure_path = output_dir / "tide_prediction.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(predictions["timestamp"], predictions["water_level_m"], label="Observed", linewidth=1.5)
    ax.plot(
        predictions["timestamp"],
        predictions["predicted_water_level_m"],
        label="Predicted",
        linewidth=1.5,
    )
    ax.set_title("Tide demonstration prediction")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Water level (m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)

    return TidePredictionResult(metrics=metrics, predictions=predictions, figure_path=figure_path)
