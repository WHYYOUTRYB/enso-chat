"""Test the extracted forecast-on-enso core used by both load_enso_data and
load_user_enso tools."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.sample_generator import generate_sample_enso
from src.pipeline.run_enso_forecast import run_forecast_on_enso


def _enso_with_exog(periods: int = 540):
    enso = generate_sample_enso(periods=periods)
    rng = np.random.default_rng(7)
    enso["soi"] = rng.normal(0, 1, len(enso))
    enso["nino12"] = rng.normal(0, 1, len(enso))
    return enso


def test_run_forecast_on_enso_produces_results_and_predictions(tmp_path: Path):
    enso = generate_sample_enso()
    outputs_dir = tmp_path / "reports" / "outputs"
    results, results_path, predictions_path = run_forecast_on_enso(
        enso, outputs_dir=outputs_dir, data_source_info={"used": "user", "fallback_reason": None}
    )
    assert results_path.exists()
    assert predictions_path.exists()
    assert set(results["leads"]) == {"1", "3", "6"}
    assert "best_model_by_lead" in results
    assert "latest_forecast" in results
    assert results["data_source"]["used"] == "user"
    # predictions CSV has rows
    preds = pd.read_csv(predictions_path)
    assert len(preds) > 0
    assert {"date", "lead", "model", "observed", "predicted"} <= set(preds.columns)


def test_run_forecast_on_enso_is_deterministic(tmp_path: Path):
    enso = generate_sample_enso()
    out1 = run_forecast_on_enso(enso, outputs_dir=tmp_path / "a", data_source_info={"used": "user", "fallback_reason": None})
    out2 = run_forecast_on_enso(enso, outputs_dir=tmp_path / "b", data_source_info={"used": "user", "fallback_reason": None})
    assert out1[0]["latest_forecast"] == out2[0]["latest_forecast"]


def test_metrics_include_acc(tmp_path: Path):
    """Every model's metrics now carry an ACC (anomaly correlation) field."""
    enso = generate_sample_enso()
    results, _, _ = run_forecast_on_enso(
        enso, outputs_dir=tmp_path, data_source_info={"used": "user", "fallback_reason": None}
    )
    for lead, models in results["leads"].items():
        for model, metrics in models.items():
            assert "acc" in metrics
            assert -1.0 <= metrics["acc"] <= 1.0


def test_exog_cols_runs_and_records(tmp_path: Path):
    """The enhanced path trains with exogenous lags and records exog_cols."""
    enso = _enso_with_exog()
    results, _, _ = run_forecast_on_enso(
        enso, outputs_dir=tmp_path, data_source_info={"used": "enhanced", "fallback_reason": None},
        exog_cols=["soi", "nino12"],
    )
    assert results["exog_cols"] == ["soi", "nino12"]
    # ACC still present on the enhanced run.
    assert "acc" in results["leads"]["1"]["random_forest"]

