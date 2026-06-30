"""Test the extracted forecast-on-enso core used by both load_enso_data and
load_user_enso tools."""

from pathlib import Path

import pandas as pd

from src.data.sample_generator import generate_sample_enso
from src.pipeline.run_enso_forecast import run_forecast_on_enso


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
