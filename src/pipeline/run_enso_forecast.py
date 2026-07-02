from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.analysis.enso_phase import classify_enso_phase
from src.config import (
    DEFAULT_LEADS,
    NOAA_ENSO_PROCESSED_PATH,
    NOAA_ENSO_RAW_PATH,
    OUTPUTS_DIR,
    SAMPLE_DATA_DIR,
)
from src.data.loaders import load_enso_csv
from src.data.noaa_enso import NoaaEnsoDownloadError, load_or_download_noaa_enso
from src.data.sample_generator import (
    generate_sample_precipitation,
    generate_sample_tide,
    write_sample_datasets,
)
from src.features.enso_features import make_enso_supervised_table
from src.models.baseline import persistence_predict
from src.models.enso_ml import (
    build_model_suite,
    fit_models_for_latest_forecast,
    train_and_predict_for_lead,
)
from src.models.evaluation import calculate_acc, calculate_regression_metrics, temporal_train_test_split


@dataclass(frozen=True)
class EnsoForecastOutput:
    results: dict
    results_path: Path
    predictions_path: Path
    sample_data_dir: Path
    enso: pd.DataFrame


def _round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 4) for key, value in metrics.items()}


def _metrics_with_acc(y_true, y_pred) -> dict[str, float]:
    """RMSE/MAE/corr plus ACC — ACC is the ENSO skill standard (anomaly corr)."""
    base = calculate_regression_metrics(y_true, y_pred)
    base["acc"] = calculate_acc(y_true, y_pred)
    return base


def _ensure_precip_and_tide_samples(sample_dir: Path) -> None:
    """Make sure the sample precipitation/tide CSVs exist.

    Per the design (NOAA data spec §6), precipitation and tide modules always
    use sample data, even when ENSO comes from NOAA. Previously these CSVs were
    only written by ``write_sample_datasets`` on the sample/fallback path, so a
    successful NOAA run left them missing and crashed every downstream module.
    Writes them lazily (no overwrite) so an existing sample dir is untouched.
    """
    sample_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "precipitation": sample_dir / "sample_precipitation.csv",
        "tide": sample_dir / "sample_tide.csv",
    }
    if not paths["precipitation"].exists() or not paths["tide"].exists():
        enso = load_enso_csv(write_sample_datasets(sample_dir)["enso"])
        write_sample_precip_and_tide(sample_dir, enso)


def write_sample_precip_and_tide(sample_dir: Path, enso: pd.DataFrame) -> None:
    """Write only the precipitation/tide sample CSVs derived from ``enso``."""
    generate_sample_precipitation(enso).to_csv(sample_dir / "sample_precipitation.csv", index=False)
    generate_sample_tide().to_csv(sample_dir / "sample_tide.csv", index=False)


def _resolve_enso_data(
    sample_dir: Path,
    data_source: str,
    refresh_noaa: bool,
    base_dir: Path | None,
    *,
    timings: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    allowed = {"auto", "noaa", "sample"}
    if data_source not in allowed:
        raise ValueError(f"data_source must be one of {sorted(allowed)}")

    if base_dir is None:
        noaa_raw_path = NOAA_ENSO_RAW_PATH
        noaa_processed_path = NOAA_ENSO_PROCESSED_PATH
    else:
        noaa_raw_path = base_dir / "data" / "raw" / "noaa_nino34_raw.txt"
        noaa_processed_path = base_dir / "data" / "processed" / "noaa_nino34.csv"

    if data_source in {"auto", "noaa"}:
        try:
            enso = load_or_download_noaa_enso(
                raw_path=noaa_raw_path,
                processed_path=noaa_processed_path,
                refresh=refresh_noaa,
                timings=timings,
            )
            # Precipitation and tide always use sample data (design §6); make
            # sure those CSVs exist so downstream modules work in NOAA mode.
            _ensure_precip_and_tide_samples(sample_dir)
            return enso, {"requested": data_source, "used": "noaa", "fallback_reason": None}
        except NoaaEnsoDownloadError as exc:
            if data_source == "noaa":
                raise
            sample_paths = write_sample_datasets(sample_dir)
            enso = load_enso_csv(sample_paths["enso"])
            return enso, {
                "requested": data_source,
                "used": "sample",
                "fallback_reason": str(exc),
            }

    sample_paths = write_sample_datasets(sample_dir)
    enso = load_enso_csv(sample_paths["enso"])
    return enso, {"requested": data_source, "used": "sample", "fallback_reason": None}


def run_forecast_on_enso(
    enso: pd.DataFrame,
    *,
    outputs_dir: Path,
    data_source_info: dict,
    exog_cols: list[str] | None = None,
    timings: dict | None = None,
) -> tuple[dict, Path, Path]:
    """Run the ENSO modeling pipeline on an already-loaded ENSO DataFrame.

    Shared by ``run_enso_forecast`` (sample/NOAA) and the ``load_user_enso``
    tool (user-uploaded CSV). Builds features, trains Persistence + Ridge +
    RandomForest for 1/3/6-month leads, evaluates, and writes the results
    JSON + predictions CSV to ``outputs_dir``.

    When ``exog_cols`` is given (e.g. ``["soi", "nino12"]``), those columns
    must already be merged into ``enso`` and are added as lag features — the
    "enhanced" track. ``None`` keeps the original Niño3.4-only behavior.

    Returns ``(results, results_path, predictions_path)``. Each model's metrics
    include ``acc`` (anomaly correlation) for data-driven lead-confidence.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    import time as _time
    _t_feat = _time.perf_counter()
    table, feature_cols = make_enso_supervised_table(enso, leads=DEFAULT_LEADS, max_lag=12, exog_cols=exog_cols)
    train, test = temporal_train_test_split(table, test_fraction=0.25)
    if timings is not None:
        timings["features"] = round(_time.perf_counter() - _t_feat, 3)

    results: dict = {
        "target": "Niño3.4 index",
        "data_source": data_source_info,
        "exog_cols": list(exog_cols) if exog_cols else [],
        "leads": {},
        "best_model_by_lead": {},
        "latest_forecast": {},
    }
    prediction_rows: list[dict] = []
    _t_train = _time.perf_counter()
    for lead in DEFAULT_LEADS:
        target_col = f"target_lead_{lead}"
        y_true = test[target_col].to_numpy(dtype=float)

        lead_metrics: dict[str, dict[str, float]] = {}
        persistence_pred = persistence_predict(test)
        lead_metrics["persistence"] = _round_metrics(
            _metrics_with_acc(y_true, persistence_pred)
        )

        ml_predictions = train_and_predict_for_lead(
            models=build_model_suite(random_state=42),
            train_df=train,
            test_df=test,
            feature_cols=feature_cols,
            lead=lead,
        )
        for model_name, y_pred in ml_predictions.items():
            lead_metrics[model_name] = _round_metrics(_metrics_with_acc(y_true, y_pred))
            for date_value, observed, predicted in zip(test["date"], y_true, y_pred):
                prediction_rows.append(
                    {
                        "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                        "lead": lead,
                        "model": model_name,
                        "observed": float(observed),
                        "predicted": float(predicted),
                    }
                )

        for date_value, observed, predicted in zip(test["date"], y_true, persistence_pred):
            prediction_rows.append(
                {
                    "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                    "lead": lead,
                    "model": "persistence",
                    "observed": float(observed),
                    "predicted": float(predicted),
                }
            )

        best_model = min(lead_metrics, key=lambda name: lead_metrics[name]["rmse"])
        latest_forecasts = fit_models_for_latest_forecast(
            models=build_model_suite(random_state=42),
            table=table,
            feature_cols=feature_cols,
            lead=lead,
        )
        latest_forecasts["persistence"] = float(table.iloc[-1]["nino34_lag_0"])
        best_forecast_value = latest_forecasts[best_model]

        results["leads"][str(lead)] = lead_metrics
        results["best_model_by_lead"][str(lead)] = best_model
        results["latest_forecast"][str(lead)] = {
            "model": best_model,
            "value": round(float(best_forecast_value), 4),
            "phase": classify_enso_phase(float(best_forecast_value)),
        }

    results_path = outputs_dir / "enso_results.json"
    predictions_path = outputs_dir / "enso_predictions.csv"
    if timings is not None:
        timings["train"] = round(_time.perf_counter() - _t_train, 3)
    _t_write = _time.perf_counter()
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    if timings is not None:
        timings["write"] = round(_time.perf_counter() - _t_write, 3)

    return results, results_path, predictions_path


def run_enso_forecast(
    base_dir: Path | None = None,
    data_source: str = "auto",
    refresh_noaa: bool = False,
    *,
    timings: dict | None = None,
) -> EnsoForecastOutput:
    if base_dir is None:
        sample_dir = SAMPLE_DATA_DIR
        outputs_dir = OUTPUTS_DIR
    else:
        sample_dir = base_dir / "data" / "sample"
        outputs_dir = base_dir / "reports" / "outputs"

    enso, data_source_info = _resolve_enso_data(
        sample_dir=sample_dir,
        data_source=data_source,
        refresh_noaa=refresh_noaa,
        base_dir=base_dir,
        timings=timings,
    )
    results, results_path, predictions_path = run_forecast_on_enso(
        enso, outputs_dir=outputs_dir, data_source_info=data_source_info, timings=timings
    )

    return EnsoForecastOutput(
        results=results,
        results_path=results_path,
        predictions_path=predictions_path,
        sample_data_dir=sample_dir,
        enso=enso,
    )


def main() -> None:
    output = run_enso_forecast()
    print(f"Wrote ENSO results to {output.results_path}")
    print(f"Wrote ENSO predictions to {output.predictions_path}")


if __name__ == "__main__":
    main()
