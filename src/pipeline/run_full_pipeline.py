from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analysis.precipitation_analysis import analyze_precipitation_by_enso_phase
from src.config import FIGURES_DIR, OUTPUTS_DIR, SAMPLE_DATA_DIR
from src.data.loaders import load_precipitation_csv, load_tide_csv
from src.models.tide_model import run_tide_demo_prediction
from src.pipeline.run_enso_forecast import run_enso_forecast
from src.reporting.llm_interpretation import resolve_interpretation
from src.reporting.report_context import build_report_context
from src.reporting.report_writer import write_markdown_report
from src.visualization.plots import (
    plot_enso_phase_timeline,
    plot_enso_rmse_by_model,
    plot_enso_timeseries,
    plot_observed_vs_predicted,
)


def run_full_pipeline(
    base_dir: Path | None = None,
    data_source: str = "auto",
    refresh_noaa: bool = False,
) -> dict:
    if base_dir is None:
        figures_dir = FIGURES_DIR
        outputs_dir = OUTPUTS_DIR
        sample_dir = SAMPLE_DATA_DIR
    else:
        figures_dir = base_dir / "reports" / "figures"
        outputs_dir = base_dir / "reports" / "outputs"
        sample_dir = base_dir / "data" / "sample"

    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    enso_output = run_enso_forecast(
        base_dir=base_dir,
        data_source=data_source,
        refresh_noaa=refresh_noaa,
    )
    # Use the ENSO DataFrame returned by the forecast (works for NOAA, sample,
    # and auto-fallback paths); do not read sample_enso.csv, which may not exist
    # in NOAA mode.
    enso = enso_output.enso
    precipitation = load_precipitation_csv(sample_dir / "sample_precipitation.csv")
    tide = load_tide_csv(sample_dir / "sample_tide.csv")

    predictions = pd.read_csv(enso_output.predictions_path, parse_dates=["date"])
    results = enso_output.results
    lead_for_plot = 1
    model_for_plot = results["best_model_by_lead"][str(lead_for_plot)]

    figure_paths = [
        plot_enso_timeseries(enso, figures_dir),
        plot_observed_vs_predicted(predictions, figures_dir, lead=lead_for_plot, model=model_for_plot),
        plot_enso_rmse_by_model(results, figures_dir),
        plot_enso_phase_timeline(enso, figures_dir),
    ]

    precipitation_result = analyze_precipitation_by_enso_phase(enso, precipitation, figures_dir)
    tide_result = run_tide_demo_prediction(tide, figures_dir)

    # Optional LLM-polished interpretation (falls back to rule-based without a key).
    interp_context = {
        "enso": results,
        "precipitation": precipitation_result.summary,
        "tide": {"metrics": tide_result.metrics},
    }
    interpretation = resolve_interpretation(interp_context)

    context = build_report_context(
        enso_results_path=enso_output.results_path,
        precipitation_summary=precipitation_result.summary,
        precipitation_figure=precipitation_result.figure_path,
        tide_metrics=tide_result.metrics,
        tide_figure=tide_result.figure_path,
        figure_paths=figure_paths,
        interpretation=interpretation,
    )

    context_path = outputs_dir / "report_context.json"
    context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_markdown_report(context, output_dir=outputs_dir)

    all_figures = figure_paths + [precipitation_result.figure_path, tide_result.figure_path]
    return {
        "report_path": report_path,
        "report_context_path": context_path,
        "figures": all_figures,
        "enso_results_path": enso_output.results_path,
    }


def main() -> None:
    result = run_full_pipeline()
    print(f"Wrote report to {result['report_path']}")
    print(f"Wrote report context to {result['report_context_path']}")
    print("Figures:")
    for figure in result["figures"]:
        print(f"- {figure}")


if __name__ == "__main__":
    main()
