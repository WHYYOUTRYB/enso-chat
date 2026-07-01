"""Tool layer: wraps existing ``src/`` functions as agent-callable tools.

Each tool has a name, description, JSON-Schema parameters, and a callable.
Tools return compact string/JSON summaries (paths + headline numbers) — never
raw DataFrames — so the conversation context stays small. Heavy objects live on
a shared :class:`ToolContext` that persists across the agentic loop.

The wrapped logic is reused verbatim from the existing modules; this file only
adapts their signatures and serializes their outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.analysis.enso_phase import classify_enso_phase
from src.analysis.precipitation_analysis import analyze_precipitation_by_enso_phase
from src.config import ACC_LOW_CONF, ACC_REFUSE, DEFAULT_LEADS, FIGURES_DIR, OUTPUTS_DIR, PROJECT_ROOT, SAMPLE_DATA_DIR
from src.data.loaders import load_enso_csv, load_precipitation_csv, load_tide_csv
from src.data.source_registry import IndexLoadError, list_sources, load_index as _registry_load_index
from src.features.enso_features import make_enso_supervised_table
from src.models.enso_ml import build_model_suite, fit_models_for_latest_forecast
from src.models.tide_model import run_tide_demo_prediction
from src.pipeline.run_enso_forecast import run_enso_forecast, run_forecast_on_enso
from src.visualization.plots import (
    plot_enso_phase_timeline,
    plot_enso_rmse_by_model,
    plot_enso_timeseries,
    plot_observed_vs_predicted,
)


@dataclass
class ToolContext:
    """Mutable run state shared across tools for one agent run."""

    base_dir: Path | None = None
    enso: pd.DataFrame | None = None
    precipitation: pd.DataFrame | None = None
    tide: pd.DataFrame | None = None
    predictions: pd.DataFrame | None = None
    results: dict | None = None
    # Data source used for the cached ENSO run, so identical load_enso_data calls
    # can be served from cache and train_and_evaluate can re-run with the same
    # source instead of silently falling back to "sample".
    enso_data_source: str | None = None
    enso_results_path: Path | None = None
    predictions_path: Path | None = None
    figure_paths: list[Path] = field(default_factory=list)
    precipitation_summary: dict | None = None
    precipitation_figure: Path | None = None
    tide_metrics: dict | None = None
    tide_figure: Path | None = None
    report_path: Path | None = None
    # CNN-LSTM track (spatial-field model). Kept on a separate slot so it never
    # overwrites the Ridge/RF results cached on `results`; the two methods coexist.
    cnn_forecasts: dict | None = None
    # Enhanced track (Ridge/RF + exogenous SOI/Niño1+2). Separate slot for the
    # same reason — coexists with the baseline `results` and `cnn_forecasts`.
    enhanced_results: dict | None = None
    # Cached exogenous index series loaded via load_index (name -> DataFrame).
    loaded_indices: dict | None = None

    @property
    def figures_dir(self) -> Path:
        return FIGURES_DIR if self.base_dir is None else self.base_dir / "reports" / "figures"

    @property
    def outputs_dir(self) -> Path:
        return OUTPUTS_DIR if self.base_dir is None else self.base_dir / "reports" / "outputs"

    @property
    def sample_dir(self) -> Path:
        return SAMPLE_DATA_DIR if self.base_dir is None else self.base_dir / "data" / "sample"


@dataclass
class Tool:
    """A single agent-callable tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., str]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Holds the available tools and dispatches calls by name."""

    def __init__(self, tools: list[Tool]):
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            available = ", ".join(self._tools)
            return f"Error: unknown tool '{name}'. Available: {available}"
        tool = self._tools[name]
        try:
            return tool.fn(**arguments)
        except Exception as exc:  # surface errors back to the LLM, don't crash the loop
            return f"Error executing tool '{name}': {exc.__class__.__name__}: {exc}"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _load_enso_data(ctx: ToolContext, data_source: str = "sample", refresh_noaa: bool = False) -> str:
    """Run ENSO modeling end-to-end (reuses run_enso_forecast) and load the series.

    Returns a compact summary: data source used, row count, date range, and the
    best model per lead time. Heavy artifacts (results JSON, predictions CSV)
    are cached on the context for later tools.

    Idempotent: a second call with the same ``data_source`` and ``refresh_noaa=False``
    reuses the cached results instead of re-running the whole forecast (which
    would re-download NOAA and re-train every model).
    """
    if (
        ctx.results is not None
        and ctx.enso_data_source == data_source
        and not refresh_noaa
    ):
        return _enso_summary(ctx, cached=True)

    output = run_enso_forecast(
        base_dir=ctx.base_dir,
        data_source=data_source,
        refresh_noaa=refresh_noaa,
    )
    ctx.enso_results_path = output.results_path
    ctx.predictions_path = output.predictions_path
    ctx.results = output.results
    ctx.enso_data_source = data_source
    # Use the ENSO DataFrame returned by the forecast — works for NOAA, sample,
    # and auto-fallback. Reading sample_enso.csv would break in NOAA mode where
    # that file is never written.
    ctx.enso = output.enso
    ctx.predictions = pd.read_csv(output.predictions_path, parse_dates=["date"])

    return _enso_summary(ctx, cached=False)


def _enso_summary(ctx: ToolContext, *, cached: bool) -> str:
    """Render the compact ENSO summary string for a (cached or fresh) load."""
    info = ctx.results["data_source"]
    best = ctx.results["best_model_by_lead"]
    best_json = json.dumps(best, separators=(",", ":"))
    prefix = "ENSO modeling already cached" if cached else "ENSO modeling complete"
    return (
        f"{prefix}. data_source used: {info['used']}"
        f"{' (fallback: ' + info['fallback_reason'] + ')' if info['fallback_reason'] else ''}. "
        f"rows={len(ctx.enso)}, "
        f"date_range={ctx.enso['date'].min().date()}_to_{ctx.enso['date'].max().date()}. "
        f"best_model_by_lead={best_json}. "
        f"results={ctx.enso_results_path.as_posix()}, predictions={ctx.predictions_path.as_posix()}"
    )


def _load_user_enso(ctx: ToolContext, path: str) -> str:
    """Load a user-uploaded ENSO CSV and run the modeling pipeline on it.

    The CSV must have ``date`` and ``nino34`` columns (same format as the
    sample ENSO data). On success, replaces the ENSO series/results cached on
    the context so subsequent tools (forecast_for_month, plots, etc.) use the
    user's data. On any failure (missing file, missing columns, parse error)
    returns an Error string and leaves the context untouched.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        return f"Error: file not found: {path}"
    try:
        enso = load_enso_csv(csv_path)
    except ValueError as exc:
        return f"Error: {exc}"
    if len(enso) < 30:
        return (
            f"Error: uploaded ENSO CSV has only {len(enso)} rows; need at least ~30 "
            f"(2+ years) to train the models."
        )

    data_source_info = {"requested": "user", "used": "user", "fallback_reason": None}
    results, results_path, predictions_path = run_forecast_on_enso(
        enso, outputs_dir=ctx.outputs_dir, data_source_info=data_source_info
    )
    ctx.enso_results_path = results_path
    ctx.predictions_path = predictions_path
    ctx.results = results
    ctx.enso_data_source = "user"
    ctx.enso = enso
    ctx.predictions = pd.read_csv(predictions_path, parse_dates=["date"])

    best = results["best_model_by_lead"]
    best_json = json.dumps(best, separators=(",", ":"))
    return (
        f"User ENSO data loaded. rows={len(enso)}, "
        f"date_range={enso['date'].min().date()}_to_{enso['date'].max().date()}. "
        f"best_model_by_lead={best_json}. "
        f"results={results_path.as_posix()}, predictions={predictions_path.as_posix()}"
    )


def _forecast_latest(ctx: ToolContext, lead: int) -> str:
    """Return the latest (most-recent) ENSO forecast for a lead time + its phase."""
    if ctx.results is None:
        return "Error: run load_enso_data or train_and_evaluate first."
    fc = ctx.results["latest_forecast"][str(lead)]
    return f"lead={lead}: value={fc['value']}, phase={fc['phase']}, model={fc['model']}"


def _classify_phase(value: float) -> str:
    """Classify a Niño3.4 value into El Niño / La Niña / Neutral (±0.5 threshold)."""
    return classify_enso_phase(float(value))


def _analyze_precipitation(ctx: ToolContext) -> str:
    """Summarize precipitation anomaly by ENSO phase and save the box plot."""
    if ctx.enso is None:
        return "Error: load ENSO data first."
    ctx.precipitation = load_precipitation_csv(ctx.sample_dir / "sample_precipitation.csv")
    result = analyze_precipitation_by_enso_phase(ctx.enso, ctx.precipitation, ctx.figures_dir)
    ctx.precipitation_summary = result.summary
    ctx.precipitation_figure = result.figure_path
    stats = result.summary["phase_statistics"]
    lines = [f"{phase}: mean={s['mean']}, std={s['std']}, n={s['count']}" for phase, s in stats.items()]
    return "Precipitation by ENSO phase:\n" + "\n".join(lines) + f"\nfigure={result.figure_path.as_posix()}"


def _run_tide_prediction(ctx: ToolContext) -> str:
    """Run the tide demonstration prediction; return RMSE/MAE/corr and figure path."""
    ctx.tide = load_tide_csv(ctx.sample_dir / "sample_tide.csv")
    result = run_tide_demo_prediction(ctx.tide, ctx.figures_dir)
    ctx.tide_metrics = result.metrics
    ctx.tide_figure = result.figure_path
    return f"Tide prediction metrics: {json.dumps(result.metrics, ensure_ascii=False)}; figure={result.figure_path.as_posix()}"


def _plot_enso_timeseries(ctx: ToolContext) -> str:
    """Plot the Niño3.4 time series with El Niño/La Niña thresholds."""
    if ctx.enso is None:
        return "Error: load ENSO data first."
    path = plot_enso_timeseries(ctx.enso, ctx.figures_dir)
    ctx.figure_paths.append(path)
    return f"Saved {path.as_posix()}"


def _plot_observed_vs_predicted(ctx: ToolContext, lead: int, model: str) -> str:
    """Plot observed vs predicted Niño3.4 for a given lead time and model."""
    if ctx.predictions is None:
        return "Error: run load_enso_data first to produce predictions."
    path = plot_observed_vs_predicted(ctx.predictions, ctx.figures_dir, lead=lead, model=model)
    ctx.figure_paths.append(path)
    return f"Saved {path.as_posix()}"


def _plot_rmse_by_model(ctx: ToolContext) -> str:
    """Bar chart of RMSE across models and lead times."""
    if ctx.results is None:
        return "Error: run load_enso_data first."
    path = plot_enso_rmse_by_model(ctx.results, ctx.figures_dir)
    ctx.figure_paths.append(path)
    return f"Saved {path.as_posix()}"


def _plot_phase_timeline(ctx: ToolContext) -> str:
    """Scatter the Niño3.4 series colored by ENSO phase."""
    if ctx.enso is None:
        return "Error: load ENSO data first."
    path = plot_enso_phase_timeline(ctx.enso, ctx.figures_dir)
    ctx.figure_paths.append(path)
    return f"Saved {path.as_posix()}"


def _read_results(ctx: ToolContext) -> str:
    """Return the ENSO results JSON as a compact summary."""
    if ctx.results is None:
        return "Error: run load_enso_data first."
    r = ctx.results
    best = r["best_model_by_lead"]
    latest = r["latest_forecast"]
    return (
        f"target={r['target']}, best_model_by_lead={best}, "
        f"latest_forecast={json.dumps(latest, ensure_ascii=False)}"
    )


# ---------------------------------------------------------------------------
# Target-month forecasting (forecast_for_month + supporting diagnostics)
# ---------------------------------------------------------------------------
# These tools let the agent answer "what about next March?" instead of being
# locked to the 1/3/6-month leads trained by load_enso_data. The lead is
# derived from the target month minus the most-recent data month.
#
# Lead buckets (matches the agreed policy):
#   lead <= 0      -> target already passed / current month; no forecast.
#   lead in {1,3,6}-> reuse the cached latest_forecast from load_enso_data.
#   lead 2,4,5     -> train that single lead on the fly, normal confidence.
#   lead 7..11     -> train on the fly, flag 低可信度 (low confidence).
#   lead >= 12     -> HARD WARNING, do NOT predict (ENSO predictability ceiling).

_HARD_WARN_LEAD = 12
_LOW_CONF_LEAD = 7  # leads >= this and < _HARD_WARN_LEAD are low-confidence


def _compute_lead(last_date: pd.Timestamp, target_year: int, target_month: int) -> int:
    """Months from the most-recent data month to the target month."""
    return (target_year - last_date.year) * 12 + (target_month - last_date.month)


def _forecast_value_for_lead(ctx: ToolContext, lead: int) -> dict:
    """Train a single lead on the fly and return {value, phase, model}.

    Uses the ENSO series cached on the context (``ctx.enso``) — never reloads
    data. Selects the best of {linear_ridge, random_forest} by in-sample fit
    on the supervised table (mirrors run_enso_forecast's selection logic).
    """
    table, feature_cols = make_enso_supervised_table(ctx.enso, leads=(lead,), max_lag=12)
    forecasts = fit_models_for_latest_forecast(
        models=build_model_suite(random_state=42),
        table=table,
        feature_cols=feature_cols,
        lead=lead,
    )
    # Drop persistence (it has no 'model' fit path here) and pick best by value
    # magnitude proximity — but we don't have a test split for an ad-hoc lead,
    # so just report the ML models and pick random_forest as primary (matches
    # the suite's usual winner on the sample data).
    primary_name = "random_forest" if "random_forest" in forecasts else next(iter(forecasts))
    value = float(forecasts[primary_name])
    return {"value": round(value, 4), "phase": classify_enso_phase(value), "model": primary_name}


# ---------------------------------------------------------------------------
# CNN-LSTM track (spatial-field model)
# ---------------------------------------------------------------------------
# Second prediction track: a CNN-LSTM trained offline on SODA sst/t300/ua/va
# spatial fields (see src/models/cnn_lstm.py + scripts/train_cnn_lstm.py). The
# online tool only does a CPU forward pass. Because real-time spatial fields are
# not wired up yet, the input window is SODA's tail months — forecasts are
# labeled as such (not real-time). Coexists with the Ridge/RF track above; the
# two never share the `results` slot.

CNN_LSTM_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "cnn_lstm_soda.pth"
SODA_TRAIN_PATH = PROJECT_ROOT / "data" / "SODA_train.nc"
SODA_LABEL_PATH = PROJECT_ROOT / "data" / "SODA_label.nc"
HINDCAST_REPORT_PATH = OUTPUTS_DIR / "cnn_lstm_hindcast.json"
REALTIME_HINDCAST_REPORT_PATH = OUTPUTS_DIR / "cnn_lstm_realtime_hindcast.json"


def _forecast_cnn_lstm(ctx: ToolContext, lead: int, mode: str = "soda_tail") -> str:
    """Run the CNN-LSTM forward pass for one lead (1..24).

    Args:
        lead: 1..24.
        mode: ``"soda_tail"`` (default, backward compatible) uses SODA's last
            window — non-real-time. ``"realtime"`` fetches live sst/t300/ua/va
            fields from OISST/GODAS/NCEP, anomalizes them against precomputed
            climatologies, and runs the same model. Realtime results are
            **cross-domain** (trained SODA / inferred other sources) and must be
            labeled as such — SODA-hindcast ACC does not transfer.

    Results are cached on ``ctx.cnn_forecasts`` (keyed by mode) so a second call
    for a different lead reuses the same 24-lead prediction.
    """
    if not (1 <= lead <= 24):
        return f"Error: lead must be 1..24, got {lead}."
    if mode not in ("soda_tail", "realtime"):
        return f"Error: mode must be 'soda_tail' or 'realtime', got {mode!r}."

    # Reuse the cached 24-lead prediction if this mode's window has run.
    if ctx.cnn_forecasts is not None and ctx.cnn_forecasts.get("mode") == mode and "leads" in ctx.cnn_forecasts:
        leads = ctx.cnn_forecasts["leads"]
        if str(lead) in leads or lead in leads:
            entry = leads.get(str(lead)) or leads.get(lead)
            src = ctx.cnn_forecasts.get("source", "?")
            we = ctx.cnn_forecasts.get("window_end", "?")
            return f"lead={lead}: value={entry['value']}, phase={entry['phase']} (CNN-LSTM, cached). source={src}, window_end={we}."

    # Lazy import: torch is a heavy, training-only dep — don't fail `import tools`.
    try:
        from src.models.cnn_lstm import load_soda_tail_window, predict_cnn_lstm, predict_cnn_lstm_realtime
    except ImportError as exc:
        return f"Error: CNN-LSTM backend unavailable (missing dependency): {exc}"

    if not CNN_LSTM_WEIGHTS_PATH.exists():
        return (
            "Error: CNN-LSTM weights not found at "
            f"{CNN_LSTM_WEIGHTS_PATH.as_posix()}. Train first: python scripts/train_cnn_lstm.py"
        )

    if mode == "soda_tail":
        if not SODA_TRAIN_PATH.exists():
            return f"Error: SODA train data not found: {SODA_TRAIN_PATH.as_posix()}"
        try:
            window, window_end = load_soda_tail_window(SODA_TRAIN_PATH)
            all_leads = predict_cnn_lstm(window, CNN_LSTM_WEIGHTS_PATH)
        except FileNotFoundError as exc:
            return f"Error: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"Error running CNN-LSTM inference: {exc.__class__.__name__}: {exc}"
        source = "SODA末端窗口(非实时空间场)"
        note = "Note: based on SODA reanalysis tail, not real-time spatial fields."
    else:  # realtime
        # Check climatologies exist BEFORE fetching (avoids a slow OISST download
        # only to fail at anomalize time). Climatologies are stored as .npz
        # (load_climatology rewrites a .nc path to .npz automatically).
        clim_dir = PROJECT_ROOT / "data" / "processed"
        clim_names = ["sst", "t300", "uwnd", "vwnd"]
        missing_clim = [n for n in clim_names
                        if not (clim_dir / f"{n}_climatology.npz").exists()
                        and not (clim_dir / f"{n}_climatology.nc").exists()]
        if missing_clim:
            return (
                f"Error: missing climatologies {missing_clim} in {clim_dir.as_posix()}. "
                "Run: python scripts/build_climatology.py (one-time offline precompute)."
            )
        try:
            from src.data.realtime_fetch import fetch_realtime_window, RealtimeFetchError
        except ImportError as exc:
            return f"Error: realtime backend unavailable: {exc}"
        try:
            window, window_end, missing = fetch_realtime_window()
        except RealtimeFetchError as exc:
            return f"Error: realtime fetch failed: {exc}."
        except FileNotFoundError as exc:
            return f"Error: {exc}. Run scripts/build_climatology.py first."
        except Exception as exc:  # noqa: BLE001
            return f"Error running realtime fetch: {exc.__class__.__name__}: {exc}"
        try:
            all_leads = predict_cnn_lstm_realtime(window, CNN_LSTM_WEIGHTS_PATH)
        except Exception as exc:  # noqa: BLE001
            return f"Error running CNN-LSTM realtime inference: {exc.__class__.__name__}: {exc}"
        source = f"实时空间场(OISST+GODAS+NCEP), data_through={window_end}, cross-domain"
        if missing:
            source += f", missing={missing}(degraded)"
        note = (
            "Note: REALTIME cross-domain — trained on SODA, inferred on OISST/GODAS/NCEP. "
            "Precision is lower than SODA hindcast; do NOT apply hindcast ACC here. "
            "Window cut off at wind channel's latest month (~5-month lag)."
        )

    ctx.cnn_forecasts = {
        "mode": mode,
        "window_end": window_end,
        "source": source,
        "leads": {item["lead"]: {"value": item["value"], "phase": item["phase"]} for item in all_leads},
    }
    entry = ctx.cnn_forecasts["leads"][lead]
    guide = ""
    if mode == "realtime":
        guide = " [可靠性请调 report_realtime_skill(lead="
        guide += f"{lead}) 查跨域 ACC，勿用 report_hindcast_skill]"
    return (
        f"lead={lead}: value={entry['value']}, phase={entry['phase']} "
        f"(CNN-LSTM, spatial sst/t300/ua/va, mode={mode}). source={source}. {note}{guide}"
    )


# ---------------------------------------------------------------------------
# Enhanced track: Ridge/RF + exogenous climate indices (SOI, Niño1+2)
# ---------------------------------------------------------------------------
# The real-time-capable prediction track. Unlike CNN-LSTM (spatial fields,
# non-real-time), the exogenous indices are 1-D monthly series downloadable
# from NOAA/PSL on demand, so enhanced forecasts can genuinely answer "next
# March". Lead confidence is data-driven from per-lead ACC rather than the
# hard-coded 7/12 thresholds used by the baseline _forecast_for_month.

EXOG_INDICES = ("soi", "nino12")


def _confidence_from_acc(acc: float) -> tuple[str, str]:
    """Map a per-lead ACC to a (confidence, tag) pair using config thresholds."""
    if acc < ACC_REFUSE:
        return "refuse", f" [拒绝/refused: ACC={acc:.2f}<{ACC_REFUSE}, below reliable range]"
    if acc < ACC_LOW_CONF:
        return "low_confidence", f" [低可信度/low_confidence: ACC={acc:.2f}<{ACC_LOW_CONF}, indicative only]"
    return "normal", f" [ACC={acc:.2f}]"


def _load_exog_into_ctx(ctx: ToolContext, cache_dir=None) -> tuple[pd.DataFrame, list[str]]:
    """Load SOI + Niño1+2, merge onto ctx.enso, return (merged_df, available_cols).

    On any index failure, skips that index and continues — the caller reports
    which exog variables were actually available. ``ctx.enso`` must be loaded
    first.
    """
    if ctx.enso is None:
        raise RuntimeError("ENSO data not loaded; call load_enso_data first.")
    if ctx.loaded_indices is None:
        ctx.loaded_indices = {}
    merged = ctx.enso.copy()
    available: list[str] = []
    for name in EXOG_INDICES:
        if name in ctx.loaded_indices:
            idx = ctx.loaded_indices[name]
        else:
            try:
                idx = _registry_load_index(name, cache_dir=cache_dir)
            except IndexLoadError:
                continue
            ctx.loaded_indices[name] = idx
        if "value" not in idx.columns and name not in idx.columns:
            # source_registry returns the column named by value_col (= name).
            pass
        col = name if name in idx.columns else "value"
        idx_s = idx[["date", col]].rename(columns={col: name})
        merged = merged.merge(idx_s, on="date", how="left")
        if name in merged.columns and merged[name].notna().any():
            available.append(name)
    # Forward-fill any trailing gaps from non-overlapping index coverage, then
    # drop rows still missing exog values (feature construction needs them).
    for name in available:
        merged[name] = merged[name].ffill().bfill()
    return merged, available


def _ensure_enhanced_results(ctx: ToolContext) -> str | None:
    """Run the enhanced pipeline once and cache on ctx.enhanced_results.

    Returns an error string if ENSO data can't be loaded; None on success.
    Reuses the cached results on repeat calls.
    """
    if ctx.enhanced_results is not None:
        return None
    if ctx.enso is None or ctx.results is None:
        _load_enso_data(ctx, data_source="auto", refresh_noaa=False)
    if ctx.enso is None:
        return "Error: could not load ENSO data for enhanced forecast."

    try:
        merged, available = _load_exog_into_ctx(ctx)
    except Exception as exc:  # noqa: BLE001 — surface to LLM
        return f"Error loading exogenous indices: {exc.__class__.__name__}: {exc}"

    if not available:
        # Fall back to baseline (no exog) but flag it clearly.
        data_source_info = {
            "requested": "enhanced",
            "used": "nino34_only_fallback",
            "fallback_reason": "exogenous indices (SOI/Niño1+2) unavailable",
        }
        results, results_path, predictions_path = run_forecast_on_enso(
            ctx.enso, outputs_dir=ctx.outputs_dir, data_source_info=data_source_info
        )
        ctx.enhanced_results = {**results, "_exog_used": [], "_fallback": True}
        return None

    data_source_info = {"requested": "enhanced", "used": "enhanced", "fallback_reason": None}
    results, results_path, predictions_path = run_forecast_on_enso(
        merged, outputs_dir=ctx.outputs_dir, data_source_info=data_source_info, exog_cols=available
    )
    ctx.enhanced_results = {**results, "_exog_used": available, "_fallback": False}
    return None


def _enhanced_latest_for_lead(ctx: ToolContext, lead: int) -> tuple[float, str, float] | None:
    """Train a single enhanced lead on the fly; return (value, phase, acc).

    Mirrors _forecast_value_for_lead but on the enhanced (exog) table. ACC is
    computed on the in-sample fit (no held-out test for an ad-hoc lead) so it
    is optimistic — used only as a coarse confidence bucket, reported honestly.
    """
    merged, available = _load_exog_into_ctx(ctx)
    if not available:
        return None
    table, feature_cols = make_enso_supervised_table(merged, leads=(lead,), max_lag=12, exog_cols=available)
    from sklearn.metrics import mean_absolute_error  # local import keeps module import light
    from src.models.evaluation import calculate_acc, calculate_regression_metrics
    from src.models.enso_ml import build_model_suite, fit_models_for_latest_forecast

    models = build_model_suite(random_state=42)
    forecasts = fit_models_for_latest_forecast(models=models, table=table, feature_cols=feature_cols, lead=lead)
    primary = "random_forest" if "random_forest" in forecasts else next(iter(forecasts))
    value = float(forecasts[primary])
    # In-sample ACC (optimistic) for a coarse confidence bucket.
    from src.models.baseline import persistence_predict
    from src.models.evaluation import temporal_train_test_split
    train, test = temporal_train_test_split(table, test_fraction=0.25)
    suite = build_model_suite(random_state=42)
    from src.models.enso_ml import train_and_predict_for_lead
    preds = train_and_predict_for_lead(models=suite, train_df=train, test_df=test, feature_cols=feature_cols, lead=lead)
    y_true = test[f"target_lead_{lead}"].to_numpy(dtype=float)
    acc = calculate_acc(y_true, preds.get(primary, y_true)) if primary in preds else 0.0
    return value, classify_enso_phase(value), acc


def _list_data_sources(ctx: ToolContext) -> str:
    """List all registered climate-index data sources."""
    return "\n".join(
        f"- {s['name']}: {s['description']} (coverage {s['coverage']})"
        for s in list_sources()
    ) + "\n\nUse load_index(name) to load one; forecast_enhanced uses soi+nino12 automatically."


def _load_index_tool(ctx: ToolContext, name: str) -> str:
    """Load a single registered index and cache it on the context."""
    if ctx.loaded_indices is None:
        ctx.loaded_indices = {}
    try:
        df = _registry_load_index(name)
    except IndexLoadError as exc:
        return f"Error loading index '{name}': {exc}"
    ctx.loaded_indices[name] = df
    val_col = name if name in df.columns else df.columns[-1]
    return (
        f"Loaded index '{name}'. rows={len(df)}, "
        f"date_range={df['date'].min().date()}_to_{df['date'].max().date()}, "
        f"column={val_col}. Cached for use by forecast_enhanced."
    )


def _forecast_enhanced(
    ctx: ToolContext, target_year: int, target_month: int, data_source: str = "auto"
) -> str:
    """Forecast Niño3.4 for a target month using Ridge/RF + exogenous indices.

    Loads Niño3.4 (cached) plus SOI + Niño1+2, trains the enhanced model, and
    dispatches by lead. Confidence is data-driven from per-lead ACC: ACC<
    {ACC_REFUSE} refuses, ACC<{ACC_LOW_CONF} flags low confidence, else normal.
    If exogenous indices are unreachable, falls back to Niño3.4-only and flags it.
    """
    if not (1 <= target_month <= 12):
        return f"Error: target_month must be 1..12, got {target_month}."
    err = _ensure_enhanced_results(ctx)
    if err is not None:
        return err
    res = ctx.enhanced_results
    exog_used = res.get("_exog_used", [])
    fallback = res.get("_fallback", False)

    last_date = pd.Timestamp(ctx.enso["date"].max())
    lead = _compute_lead(last_date, target_year, target_month)
    target = f"{target_year}-{target_month:02d}"
    last_iso = last_date.strftime("%Y-%m")

    exog_tag = f" exog={exog_used}" if exog_used else " exog=[](fallback to nino34-only)"
    if fallback:
        exog_tag = " exog=UNAVAILABLE(fallback nino34-only)"

    if lead <= 0:
        return f"target={target} (lead={lead}): at or before latest data ({last_iso}); no forecast needed.{exog_tag}"

    if str(lead) in res["latest_forecast"]:
        fc = res["latest_forecast"][str(lead)]
        best = res["best_model_by_lead"][str(lead)]
        acc = res["leads"][str(lead)][best].get("acc", 0.0)
        conf, tag = _confidence_from_acc(acc)
        if conf == "refuse":
            return f"target={target} lead={lead} (enhanced, cached): refusing — ACC={acc:.2f} below reliable range. Recommend a shorter lead or refresh data.{exog_tag}"
        return (
            f"target={target} lead={lead} (enhanced, cached): value={fc['value']}, "
            f"phase={fc['phase']}, model={fc['model']}, acc={acc:.2f}, "
            f"data_through={last_iso}.{tag}{exog_tag}"
        )

    # On-the-fly lead (2/4/5/7-11/...): train single lead, use in-sample ACC bucket.
    try:
        got = _enhanced_latest_for_lead(ctx, lead)
    except Exception as exc:  # noqa: BLE001
        return f"Error training enhanced lead={lead}: {exc.__class__.__name__}: {exc}"
    if got is None:
        return f"target={target} lead={lead}: enhanced unavailable (no exog indices loaded).{exog_tag}"
    value, phase, acc = got
    conf, tag = _confidence_from_acc(acc)
    if conf == "refuse":
        return f"target={target} lead={lead} (enhanced, on-the-fly): refusing — ACC={acc:.2f} below reliable range.{exog_tag}"
    return (
        f"target={target} lead={lead} (enhanced, on-the-fly): value={round(value,4)}, "
        f"phase={phase}, model=random_forest, acc={acc:.2f}, data_through={last_iso}.{tag}{exog_tag}"
    )


def _compare_methods(
    ctx: ToolContext, target_year: int, target_month: int, data_source: str = "auto"
) -> str:
    """Run baseline / enhanced / CNN-LSTM side by side for one target month.

    Each method is invoked independently and its result line collected; a
    missing method (e.g. CNN-LSTM weights absent) is reported as 'unavailable'
    rather than aborting the comparison.
    """
    if not (1 <= target_month <= 12):
        return f"Error: target_month must be 1..12, got {target_month}."
    target = f"{target_year}-{target_month:02d}"

    # Baseline (Ridge/RF): reuse forecast_for_month logic.
    baseline = _forecast_for_month(ctx, target_year, target_month, data_source)

    # Enhanced.
    enhanced = _forecast_enhanced(ctx, target_year, target_month, data_source)

    # CNN-LSTM: pick the lead matching the target month's lead for comparability.
    cnn_line = "cnn_lstm: unavailable (weights not trained or SODA missing)"
    if CNN_LSTM_WEIGHTS_PATH.exists() and SODA_TRAIN_PATH.exists():
        if ctx.cnn_forecasts is None:
            # Trigger a CNN run at lead 1 to populate the cache (the 24-lead
            # prediction is cached, then we read the matching lead below).
            _forecast_cnn_lstm(ctx, lead=1)
        if ctx.cnn_forecasts is not None and "leads" in ctx.cnn_forecasts:
            # Use the baseline lead for an apples-to-apples comparison.
            last_date = pd.Timestamp(ctx.enso["date"].max()) if ctx.enso is not None else None
            if last_date is not None:
                lead = _compute_lead(last_date, target_year, target_month)
                if 1 <= lead <= 24 and lead in ctx.cnn_forecasts["leads"]:
                    e = ctx.cnn_forecasts["leads"][lead]
                    cnn_line = (
                        f"cnn_lstm (lead={lead}, SODA末端非实时): value={e['value']}, phase={e['phase']}"
                    )
                else:
                    e = ctx.cnn_forecasts["leads"][12]
                    cnn_line = (
                        f"cnn_lstm (lead=12 default, SODA末端非实时): value={e['value']}, phase={e['phase']}"
                    )

    lines = [f"Method comparison for target={target}:", f"- baseline (Ridge/RF, nino34-only): {baseline}",
             f"- enhanced (Ridge/RF + SOI/Niño1+2): {enhanced}", f"- {cnn_line}"]
    return "\n".join(lines)


def _report_hindcast_skill(ctx: ToolContext, lead: int | None = None) -> str:
    """Report CNN-LSTM hindcast skill vs the Persistence null model.

    Answers "is this forecast trustworthy?" the way Ham et al. 2019 (Nature) do:
    all-season ACC per lead, benchmarked against Persistence (forecast = last
    observed month). The CNN must beat Persistence — and stay above ACC=0.5 —
    to claim skill. Reads the precomputed report if present; otherwise runs the
    hindcast on the fly (needs trained weights).

    Args:
        lead: if given, return that single lead's CNN-vs-Persistence line plus a
            reliability verdict; if None, return the full per-lead table.
    """
    import json as _json

    data = None
    if HINDCAST_REPORT_PATH.exists():
        try:
            data = _json.loads(HINDCAST_REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — corrupt cache, fall through to recompute
            data = None

    if data is None:
        if not CNN_LSTM_WEIGHTS_PATH.exists() or not SODA_TRAIN_PATH.exists() or not SODA_LABEL_PATH.exists():
            return (
                "Error: hindcast unavailable — need trained weights + SODA data. "
                "Run: python scripts/train_cnn_lstm.py && python scripts/run_hindcast.py"
            )
        try:
            from src.models.hindcast import run_hindcast

            res = run_hindcast(CNN_LSTM_WEIGHTS_PATH, SODA_TRAIN_PATH, SODA_LABEL_PATH)
        except Exception as exc:  # noqa: BLE001
            return f"Error running hindcast: {exc.__class__.__name__}: {exc}"
        cnn_acc, pers_acc, gap = res.cnn_acc, res.persistence_acc, res.skill_gap
        leads = res.leads
        n = res.n_samples
    else:
        cnn_acc, pers_acc, gap = data["cnn_acc"], data["persistence_acc"], data["skill_gap"]
        leads = data["leads"]
        n = data["n_samples"]

    if lead is not None:
        if not (1 <= lead <= len(leads)):
            return f"Error: lead must be 1..{len(leads)}, got {lead}."
        i = lead - 1
        c, p, g = cnn_acc[i], pers_acc[i], gap[i]
        if g > 0 and c >= 0.5:
            verdict = "reliable — CNN beats Persistence and ACC>=0.5."
        elif g > 0:
            verdict = f"skillful vs Persistence (gap=+{g:.2f}) but ACC={c:.2f}<0.5 — indicative only."
        else:
            verdict = f"NO skill over Persistence (gap={g:+.2f}) — prefer Persistence or a shorter lead."
        return (
            f"lead={lead}: CNN-ACC={c:.3f}, Persistence-ACC={p:.3f}, gap={g:+.3f}. {verdict} "
            f"(all-season ACC, Ham et al. 2019 metric; n={n} test windows.)"
        )

    # Full table.
    lines = [
        f"Hindcast skill (n={n} test windows). All-season ACC — Ham et al. 2019 metric "
        f"(their CNN >0.5 to lead≈17). CNN must beat Persistence to claim skill.",
        f"{'lead':>4} {'CNN-ACC':>8} {'Persist':>8} {'gap':>7}",
    ]
    for i, ld in enumerate(leads):
        lines.append(f"{ld:>4} {cnn_acc[i]:>8.3f} {pers_acc[i]:>8.3f} {gap[i]:>+7.3f}")
    above_pers = [leads[i] for i in range(len(leads)) if gap[i] > 0]
    above_05 = [leads[i] for i in range(len(leads)) if cnn_acc[i] >= 0.5]
    lines.append(f"CNN beats Persistence at leads={above_pers}.")
    lines.append(f"CNN ACC>=0.5 at leads={above_05}.")
    lines.append(
        "Reliability: short leads (1-3) Persistence is stronger (ENSO autocorrelation) — "
        "use Persistence or Ridge/RF there; CNN's value is lead 4-23 where Persistence fails."
    )
    return "\n".join(lines)


def _report_realtime_skill(ctx: ToolContext, lead: int | None = None) -> str:
    """Report CNN-LSTM skill on the REALTIME domain (OISST/GODAS/NCEP).

    This is the only ACC that legitimately judges realtime predictions — the
    SODA hindcast ACC does NOT transfer across domains. Reads the precomputed
    realtime hindcast report (``cnn_lstm_realtime_hindcast.json``); if absent,
    tells the user to run ``scripts/run_realtime_hindcast.py``.

    Pass an optional lead (1..24) for a single-lead cross-domain verdict.
    """
    import json as _json

    if not REALTIME_HINDCAST_REPORT_PATH.exists():
        return (
            "Error: realtime-domain hindcast not computed yet. "
            "Run: python scripts/run_realtime_hindcast.py "
            "(builds a leakage-free climatology + evaluates cross-domain ACC). "
            "Until then, realtime prediction skill is unquantified — the SODA "
            "hindcast ACC does NOT apply cross-domain."
        )
    try:
        data = _json.loads(REALTIME_HINDCAST_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"Error reading realtime hindcast report: {exc}"
    cnn_acc, pers_acc, gap = data["cnn_acc"], data["persistence_acc"], data["skill_gap"]
    leads = data["leads"]
    n = data["n_windows"]

    if lead is not None:
        if not (1 <= lead <= len(leads)):
            return f"Error: lead must be 1..{len(leads)}, got {lead}."
        i = lead - 1
        c, p, g = cnn_acc[i], pers_acc[i], gap[i]
        if g > 0 and c >= 0.5:
            verdict = "cross-domain reliable — beats Persistence and ACC>=0.5."
        elif g > 0:
            verdict = f"cross-domain skillful vs Persistence (gap=+{g:.2f}) but ACC={c:.2f}<0.5 — indicative only."
        else:
            verdict = f"NO cross-domain skill over Persistence (gap={g:+.2f}) — do not trust this realtime lead."
        return (
            f"lead={lead}: CNN-ACC={c:.3f}, Persistence-ACC={p:.3f}, gap={g:+.3f} (REALTIME domain, n={n} windows). {verdict}"
        )

    lines = [
        f"Realtime-domain hindcast skill (n={n} windows, eval={data.get('eval_period','?')}). "
        f"Cross-domain ACC on OISST/GODAS/NCEP — the ONLY metric that judges realtime predictions.",
        f"{'lead':>4} {'CNN-ACC':>8} {'Persist':>8} {'gap':>7}",
    ]
    for i, ld in enumerate(leads):
        lines.append(f"{ld:>4} {cnn_acc[i]:>8.3f} {pers_acc[i]:>8.3f} {gap[i]:>+7.3f}")
    above_pers = [leads[i] for i in range(len(leads)) if gap[i] > 0]
    above_05 = [leads[i] for i in range(len(leads)) if cnn_acc[i] >= 0.5]
    lines.append(f"CNN beats Persistence at leads={above_pers}.")
    lines.append(f"CNN ACC>=0.5 at leads={above_05}.")
    lines.append("Note: this is cross-domain (trained SODA / evaluated realtime). Compare to SODA hindcast for the domain gap.")
    return "\n".join(lines)


def _forecast_for_month(
    ctx: ToolContext, target_year: int, target_month: int, data_source: str = "auto"
) -> str:
    """Forecast Niño3.4 for a specific target month, dispatching by lead.

    Loads ENSO data (reusing the cache) if not already loaded, then computes
    the lead from the target month vs. the latest data month and dispatches:
    past / cached lead / on-the-fly training / low-confidence / hard warning.
    """
    if not (1 <= target_month <= 12):
        return f"Error: target_month must be 1..12, got {target_month}."
    if ctx.enso is None or ctx.results is None:
        _load_enso_data(ctx, data_source=data_source, refresh_noaa=False)
    if ctx.enso is None:
        return "Error: could not load ENSO data."

    last_date = pd.Timestamp(ctx.enso["date"].max())
    lead = _compute_lead(last_date, target_year, target_month)
    target = f"{target_year}-{target_month:02d}"
    last_iso = last_date.strftime("%Y-%m")

    if lead <= 0:
        return (
            f"target={target} (lead={lead}): target month is at or before the latest data "
            f"({last_iso}); no forecast needed (data already covers it)."
        )

    if str(lead) in ctx.results["latest_forecast"]:
        fc = ctx.results["latest_forecast"][str(lead)]
        return (
            f"target={target} lead={lead} (cached): value={fc['value']}, "
            f"phase={fc['phase']}, model={fc['model']}, data_through={last_iso}."
        )

    if lead >= _HARD_WARN_LEAD:
        return (
            f"target={target} requires lead={lead} months (data through {last_iso}). "
            f"This exceeds the reliable forecast range (lead < {_HARD_WARN_LEAD}); "
            f"ENSO predictability decays sharply past ~6 months. "
            f"Refusing to predict — recommend re-running closer to the target month, "
            f"or updating ENSO data first (load_enso_data data_source='auto' refresh_noaa=True)."
        )

    fc = _forecast_value_for_lead(ctx, lead)
    confidence = "low_confidence" if lead >= _LOW_CONF_LEAD else "normal"
    tag = " [低可信度/low_confidence: lead>=7, treat as indicative only]" if lead >= _LOW_CONF_LEAD else ""
    return (
        f"target={target} lead={lead} (trained on the fly, confidence={confidence}): "
        f"value={fc['value']}, phase={fc['phase']}, model={fc['model']}, "
        f"data_through={last_iso}.{tag}"
    )


def _diagnose_local_data(ctx: ToolContext) -> str:
    """Report what ENSO data is available locally and how fresh it is.

    Scans the sample and processed (NOAA) CSVs without requiring load_enso_data
    to have run. Useful before forecasting to check coverage and freshness.
    """
    candidates = {
        "sample": ctx.sample_dir / "sample_enso.csv",
        "noaa": (ctx.base_dir / "data" / "processed" / "noaa_nino34.csv") if ctx.base_dir else None,
    }
    found: list[str] = []
    for name, path in candidates.items():
        if path is None or not path.exists():
            continue
        try:
            df = pd.read_csv(path, parse_dates=["date"])
        except Exception as exc:  # noqa: BLE001 — diagnose must never crash
            found.append(f"{name}: present but unreadable ({exc})")
            continue
        found.append(
            f"{name}: rows={len(df)}, "
            f"date_range={df['date'].min().date()}_to_{df['date'].max().date()}"
        )

    if not found:
        return (
            "No local ENSO data found. Run load_enso_data(data_source='sample') "
            "for offline data, or data_source='auto' to try NOAA (falls back to sample)."
        )

    # Also surface the cached in-memory series if load_enso_data already ran.
    if ctx.enso is not None:
        last = pd.Timestamp(ctx.enso["date"].max())
        used = (ctx.results or {}).get("data_source", {}).get("used", "?") if ctx.results else "?"
        found.append(f"in-memory: data_source={used}, last_date={last.strftime('%Y-%m')}")
    return "Local ENSO data:\n" + "\n".join(found)


def recommend_data_range_dict(
    ctx: ToolContext, target_year: int, target_month: int
) -> dict:
    """Structured data-range/lead assessment for a target month (UI-facing).

    Returns a dict the UI reads directly (``bucket`` / ``allow_run``) so it
    never parses the LLM-facing string. ``bucket`` aligns with the lead dispatch
    in :func:`_forecast_for_month`:

        lead <= 0              -> past
        lead in DEFAULT_LEADS  -> cached
        lead < _LOW_CONF_LEAD  -> short
        lead < _HARD_WARN_LEAD -> low_confidence
        lead >= _HARD_WARN_LEAD -> out_of_range

    ``allow_run`` is False for past / out_of_range / invalid / no_data.
    Assessment auto-loads SAMPLE data only — never NOAA, never DeepSeek — so it
    is safe to call repeatedly as a zero-cost preflight.
    """
    target = f"{target_year}-{target_month:02d}"
    if not (1 <= target_month <= 12):
        return {
            "target": target, "lead": None, "bucket": "invalid",
            "data_through": None, "history_years": None,
            "recommendation": f"Error: target_month must be 1..12, got {target_month}.",
            "allow_run": False,
        }
    if ctx.enso is None:
        _load_enso_data(ctx, data_source="sample", refresh_noaa=False)
    if ctx.enso is None:
        return {
            "target": target, "lead": None, "bucket": "no_data",
            "data_through": None, "history_years": None,
            "recommendation": "Error: could not load ENSO data to assess.",
            "allow_run": False,
        }

    last_date = pd.Timestamp(ctx.enso["date"].max())
    lead = _compute_lead(last_date, target_year, target_month)
    last_iso = last_date.strftime("%Y-%m")
    n_years = len(ctx.enso) / 12.0

    if lead <= 0:
        bucket = "past"
    elif lead in DEFAULT_LEADS:
        bucket = "cached"
    elif lead < _LOW_CONF_LEAD:
        bucket = "short"
    elif lead < _HARD_WARN_LEAD:
        bucket = "low_confidence"
    else:
        bucket = "out_of_range"

    if bucket == "out_of_range":
        recommendation = (
            f"lead>={_HARD_WARN_LEAD} exceeds the reliable range — do NOT forecast. "
            "Re-run closer to the target month or refresh ENSO data."
        )
    elif bucket == "low_confidence":
        recommendation = (
            "forecast possible but 低可信度 — present as indicative only, "
            "state uncertainty in the report."
        )
    else:
        recommendation = "within reliable range; forecast is reasonable."

    return {
        "target": target,
        "lead": lead,
        "bucket": bucket,
        "data_through": last_iso,
        "history_years": round(n_years),
        "recommendation": recommendation,
        "allow_run": bucket not in {"past", "out_of_range"},
    }


def _recommend_data_range(ctx: ToolContext, target_year: int, target_month: int) -> str:
    """LLM-facing string wrapper over :func:`recommend_data_range_dict`.

    Kept as a multi-line string (with ``bucket=`` and ``lead=`` tokens) so the
    existing tool contract and tests are unchanged.
    """
    d = recommend_data_range_dict(ctx, target_year, target_month)
    if d["bucket"] in ("invalid", "no_data"):
        return d["recommendation"]
    lines = [
        f"target={d['target']}, lead={d['lead']} months (data through {d['data_through']}).",
        f"bucket={d['bucket']}.",
        f"available history ~{d['history_years']} years (>=30 years is generally adequate for ENSO ML).",
        f"Recommendation: {d['recommendation']}",
    ]
    if d["history_years"] < 30:
        lines.append("Note: shorter history (<30y) may reduce model stability.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------


def build_tools(ctx: ToolContext) -> ToolRegistry:
    """Build the full tool registry bound to a shared ToolContext."""
    tools = [
        Tool(
            name="load_enso_data",
            description=(
                "Run ENSO Niño3.4 modeling end-to-end (Persistence, Ridge, Random Forest "
                "for 1/3/6-month leads), cache the results JSON and predictions CSV, and load "
                "the ENSO series. Call this FIRST to produce the core results and artifacts. "
                "data_source options: 'sample' (offline), 'noaa' (require NOAA), 'auto' (NOAA then sample)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "data_source": {
                        "type": "string",
                        "enum": ["sample", "noaa", "auto"],
                        "description": "ENSO data source. Use 'sample' for offline runs.",
                    },
                    "refresh_noaa": {
                        "type": "boolean",
                        "description": "Re-download NOAA data even if cached. Only relevant for noaa/auto.",
                    },
                },
                "required": ["data_source"],
                "additionalProperties": False,
            },
            fn=lambda data_source="sample", refresh_noaa=False: _load_enso_data(ctx, data_source, refresh_noaa),
        ),
        Tool(
            name="load_user_enso",
            description=(
                "Load a user-uploaded ENSO CSV (must have 'date' and 'nino34' columns, "
                "monthly Niño3.4 values) and run the modeling pipeline on it, replacing the "
                "current ENSO data. Use this when the user has uploaded their own CSV via the "
                "sidebar. The path is provided by the UI after upload; pass it verbatim."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the uploaded ENSO CSV file (date+nino34 columns).",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            fn=lambda path: _load_user_enso(ctx, path),
        ),
        Tool(
            name="forecast_latest",
            description="Return the latest (most recent) Niño3.4 forecast value and ENSO phase for a lead time.",
            parameters={
                "type": "object",
                "properties": {"lead": {"type": "integer", "enum": [1, 3, 6]}},
                "required": ["lead"],
                "additionalProperties": False,
            },
            fn=lambda lead: _forecast_latest(ctx, lead),
        ),
        Tool(
            name="forecast_cnn_lstm",
            description=(
                "Forecast Niño3.4 for a lead time (1..24 months) using the CNN-LSTM "
                "spatial-field model (trained on SODA sst/t300/ua/va). mode='soda_tail' "
                "(default) uses SODA's last window — NOT real-time. mode='realtime' fetches "
                "live OISST+GODAS+NCEP fields, anomalizes against climatologies, and runs the "
                "same model — real-time but CROSS-DOMAIN (trained SODA / inferred other sources), "
                "so precision is lower than the SODA hindcast; results are labeled as such and "
                "hindcast ACC does not apply. Realtime window is cut off at the wind channel's "
                "latest month (~5-month lag). Use realtime for genuine now-casting, soda_tail for "
                "method demonstration/hindcast comparison."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "lead": {"type": "integer", "minimum": 1, "maximum": 24},
                    "mode": {"type": "string", "enum": ["soda_tail", "realtime"], "description": "Default 'soda_tail'."},
                },
                "required": ["lead"],
                "additionalProperties": False,
            },
            fn=lambda lead, mode="soda_tail": _forecast_cnn_lstm(ctx, lead, mode),
        ),
        Tool(
            name="list_data_sources",
            description=(
                "List all registered climate-index data sources (Niño3.4, SOI, Niño1+2) "
                "with descriptions and coverage. Use when the user asks what data is "
                "available, or before loading an index."
            ),
            parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            fn=lambda: _list_data_sources(ctx),
        ),
        Tool(
            name="load_index",
            description=(
                "Download and cache one registered climate index by name (e.g. 'soi', "
                "'nino12', 'nino34') from NOAA/PSL. Returns row count and date range. "
                "Cached on the session for reuse by forecast_enhanced."
            ),
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "enum": ["nino34", "soi", "nino12"]}},
                "required": ["name"],
                "additionalProperties": False,
            },
            fn=lambda name: _load_index_tool(ctx, name),
        ),
        Tool(
            name="forecast_enhanced",
            description=(
                "Forecast Niño3.4 for a target month using Ridge/RF augmented with "
                "exogenous climate indices (SOI + Niño1+2) — the real-time-capable, "
                "more skillful track. Confidence is data-driven from per-lead ACC: "
                "below ~0.3 refuses, below ~0.5 flags low confidence. Falls back to "
                "Niño3.4-only if indices are unreachable. Use for the best real-time "
                "forecast or when the user wants the enhanced/multivariate model."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target_year": {"type": "integer", "description": "Target year, e.g. 2027."},
                    "target_month": {"type": "integer", "description": "Target month 1..12."},
                    "data_source": {
                        "type": "string",
                        "enum": ["sample", "noaa", "auto"],
                        "description": "ENSO data source. Default 'auto'.",
                    },
                },
                "required": ["target_year", "target_month"],
                "additionalProperties": False,
            },
            fn=lambda target_year, target_month, data_source="auto": _forecast_enhanced(ctx, target_year, target_month, data_source),
        ),
        Tool(
            name="compare_methods",
            description=(
                "Run baseline (Ridge/RF Niño3.4-only), enhanced (Ridge/RF + SOI/Niño1+2), "
                "and CNN-LSTM side by side for one target month, returning a comparison "
                "table. Use when the user asks to compare methods/精度 or wants a "
                "comprehensive forecast. Methods that are unavailable (e.g. CNN-LSTM "
                "weights not trained) are reported as 'unavailable' without aborting."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target_year": {"type": "integer", "description": "Target year, e.g. 2027."},
                    "target_month": {"type": "integer", "description": "Target month 1..12."},
                    "data_source": {
                        "type": "string",
                        "enum": ["sample", "noaa", "auto"],
                        "description": "ENSO data source. Default 'auto'.",
                    },
                },
                "required": ["target_year", "target_month"],
                "additionalProperties": False,
            },
            fn=lambda target_year, target_month, data_source="auto": _compare_methods(ctx, target_year, target_month, data_source),
        ),
        Tool(
            name="report_hindcast_skill",
            description=(
                "Report CNN-LSTM skill on the SODA TRAINING domain (all-season ACC per lead "
                "vs Persistence). Use for: 'SODA/训练域/方法上限' questions, or to show the "
                "model's best-case skill. **Do NOT use this for realtime/live forecast "
                "reliability** — that is a different domain; use report_realtime_skill instead. "
                "Pass an optional lead (1..24) for a single-lead verdict."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "lead": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 24,
                        "description": "Optional: return a single lead's verdict instead of the full table.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            fn=lambda lead=None: _report_hindcast_skill(ctx, lead),
        ),
        Tool(
            name="report_realtime_skill",
            description=(
                "Report CNN-LSTM skill on the REALTIME domain (OISST/GODAS/NCEP fields) — "
                "the ONLY ACC that judges realtime/live predictions. Use for: '实时/realtime/"
                "live 预测准不准/可不可靠' questions, or after a forecast_cnn_lstm(mode=realtime) "
                "call when the user cares about reliability. The SODA report_hindcast_skill "
                "does NOT transfer cross-domain — do not use that for realtime. Pass an "
                "optional lead (1..24) for a single-lead cross-domain verdict. Requires the "
                "precomputed realtime hindcast (scripts/run_realtime_hindcast.py)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "lead": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 24,
                        "description": "Optional: return a single lead's cross-domain verdict.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            fn=lambda lead=None: _report_realtime_skill(ctx, lead),
        ),
        Tool(
            name="classify_phase",
            description="Classify a Niño3.4 anomaly value into El Niño (>=0.5), La Niña (<=-0.5), or Neutral.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "number", "description": "Niño3.4 anomaly value."}},
                "required": ["value"],
                "additionalProperties": False,
            },
            fn=lambda value: _classify_phase(value),
        ),
        Tool(
            name="analyze_precipitation",
            description="Summarize precipitation anomaly statistics by ENSO phase and save a box plot. Requires load_enso_data first.",
            parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            fn=lambda: _analyze_precipitation(ctx),
        ),
        Tool(
            name="run_tide_prediction",
            description="Run the tide demonstration prediction (Ridge on harmonic features); returns RMSE/MAE/corr and a figure.",
            parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            fn=lambda: _run_tide_prediction(ctx),
        ),
        Tool(
            name="plot_enso_timeseries",
            description="Plot the Niño3.4 time series with El Niño/La Niña threshold lines. Requires load_enso_data.",
            parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            fn=lambda: _plot_enso_timeseries(ctx),
        ),
        Tool(
            name="plot_observed_vs_predicted",
            description="Plot observed vs predicted Niño3.4 for a given lead time and model name (e.g. random_forest, linear_ridge, persistence).",
            parameters={
                "type": "object",
                "properties": {
                    "lead": {"type": "integer", "enum": [1, 3, 6]},
                    "model": {"type": "string", "description": "Model name from best_model_by_lead or the metrics table."},
                },
                "required": ["lead", "model"],
                "additionalProperties": False,
            },
            fn=lambda lead, model: _plot_observed_vs_predicted(ctx, lead, model),
        ),
        Tool(
            name="plot_rmse_by_model",
            description="Bar chart of RMSE across all models and lead times. Requires load_enso_data.",
            parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            fn=lambda: _plot_rmse_by_model(ctx),
        ),
        Tool(
            name="plot_phase_timeline",
            description="Scatter the Niño3.4 series colored by ENSO phase. Requires load_enso_data.",
            parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            fn=lambda: _plot_phase_timeline(ctx),
        ),
        Tool(
            name="read_results",
            description="Return a compact summary of the ENSO results (best model per lead, latest forecast). Requires load_enso_data.",
            parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            fn=lambda: _read_results(ctx),
        ),
        Tool(
            name="forecast_for_month",
            description=(
                "Forecast Niño3.4 for a specific target month (e.g. 'next March' = "
                "target_year=2027, target_month=3). Derives the lead from the target minus the "
                "latest data month. lead in {1,3,6} reuses cached results; lead 2/4/5 trains on the "
                "fly; lead 7-11 trains but flags 低可信度 (low confidence); lead>=12 refuses to "
                "predict (out of reliable range). Loads data via 'auto' if not already loaded."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target_year": {"type": "integer", "description": "Target year, e.g. 2027."},
                    "target_month": {"type": "integer", "description": "Target month 1..12."},
                    "data_source": {
                        "type": "string",
                        "enum": ["sample", "noaa", "auto"],
                        "description": "Only used if data must be loaded. Default 'auto'.",
                    },
                },
                "required": ["target_year", "target_month"],
                "additionalProperties": False,
            },
            fn=lambda target_year, target_month, data_source="auto": _forecast_for_month(
                ctx, target_year, target_month, data_source
            ),
        ),
        Tool(
            name="diagnose_local_data",
            description=(
                "Report what ENSO data is available locally (sample + processed NOAA CSVs) and "
                "how fresh it is (last date, row count). Does NOT require load_enso_data. Useful "
                "before forecasting to check data coverage."
            ),
            parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            fn=lambda: _diagnose_local_data(ctx),
        ),
        Tool(
            name="recommend_data_range",
            description=(
                "Given a target month, recommend whether the available data range / lead is "
                "adequate for a reliable forecast, and which confidence bucket applies "
                "(cached/short/low_confidence/out_of_range). Loads data via 'auto' if needed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target_year": {"type": "integer", "description": "Target year, e.g. 2027."},
                    "target_month": {"type": "integer", "description": "Target month 1..12."},
                },
                "required": ["target_year", "target_month"],
                "additionalProperties": False,
            },
            fn=lambda target_year, target_month: _recommend_data_range(ctx, target_year, target_month),
        ),
    ]
    return ToolRegistry(tools)
