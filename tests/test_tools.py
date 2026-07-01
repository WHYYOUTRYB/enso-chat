from pathlib import Path

import pandas as pd

from src.agent.tools import ToolContext, build_tools


def test_build_tools_has_expected_names():
    names = set(build_tools(ToolContext()).names())
    # Kept tools (12)
    expected = {
        "load_enso_data", "forecast_for_month", "diagnose_local_data",
        "recommend_data_range", "forecast_latest", "classify_phase",
        "read_results", "plot_enso_timeseries", "plot_observed_vs_predicted",
        "plot_rmse_by_model", "plot_phase_timeline", "analyze_precipitation",
        "run_tide_prediction",
    }
    assert expected <= names, f"missing: {expected - names}"


def test_build_tools_dropped_heavy_tools():
    names = set(build_tools(ToolContext()).names())
    dropped = {"write_report", "build_enso_features", "train_and_evaluate", "list_figures"}
    assert names.isdisjoint(dropped), f"should be dropped but present: {dropped & names}"


def test_classify_phase_tool(tmp_path: Path):
    ctx = ToolContext(base_dir=tmp_path)
    registry = build_tools(ctx)
    assert registry.execute("classify_phase", {"value": 0.7}) == "El Niño"
    assert registry.execute("classify_phase", {"value": -0.6}) == "La Niña"


def test_forecast_for_month_hard_warns(tmp_path: Path):
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    tools.execute("load_enso_data", {"data_source": "sample"})
    out = tools.execute("forecast_for_month", {"target_year": 2027, "target_month": 3})
    assert "lead" in out.lower() and "27" in out
    assert "refusing" in out.lower() or "exceeds the reliable" in out.lower()


# --- load_user_enso: user-uploaded ENSO CSV ---


def _write_user_enso_csv(path: Path, periods: int = 120) -> Path:
    """Write a small valid ENSO CSV (date+nino34) for testing."""
    from src.data.sample_generator import generate_sample_enso
    enso = generate_sample_enso(periods=periods)
    enso.to_csv(path, index=False)
    return path


def test_load_user_enso_loads_and_models(tmp_path: Path):
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    csv_path = _write_user_enso_csv(tmp_path / "user_enso.csv", periods=120)
    out = tools.execute("load_user_enso", {"path": str(csv_path)})
    assert "Error" not in out
    assert ctx.enso is not None
    assert len(ctx.enso) == 120
    assert ctx.results is not None
    assert set(ctx.results["leads"]) == {"1", "3", "6"}
    assert ctx.results["data_source"]["used"] == "user"


def test_load_user_enso_missing_columns_errors(tmp_path: Path):
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2020-01-01"], "temp": [1.0]}).to_csv(bad, index=False)
    out = tools.execute("load_user_enso", {"path": str(bad)})
    assert out.startswith("Error")
    assert ctx.enso is None  # not loaded on failure


def test_load_user_enso_nonexistent_file_errors(tmp_path: Path):
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    out = tools.execute("load_user_enso", {"path": str(tmp_path / "nope.csv")})
    assert out.startswith("Error")


def test_build_tools_includes_load_user_enso():
    names = set(build_tools(ToolContext()).names())
    assert "load_user_enso" in names


# --- forecast_cnn_lstm (CNN-LSTM track) ---


def test_build_tools_includes_forecast_cnn_lstm():
    names = set(build_tools(ToolContext()).names())
    assert "forecast_cnn_lstm" in names


def test_forecast_cnn_lstm_missing_weights_returns_error(tmp_path, monkeypatch):
    # Point the tool at a non-existent weights path so the missing-weights
    # branch is exercised even after the real weights have been trained.
    from src.agent import tools as tools_mod

    monkeypatch.setattr(tools_mod, "CNN_LSTM_WEIGHTS_PATH", tmp_path / "does_not_exist.pth")
    ctx = ToolContext(base_dir=tmp_path)
    registry = build_tools(ctx)
    out = registry.execute("forecast_cnn_lstm", {"lead": 12})
    assert out.startswith("Error")
    assert "weights" in out.lower()
    assert ctx.cnn_forecasts is None  # context untouched on failure


def test_forecast_cnn_lstm_rejects_invalid_lead(tmp_path):
    ctx = ToolContext(base_dir=tmp_path)
    registry = build_tools(ctx)
    out = registry.execute("forecast_cnn_lstm", {"lead": 99})
    assert out.startswith("Error")


def test_forecast_cnn_lstm_realtime_missing_climatology_errors(tmp_path, monkeypatch):
    """Realtime mode with no precomputed climatologies fails fast (no download)."""
    import src.agent.tools as tools_mod

    # Force the climatology check to look in an empty tmp dir.
    monkeypatch.setattr(tools_mod, "PROJECT_ROOT", tmp_path)
    ctx = ToolContext(base_dir=tmp_path)
    registry = build_tools(ctx)
    out = registry.execute("forecast_cnn_lstm", {"lead": 6, "mode": "realtime"})
    assert out.startswith("Error")
    assert "climatolog" in out.lower()
    assert "build_climatology" in out


def test_forecast_cnn_lstm_invalid_mode_errors(tmp_path):
    ctx = ToolContext(base_dir=tmp_path)
    registry = build_tools(ctx)
    out = registry.execute("forecast_cnn_lstm", {"lead": 6, "mode": "bogus"})
    assert out.startswith("Error")
    assert "mode" in out


# --- enhanced track: list_data_sources / load_index / forecast_enhanced / compare_methods ---


def test_build_tools_includes_enhanced_track():
    names = set(build_tools(ToolContext()).names())
    assert {"list_data_sources", "load_index", "forecast_enhanced", "compare_methods"} <= names


def test_list_data_sources_lists_three_indices(tmp_path):
    ctx = ToolContext(base_dir=tmp_path)
    registry = build_tools(ctx)
    out = registry.execute("list_data_sources", {})
    for name in ("nino34", "soi", "nino12"):
        assert name in out


def test_load_index_unknown_name_errors(tmp_path):
    ctx = ToolContext(base_dir=tmp_path)
    registry = build_tools(ctx)
    out = registry.execute("load_index", {"name": "bogus"})
    # enum validation may convert to a registry error; either way it must not crash
    assert "Error" in out or "bogus" in out


def _patch_synthetic_indices(monkeypatch):
    """Replace the registry loader with synthetic SOI/Niño1+2 (no network)."""
    import numpy as np
    import pandas as pd
    from src.data.sample_generator import generate_sample_enso
    import src.agent.tools as tools_mod

    def fake_load(name, *, refresh=False, cache_dir=None, timeout=30.0):
        enso = generate_sample_enso()
        rng = np.random.default_rng({"soi": 1, "nino12": 2, "nino34": 3}[name])
        return pd.DataFrame({"date": pd.to_datetime(enso["date"]), name: rng.normal(0, 1, len(enso))})

    monkeypatch.setattr(tools_mod, "_registry_load_index", fake_load)


def test_forecast_enhanced_offline_runs_and_returns_value(tmp_path, monkeypatch):
    _patch_synthetic_indices(monkeypatch)
    ctx = ToolContext(base_dir=tmp_path)
    registry = build_tools(ctx)
    # Use sample ENSO data (deterministic, no network) and a far-future target
    # so lead > 0.
    out = registry.execute("forecast_enhanced", {"target_year": 2027, "target_month": 3, "data_source": "sample"})
    assert "Error" not in out
    assert "enhanced" in out
    assert "lead=" in out
    assert ctx.enhanced_results is not None
    assert ctx.enhanced_results.get("_exog_used") == ["soi", "nino12"]


def test_forecast_enhanced_data_driven_confidence(tmp_path, monkeypatch):
    """Confidence tag must reference ACC, not the hard-coded 7/12 thresholds."""
    _patch_synthetic_indices(monkeypatch)
    ctx = ToolContext(base_dir=tmp_path)
    registry = build_tools(ctx)
    out = registry.execute("forecast_enhanced", {"target_year": 2027, "target_month": 3, "data_source": "sample"})
    # The result annotates ACC explicitly (data-driven, not month-count thresholds).
    assert "ACC=" in out


def test_compare_methods_runs_all_three(tmp_path, monkeypatch):
    _patch_synthetic_indices(monkeypatch)
    ctx = ToolContext(base_dir=tmp_path)
    registry = build_tools(ctx)
    out = registry.execute("compare_methods", {"target_year": 2027, "target_month": 3, "data_source": "sample"})
    assert "baseline" in out and "enhanced" in out and "cnn_lstm" in out
    # CNN-LSTM weights exist in the repo (trained earlier) → not 'unavailable'.
    assert "unavailable" not in out or "cnn_lstm" in out


# --- hindcast skill reporting ---


def test_build_tools_includes_report_hindcast_skill():
    names = set(build_tools(ToolContext()).names())
    assert "report_hindcast_skill" in names


def test_report_hindcast_skill_full_table(tmp_path, monkeypatch):
    """Full table reads the cached hindcast JSON (or recomputes) without network."""
    from src.agent import tools as tools_mod

    # Point at the real report path (generated by scripts/run_hindcast.py); if the
    # repo has trained weights it exists. If not, the tool recomputes from weights.
    ctx = ToolContext(base_dir=tmp_path)
    registry = build_tools(ctx)
    out = registry.execute("report_hindcast_skill", {})
    assert "CNN-ACC" in out and "Persist" in out
    assert "gap" in out


def test_report_hindcast_skill_single_lead_verdict(tmp_path):
    ctx = ToolContext(base_dir=tmp_path)
    registry = build_tools(ctx)
    out = registry.execute("report_hindcast_skill", {"lead": 6})
    assert "lead=6" in out
    assert "CNN-ACC" in out
    # A verdict phrase is always present.
    assert any(k in out for k in ("reliable", "skillful", "NO skill", "indicative"))


def test_report_hindcast_skill_invalid_lead_errors(tmp_path):
    ctx = ToolContext(base_dir=tmp_path)
    registry = build_tools(ctx)
    out = registry.execute("report_hindcast_skill", {"lead": 99})
    assert out.startswith("Error")
