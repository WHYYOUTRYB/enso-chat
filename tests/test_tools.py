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
