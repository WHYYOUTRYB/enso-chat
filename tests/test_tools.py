from pathlib import Path

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
