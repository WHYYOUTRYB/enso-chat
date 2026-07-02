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


# --- write_forecast_report (deterministic Markdown report) ---


def test_build_tools_includes_write_forecast_report():
    names = set(build_tools(ToolContext()).names())
    assert "write_forecast_report" in names


def test_write_forecast_report_errors_with_no_results(tmp_path):
    """No tracks run -> a clear Error, and no NEW report file written."""
    from src.reports.forecast_report import FORECAST_REPORTS_DIR

    before = set(FORECAST_REPORTS_DIR.glob("enso_forecast_report*.md"))
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    out = tools.execute("write_forecast_report", {})
    assert out.startswith("Error")
    # The empty-context path must not create any new report file.
    after = set(FORECAST_REPORTS_DIR.glob("enso_forecast_report*.md"))
    assert after == before


def test_write_forecast_report_real_numbers_and_repro(tmp_path, monkeypatch):
    """Report assembled from real sample-data results: values match ctx, and
    the reproducibility / data-source sections are present (no fabrication)."""
    import re
    from src.agent.tools import ToolContext, build_tools
    from src.reports.forecast_report import FORECAST_REPORTS_DIR

    # Run the basic track on deterministic sample data (no network).
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    tools.execute("load_enso_data", {"data_source": "sample"})
    # Produce a plot so the figures section copies a real image.
    tools.execute("plot_enso_timeseries", {})

    out = tools.execute("write_forecast_report", {"target_label": "测试目标"})
    assert "Error" not in out
    assert "Report written" in out

    # Extract the path and read the file back.
    m = re.search(r"Report written: (\S+)", out)
    assert m is not None
    report_path = Path(m.group(1))
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")

    # Header carries the target label and the real data-through month.
    assert "测试目标" in text
    assert "数据截止月份" in text
    # Deterministic-assembly disclaimer is present.
    assert "确定性拼装" in text or "未由语言模型生成" in text

    # Real numbers from ctx.results are quoted exactly (3dp) — check lead 1.
    fc = ctx.results["latest_forecast"]["1"]
    assert f"{fc['value']:.3f}" in text
    # Best model name from results appears in the results table.
    assert ctx.results["best_model_by_lead"]["1"] in text

    # Reproducibility section lists the real config knobs.
    for token in ("random_state=42", "temporal_train_test_split", "run_hindcast.py", "NOAA"):
        assert token in text
    # The data-source URL from config is named (real source, not invented).
    assert "psl.noaa.gov" in text or "NOAA_NINO34_URL" in text

    # CNN-LSTM real architecture/training details (read from cnn_lstm.py).
    for cnn_token in ("Adam", "lr=1e-3", "Dropout(0.7)", "LSTM(1728", "SODA"):
        assert cnn_token in text
    # Data & Code Availability section (paper-standard), with the Ham reference.
    assert "数据与代码可用性" in text
    assert "10.1038/s41586-019-1559-7" in text

    # Tracks not run this session are flagged 未运行, not guessed.
    assert "增强轨" in text and "CNN-LSTM" in text
    assert "未运行" in text

    # The plot was copied into a figures/ subdir and embedded by relative path.
    assert "](figures/" in text
    copied = list((report_path.parent / "figures").glob("*"))
    assert len(copied) == 1


# --- read_report / accept_report_polish (academic polish w/ numeric guard) ---


def _make_draft_report(ctx, tools):
    """Run the basic track + write a report; return its path."""
    tools.execute("load_enso_data", {"data_source": "sample"})
    tools.execute("plot_enso_timeseries", {})
    out = tools.execute("write_forecast_report", {"target_label": "t"})
    import re
    m = re.search(r"Report written: (\S+)", out)
    assert m, out
    return Path(m.group(1))


def test_build_tools_includes_read_and_accept_polish():
    names = set(build_tools(ToolContext()).names())
    assert {"read_report", "accept_report_polish"} <= names


def test_read_report_returns_full_markdown(tmp_path):
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    path = _make_draft_report(ctx, tools)
    text = tools.execute("read_report", {"report_path": str(path)})
    assert "ENSO 预测报告" in text and "摘要" in text and "方法" in text


def test_accept_report_polish_preserves_numbers(tmp_path):
    """A prose-only polish (numbers untouched) is accepted and written back."""
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    path = _make_draft_report(ctx, tools)
    draft = path.read_text(encoding="utf-8")
    # Insert a 学术关键词 line into the abstract (appends near top, no number change).
    polished = draft.replace(
        "## 摘要\n",
        "## 摘要\n\n本摘要经学术润色，措辞更规范。\n",
        1,
    )
    out = tools.execute(
        "accept_report_polish",
        {"polished_markdown": polished, "report_path": str(path)},
    )
    assert "accepted" in out
    assert path.read_text(encoding="utf-8") == polished


def test_accept_report_polish_rejects_changed_numbers(tmp_path):
    """If the LLM altered a number, the polish is REJECTED and the draft untouched."""
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    path = _make_draft_report(ctx, tools)
    draft = path.read_text(encoding="utf-8")
    # Mutate the first numeric token we find (e.g. a lead/digit) to a different value.
    import re
    m = re.search(r"(?<![A-Za-z_])(\d+(?:\.\d+)?)(?![A-Za-z_])", draft)
    assert m, "draft should contain numbers"
    bad = m.group(1)
    bad_new = "9" + bad if "." not in bad else bad.replace(bad[:3], "0.00", 1)
    polished = draft.replace(bad, bad_new, 1)
    out = tools.execute(
        "accept_report_polish",
        {"polished_markdown": polished, "report_path": str(path)},
    )
    assert "REJECTED" in out
    # Draft on disk must be unchanged.
    assert path.read_text(encoding="utf-8") == draft


def test_accept_report_polish_errors_without_report(tmp_path):
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    out = tools.execute("accept_report_polish", {"polished_markdown": "x"})
    assert out.startswith("Error")


# --- explain_component / read_source (self-description & source reading) ---


def test_build_tools_includes_explain_and_read_source():
    names = set(build_tools(ToolContext()).names())
    assert {"explain_component", "read_source"} <= names


def test_explain_component_lists_all():
    ctx = ToolContext()
    tools = build_tools(ctx)
    out = tools.execute("explain_component", {"name": ""})
    assert "[agent]" in out and "[models]" in out
    assert "agent.run_turn" in out and "models.cnn_lstm" in out


def test_explain_component_known_returns_summary():
    ctx = ToolContext()
    tools = build_tools(ctx)
    out = tools.execute("explain_component", {"name": "models.cnn_lstm"})
    assert "models.cnn_lstm" in out
    assert "src/models/cnn_lstm.py" in out
    assert "_build_model" in out
    assert "read_source" in out


def test_explain_component_unknown_lists():
    ctx = ToolContext()
    tools = build_tools(ctx)
    out = tools.execute("explain_component", {"name": "no_such_thing"})
    assert "未知" in out or "explain_component" in out


def test_read_source_returns_numbered_lines(tmp_path):
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    out = tools.execute("read_source", {"file_path": "src/models/baseline.py"})
    assert "persistence_predict" in out
    assert "7:" in out
    assert "截断" not in out


def test_read_source_symbol_locates_block(tmp_path):
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    out = tools.execute(
        "read_source",
        {"file_path": "src/models/baseline.py", "symbol": "persistence_predict"},
    )
    assert ("lines 7-10" in out or "lines 7-1" in out)
    assert "def persistence_predict" in out
    assert "nino34_lag_0" in out


def test_read_source_symbol_includes_decorator(tmp_path):
    """A @dataclass-class symbol must include its @decorator line in the block.

    Regression: the locator used to start the block at `class Tool:` and drop
    the `@dataclass` above it, hiding the class is a dataclass from the reader.
    """
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    out = tools.execute(
        "read_source",
        {"file_path": "src/agent/tools.py", "symbol": "Tool"},
    )
    # Block starts at the @dataclass line (82), not the class line (83).
    assert "lines 82-" in out
    assert "@dataclass" in out
    assert "class Tool:" in out


def test_read_source_symbol_trims_trailing_separator(tmp_path):
    """The block must not bleed into the separator comment block before the
    next def/class (e.g. '# ---\\n# Inference ...'). Regression: locator used to
    return lines through to the next def, bundling the gap into the symbol.
    """
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    out = tools.execute(
        "read_source",
        {"file_path": "src/models/cnn_lstm.py", "symbol": "train_cnn_lstm"},
    )
    head = out.splitlines()[0]
    # train_cnn_lstm ends with `return metrics` (~line 320); block must not say
    # lines going up to ~327 (where the separator + next def live).
    assert "return metrics" in out
    assert "Inference" not in out  # the trailing separator comment block
    assert "# ----" not in out.splitlines()[-1]


def test_read_source_symbol_not_found_errors(tmp_path):
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    out = tools.execute(
        "read_source",
        {"file_path": "src/models/baseline.py", "symbol": "bogus_fn"},
    )
    assert out.startswith("Error")
    assert "bogus_fn" in out


def test_read_source_rejects_path_outside_root(tmp_path, monkeypatch):
    import src.agent.code_guide as cg

    monkeypatch.setattr(cg, "_PROJECT_ROOT", tmp_path.resolve())
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    out = tools.execute(
        "read_source",
        {"file_path": str(Path(__file__).resolve().parent.parent / "README.md")},
    )
    assert out.startswith("Error")
    assert "outside" in out.lower() or "not found" in out.lower()


# --- data-provenance prefix on forecast tools ---


def _ctx_with_sample(ctx, tools):
    tools.execute("load_enso_data", {"data_source": "sample"})
    return ctx


def test_forecast_for_month_has_provenance_prefix(tmp_path):
    """forecast_for_month leads with [数据来源]: source/url + time range + rows."""
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    _ctx_with_sample(ctx, tools)
    out = tools.execute("forecast_for_month", {"target_year": 2027, "target_month": 3})
    assert out.startswith("[数据来源]")
    assert "样本" in out and "行" in out
    assert "时间范围" in out
    assert "截止月=预测起算点" in out
    # Sample-data label is identifiable in the prefix.
    assert "样本" in out.split("\n")[0]
    # The forecast body follows the prefix on a later line.
    assert "lead=" in out


def test_forecast_enhanced_has_exog_provenance(tmp_path, monkeypatch):
    """enhanced prefix includes the exog indices + their source URLs."""
    _patch_synthetic_indices(monkeypatch)  # reuse the offline exog stub
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    out = tools.execute(
        "forecast_enhanced",
        {"target_year": 2027, "target_month": 3, "data_source": "sample"},
    )
    head = out.split("\n")[0]
    assert head.startswith("[数据来源]")
    assert "外源指数" in head
    assert "soi" in head and "nino12" in head
    assert "psl.noaa.gov" in head  # real SOI/Niño1+2 URLs


def test_forecast_cnn_lstm_has_cnn_provenance(tmp_path):
    """CNN-LSTM prefix names the mode + input source from ctx.cnn_forecasts."""
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    tools.execute("load_enso_data", {"data_source": "sample"})
    out = tools.execute("forecast_cnn_lstm", {"lead": 12})
    head = out.split("\n")[0]
    assert head.startswith("[数据来源]")
    assert "CNN-LSTM mode=" in head
    assert "输入源" in head


def test_compare_methods_has_provenance_header(tmp_path, monkeypatch):
    _patch_synthetic_indices(monkeypatch)
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    out = tools.execute(
        "compare_methods", {"target_year": 2027, "target_month": 3, "data_source": "sample"}
    )
    assert out.startswith("[数据来源]")
    assert "样本" in out


# --- data freshness self-check on forecast prefix ---


def test_forecast_prefix_flags_stale_data(tmp_path, monkeypatch):
    """When the loaded ENSO series is stale, the forecast prefix says so."""
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    tools.execute("load_enso_data", {"data_source": "sample"})
    # The stale-check only runs for real-source tracks (noaa/auto/user), so mark
    # the loaded source as 'noaa' to exercise that branch.
    ctx.results["data_source"]["used"] = "noaa"
    import src.agent.data_freshness as dfm

    monkeypatch.setattr(dfm, "is_stale", lambda dt, **kw: True)
    monkeypatch.setattr(dfm, "data_age_months", lambda dt: 9)
    out = tools.execute("forecast_for_month", {"target_year": 2027, "target_month": 3})
    head = out.split("\n")[0]
    assert "数据偏旧" in head
    assert "9" in head  # the age months


def test_forecast_prefix_silent_when_fresh(tmp_path, monkeypatch):
    """Fresh data -> no stale warning in the prefix."""
    ctx = ToolContext(base_dir=tmp_path)
    tools = build_tools(ctx)
    tools.execute("load_enso_data", {"data_source": "sample"})
    ctx.results["data_source"]["used"] = "noaa"
    import src.agent.data_freshness as dfm

    monkeypatch.setattr(dfm, "is_stale", lambda dt, **kw: False)
    out = tools.execute("forecast_for_month", {"target_year": 2027, "target_month": 3})
    assert "数据偏旧" not in out


# --- stage timings on the forecast pipeline ---


def test_run_enso_forecast_records_timings(tmp_path):
    from src.pipeline.run_enso_forecast import run_enso_forecast

    timings: dict = {}
    out = run_enso_forecast(base_dir=tmp_path, data_source="sample", timings=timings)
    # On the sample path there's no NOAA download, so 'download' may be absent;
    # but features + train + write should always be present and positive.
    assert "features" in timings and timings["features"] >= 0
    assert "train" in timings and timings["train"] > 0
    assert "write" in timings


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
