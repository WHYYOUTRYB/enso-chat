"""Tests for the realtime-domain hindcast reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.tools import ToolContext, build_tools


def test_build_tools_includes_report_realtime_skill():
    names = set(build_tools(ToolContext()).names())
    assert "report_realtime_skill" in names


def test_report_realtime_skill_missing_report_errors(tmp_path, monkeypatch):
    """With no precomputed realtime hindcast, the tool returns a clear error."""
    import src.agent.tools as tools_mod

    monkeypatch.setattr(tools_mod, "REALTIME_HINDCAST_REPORT_PATH", tmp_path / "nope.json")
    ctx = ToolContext(base_dir=tmp_path)
    registry = build_tools(ctx)
    out = registry.execute("report_realtime_skill", {})
    assert out.startswith("Error")
    assert "run_realtime_hindcast" in out


def test_report_realtime_skill_reads_cached_report(tmp_path, monkeypatch):
    """A cached realtime hindcast JSON is read and formatted into the table."""
    import src.agent.tools as tools_mod

    report = tmp_path / "rt_hindcast.json"
    payload = {
        "eval_period": "2020-01_to_2021-12",
        "n_windows": 12,
        "leads": [1, 2, 3],
        "cnn_acc": [0.6, 0.4, 0.2],
        "persistence_acc": [0.8, 0.3, 0.1],
        "skill_gap": [-0.2, 0.1, 0.1],
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(tools_mod, "REALTIME_HINDCAST_REPORT_PATH", report)
    ctx = ToolContext(base_dir=tmp_path)
    registry = build_tools(ctx)

    out = registry.execute("report_realtime_skill", {})
    assert "CNN-ACC" in out and "Persist" in out
    assert "leads=[2, 3]" in out  # only leads 2,3 have gap>0

    single = registry.execute("report_realtime_skill", {"lead": 1})
    assert "lead=1" in single
    assert "NO cross-domain skill" in single  # lead 1 gap=-0.2


def test_report_realtime_skill_invalid_lead(tmp_path, monkeypatch):
    import src.agent.tools as tools_mod

    report = tmp_path / "rt.json"
    report.write_text(json.dumps({"leads": [1, 2], "cnn_acc": [0.5, 0.4],
                                  "persistence_acc": [0.3, 0.2], "skill_gap": [0.2, 0.2],
                                  "n_windows": 5, "eval_period": "x"}), encoding="utf-8")
    monkeypatch.setattr(tools_mod, "REALTIME_HINDCAST_REPORT_PATH", report)
    ctx = ToolContext(base_dir=tmp_path)
    registry = build_tools(ctx)
    out = registry.execute("report_realtime_skill", {"lead": 99})
    assert out.startswith("Error")
