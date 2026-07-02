"""Tests for data-freshness self-check + background async retrain."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from src.agent.data_freshness import (
    BackgroundRetrainer,
    data_age_months,
    freshness_note,
    is_stale,
)


def test_data_age_months_arithmetic():
    # Year/month only — day-agnostic.
    assert data_age_months("2026-04") is not None
    assert data_age_months("garbage") is None
    assert data_age_months(None) is None


def test_is_stale_threshold(monkeypatch):
    import src.agent.data_freshness as dfm

    monkeypatch.setattr(dfm, "data_age_months", lambda dt: 4)
    assert is_stale("2026-03") is True
    monkeypatch.setattr(dfm, "data_age_months", lambda dt: 1)
    assert is_stale("2026-06") is False
    monkeypatch.setattr(dfm, "data_age_months", lambda dt: None)
    assert is_stale("garbage") is False  # unknown age -> not stale (don't block)


def test_freshness_note_stale_and_fresh(monkeypatch):
    import src.agent.data_freshness as dfm

    monkeypatch.setattr(dfm, "data_age_months", lambda dt: 4)
    note = freshness_note("2026-03")
    assert "偏旧" in note and "4" in note

    monkeypatch.setattr(dfm, "data_age_months", lambda dt: 1)
    note = freshness_note("2026-06")
    assert "新鲜" in note and "1" in note

    monkeypatch.setattr(dfm, "data_age_months", lambda dt: None)
    note = freshness_note("garbage")
    assert "无法解析" in note


def test_background_retrainer_runs_and_completes(tmp_path):
    """A background retrain on sample data finishes and yields an output."""
    r = BackgroundRetrainer()
    assert r.running is False
    started = r.start_if_idle(base_dir=tmp_path, data_source="sample")
    assert started is True
    # Dedup: a second start while running is a no-op.
    assert r.start_if_idle(base_dir=tmp_path, data_source="sample") is False

    out = None
    for _ in range(150):  # up to ~30s
        out = r.take_completed()
        if out is not None or r.last_error:
            break
        time.sleep(0.2)
    assert r.last_error is None, r.last_error
    assert out is not None
    assert len(out.enso) > 0
    assert out.results is not None
    # After taking, the holder is reset — a fresh start works again.
    assert r.running is False


def test_background_retrainer_dedups(tmp_path):
    r = BackgroundRetrainer()
    r.start_if_idle(base_dir=tmp_path, data_source="sample")
    second = r.start_if_idle(base_dir=tmp_path, data_source="sample")
    assert second is False  # already running
    # Drain.
    for _ in range(150):
        if r.take_completed() is not None or r.last_error:
            break
        time.sleep(0.2)
