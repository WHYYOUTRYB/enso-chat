"""Tests for the hindcast skill evaluation (CNN vs Persistence)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.cnn_lstm import LEAD_MONTHS, INPUT_MONTHS
from src.models.hindcast import (
    HindcastResult,
    _persistence_predictions,
    hindcast_report_text,
)

SODA_TRAIN = Path(__file__).resolve().parents[1] / "data" / "SODA_train.nc"
SODA_LABEL = Path(__file__).resolve().parents[1] / "data" / "SODA_label.nc"
WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "cnn_lstm_soda.pth"
HAS_ALL = SODA_TRAIN.exists() and SODA_LABEL.exists() and WEIGHTS.exists()


def test_persistence_predictions_shape_and_value():
    """Persistence = last observed month, held constant across all leads."""
    nino = np.arange(100, dtype=np.float32)
    starts = [0, 10, 20]
    preds = _persistence_predictions(nino, starts)
    assert preds.shape == (len(starts), LEAD_MONTHS)
    # For start=0, last input month = nino[0+11] = 11, held for every lead.
    assert np.all(preds[0] == nino[0 + INPUT_MONTHS - 1])
    # Every lead column is identical (persistence is lead-invariant).
    assert np.all(preds[0, 0] == preds[0, :])


def test_persistence_uses_last_input_month():
    nino = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 99], dtype=np.float32)
    # window start=0 → input months 0..11, last = nino[11] = 99.
    preds = _persistence_predictions(nino, [0])
    assert preds[0, 0] == 99.0


def test_hindcast_report_text_formats_table():
    res = HindcastResult(
        leads=[1, 2],
        cnn_acc=[0.7, 0.4],
        persistence_acc=[0.9, 0.1],
        skill_gap=[-0.2, 0.3],
        n_samples=10,
        split="test",
    )
    text = hindcast_report_text(res)
    assert "CNN-ACC" in text and "Persist" in text and "gap" in text
    assert "CNN beats Persistence at leads=[2]" in text  # only lead 2 has gap>0


@pytest.mark.skipif(not HAS_ALL, reason="needs SODA data + trained weights")
def test_run_hindcast_end_to_end():
    from src.models.hindcast import run_hindcast

    res = run_hindcast(WEIGHTS, SODA_TRAIN, SODA_LABEL)
    assert len(res.leads) == LEAD_MONTHS
    assert len(res.cnn_acc) == LEAD_MONTHS
    assert len(res.persistence_acc) == LEAD_MONTHS
    assert len(res.skill_gap) == LEAD_MONTHS
    assert res.n_samples > 100
    # ACC is a correlation → bounded in [-1, 1].
    assert all(-1.0 <= a <= 1.0 for a in res.cnn_acc)
    assert all(-1.0 <= a <= 1.0 for a in res.persistence_acc)
    # skill_gap == cnn - persistence (within rounding).
    for c, p, g in zip(res.cnn_acc, res.persistence_acc, res.skill_gap):
        assert abs(g - round(c - p, 4)) < 1e-3


@pytest.mark.skipif(not HAS_ALL, reason="needs SODA data + trained weights")
def test_hindcast_cnn_beats_persistence_at_medium_leads():
    """The CNN's value is at medium leads where Persistence fails (the paper's point)."""
    from src.models.hindcast import run_hindcast

    res = run_hindcast(WEIGHTS, SODA_TRAIN, SODA_LABEL)
    # Around lead 6-12 Persistence should have collapsed (near 0 or negative)
    # while CNN stays positive — that gap is the CNN's reason to exist.
    mid = res.skill_gap[5:12]  # leads 6..12
    assert all(g > 0 for g in mid), f"expected CNN > Persistence at mid leads, got {mid}"
