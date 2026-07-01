"""Tests for make_enso_supervised_table exogenous-variable support."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.sample_generator import generate_sample_enso
from src.features.enso_features import make_enso_supervised_table


def _enso_with_exog():
    enso = generate_sample_enso(periods=120)
    rng = np.random.default_rng(7)
    enso["soi"] = rng.normal(0, 1, len(enso))
    enso["nino12"] = rng.normal(0, 1, len(enso))
    return enso


def test_exog_cols_add_lag_features():
    enso = _enso_with_exog()
    table, feats = make_enso_supervised_table(enso, exog_cols=["soi", "nino12"])
    # Each exog col adds max_lag+1 = 13 lag features.
    assert "soi_lag_0" in feats and "soi_lag_12" in feats
    assert "nino12_lag_0" in feats and "nino12_lag_12" in feats
    assert "soi_lag_13" not in feats  # max_lag cap respected


def test_exog_cols_in_dataframe():
    enso = _enso_with_exog()
    table, _ = make_enso_supervised_table(enso, exog_cols=["soi", "nino12"])
    assert "soi_lag_0" in table.columns
    assert "nino12_lag_5" in table.columns


def test_no_exog_is_backward_compatible():
    """exog_cols=None must produce the exact original feature set."""
    enso = _enso_with_exog()
    _, feats_none = make_enso_supervised_table(enso)
    # Original 17 features: 13 nino34 lags + 2 roll means + 2 month sin/cos.
    assert len(feats_none) == 17
    assert not any(f.startswith("soi_") or f.startswith("nino12_") for f in feats_none)


def test_exog_missing_column_raises():
    enso = generate_sample_enso(periods=60)  # no soi/nino12 columns
    with pytest.raises(ValueError, match="missing required columns"):
        make_enso_supervised_table(enso, exog_cols=["soi"])
