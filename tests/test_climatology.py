"""Tests for climatology computation and anomalization."""

from __future__ import annotations

import numpy as np

from src.data.climatology import (
    CLIMATOLOGY_YEARS,
    anomalize,
    compute_monthly_climatology,
)


def _make_absolute_fields(n_years: int = 3, lat: int = 4, lon: int = 5):
    """Fields with a month-dependent mean (the climatology signal) + noise."""
    rng = np.random.default_rng(0)
    months = np.tile(np.arange(1, 13), n_years)
    base = np.zeros((len(months), lat, lon), dtype=np.float32)
    for i, m in enumerate(months):
        base[i] = m * 0.5  # January ~0.5, December ~6.0
    fields = base + rng.normal(0, 0.1, (len(months), lat, lon)).astype(np.float32)
    return fields, months


def test_climatology_shape_is_12_months():
    fields, months = _make_absolute_fields()
    clim = compute_monthly_climatology(fields, months)
    assert clim.shape == (12, fields.shape[1], fields.shape[2])


def test_climatology_recovers_month_signal():
    fields, months = _make_absolute_fields()
    clim = compute_monthly_climatology(fields, months)
    # January mean ~0.5, December mean ~6.0 (noise averages out over 3 years).
    assert abs(float(clim[0].mean()) - 0.5) < 0.1
    assert abs(float(clim[11].mean()) - 6.0) < 0.1


def test_anomalize_yields_near_zero_mean():
    fields, months = _make_absolute_fields()
    clim = compute_monthly_climatology(fields, months)
    anom = anomalize(fields, clim, months)
    # Anomalies should have ~0 mean (the climatology signal removed).
    assert abs(float(np.nanmean(anom))) < 1e-3


def test_anomalize_month_alignment():
    """A field exactly equal to its month's climatology → anomaly 0."""
    clim = np.zeros((12, 2, 2), dtype=np.float32)
    for m in range(12):
        clim[m] = m + 1  # Jan=1, Feb=2, ...
    fields = np.stack([clim[0], clim[5], clim[11]])  # Jan, Jun, Dec values
    months = np.array([1, 6, 12])
    anom = anomalize(fields, clim, months)
    assert np.allclose(anom, 0.0)


def test_climatology_years_default_is_30yr_normal():
    assert list(CLIMATOLOGY_YEARS) == list(range(1991, 2021))


def test_compute_rejects_bad_shapes():
    import pytest

    with pytest.raises(ValueError):
        compute_monthly_climatology(np.zeros((5, 4)), np.array([1, 2, 3, 4, 5]))  # 2D not 3D
    with pytest.raises(ValueError):
        compute_monthly_climatology(np.zeros((5, 4, 4)), np.array([1, 2, 3]))  # len mismatch
