"""Tests for the realtime spatial-field fetcher (network-isolated)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.realtime_fetch import (
    ChannelResult,
    TARGET_LAT,
    TARGET_LON,
    _align_to_cutoff,
    _resample_to_target,
    fetch_realtime_window,
)


def test_resample_to_soda_grid():
    """Native 2.5° grid → SODA (24 lat × 72 lon, 5° steps)."""
    lat = np.arange(-90, 90.1, 2.5)
    lon = np.arange(0, 360, 2.5)
    data = np.random.RandomState(0).rand(3, len(lat), len(lon)).astype("float32")
    da = xr.DataArray(data, dims=("t", "lat", "lon"), coords={"lat": lat, "lon": lon})
    out = _resample_to_target(da)
    assert out.shape == (3, len(TARGET_LAT), len(TARGET_LON))
    assert float(out.lat.min()) == -55.0 and float(out.lat.max()) == 60.0
    assert float(out.lon.min()) == 0.0 and float(out.lon.max()) == 355.0


def test_resample_normalizes_negative_longitude():
    """A -180..180 longitude grid is normalized to 0..360."""
    lat = np.arange(-90, 90.1, 2.5)
    lon = np.arange(-180, 180, 2.5)
    data = np.random.RandomState(1).rand(1, len(lat), len(lon)).astype("float32")
    da = xr.DataArray(data, dims=("t", "lat", "lon"), coords={"lat": lat, "lon": lon})
    out = _resample_to_target(da)
    assert out.lon.min() >= 0.0


def test_align_to_cutoff_uses_earliest_channel():
    """The common window ends at the earliest (most lagging) channel cutoff."""
    chans = {
        "sst": ChannelResult(np.zeros((12, 24, 72), "float32"), None, "2026-06", None),
        "t300": ChannelResult(np.zeros((12, 24, 72), "float32"), None, "2026-05", None),
        "ua": ChannelResult(np.zeros((12, 24, 72), "float32"), None, "2026-02", None),  # bottleneck
        "va": ChannelResult(np.zeros((12, 24, 72), "float32"), None, "2026-02", None),
    }
    window, cutoff, missing = _align_to_cutoff(chans)
    assert window.shape == (12, 24, 72, 4)
    assert cutoff == "2026-02"
    assert missing == []


def test_align_to_cutoff_zero_fills_missing_channel():
    chans = {
        "sst": ChannelResult(np.zeros((12, 24, 72), "float32"), None, "2026-06", None),
        "t300": ChannelResult(np.zeros((12, 24, 72), "float32"), None, "2026-05", None),
        "ua": ChannelResult(np.zeros((12, 24, 72), "float32"), None, "2026-02", None),
        "va": ChannelResult(None, None, None, "download failed"),
    }
    window, cutoff, missing = _align_to_cutoff(chans)
    assert "va" in missing
    # The va channel (index 3) is all zeros.
    assert np.all(window[..., 3] == 0.0)


def test_align_raises_when_all_fail():
    from src.data.realtime_fetch import RealtimeFetchError

    chans = {n: ChannelResult(None, None, None, "fail") for n in ("sst", "t300", "ua", "va")}
    with pytest.raises(RealtimeFetchError):
        _align_to_cutoff(chans)


def test_fetch_realtime_window_uses_monkeypatched_fetchers(monkeypatch, tmp_path):
    """End-to-end assembly with all four fetchers stubbed (no network)."""
    import src.data.realtime_fetch as rtf

    months = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2], dtype=int)
    fields = np.random.RandomState(2).rand(12, 24, 72).astype("float32")

    def fake_sst(n, cache):
        return ChannelResult(fields.copy(), months.copy(), "2026-06", None)

    def fake_t300(n, cache):
        return ChannelResult(fields.copy(), months.copy(), "2026-05", None)

    def fake_wind(comp, n, cache):
        return ChannelResult(fields.copy(), months.copy(), "2026-02", None)  # bottleneck

    monkeypatch.setattr(rtf, "_fetch_sst", fake_sst)
    monkeypatch.setattr(rtf, "_fetch_t300", fake_t300)
    monkeypatch.setattr(rtf, "_fetch_wind", fake_wind)

    # Stub anomalize to pass-through (skip climatology files).
    monkeypatch.setattr(rtf, "anomalize", lambda f, c, m: f)
    monkeypatch.setattr(rtf, "load_climatology", lambda p: np.zeros((12, 24, 72), "float32"))

    window, cutoff, missing = fetch_realtime_window(cache_dir=tmp_path)
    assert window.shape == (12, 24, 72, 4)
    assert cutoff == "2026-02"  # wind-limited
    assert missing == []
