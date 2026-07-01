"""Tests for the data-source registry and the generalized year/month parser."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.source_registry import (
    REGISTRY,
    IndexLoadError,
    describe_sources,
    list_sources,
    parse_year_month_table,
)

# Minimal PSL-format sample: a "start end" header line, two data years, a
# missing value (-99.99 in March of 1866), and trailing metadata.
_SAMPLE_SOI = """1866 2025
1866  -0.6  -0.3  -99.99  0.5  0.2  -0.1  0.3  0.4  -0.2  0.1  -0.5  0.6
1867   0.1   0.2   0.3   0.4  0.5  0.6  0.7  0.8  0.9  1.0  1.1  1.2
https://psl.noaa.gov/...
reference: Ropelewski & Jones
units=norm
"""


def test_parse_year_month_table_basic():
    df = parse_year_month_table(_SAMPLE_SOI, value_col="soi")
    assert list(df.columns) == ["date", "soi"]
    # 12 months of 1866 (March missing dropped → 11) + 12 of 1867 = 23 rows.
    assert len(df) == 23
    assert -99.99 not in df["soi"].values  # missing values dropped
    assert df["date"].iloc[0] == pd.Timestamp("1866-01-01")
    # March 1866 (-99.99) skipped — February then April.
    assert df["date"].iloc[1] == pd.Timestamp("1866-02-01")
    assert df["date"].iloc[2] == pd.Timestamp("1866-04-01")


def test_parse_year_month_table_skips_short_metadata_lines():
    # Header (2 tokens) and trailing metadata (<13 tokens) must be ignored.
    df = parse_year_month_table(_SAMPLE_SOI, value_col="soi")
    # No row should come from a metadata line.
    assert all(y in (1866, 1867) for y in df["date"].dt.year.unique())


def test_parse_year_month_table_empty_raises():
    with pytest.raises(IndexLoadError):
        parse_year_month_table("only metadata here\nno numbers", value_col="x")


def test_registry_has_three_sources():
    assert set(REGISTRY) == {"nino34", "soi", "nino12"}


def test_list_sources_shape():
    items = list_sources()
    assert len(items) == 3
    assert all({"name", "description", "coverage"} == set(s) for s in items)


def test_describe_sources_mentions_all():
    text = describe_sources()
    for name in ("nino34", "soi", "nino12"):
        assert name in text
