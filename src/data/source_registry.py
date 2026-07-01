"""Registry of known climate-index data sources (the "data discovery" layer).

Each source is a NOAA/PSL monthly timeseries in the same ASCII format as
Niño3.4: lines of ``YYYY  v1 v2 ... v12`` with ``-99.99`` missing-value
markers, plus a leading ``start_year end_year`` line and trailing metadata
lines (skipped automatically). Because every registered source shares the
format, :func:`parse_year_month_table` generalizes
``noaa_enso.parse_noaa_nino34_table`` and is reused for all of them.

The registry is the controlled "list/select/load" surface the agent sees via
the ``list_data_sources`` / ``load_index`` tools. Adding a source means adding
one entry here — no tool changes. Free-form web discovery (searching the open
web for arbitrary sources) is deliberately out of scope for now.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import (
    DEFAULT_NINO12_URL,
    DEFAULT_NOAA_NINO34_URL,
    DEFAULT_SOI_URL,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

MISSING_VALUE = -99.99


class IndexLoadError(RuntimeError):
    """Raised when a registered index cannot be downloaded or parsed."""


@dataclass(frozen=True)
class DataSource:
    """One downloadable climate-index series."""

    name: str  # short key, e.g. "soi"
    description: str  # human-readable, shown by list_data_sources
    url: str
    value_col: str  # column name in the returned DataFrame
    coverage: str  # human-readable date coverage


REGISTRY: dict[str, DataSource] = {
    "nino34": DataSource(
        name="nino34",
        description="Niño3.4 SST anomaly — the ENSO target index itself (central Pacific).",
        url=DEFAULT_NOAA_NINO34_URL,
        value_col="nino34",
        coverage="1870-present",
    ),
    "soi": DataSource(
        name="soi",
        description="Southern Oscillation Index (Tahiti-Darwin SLP) — atmospheric ENSO precursor.",
        url=DEFAULT_SOI_URL,
        value_col="soi",
        coverage="1866-present",
    ),
    "nino12": DataSource(
        name="nino12",
        description="Niño1+2 SST anomaly — eastern Pacific upwelling region, ENSO development precursor.",
        url=DEFAULT_NINO12_URL,
        value_col="nino12",
        coverage="1870-present",
    ),
}


def parse_year_month_table(raw_text: str, value_col: str = "value") -> pd.DataFrame:
    """Parse the PSL ``YYYY v1..v12`` ASCII format into ``(date, value_col)``.

    Generalizes ``noaa_enso.parse_noaa_nino34_table``: skips the leading
    ``start end`` year line and trailing metadata lines (anything with <13
    whitespace tokens), drops ``-99.99`` missing values.
    """
    rows: list[dict] = []
    for line in raw_text.splitlines():
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        for month_index, value_text in enumerate(parts[1:13], start=1):
            try:
                value = float(value_text)
            except ValueError:
                continue
            if value == MISSING_VALUE:
                continue
            rows.append(
                {
                    "date": pd.Timestamp(year=year, month=month_index, day=1),
                    value_col: value,
                }
            )
    if not rows:
        raise IndexLoadError(f"No valid rows were parsed for index '{value_col}'")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _cache_paths(name: str, cache_dir: Path | None) -> tuple[Path, Path]:
    base = cache_dir if cache_dir is not None else RAW_DATA_DIR
    raw = base / f"{name}_raw.txt"
    processed = (cache_dir if cache_dir is not None else PROCESSED_DATA_DIR) / f"{name}.csv"
    return raw, processed


def _download_text(url: str, timeout: float = 30.0) -> str:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "enso-chat/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise IndexLoadError(f"{url} returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise IndexLoadError(f"{url} download failed: {exc}") from exc
    text = payload.decode("utf-8", errors="replace")
    if not text.strip():
        raise IndexLoadError(f"{url} returned empty content")
    return text


def load_index(
    name: str,
    *,
    refresh: bool = False,
    cache_dir: Path | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Download (or read cached) a registered index as ``(date, value_col)``.

    Caches raw text + parsed CSV under ``data/raw`` and ``data/processed``
    (or ``cache_dir``). On any download/parse failure raises
    :class:`IndexLoadError` — callers (the tool layer) catch it and surface a
    string to the LLM rather than crashing the loop.
    """
    if name not in REGISTRY:
        raise IndexLoadError(f"Unknown index '{name}'. Available: {sorted(REGISTRY)}")
    src = REGISTRY[name]
    raw_path, processed_path = _cache_paths(name, cache_dir)

    if processed_path.exists() and not refresh:
        df = pd.read_csv(processed_path, parse_dates=["date"])
        if src.value_col not in df.columns:
            raise IndexLoadError(f"Cached {name} CSV missing column '{src.value_col}'")
        return df.sort_values("date").reset_index(drop=True)

    raw_text = _download_text(src.url, timeout=timeout)
    parsed = parse_year_month_table(raw_text, value_col=src.value_col)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(raw_text, encoding="utf-8")
    parsed.to_csv(processed_path, index=False)
    return parsed


def list_sources() -> list[dict]:
    """Return a JSON-serializable summary of every registered source."""
    return [
        {"name": s.name, "description": s.description, "coverage": s.coverage}
        for s in REGISTRY.values()
    ]


def describe_sources() -> str:
    """Multi-line human/LLM-readable summary of available sources."""
    lines = ["Available data sources:"]
    for s in REGISTRY.values():
        lines.append(f"- {s.name}: {s.description} (coverage {s.coverage})")
    return "\n".join(lines)
