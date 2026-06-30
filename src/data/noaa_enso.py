from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

from src.config import DEFAULT_NOAA_NINO34_URL


class NoaaEnsoDownloadError(RuntimeError):
    """Raised when NOAA ENSO data cannot be downloaded or parsed."""


MISSING_VALUE = -99.99


def parse_noaa_nino34_table(raw_text: str) -> pd.DataFrame:
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
                    "nino34": value,
                }
            )

    if not rows:
        raise NoaaEnsoDownloadError("No valid NOAA Niño3.4 rows were parsed")

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def download_noaa_enso_text(url: str = DEFAULT_NOAA_NINO34_URL, timeout: float = 20.0) -> str:
    try:
        with urlopen(url, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise NoaaEnsoDownloadError(f"NOAA ENSO download returned HTTP status {status}")
            payload = response.read()
    except HTTPError as exc:
        raise NoaaEnsoDownloadError(f"NOAA ENSO download returned HTTP status {exc.code}") from exc
    except URLError as exc:
        raise NoaaEnsoDownloadError(f"NOAA ENSO download failed: {exc.reason}") from exc
    except OSError as exc:
        raise NoaaEnsoDownloadError(f"NOAA ENSO download failed: {exc}") from exc

    text = payload.decode("utf-8", errors="replace")
    if not text.strip():
        raise NoaaEnsoDownloadError("NOAA ENSO download returned empty content")
    return text


def load_or_download_noaa_enso(
    raw_path: Path,
    processed_path: Path,
    url: str = DEFAULT_NOAA_NINO34_URL,
    refresh: bool = False,
    timeout: float = 20.0,
) -> pd.DataFrame:
    if processed_path.exists() and not refresh:
        df = pd.read_csv(processed_path, parse_dates=["date"])
        required = {"date", "nino34"}
        missing = required.difference(df.columns)
        if missing:
            raise NoaaEnsoDownloadError(
                f"Cached NOAA ENSO CSV missing required columns: {sorted(missing)}"
            )
        return df.sort_values("date").reset_index(drop=True)

    raw_text = download_noaa_enso_text(url=url, timeout=timeout)
    parsed = parse_noaa_nino34_table(raw_text)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(raw_text, encoding="utf-8")
    parsed.to_csv(processed_path, index=False)
    return parsed
