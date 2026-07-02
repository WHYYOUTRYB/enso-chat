"""Data-freshness self-check + background async retrain for the ENSO agent.

Two concerns that previously hurt UX:

* **"拉取最新 NOAA 很慢"** — the slowness was never the download (~1s) but the
  synchronous model retrain (~4s) that ``load_enso_data(refresh_noaa=True)``
  triggers inline. We now let a forecast proceed on current data and retrain in
  a background thread, swapping results in once ready.
* **stale data silently used as the forecast baseline** — ENSO is monthly and
  the cutoff month *is* the forecast origin, so a series lagging months behind
  "now" quietly degrades every prediction. Forecast tools now self-check
  freshness and surface a note (and can kick off a background refresh).

This module is pure logic + a threading wrapper — no Streamlit import, so it is
unit-testable. The UI layer (:mod:`src.web.app`) wires it into session_state.

Concurrency model
-----------------
:class:`BackgroundRetrainer` runs ``run_enso_forecast(refresh_noaa=True)`` in a
worker thread that **only writes artifacts to disk** (results JSON, predictions
CSV) — it never touches the live ``ToolContext``, so there is no read/write race
with the foreground ``run_turn``. When the worker finishes it stores the fresh
``EnsoForecastOutput`` on a thread-safe holder; the foreground checks the holder
at the start of the next turn and, if ready, atomically reloads results into the
context (a few-ms read from disk). If a retrain is already running, a second
request is a no-op (deduped).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import ENSO_STALE_MONTHS


def data_age_months(data_through: str | pd.Timestamp) -> int | None:
    """Whole months between the data's latest month and "now".

    ``data_through`` is the ENSO series' max date (the forecast baseline), as a
    'YYYY-MM' string or Timestamp. Returns None if it cannot be parsed. Uses
    year/month arithmetic only (day-agnostic), so '2026-04' vs a 2026-07 'now'
    is 3 months regardless of day.
    """
    try:
        ts = pd.Timestamp(data_through)
    except (ValueError, TypeError):
        return None
    if ts is pd.NaT:
        return None
    now = pd.Timestamp.now()
    return (now.year - ts.year) * 12 + (now.month - ts.month)


def is_stale(data_through: str | pd.Timestamp, *, threshold: int = ENSO_STALE_MONTHS) -> bool:
    """True if the data lags more than ``threshold`` months behind now."""
    age = data_age_months(data_through)
    return age is not None and age > threshold


def freshness_note(data_through: str | pd.Timestamp) -> str:
    """One-line freshness status for surfacing to the user/agent.

    Examples: '数据新鲜（截止 2026-06，距今 0 个月）' or
    '⚠️ 数据偏旧（截止 2026-03，距今 4 个月，超过 2 个月阈值）'.
    """
    age = data_age_months(data_through)
    if age is None:
        return f"⚠️ 无法解析数据截止月份「{data_through}」，跳过新鲜度判断。"
    if age < 0:
        return f"⚠️ 数据截止月 {data_through} 晚于当前（{age} 个月），疑似时钟异常。"
    if age <= ENSO_STALE_MONTHS:
        return f"数据新鲜（截止 {pd.Timestamp(data_through):%Y-%m}，距今 {age} 个月）。"
    return (
        f"⚠️ 数据偏旧（截止 {pd.Timestamp(data_through):%Y-%m}，距今 {age} 个月，"
        f"超过 {ENSO_STALE_MONTHS} 个月阈值）。建议刷新；后台已可自动重训，"
        f"或显式调 load_enso_data(refresh_noaa=True)。"
    )


@dataclass
class _RetrainHolder:
    """Thread-safe handoff between the retrain worker and the foreground.

    The worker writes ``output`` + ``done=True`` under ``lock``; the foreground
    reads it and clears it once it has reloaded results into the context. At most
    one worker runs per holder (``running`` flag, set under ``lock``).
    """

    running: bool = False
    done: bool = False
    output: Any = None  # EnsoForecastOutput when done
    error: str | None = None
    started_at_through: str | None = None  # the data_through the retrain began from
    lock: threading.Lock = None  # type: ignore[assignment]  # set in __post_init__

    def __post_init__(self) -> None:
        if self.lock is None:
            self.lock = threading.Lock()


class BackgroundRetrainer:
    """Runs ``run_enso_forecast(refresh_noaa=True)`` off the main thread.

    Usage (foreground, at turn start)::

        retrainer = session_state.setdefault("retrainer", BackgroundRetrainer())
        if retrainer.take_completed():
            # reload fresh results into ctx from disk
        if stale and retrainer.start_if_idle(base_dir=ctx.base_dir):
            st.toast("数据偏旧，后台刷新中…")

    The worker only writes artifacts to disk; it does NOT mutate any
    ToolContext, so it is safe to run while ``run_turn`` is using the context.
    """

    def __init__(self) -> None:
        self._holder = _RetrainHolder()

    @property
    def running(self) -> bool:
        with self._holder.lock:
            return self._holder.running

    def start_if_idle(self, *, base_dir, data_source: str = "auto") -> bool:
        """Kick off a background retrain if none is running. Returns True if started.

        Idempotent: a second call while one is running is a no-op (returns False).
        ``base_dir`` is forwarded to ``run_enso_forecast`` so artifacts land in
        the session's own outputs dir.
        """
        with self._holder.lock:
            if self._holder.running:
                return False
            self._holder.running = True
            self._holder.done = False
            self._holder.output = None
            self._holder.error = None

        thread = threading.Thread(
            target=self._worker, args=(base_dir, data_source), daemon=True
        )
        thread.start()
        return True

    def _worker(self, base_dir, data_source: str) -> None:
        try:
            from src.pipeline.run_enso_forecast import run_enso_forecast

            output = run_enso_forecast(
                base_dir=base_dir, data_source=data_source, refresh_noaa=True
            )
            with self._holder.lock:
                self._holder.output = output
                self._holder.done = True
                self._holder.running = False
        except Exception as exc:  # noqa: BLE001 — never crash the worker thread
            with self._holder.lock:
                self._holder.error = f"{exc.__class__.__name__}: {exc}"
                self._holder.done = True
                self._holder.running = False

    def take_completed(self):
        """Return the finished ``EnsoForecastOutput`` once, or None if not done.

        After a successful take, the holder is reset so a future retrain can run.
        If the worker errored, returns None and surfaces the error via
        :attr:`last_error`.
        """
        with self._holder.lock:
            if not self._holder.done:
                return None
            out = self._holder.output
            err = self._holder.error
            # Reset for the next cycle.
            self._holder.done = False
            self._holder.output = None
            self._holder.error = None
        if err is not None:
            self.last_error = err
            return None
        return out

    last_error: str | None = None
