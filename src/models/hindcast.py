"""Hindcast skill evaluation for the CNN-LSTM track.

The reliability question — "can I trust this forecast?" — is answered not by a
bigger model but by a proper hindcast protocol, as in Ham et al. 2019 (Nature,
https://www.nature.com/articles/s41586-019-1559-7). This module runs the trained
CNN-LSTM over the held-out test windows and reports, per lead:

* **all-season ACC** — the Anomaly Correlation Coefficient across all test
  samples (the metric Ham et al. report; their CNN stays >0.5 to lead≈17).
* **Persistence-baseline ACC** — the same metric for a "forecast = last observed
  month" null model. The gap between the CNN and Persistence is the *skill over
  the simplest viable baseline*; where the CNN drops to Persistence-level, the
  forecast carries no added information.
* **per-target-month ACC** — ACC grouped by the target month's phase within the
  SODA block, exposing the spring predictability barrier (SPB) dip.

SODA's year/month are anonymous competition indices (year 1-100 blocks, month
1-36 within), NOT real calendar dates, so per-target-month uses block-phase
``(start+12+lead) % 12`` and is labeled as such — it cannot be mapped to real
DJF/MJJ seasons. All-season ACC, however, is directly comparable to Ham et al.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.models.cnn_lstm import (
    INPUT_MONTHS,
    LEAD_MONTHS,
    SPLIT_MONTH_RANGES,
    _build_model,
    _open_soda,
    make_cnn_lstm_dataset,
)
from src.models.evaluation import calculate_acc


@dataclass
class HindcastResult:
    leads: list[int]               # 1..24
    cnn_acc: list[float]           # all-season ACC per lead
    persistence_acc: list[float]   # persistence-baseline ACC per lead
    skill_gap: list[float]         # cnn_acc - persistence_acc per lead
    n_samples: int
    split: str


def _load_nino_series(label_path: Path) -> np.ndarray:
    """Load the continuous monthly Niño3.4 series from the SODA label file."""
    import shutil

    _, label_ds, tmp = _open_soda(label_path, label_path)
    try:
        y = np.asarray(label_ds["nino"].values, dtype=np.float32).reshape(-1)
    finally:
        import shutil

        label_ds.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return np.nan_to_num(y, nan=0.0)


def _persistence_predictions(nino: np.ndarray, starts: list[int]) -> np.ndarray:
    """Persistence null model: forecast = last observed month, held for all leads.

    For a window starting at month ``s`` (input months s..s+11), persistence
    predicts every lead as ``nino[s+11]`` (the most recent observed value).
    Returns shape ``(n_samples, LEAD_MONTHS)``.
    """
    preds = np.empty((len(starts), LEAD_MONTHS), dtype=np.float32)
    for i, s in enumerate(starts):
        preds[i, :] = nino[s + INPUT_MONTHS - 1]
    return preds


def run_hindcast(
    weights_path: Path,
    soda_train_path: Path,
    soda_label_path: Path,
    *,
    split: str = "test",
) -> HindcastResult:
    """Run the trained CNN-LSTM over the ``split`` windows and score vs Persistence.

    Loads weights + train-set standardization stats from the checkpoint, runs a
    batched CPU forward pass, builds the persistence null model from the Niño3.4
    label series, and reports all-season ACC for both.
    """
    import torch

    if not weights_path.exists():
        raise FileNotFoundError(f"CNN-LSTM weights not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    x_std = np.where(x_std == 0, 1.0, x_std)

    x_te, y_te = make_cnn_lstm_dataset(soda_train_path, soda_label_path, split)
    n = len(y_te)
    m_start = SPLIT_MONTH_RANGES[split][0]
    starts = list(range(m_start, m_start + n))

    # CNN forward (batched).
    model = _build_model()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    x_norm = (x_te - x_mean) / x_std
    cnn_pred = np.empty((n, LEAD_MONTHS), dtype=np.float32)
    bs = 32
    with torch.no_grad():
        for i in range(0, n, bs):
            chunk = torch.tensor(x_norm[i : i + bs], dtype=torch.float32)
            cnn_pred[i : i + bs] = model(chunk)[:, -1, :].cpu().numpy()

    # Persistence null model.
    nino = _load_nino_series(soda_label_path)
    pers_pred = _persistence_predictions(nino, starts)

    cnn_acc, pers_acc, gap = [], [], []
    for lead in range(LEAD_MONTHS):
        truth = y_te[:, lead]
        c = calculate_acc(truth, cnn_pred[:, lead])
        p = calculate_acc(truth, pers_pred[:, lead])
        cnn_acc.append(round(c, 4))
        pers_acc.append(round(p, 4))
        gap.append(round(c - p, 4))

    return HindcastResult(
        leads=list(range(1, LEAD_MONTHS + 1)),
        cnn_acc=cnn_acc,
        persistence_acc=pers_acc,
        skill_gap=gap,
        n_samples=n,
        split=split,
    )


def hindcast_report_text(res: HindcastResult) -> str:
    """Human/LLM-readable summary: per-lead CNN vs Persistence ACC + skill gap."""
    lines = [
        f"Hindcast skill (split={res.split}, n={res.n_samples} windows). "
        f"All-season ACC — the Ham et al. 2019 metric (their CNN >0.5 to lead≈17).",
        f"{'lead':>4} {'CNN-ACC':>8} {'Persist':>8} {'gap':>7}",
    ]
    for i, lead in enumerate(res.leads):
        lines.append(f"{lead:>4} {res.cnn_acc[i]:>8.3f} {res.persistence_acc[i]:>8.3f} {res.skill_gap[i]:>+7.3f}")
    # headline: the lead up to which CNN stays above Persistence + above 0.5.
    cnn_above_05 = [res.leads[i] for i in range(len(res.leads)) if res.cnn_acc[i] >= 0.5]
    cnn_above_pers = [res.leads[i] for i in range(len(res.leads)) if res.skill_gap[i] > 0]
    lines.append(f"CNN ACC >= 0.5 up to lead={max(cnn_above_05) if cnn_above_05 else 0}.")
    lines.append(f"CNN beats Persistence at leads={cnn_above_pers}.")
    return "\n".join(lines)


def save_hindcast_report(res: HindcastResult, json_path: Path, figure_path: Path | None = None) -> None:
    """Write the hindcast report as JSON (+ optional ACC-vs-lead comparison plot)."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": res.split,
        "n_samples": res.n_samples,
        "leads": res.leads,
        "cnn_acc": res.cnn_acc,
        "persistence_acc": res.persistence_acc,
        "skill_gap": res.skill_gap,
        "metric": "all-season ACC (anomaly correlation), Ham et al. 2019 convention",
        "note": "SODA year/month are anonymous competition indices; all-season ACC is comparable to Ham et al., per-target-month is not mapped to real seasons.",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if figure_path is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(res.leads, res.cnn_acc, "o-", label="CNN-LSTM", linewidth=1.8)
        ax.plot(res.leads, res.persistence_acc, "s--", label="Persistence (null)", linewidth=1.5)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, label="ACC=0.5 skill threshold")
        ax.set_xlabel("Lead time (months)")
        ax.set_ylabel("All-season ACC")
        ax.set_title("ENSO hindcast skill: CNN-LSTM vs Persistence")
        ax.set_xticks(res.leads)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)
