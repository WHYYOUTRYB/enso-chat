"""Generate the CNN-LSTM hindcast skill report (ACC vs Persistence baseline).

Produces ``reports/outputs/cnn_lstm_hindcast.json`` and
``reports/figures/cnn_lstm_hindcast.png``. Run after training::

    python scripts/run_hindcast.py

The report answers "is this forecast trustworthy?" the way Ham et al. 2019
(Nature) do: all-season ACC per lead, benchmarked against a Persistence null
model. The CNN must beat Persistence (and stay above ACC=0.5) to claim skill.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import FIGURES_DIR, OUTPUTS_DIR, PROJECT_ROOT
from src.models.hindcast import hindcast_report_text, run_hindcast, save_hindcast_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the CNN-LSTM hindcast skill report.")
    parser.add_argument("--weights-path", default=str(PROJECT_ROOT / "weights" / "cnn_lstm_soda.pth"))
    parser.add_argument("--train-path", default=str(PROJECT_ROOT / "data" / "SODA_train.nc"))
    parser.add_argument("--label-path", default=str(PROJECT_ROOT / "data" / "SODA_label.nc"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--json-path", default=str(OUTPUTS_DIR / "cnn_lstm_hindcast.json"))
    parser.add_argument("--figure-path", default=str(FIGURES_DIR / "cnn_lstm_hindcast.png"))
    args = parser.parse_args()

    res = run_hindcast(
        Path(args.weights_path),
        Path(args.train_path),
        Path(args.label_path),
        split=args.split,
    )
    save_hindcast_report(res, Path(args.json_path), Path(args.figure_path))
    print(hindcast_report_text(res))
    print(f"\nSaved report → {args.json_path}")
    print(f"Saved figure → {args.figure_path}")


if __name__ == "__main__":
    main()
