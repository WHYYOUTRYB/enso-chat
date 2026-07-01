"""Offline training for the CNN-LSTM ENSO model on SODA.

Run once to produce ``weights/cnn_lstm_soda.pth`` and
``reports/outputs/cnn_lstm_metrics.json``. The Streamlit app only does
inference (``src.models.cnn_lstm.predict_cnn_lstm``); this script is never
imported by the web layer.

Usage::

    python scripts/train_cnn_lstm.py
    python scripts/train_cnn_lstm.py --epochs 80 --batch-size 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import PROJECT_ROOT
from src.models.cnn_lstm import LEAD_MONTHS, train_cnn_lstm


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN-LSTM on SODA.")
    parser.add_argument("--train-path", default=str(PROJECT_ROOT / "data" / "SODA_train.nc"))
    parser.add_argument("--label-path", default=str(PROJECT_ROOT / "data" / "SODA_label.nc"))
    parser.add_argument("--weights-path", default=str(PROJECT_ROOT / "weights" / "cnn_lstm_soda.pth"))
    parser.add_argument("--metrics-path", default=str(PROJECT_ROOT / "reports" / "outputs" / "cnn_lstm_metrics.json"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = train_cnn_lstm(
        Path(args.train_path),
        Path(args.label_path),
        weights_path=Path(args.weights_path),
        metrics_path=Path(args.metrics_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
        seed=args.seed,
    )

    print(f"\nSaved weights → {args.weights_path}")
    print(f"Saved metrics → {args.metrics_path}")
    print("\nPer-lead test metrics (lead → rmse / mae / acc):")
    for lead in range(1, LEAD_MONTHS + 1):
        m = metrics[str(lead)]
        print(f"  lead={lead:2d}  rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  acc={m['acc']:.4f}")


if __name__ == "__main__":
    main()
