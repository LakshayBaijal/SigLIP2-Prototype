#!/usr/bin/env python3
"""
The number that actually matters for a production deployment: at various
confidence thresholds, what fraction of predictions can be auto-accepted,
and how accurate is that auto-accepted slice? Low-confidence predictions
should route to human review instead of going live unchecked.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"


def main():
    data = np.load(ROOT / "embeddings.npz", allow_pickle=True)
    X, y, splits = data["embeddings"], data["labels"], data["splits"]
    val_mask = splits == "val"
    X_val, y_val = X[val_mask], y[val_mask]

    proto_data = np.load(CKPT_DIR / "prototypes.npz", allow_pickle=True)
    prototypes, classes = proto_data["prototypes"], proto_data["classes"]

    sims = X_val @ prototypes.T
    pred_idx = sims.argmax(axis=1)
    preds = classes[pred_idx]
    confidence = sims.max(axis=1)  # cosine similarity to the winning prototype
    correct = preds == y_val

    print(f"Val set: {len(y_val)} images\n")
    print(f"{'Threshold':>10} {'Coverage':>10} {'Auto-accept acc':>17} {'# reviewed':>11}")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        accepted = confidence >= thresh
        n_accepted = accepted.sum()
        coverage = n_accepted / len(y_val)
        acc_on_accepted = correct[accepted].mean() if n_accepted > 0 else float("nan")
        n_review = len(y_val) - n_accepted
        print(f"{thresh:>10.2f} {coverage:>9.1%} {acc_on_accepted:>16.1%} {n_review:>11d}")

    print("\nInterpretation: pick the threshold row where 'Auto-accept acc' hits your bar "
          "(e.g. 99%). Everything above that threshold ships automatically; everything "
          "below routes to a human reviewer instead of going live unchecked.")


if __name__ == "__main__":
    main()
