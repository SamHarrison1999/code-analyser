#!/usr/bin/env python3
"""
Threshold sweep plots for the model service.

- Reads:  CSV with columns id,text,labels   (labels are ;-separated canonical tags)
- Calls:  POST /predict on your FastAPI model service to get per-label probabilities
- Sweeps: thresholds from --start..--stop with --step, computing P/R/F1 per label
- Saves:  one PNG per label in --out, plus best_thresholds.json

Usage:
  python tools/plot_threshold_sweep.py \
    --csv datasets/eval/snippets.csv \
    --url http://127.0.0.1:8111/predict \
    --out eval/out/plots \
    --start 0.05 --stop 0.95 --step 0.02
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Set

import numpy as np
import matplotlib.pyplot as plt
import requests


LABELS_DEFAULT = ["sast_risk", "ml_signal", "best_practice"]


def parse_labels(s: str) -> Set[str]:
    s = (s or "").strip()
    return {t.strip() for t in s.split(";") if t.strip()}


def load_dataset(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    texts = [r["text"] for r in rows]
    truth = [parse_labels(r.get("labels", "")) for r in rows]
    return rows, texts, truth


def fetch_scores(url: str, texts: List[str], batch_size: int = 32, timeout: float = 30.0):
    scores: List[Dict[str, float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        r = requests.post(
            url,
            json={"texts": batch, "threshold": 0.0},
            timeout=timeout,
        )
        r.raise_for_status()
        payload = r.json()
        scores += [it.get("scores", {}) for it in payload.get("results", [])]
    return scores


def prf(y: List[bool], s: List[float], t: float):
    tp = sum((yi and si >= t) for yi, si in zip(y, s))
    fp = sum((not yi and si >= t) for yi, si in zip(y, s))
    fn = sum((yi and si < t) for yi, si in zip(y, s))
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = (2 * P * R / (P + R)) if (P + R) else 0.0
    return P, R, F1


def sweep_for_label(
    label: str, truth: List[Set[str]], scores: List[Dict[str, float]], grid: np.ndarray
):
    y = [label in t for t in truth]
    s = [float(sc.get(label, 0.0)) for sc in scores]

    P, R, F1 = [], [], []
    for t in grid:
        p, r, f1 = prf(y, s, float(t))
        P.append(p)
        R.append(r)
        F1.append(f1)

    # Robust selection of best threshold: maximise F1, tie-break on Precision
    P = np.nan_to_num(np.array(P, dtype=float), nan=0.0)
    F1 = np.nan_to_num(np.array(F1, dtype=float), nan=0.0)
    best_i = max(range(len(grid)), key=lambda i: (F1[i], P[i]))
    best_t = float(grid[best_i])

    return P, R, F1, best_t


def plot_curves(
    label: str,
    grid: np.ndarray,
    P,
    R,
    F1,
    best_t: float,
    out_dir: Path,
    marks_from: Dict[str, float] | None,
):
    plt.figure(figsize=(6, 4))
    plt.plot(grid, P, label="Precision")
    plt.plot(grid, R, label="Recall")
    plt.plot(grid, F1, label="F1")

    # Best vertical line
    plt.axvline(best_t, linestyle="--", label=f"Best F1 @ {best_t:.2f}")

    # Optional external marks (e.g., calibrated thresholds per label)
    if marks_from and label in marks_from:
        plt.axvline(float(marks_from[label]), linestyle=":", label=f"Mark {marks_from[label]:.2f}")

    plt.title(f"Threshold sweep — {label}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.xlim(min(grid), max(grid))
    plt.ylim(0.0, 1.02)
    plt.legend(loc="lower left")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"threshold_sweep_{label}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot threshold sweeps for the model service")
    ap.add_argument("--csv", required=True, help="Path to evaluation CSV (id,text,labels)")
    ap.add_argument(
        "--url",
        required=True,
        help="Model service predict endpoint, e.g. http://127.0.0.1:8111/predict",
    )
    ap.add_argument("--out", required=True, help="Output directory for PNGs")
    ap.add_argument(
        "--labels",
        nargs="*",
        default=LABELS_DEFAULT,
        help="Labels to evaluate (default: %(default)s)",
    )
    ap.add_argument("--start", type=float, default=0.05, help="Grid start (inclusive)")
    ap.add_argument("--stop", type=float, default=0.95, help="Grid stop (inclusive)")
    ap.add_argument("--step", type=float, default=0.02, help="Grid step")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for service calls")
    ap.add_argument(
        "--marks-from",
        type=str,
        default=None,
        help="Optional JSON file with {label: threshold} to overlay",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)

    # Build threshold grid (inclusive of stop, within floating tolerance)
    grid = np.arange(args.start, args.stop + 1e-9, args.step, dtype=float)

    # Load dataset and call model once
    _, texts, truth = load_dataset(csv_path)
    scores = fetch_scores(args.url, texts, batch_size=args.batch_size)

    # Optional marks overlay
    marks = None
    if args.marks_from:
        marks = json.loads(Path(args.marks_from).read_text(encoding="utf-8"))

    best_thresholds = {}
    for lab in args.labels:
        P, R, F1, best_t = sweep_for_label(lab, truth, scores, grid)
        best_thresholds[lab] = round(best_t, 3)
        plot_curves(lab, grid, P, R, F1, best_t, out_dir, marks)

    # Save chosen thresholds for convenience
    (out_dir / "best_thresholds.json").write_text(
        json.dumps(best_thresholds, indent=2), encoding="utf-8"
    )
    print("Best thresholds:", best_thresholds)


if __name__ == "__main__":
    main()
