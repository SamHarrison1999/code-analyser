# --- file: eval/eval.py ---
# Annotation: Standard library imports kept lightweight so the module always imports during tests.
import argparse
from pathlib import Path
from typing import List, Tuple, Iterable

# Annotation: Try NumPy for numeric work; provide tiny fallbacks if unavailable in the test environment.
try:
    import numpy as np
except Exception:
    class _NP:
        @staticmethod
        def array(x): return x
        @staticmethod
        def asarray(x): return x
        @staticmethod
        def unique(x):
            xs = list(x)
            seen = []
            for v in xs:
                if v not in seen: seen.append(v)
            return seen
        @staticmethod
        def zeros(shape, dtype=float):
            if len(shape)==2: return [[0 for _ in range(shape[1])] for __ in range(shape[0])]
            return [0 for _ in range(shape[0])]
        @staticmethod
        def argsort(x): return sorted(range(len(x)), key=lambda i: x[i])
        @staticmethod
        def cumsum(x):
            s=0; out=[]
            for v in x: s+=v; out.append(s)
            return out
        @staticmethod
        def diff(x):
            return [x[i+1]-x[i] for i in range(len(x)-1)]
        @staticmethod
        def trapz(y, x):
            area=0.0
            for i in range(len(y)-1):
                area += 0.5*(y[i]+y[i+1])*(x[i+1]-x[i])
            return area
    np = _NP()  # type: ignore

# Annotation: Pull common metrics from scikit-learn when present; otherwise define compact local substitutes.
try:
    from sklearn.metrics import confusion_matrix as _sk_confusion_matrix, roc_curve as _sk_roc_curve, auc as _sk_auc
    confusion_matrix = _sk_confusion_matrix
    roc_curve = _sk_roc_curve
    auc = _sk_auc
except Exception:
    # Annotation: Fallback confusion_matrix that supports arbitrary label sets.
    def confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], labels: List[int] | None = None):
        yt = list(y_true); yp = list(y_pred)
        labs = labels if labels is not None else sorted(set(yt) | set(yp))
        idx = {lab:i for i,lab in enumerate(labs)}
        mat = np.zeros((len(labs), len(labs)))
        for t,p in zip(yt, yp):
            i = idx.get(t, None); j = idx.get(p, None)
            if i is not None and j is not None:
                try:
                    mat[i][j] += 1
                except Exception:
                    mat[i][j] = mat[i][j] + 1
        return mat
    # Annotation: Fallback roc_curve for binary labels (0/1) using a simple threshold sweep.
    def roc_curve(y_true: Iterable[int], y_score: Iterable[float], pos_label: int = 1, drop_intermediate: bool = True):
        yt = [1 if y==pos_label else 0 for y in y_true]
        ys = list(y_score)
        order = sorted(range(len(ys)), key=lambda i: -ys[i])
        yt_sorted = [yt[i] for i in order]
        ys_sorted = [ys[i] for i in order]
        P = sum(yt_sorted); N = len(yt_sorted) - P
        tps = []; fps = []
        tp = 0; fp = 0
        prev_score = None
        thresholds = []
        for i,(y, s) in enumerate(zip(yt_sorted, ys_sorted)):
            if prev_score is None or s != prev_score:
                thresholds.append(s)
                tps.append(tp)
                fps.append(fp)
                prev_score = s
            if y==1: tp += 1
            else: fp += 1
        thresholds.append(-float("inf"))
        tps.append(tp)
        fps.append(fp)
        fpr = [f/ max(1, N) for f in fps]
        tpr = [t/ max(1, P) for t in tps]
        return fpr, tpr, thresholds
    # Annotation: Fallback trapezoidal AUC.
    def auc(x: Iterable[float], y: Iterable[float]) -> float:
        xs = list(x); ys = list(y)
        if len(xs) != len(ys) or len(xs) < 2: return 0.0
        return float(np.trapz(ys, xs))

# Annotation: Small helper to compute micro-averaged ROC AUC given per-sample scores and true labels.
def binary_auc(y_true: List[int], y_score: List[float]) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    return auc(fpr, tpr)

# Annotation: CLI kept minimal so the test harness can run the module with '--help' safely.
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate predictions with basic metrics (confusion matrix, ROC, AUC).")
    p.add_argument("--ytrue", type=Path, help="Path to a file containing true labels (one per line).")
    p.add_argument("--yscore", type=Path, help="Path to a file containing predicted scores (one per line).")
    p.add_argument("--ypred", type=Path, help="Path to a file containing predicted labels (one per line).")
    return p

# Annotation: Minimal main that only does I/O if explicit files are provided; otherwise it exits quietly.
def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.ytrue and args.ypred:
        yt = [int(x.strip()) for x in args.ytrue.read_text(encoding="utf-8").splitlines() if x.strip()!=""]
        yp = [int(x.strip()) for x in args.ypred.read_text(encoding="utf-8").splitlines() if x.strip()!=""]
        cm = confusion_matrix(yt, yp)
        print("confusion_matrix:", cm)
    if args.ytrue and args.yscore:
        yt = [int(x.strip()) for x in args.ytrue.read_text(encoding="utf-8").splitlines() if x.strip()!=""]
        ys = [float(x.strip()) for x in args.yscore.read_text(encoding="utf-8").splitlines() if x.strip()!=""]
        print("roc_auc:", binary_auc(yt, ys))

# Annotation: Execute only when launched as a script; tests call the module with '--help' which is handled by argparse.
if __name__ == "__main__":
    main()
