# Compare Rules vs Model vs Fusion using /batch on file paths with efficient chunking and optional parallelism to avoid timeouts.
# Reads id,text,labels or filename+numeric CSVs, writes each snippet to a temp .py file, posts chunks of paths to /batch, thresholds scores, and prints micro/macro/per-label F1.
import argparse

# Standard libraries for filesystem, temporary dirs, timing and JSON.
import os
import tempfile
import time
import json
import csv

# Cross-platform path handling.
from pathlib import Path

# Typing aids readability.
from typing import Dict, List, Tuple, Optional

# HTTP client library.
import requests

# Concurrency primitives for optional parallel chunk processing.
from concurrent.futures import ThreadPoolExecutor, as_completed

# Numerical arrays and metrics.
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score

# Canonical label order for your project.
LABELS: List[str] = ["sast_risk", "ml_signal", "best_practice"]
# Normalisation map to cope with minor label spelling variants.
ALIAS = {
    "sast-risk": "sast_risk",
    "sast_risk": "sast_risk",
    "sast": "sast_risk",
    "ml-signal": "ml_signal",
    "ml_signal": "ml_signal",
    "best-practice": "best_practice",
    "best_practice": "best_practice",
}


# Convert any label string into a canonical name or None if unknown.
def _norm_label(s: str) -> Optional[str]:
    key = str(s).strip().lower().replace(" ", "").replace("-", "_")
    if key in ALIAS:
        return ALIAS[key]
    if key in LABELS:
        return key
    return None


# Parse a comma-separated labels string into a fixed-order 0/1 vector.
def _labels_from_string(label_str: str) -> np.ndarray:
    parts = [x.strip() for x in str(label_str).split(",") if x.strip()]
    normed = {_norm_label(x) for x in parts}
    row = [1 if lbl in normed else 0 for lbl in LABELS]
    return np.array(row, dtype=np.int32)


# Load dataset from CSV; supports (id,text,labels) or (filename + numeric label columns).
def _load_from_csv(csv_path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    names: List[str] = []
    codes: List[str] = []
    y_rows: List[np.ndarray] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in (reader.fieldnames or [])]
        has_text = "text" in header
        has_labels_str = "labels" in header
        has_filename = "filename" in header
        has_all_numeric = all(lbl in header for lbl in LABELS)
        if has_text and has_labels_str:
            for row in reader:
                if not any(row.values()):
                    continue
                rid = (row.get("id") or str(len(names))).strip()
                code_text = row.get("text", "")
                y = _labels_from_string(row.get("labels", ""))
                names.append(Path(rid).stem)
                codes.append(code_text)
                y_rows.append(y)
            return (
                names,
                codes,
                (
                    np.stack(y_rows, axis=0)
                    if y_rows
                    else np.zeros((0, len(LABELS)), dtype=np.int32)
                ),
            )
        if has_filename and has_all_numeric:
            for row in reader:
                if not any(row.values()):
                    continue
                fname = Path(row.get("filename", "")).stem
                y = np.array([int(row.get(lbl, "0") or "0") for lbl in LABELS], dtype=np.int32)
                names.append(fname)
                codes.append("")
                y_rows.append(y)
            return (
                names,
                codes,
                (
                    np.stack(y_rows, axis=0)
                    if y_rows
                    else np.zeros((0, len(LABELS)), dtype=np.int32)
                ),
            )
    raise ValueError(
        f"CSV schema not recognised in {csv_path}. Expected either columns (id,text,labels) or (filename,{','.join(LABELS)})."
    )


# Helper to parse per-label thresholds from a string like "sast_risk=0.6,ml_signal=0.3".
def _parse_thresholds(s: Optional[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not s:
        return out
    for part in s.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        nl = _norm_label(k)
        if nl:
            try:
                out[nl] = float(v)
            except Exception:
                pass
    return out


# Turn a batch response into {basename → {label:bool}} decisions; supports list or dict 'results', 'prediction'/'predictions', 'probs' and 'preds'.
def _decisions_map_from_batch(
    data: dict, default_threshold: float, per_label_thr: Dict[str, float]
) -> Dict[str, Dict[str, bool]]:
    out: Dict[str, Dict[str, bool]] = {}
    results = data.get("results")
    items: List[dict] = []
    if isinstance(results, list):
        items = [x for x in results if isinstance(x, dict)]
    elif isinstance(results, dict):
        for k, v in results.items():
            if isinstance(v, dict):
                item = dict(v)
                item.setdefault("path", k)
                items.append(item)
    for it in items:
        name = it.get("path") or it.get("file") or it.get("filename") or ""
        bname = Path(str(name)).name
        thr_field = it.get("threshold", data.get("threshold", default_threshold))
        decide: Dict[str, bool] = {lbl: False for lbl in LABELS}
        seq = it.get("prediction") if isinstance(it, dict) else None
        if seq is None:
            seq = it.get("predictions")
        if isinstance(seq, list):
            for row in seq:
                if not isinstance(row, dict):
                    continue
                lbl = row.get("label")
                nl = _norm_label(lbl) if lbl is not None else None
                if not nl:
                    continue
                score = row.get("score") or row.get("prob") or row.get("p") or row.get("confidence")
                thr = per_label_thr.get(
                    nl,
                    (
                        float(thr_field)
                        if not isinstance(thr_field, dict)
                        else float(thr_field.get(nl, default_threshold))
                    ),
                )
                decide[nl] = True if score is None else float(score) >= thr
            out[bname] = decide
            continue
        probs = it.get("probs")
        if isinstance(probs, dict):
            for k, v in probs.items():
                nk = _norm_label(k)
                if not nk:
                    continue
                thr = per_label_thr.get(
                    nk,
                    (
                        float(thr_field)
                        if not isinstance(thr_field, dict)
                        else float(thr_field.get(nk, default_threshold))
                    ),
                )
                decide[nk] = float(v) >= thr
            out[bname] = decide
            continue
        preds = it.get("preds")
        if isinstance(preds, dict):
            for k, v in preds.items():
                nk = _norm_label(k)
                if nk:
                    decide[nk] = bool(v)
            out[bname] = decide
            continue
        out[bname] = decide
    return out


# Compute micro/macro F1 and per-label F1.
def _score(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_label = {}
    for i, lbl in enumerate(LABELS):
        per_label[lbl] = f1_score(y_true[:, i], y_pred[:, i], average="binary", zero_division=0)
    return {
        "p_micro": float(p_micro),
        "r_micro": float(r_micro),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        **{f"f1_{k}": float(v) for k, v in per_label.items()},
    }


# Print a formatted table row matching your dissertation style.
def _print_row(mode: str, metrics: Dict[str, float], n: int, elapsed_s: float) -> None:
    line = f"{mode.upper():<8}  P={metrics['p_micro']:.2f}  R={metrics['r_micro']:.2f}  F1={metrics['f1_micro']:.2f}  |  F1(sast)={metrics['f1_sast_risk']:.2f}  F1(ml)={metrics['f1_ml_signal']:.2f}  F1(best)={metrics['f1_best_practice']:.2f}  |  N={n:<4d}  t={elapsed_s:.2f}s"
    print(line)


# Yield successive chunks of a list with a fixed batch size.
def _chunks(seq: List[Path], size: int) -> List[List[Path]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


# Post one chunk of paths to /batch with retries; returns a decisions map {basename → {label:bool}}.
def _post_chunk(
    base_url: str,
    mode: str,
    paths: List[Path],
    language: str,
    ret: Optional[str],
    timeout: float,
    default_thr: float,
    per_label_thr: Dict[str, float],
    retries: int = 3,
    backoff: float = 2.0,
) -> Dict[str, Dict[str, bool]]:
    payload = {"paths": [str(p) for p in paths], "language": language}
    params = {"mode": mode}
    if ret:
        params["return"] = ret
    for attempt in range(retries):
        try:
            with requests.Session() as sess:
                resp = sess.post(
                    f"{base_url.rstrip('/')}/batch", params=params, json=payload, timeout=timeout
                )
                resp.raise_for_status()
                data = resp.json()
                return _decisions_map_from_batch(data, default_thr, per_label_thr)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2**attempt))
    return {}


# Main: write temp files, send chunks (optionally in parallel), score, and optionally dump JSON.
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare Rules vs Model vs Fusion using chunked /batch calls."
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8111", help="FastAPI base URL")
    ap.add_argument(
        "--truth-csv",
        required=True,
        help="CSV with id,text,labels or filename plus numeric columns",
    )
    ap.add_argument("--language", default="python", help="Language hint to include in payload")
    ap.add_argument(
        "--return", dest="ret", default="scores", help="Optional return hint (scores or probs)"
    )
    ap.add_argument(
        "--threshold", type=float, default=0.5, help="Default decision threshold for scores/probs"
    )
    ap.add_argument(
        "--threshold-per-label",
        dest="thr_map",
        default=None,
        help="Per-label thresholds, e.g. 'sast_risk=0.6,ml_signal=0.3,best_practice=0.4'",
    )
    ap.add_argument(
        "--modes", default="rules,model,fusion", help="Comma-separated: rules,model,fusion"
    )
    ap.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds")
    ap.add_argument("--batch-size", type=int, default=24, help="Number of files per /batch request")
    ap.add_argument(
        "--workers", type=int, default=1, help="Number of parallel chunks to process per mode"
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="Optional cap on number of samples for a quick run"
    )
    ap.add_argument(
        "--out-json", default=None, help="Optional path to write per-mode metrics as JSON"
    )
    args = ap.parse_args()
    try:
        requests.get(f"{args.base_url.rstrip('/')}/healthz", timeout=5).raise_for_status()
    except Exception as e:
        raise SystemExit(f"❌ Cannot reach {args.base_url} (/healthz): {e}")
    csv_path = Path(args.truth_csv).resolve()
    names, codes, y_true = _load_from_csv(csv_path)
    if args.limit is not None:
        names = names[: args.limit]
        codes = codes[: args.limit]
        y_true = y_true[: args.limit]
    pos = y_true.sum(axis=0) if y_true.size else np.zeros(len(LABELS), dtype=np.int32)
    print(
        f"[truth] positives → sast_risk={int(pos[0])}, ml_signal={int(pos[1])}, best_practice={int(pos[2])}, N={len(names)}"
    )
    per_label_thr = _parse_thresholds(args.thr_map)
    with tempfile.TemporaryDirectory(prefix="cmp_paths_") as tmpdir:
        tmp_root = Path(tmpdir)
        file_paths: List[Path] = []
        for name, code in zip(names, codes):
            p = tmp_root / f"{name}.py"
            p.write_text(code, encoding="utf-8")
            file_paths.append(p)
        chunks = _chunks(file_paths, args.batch_size)
        all_metrics: Dict[str, Dict[str, float]] = {}
        print("MODE      P     R     F1    |  F1(sast)  F1(ml)  F1(best)  |  N     t")
        for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
            y_pred = np.zeros_like(y_true, dtype=np.int32)
            t0 = time.time()
            if args.workers > 1:
                futures = {}
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    for idx, chunk in enumerate(chunks):
                        futures[
                            ex.submit(
                                _post_chunk,
                                args.base_url,
                                mode,
                                chunk,
                                args.language,
                                args.ret,
                                args.timeout,
                                args.threshold,
                                per_label_thr,
                            )
                        ] = (idx, chunk)
                    for fut in as_completed(futures):
                        idx, chunk = futures[fut]
                        dec_map = fut.result()
                        for p in chunk:
                            b = p.name
                            dm = dec_map.get(b, {lbl: False for lbl in LABELS})
                            i = file_paths.index(p)
                            y_pred[i, :] = np.array(
                                [1 if dm.get(lbl, False) else 0 for lbl in LABELS], dtype=np.int32
                            )
                        print(f"[{mode}] chunk {idx+1}/{len(chunks)} done")
            else:
                for idx, chunk in enumerate(chunks):
                    dec_map = _post_chunk(
                        args.base_url,
                        mode,
                        chunk,
                        args.language,
                        args.ret,
                        args.timeout,
                        args.threshold,
                        per_label_thr,
                    )
                    for p in chunk:
                        b = p.name
                        dm = dec_map.get(b, {lbl: False for lbl in LABELS})
                        i = file_paths.index(p)
                        y_pred[i, :] = np.array(
                            [1 if dm.get(lbl, False) else 0 for lbl in LABELS], dtype=np.int32
                        )
                    print(f"[{mode}] chunk {idx+1}/{len(chunks)} done")
            elapsed = time.time() - t0
            metrics = _score(y_true, y_pred)
            all_metrics[mode] = metrics
            _print_row(mode, metrics, n=len(names), elapsed_s=elapsed)
        if args.out_json:
            Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out_json).write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")


# Entry point.
if __name__ == "__main__":
    main()
