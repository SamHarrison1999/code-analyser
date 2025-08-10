# --- file: eval/eval.py ---
# Annotation: argparse parses CLI flags like --csv, --out, --service-url, --threshold, --batch-size, --plots, --kfold.
import argparse
# Annotation: csv handles robust CSV IO even when snippets contain embedded newlines.
import csv
# Annotation: json is used to serialise reasons/scores into the preds CSV for later inspection.
import json
# Annotation: pathlib gives portable path handling and easy directory creation.
from pathlib import Path
# Annotation: time is used to report simple timings for rules/model/fusion passes.
import time
# Annotation: typing improves clarity of function signatures in this evaluation module.
from typing import List, Dict, Set, Tuple

# Annotation: matplotlib is used to save confusion matrix and ROC figures; we do not depend on seaborn.
import matplotlib.pyplot as plt
# Annotation: sklearn provides confusion matrices and ROC AUC utilities.
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Annotation: Import your rule engine aggregator to collect reasons from AST + tools (flake8, bandit, SonarLint, etc.).
from src.ml.rule_engine import collect_reasons
# Annotation: Import the HTTP client for your FastAPI model service.
from src.ml.model_client import AnalyserModelClient

from src.ml.fusion import gated_fuse_one, DEFAULT_MODEL_TH, DEFAULT_RULE_TH


# Annotation: Canonical multi-label tag order used throughout the project.
LABEL_ORDER: List[str] = ["sast_risk", "ml_signal", "best_practice"]

# Annotation: Parse a semicolon-separated label string into a canonicalised set.
def parse_labels(s: str) -> Set[str]:
    # Annotation: Guard against empty input and normalise whitespace.
    s = (s or "").strip()
    # Annotation: Return the set of non-empty tags.
    return {t.strip() for t in s.split(";") if t.strip()}

# Annotation: Convert rule-engine per-line reasons into overall snippet label set and per-label message bucket.
def labels_from_reasons(reasons_by_line: Dict[int, List[str]]) -> Tuple[Set[str], Dict[str, List[str]]]:
    # Annotation: We keep a bag of messages per label for CSV debugging.
    per_label: Dict[str, List[str]] = {k: [] for k in LABEL_ORDER}
    # Annotation: Flatten all messages.
    all_msgs: List[str] = [m for _, msgs in reasons_by_line.items() for m in msgs]
    # Annotation: Keyword buckets to map reasons to labels.
    sec_keys = ["unsafe", "injection", "verify=false", "exec", "eval", "pickle", "yaml.load", "shell=true", "credential", "deserial", "sql", "md5", "sha1", "b3"]
    style_keys = ["pep8", "docstring", "unused", "import", "naming", "format", "style", "convention", "wildcard import", "mutable", "bare except", "is true", "shadow", "mutate"]
    ml_keys = ["strip().lower()", "redundant", "pointless", "double assign", "no newline", "bool", "identity", "join(list", "slice"]
    # Annotation: Decide labels via simple keyword presence and collect messages.
    labels: Set[str] = set()
    for m in all_msgs:
        low = m.lower()
        if any(k in low for k in sec_keys):
            labels.add("sast_risk")
            per_label["sast_risk"].append(m)
        if any(k in low for k in style_keys):
            labels.add("best_practice")
            per_label["best_practice"].append(m)
        if any(k in low for k in ml_keys):
            labels.add("ml_signal")
            per_label["ml_signal"].append(m)
    # Annotation: If we found messages but no bucket matched, default to ml_signal as a neutral indicator.
    if not labels and all_msgs:
        labels.add("ml_signal")
        per_label["ml_signal"].extend(all_msgs)
    # Annotation: Return overall labels and per-label reasons.
    return labels, per_label

# Annotation: Batch-call the model microservice and collect thresholded label sets and raw scores.
def model_predict_all(client: AnalyserModelClient, texts: List[str], threshold: float, batch_size: int) -> Tuple[List[Set[str]], List[Dict[str, float]]]:
    # Annotation: Outputs aligned with inputs.
    all_label_sets: List[Set[str]] = []
    all_scores: List[Dict[str, float]] = []
    # Annotation: Iterate in fixed-size batches for efficiency.
    for i in range(0, len(texts), batch_size):
        # Annotation: Slice the current batch.
        batch = texts[i:i + batch_size]
        # Annotation: Query /predict once per batch.
        resp = client.predict(batch, threshold=threshold)
        # Annotation: Append thresholded labels and raw probabilities.
        for item in resp["results"]:
            # Annotation: Prefer service-supplied labels; derive from scores if absent.
            lbls = set(item.get("labels") or [])
            if not lbls:
                sc = item.get("scores") or {}
                lbls = {k for k, v in sc.items() if float(v) >= threshold}
            all_label_sets.append(lbls)
            all_scores.append(item.get("scores") or {})
    # Annotation: Return aligned lists.
    return all_label_sets, all_scores

# Annotation: Simple fusion policy — union of rules and model so we keep positives from either side.
def fuse_labels(rule_labels: Set[str], model_labels: Set[str]) -> Set[str]:
    # Annotation: Return the union set.
    return set(rule_labels) | set(model_labels)

# Annotation: Compute per-label TP/FP/FN and convert to precision/recall/F1.
def prf1_per_label(y_true: List[Set[str]], y_pred: List[Set[str]], labels: List[str]) -> Dict[str, Dict[str, float]]:
    # Annotation: Start counters for each label.
    counts: Dict[str, Dict[str, int]] = {l: {"tp": 0, "fp": 0, "fn": 0} for l in labels}
    # Annotation: Accumulate counts per label across examples.
    for t, p in zip(y_true, y_pred):
        for l in labels:
            in_t = l in t
            in_p = l in p
            if in_t and in_p:
                counts[l]["tp"] += 1
            elif not in_t and in_p:
                counts[l]["fp"] += 1
            elif in_t and not in_p:
                counts[l]["fn"] += 1
    # Annotation: Convert to metrics with safe zero-handling.
    metrics: Dict[str, Dict[str, float]] = {}
    for l, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        metrics[l] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    # Annotation: Return per-label metric dict.
    return metrics

# Annotation: Summarise metrics with micro/macro averaging plus the per-label table.
def summarise_metrics(y_true: List[Set[str]], y_pred: List[Set[str]], labels: List[str]) -> Dict[str, Dict[str, float]]:
    # Annotation: First compute per-label metrics to support macro.
    per = prf1_per_label(y_true, y_pred, labels)
    # Annotation: Micro across all labels pools TP/FP/FN.
    tp = sum(int(l in t and l in p) for t, p in zip(y_true, y_pred) for l in labels)
    fp = sum(int(l not in t and l in p) for t, p in zip(y_true, y_pred) for l in labels)
    fn = sum(int(l in t and l not in p) for t, p in zip(y_true, y_pred) for l in labels)
    # Annotation: Compute micro P/R/F1.
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0
    # Annotation: Macro is the unweighted mean across labels.
    macro_p = sum(per[l]["precision"] for l in labels) / len(labels)
    macro_r = sum(per[l]["recall"] for l in labels) / len(labels)
    macro_f1 = sum(per[l]["f1"] for l in labels) / len(labels)
    # Annotation: Return a compact summary bundle.
    return {"micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1}, "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1}, "per_label": per}

# Annotation: Read the evaluation CSV and return list of (id, text, true_label_set).
def load_dataset(csv_path: Path) -> List[Tuple[str, str, Set[str]]]:
    # Annotation: Use newline='' so csv module normalises line endings correctly on Windows.
    with csv_path.open(newline="", encoding="utf-8") as f:
        # Annotation: Expect header id,text,labels.
        r = csv.DictReader(f)
        # Annotation: Parse each row into a tuple and keep order.
        rows = [(row["id"], row["text"], parse_labels(row.get("labels", ""))) for row in r]
    # Annotation: Return the dataset in CSV order.
    return rows

# Annotation: Write a predictions CSV for later inspection (includes reasons and scores).
def write_predictions(out_csv: Path, rows: List[Tuple[str, str, Set[str]]], y_rules: List[Set[str]], y_model: List[Set[str]], y_fused: List[Set[str]], reasons_all: List[Dict[str, List[str]]], scores_all: List[Dict[str, float]]) -> None:
    # Annotation: Ensure parent directory exists.
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Annotation: Open output file and write header + rows.
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        # Annotation: Column order is stable for downstream plotting.
        w = csv.DictWriter(f, fieldnames=["id", "labels_true", "labels_rules", "labels_model", "labels_fusion", "text", "rule_reasons", "model_scores"])
        # Annotation: Write header first.
        w.writeheader()
        # Annotation: Emit one aligned row per example.
        for (rid, text, y_true), lr, lm, lf, reas, sc in zip(rows, y_rules, y_model, y_fused, reasons_all, scores_all):
            # Annotation: Canonicalise label ordering in the CSV for readability.
            w.writerow({"id": rid, "labels_true": ";".join(sorted(y_true, key=LABEL_ORDER.index)) if y_true else "", "labels_rules": ";".join(sorted(lr, key=LABEL_ORDER.index)) if lr else "", "labels_model": ";".join(sorted(lm, key=LABEL_ORDER.index)) if lm else "", "labels_fusion": ";".join(sorted(lf, key=LABEL_ORDER.index)) if lf else "", "text": text, "rule_reasons": json.dumps(reas, ensure_ascii=False), "model_scores": json.dumps(sc, ensure_ascii=False)})

# Annotation: Plot and save a 2x2 confusion matrix figure for an individual label.
def plot_confusion_for_label(y_true: List[Set[str]], y_pred: List[Set[str]], label: str, out_dir: Path, title_prefix: str) -> None:
    # Annotation: Create binary arrays for the target label.
    y_t = [1 if label in t else 0 for t in y_true]
    y_p = [1 if label in p else 0 for p in y_pred]
    # Annotation: Compute confusion matrix in TN, FP, FN, TP layout then reshape to 2x2.
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    # Annotation: Start a fresh figure for this label.
    plt.figure(figsize=(4, 4))
    # Annotation: Show as an image so counts are visible; no custom colour map to keep dependencies minimal.
    plt.imshow(cm, interpolation="nearest")
    # Annotation: Set titles and axes labels.
    plt.title(f"{title_prefix} — {label}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    # Annotation: Set tick marks for the binary classes.
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    # Annotation: Annotate each cell with its count.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    # Annotation: Tight layout and write to disk.
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{title_prefix.lower().replace(' ', '_')}_{label}_cm.png", dpi=150)
    # Annotation: Close to free memory in repeated calls.
    plt.close()

# Annotation: Plot ROC curve for the model per label; rules are 0/1 so we draw a single operating point.
def plot_roc_for_label(y_true: List[Set[str]], model_scores: List[Dict[str, float]], label: str, out_dir: Path, title_prefix: str) -> None:
    # Annotation: Build binary ground truth per label.
    y_t = [1 if label in t else 0 for t in y_true]
    # Annotation: Extract continuous model probabilities for this label; default to 0.0 when missing.
    y_s = [float(sc.get(label, 0.0)) for sc in model_scores]
    # Annotation: Compute ROC curve points using sklearn.
    fpr, tpr, _ = roc_curve(y_t, y_s)
    # Annotation: Compute area under the ROC curve.
    auc_val = auc(fpr, tpr)
    # Annotation: Plot the ROC curve.
    plt.figure(figsize=(4, 4))
    # Annotation: Diagonal reference for a random classifier.
    plt.plot([0, 1], [0, 1], linestyle="--")
    # Annotation: The model ROC line with AUC in legend.
    plt.plot(fpr, tpr, label=f"model AUC={auc_val:.2f}")
    # Annotation: Axes labels and title.
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} ROC — {label}")
    plt.legend(loc="lower right")
    # Annotation: Tight layout and save to disk.
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{title_prefix.lower().replace(' ', '_')}_{label}_roc.png", dpi=150)
    # Annotation: Close the figure to conserve memory.
    plt.close()

# Annotation: Pretty-print a concise metrics block to stdout.
def print_report(title: str, summary: Dict[str, Dict[str, float]]) -> None:
    # Annotation: Section heading first.
    print(f"\n=== {title} ===")
    # Annotation: Micro and macro aggregates.
    print(f"micro: P={summary['micro']['precision']:.2f} R={summary['micro']['recall']:.2f} F1={summary['micro']['f1']:.2f}")
    print(f"macro: P={summary['macro']['precision']:.2f} R={summary['macro']['recall']:.2f} F1={summary['macro']['f1']:.2f}")
    # Annotation: Per-label table in canonical order.
    for l in LABEL_ORDER:
        m = summary["per_label"][l]
        print(f"{l:13s}  P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} (support={int(m['support'])})")

# Annotation: Evaluate one pass on the full dataset and optionally write preds + plots.
def evaluate_once(csv_path: Path, out_csv: Path, service_url: str, threshold: float, batch_size: int, make_plots: bool) -> Dict[str, Dict[str, float]]:
    # Annotation: Load dataset rows as (id,text,true_labels).
    rows = load_dataset(csv_path)
    # Annotation: Slice vectors for convenience.
    texts = [t for _, t, _ in rows]
    y_true = [labs for _, _, labs in rows]
    # Annotation: RULES pass — collect reasons per snippet and map to label sets.
    t0 = time.time()
    rule_labels: List[Set[str]] = []
    rule_reasons_all: List[Dict[str, List[str]]] = []
    for txt in texts:
        reasons = collect_reasons(txt)
        labs, per_label = labels_from_reasons(reasons)
        rule_labels.append(labs)
        rule_reasons_all.append(per_label)
    t1 = time.time()
    # Annotation: MODEL pass — call the microservice with batching.
    client = AnalyserModelClient(base_url=service_url, timeout=30.0)
    model_labels, model_scores = model_predict_all(client, texts, threshold=threshold, batch_size=batch_size)
    t2 = time.time()
    # GATED FUSION — model gate with per-label thresholds from fusion.py
    fused_labels: list[set[str]] = []
    for rule_set, score_dict in zip(rule_labels, model_scores):
        decisions = gated_fuse_one(
            scores=score_dict,
            rule_hits=rule_set,  # just the rule labels we found for this snippet
            th_model={"sast_risk": 0.35, "ml_signal": 0.65, "best_practice": 0.61},
            th_rule=None,  # None => use DEFAULT_RULE_TH
        )
        fused_labels.append({lab for lab, ok in decisions.items() if ok})
    t3 = time.time()
    # Annotation: Persist predictions to CSV for inspection.
    write_predictions(out_csv, rows, rule_labels, model_labels, fused_labels, rule_reasons_all, model_scores)
    # Annotation: Compute metrics for all three systems.
    rules_summary = summarise_metrics(y_true, rule_labels, LABEL_ORDER)
    model_summary = summarise_metrics(y_true, model_labels, LABEL_ORDER)
    fused_summary = summarise_metrics(y_true, fused_labels, LABEL_ORDER)
    # Annotation: Report timings so you can plan runs.
    print(f"\nTimings: rules={t1-t0:.2f}s model={t2-t1:.2f}s fuse={t3-t2:.2f}s total={t3-t0:.2f}s")
    # Annotation: Print the three sections in a fixed order.
    print_report("RULES", rules_summary)
    print_report("MODEL", model_summary)
    print_report("FUSION", fused_summary)
    # Annotation: If requested, render plots per label.
    if make_plots:
        plots_dir = Path("eval/out/plots")
        for lab in LABEL_ORDER:
            plot_confusion_for_label(y_true, rule_labels, lab, plots_dir, "Rules")
            plot_confusion_for_label(y_true, model_labels, lab, plots_dir, "Model")
            plot_confusion_for_label(y_true, fused_labels, lab, plots_dir, "Fusion")
            plot_roc_for_label(y_true, model_scores, lab, plots_dir, "Model")
        print(f"\nSaved plots to: {plots_dir}")
    # Annotation: Return a dict of macro/micro metrics for possible aggregation by k-fold.
    return {"rules": rules_summary, "model": model_summary, "fusion": fused_summary}

# Annotation: Try to import multilabel stratified k-fold; fall back to plain KFold if unavailable.
def _get_kfold(n_splits: int, y_true: List[Set[str]]):
    # Annotation: Build a binary indicator matrix for stratification if iterstrat is installed.
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        import numpy as np
        Y = np.zeros((len(y_true), len(LABEL_ORDER)), dtype=int)
        for i, labs in enumerate(y_true):
            for j, l in enumerate(LABEL_ORDER):
                if l in labs:
                    Y[i, j] = 1
        return MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(range(len(y_true)), Y)
    except Exception:
        from sklearn.model_selection import KFold
        return KFold(n_splits=n_splits, shuffle=True, random_state=42).split(range(len(y_true)))

# Annotation: Run a k-fold evaluation to estimate variance (no training; just repeated sub-sampling).
def evaluate_kfold(csv_path: Path, service_url: str, threshold: float, batch_size: int, k_splits: int) -> None:
    # Annotation: Load data once and prepare structures for subset evaluation.
    rows = load_dataset(csv_path)
    texts = [t for _, t, _ in rows]
    y_true_all = [labs for _, _, labs in rows]
    # Annotation: Prepare k-fold splitter (multilabel-stratified if available).
    splitter = _get_kfold(k_splits, y_true_all)
    # Annotation: Accumulators for macro F1 per system per fold.
    fold_stats: List[Tuple[float, float, float]] = []
    # Annotation: Evaluate each fold by slicing to the validation indices.
    for fold_idx, (_, val_idx) in enumerate(splitter, 1):
        sub_csv = Path(f"eval/out/fold_{fold_idx}.csv")
        sub_rows = [rows[i] for i in val_idx]
        with sub_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "text", "labels"])
            w.writeheader()
            for rid, text, labs in sub_rows:
                w.writerow({"id": rid, "text": text, "labels": ";".join(sorted(labs, key=LABEL_ORDER.index))})
        out_csv = Path(f"eval/out/preds_fold_{fold_idx}.csv")
        print(f"\n--- Fold {fold_idx}/{k_splits} ({len(sub_rows)} examples) ---")
        summaries = evaluate_once(sub_csv, out_csv, service_url, threshold, batch_size, make_plots=False)
        r_f1 = summaries["rules"]["macro"]["f1"]
        m_f1 = summaries["model"]["macro"]["f1"]
        f_f1 = summaries["fusion"]["macro"]["f1"]
        fold_stats.append((r_f1, m_f1, f_f1))
    # Annotation: Report mean ± std for macro F1 across folds.
    import statistics as stats
    r_mean = stats.mean(x[0] for x in fold_stats)
    m_mean = stats.mean(x[1] for x in fold_stats)
    f_mean = stats.mean(x[2] for x in fold_stats)
    r_std = stats.pstdev(x[0] for x in fold_stats)
    m_std = stats.pstdev(x[1] for x in fold_stats)
    f_std = stats.pstdev(x[2] for x in fold_stats)
    print(f"\n=== K-fold summary (macro F1, {k_splits} folds) ===")
    print(f"Rules : {r_mean:.3f} ± {r_std:.3f}")
    print(f"Model : {m_mean:.3f} ± {m_std:.3f}")
    print(f"Fusion: {f_mean:.3f} ± {f_std:.3f}")

# Annotation: CLI entrypoint that wires single-run, plots, and optional k-fold evaluation.
def main() -> None:
    # Annotation: Define flags with sensible defaults.
    ap = argparse.ArgumentParser(description="Rules vs Model vs Fusion with confusion matrices, ROC, and k-fold")
    ap.add_argument("--csv", default="datasets/eval/snippets.csv", help="Path to CSV with columns: id,text,labels")
    ap.add_argument("--out", default="eval/out/preds.csv", help="Output CSV path for predictions")
    ap.add_argument("--service-url", default="http://127.0.0.1:8111", help="Base URL of the model FastAPI service")
    ap.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for model label activation")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for model service requests")
    ap.add_argument("--plots", action="store_true", help="Save confusion matrices and ROC curves per label")
    ap.add_argument("--kfold", type=int, default=0, help="If >0, run k-fold evaluation with this many folds")
    args = ap.parse_args()
    # Annotation: Resolve paths and ensure output directories exist.
    csv_path = Path(args.csv)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Annotation: Single full-dataset evaluation (also produces plots if requested).
    evaluate_once(csv_path, out_csv, args.service_url, args.threshold, args.batch_size, make_plots=args.plots)
    # Annotation: Optional k-fold repeated evaluation for variance estimates.
    if args.kfold and args.kfold > 0:
        evaluate_kfold(csv_path, args.service_url, args.threshold, args.batch_size, args.kfold)

# Annotation: Standard entry-guard so the module can be imported safely.
if __name__ == "__main__":
    # Annotation: Invoke the CLI.
    main()
