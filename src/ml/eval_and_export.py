# src/ml/eval_and_export.py
# Import standard libraries for filesystem, arguments, JSON, typing and progress.
# Path utilities.
import argparse

# JSON serialisation for metrics and overlays.
import json
import os

# Typing helpers.
from typing import Dict, List, Tuple, Optional

# Plotting for reliability diagrams.
import matplotlib.pyplot as plt

# Numerical work.
import numpy as np

# Metrics for classification.
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

# Filesystem paths.
from pathlib import Path

# Import Transformers APIs, but fall back to light stubs when unavailable in tests.
try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        AutoConfig,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )
except Exception:

    class AutoModelForSequenceClassification:
        @classmethod
        def from_pretrained(cls, *a, **k):
            class _M:
                config = type(
                    "C", (object,), {"num_labels": 2, "id2label": {"0": "LABEL_0", "1": "LABEL_1"}}
                )()

                def to(self, *a, **k):
                    return self

                def eval(self):
                    return self

            return _M()

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *a, **k):
            class _T:
                model_max_length = 128

                def __call__(self, texts, **kw):
                    if isinstance(texts, str):
                        texts = [texts]
                    return {
                        "input_ids": [[1, 2, 3] for _ in texts],
                        "attention_mask": [[1, 1, 1] for _ in texts],
                    }

            return _T()

    class AutoConfig:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return type(
                "C", (object,), {"num_labels": 2, "id2label": {"0": "LABEL_0", "1": "LABEL_1"}}
            )()

    class TrainingArguments:
        def __init__(self, output_dir, **kw):
            self.output_dir = output_dir

    class Trainer:
        def __init__(self, *a, **k):
            pass

        def train(self):
            return {"train_loss": 0.0}

        def evaluate(self, *a, **k):
            return {"eval_loss": 0.0}

        def save_model(self, *a, **k):
            pass

    def default_data_collator(*a, **k):
        return None

# Annotation: Use a relative import so Python resolves the module inside the 'ml' package when running 'python -m ml.eval_and_export'.
from .dataset_loader import load_local_annotated_dataset

from huggingface_hub import upload_folder

# Label order must match training; adjust if you changed it there.
LABEL_MAP: Dict[str, int] = {"sast_risk": 0, "ml_signal": 1, "best_practice": 2}

# Simple dataset adapter compatible with HF Trainer predict().
class HFDataset:
    # Construct with a list of records produced by your loader.
    def __init__(self, records: List[Dict]):
        # Store records.
        self.records = records

    # Return dataset length.
    def __len__(self) -> int:
        # Number of examples.
        return len(self.records)

    # Provide items for the Trainer; labels are floats for multi-label BCE.
    def __getitem__(self, idx: int) -> Dict:
        # Build the item with input ids and mask.
        item = {
            "input_ids": self.records[idx]["input_ids"],
            "attention_mask": self.records[idx]["attention_mask"],
        }
        # Add labels as float32.
        item["labels"] = np.array(self.records[idx]["labels"], dtype=np.float32)
        # Keep filename for overlays (ignored by forward pass).
        item["filename"] = self.records[idx]["filename"]
        # Return item.
        return item

# Stable sigmoid for converting logits to probabilities.
def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable logistic.
    return 1.0 / (1.0 + np.exp(-x))

# Compute Expected Calibration Error and provide bin curves.
def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    # Create equal-width bins in [0,1].
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # Assign each probability to a bin.
    bin_ids = np.digitize(y_prob, bins) - 1
    # Prepare accumulators.
    ece = 0.0
    bin_acc = np.zeros(n_bins, dtype=np.float32)
    bin_conf = np.zeros(n_bins, dtype=np.float32)
    # Count per bin.
    for b in range(n_bins):
        # Mask for current bin.
        mask = bin_ids == b
        # Skip empty bins.
        if not np.any(mask):
            continue
        # Empirical accuracy in bin.
        acc = float(y_true[mask].mean())
        # Mean confidence in bin.
        conf = float(y_prob[mask].mean())
        # Bin weight (fraction of samples).
        w = float(mask.mean())
        # Add contribution to ECE.
        ece += abs(acc - conf) * w
        # Store curves.
        bin_acc[b] = acc
        bin_conf[b] = conf
    # Return ECE and curves.
    return float(ece), bins, bin_acc, bin_conf

# Save a reliability diagram to a PNG path.
def save_reliability_plot(
    label_name: str, bins: np.ndarray, acc: np.ndarray, conf: np.ndarray, out_path: str
) -> None:
    # Create a fresh figure.
    fig = plt.figure(figsize=(5, 4))
    # X positions at bin midpoints.
    mids = (bins[:-1] + bins[1:]) / 2.0
    # Plot accuracy bars.
    plt.bar(
        mids,
        acc,
        width=(bins[1] - bins[0]) * 0.9,
        alpha=0.6,
        label="empirical accuracy",
    )
    # Plot confidence mean as a line.
    plt.plot(mids, conf, linewidth=2, label="mean confidence")
    # Ideal calibration diagonal.
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    # Axes labels and title.
    plt.xlabel("confidence")
    plt.ylabel("accuracy")
    plt.title(f"Reliability – {label_name}")
    # Legend.
    plt.legend()
    # Tight layout and save to disk.
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

# Escape HTML for safe code rendering.
def html_escape(s: str) -> str:
    # Replace special characters.
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# Render a per-file overlay HTML with confidences.
def render_overlay_html(
    filename: str, code_text: str, probs: Dict[str, float], threshold: float
) -> str:
    # Badge row for label confidences.
    badges = "".join(
        f"<span style='padding:2px 6px;border-radius:10px;border:1px solid #888;margin-right:6px'>{lbl}: {probs[lbl]:.2f}</span>"
        for lbl in LABEL_MAP.keys()
    )
    # Header with filename and badges.
    header = f"<h3 style='font-family:Inter,system-ui'>File: {html_escape(filename)}</h3><div>{badges}</div>"
    # Code block for context.
    pre = f"<pre style='background:#0b1021;color:#ebedf5;padding:12px;border-radius:8px;overflow:auto;font-size:12px;line-height:1.4'>{html_escape(code_text)}</pre>"
    # Footnote with threshold.
    foot = f"<div style='color:#666;font-size:12px'>Threshold {threshold:.2f}. Ticks on dashboard indicate predictions ≥ threshold.</div>"
    # Compose final minimal document.
    return f"<!doctype html><meta charset='utf-8'><title>{html_escape(filename)} – overlay</title><div style='max-width:980px;margin:24px auto'>{header}{pre}{foot}</div>"

# Try to locate original code path by filename stem within code root.
def find_code_path(code_root: str, filename_no_ext: str) -> Optional[str]:
    # Walk the repository for a matching .py file.
    for root, _, files in os.walk(code_root):
        for f in files:
            if f.endswith(".py") and os.path.splitext(f)[0] == filename_no_ext:
                return os.path.join(root, f)
    # Not found.
    return None

# Infer a sensible model_type for AutoConfig from a checkpoint folder.
def _infer_model_type(checkpoint_dir: str) -> str:
    # If training_args.bin exists, use the original model name to infer the type.
    ta = os.path.join(checkpoint_dir, "training_args.bin")
    if os.path.exists(ta):
        try:
            # Lazy import to avoid torch dependency at module import time.
            import torch

            tr_args = torch.load(ta, map_location="cpu")
            name = str(getattr(tr_args, "model_name_or_path", "")).lower()
            # Heuristics based on common substrings.
            if any(k in name for k in ["roberta", "codebert", "xlm-roberta"]):
                return "roberta"
            if "bert" in name:
                return "bert"
            if "deberta" in name:
                return "deberta"
            if "mpnet" in name:
                return "mpnet"
            if "distilbert" in name:
                return "distilbert"
        except Exception:
            # Fall back to file-based heuristics if loading fails.
            pass
    # File-based heuristics when training args are unavailable.
    has_merges = os.path.exists(os.path.join(checkpoint_dir, "merges.txt"))
    has_vocab = os.path.exists(os.path.join(checkpoint_dir, "vocab.json"))
    # RoBERTa/CodeBERT checkpoints ship merges.txt + vocab.json.
    if has_merges and has_vocab:
        return "roberta"
    # Fallback to roberta which works for microsoft/codebert-base.
    return "roberta"

# Ensure a valid config.json exists; create a minimal one if missing or unreadable.
def _ensure_config_json(checkpoint_dir: str, label_map: Dict[str, int]) -> str:
    # Compute paths and label dictionaries.
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    id2label = {i: lbl for lbl, i in label_map.items()}
    label2id = {lbl: i for lbl, i in label_map.items()}
    num_labels = max(label_map.values()) + 1
    # If config.json already exists, verify it is loadable; otherwise reconstruct.
    need_write = False
    if not os.path.exists(cfg_path):
        # Absent config; we will write one.
        need_write = True
    else:
        try:
            # Try to parse through AutoConfig to validate.
            _ = AutoConfig.from_pretrained(checkpoint_dir)
        except Exception:
            # Present but unreadable; rewrite.
            need_write = True
    # Write a minimal but sufficient config.json when required.
    if need_write:
        model_type = _infer_model_type(checkpoint_dir)
        # Minimal configuration dictionary with safe defaults; HF fills the rest.
        cfg = {
            "model_type": model_type,
            "num_labels": num_labels,
            "id2label": {str(k): v for k, v in id2label.items()},
            "label2id": label2id,
            "problem_type": "multi_label_classification",
        }
        # Persist the configuration so future tools can load it.
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        # Informative print for transparency.
        print(
            f"🧩 Reconstructed missing/invalid config.json at: {cfg_path} (model_type={model_type}, num_labels={num_labels})"
        )
    # Return the path for convenience.
    return cfg_path

# Main routine to evaluate an existing checkpoint, export artefacts and optionally push to hub.
def main() -> None:
    # Define CLI arguments.
    ap = argparse.ArgumentParser()
    # Where your original code lives (for linking overlays).
    ap.add_argument("--code-dir", default="datasets/github_fintech")
    # Where your annotations live (for dataset loader).
    ap.add_argument("--annotation-dir", default="datasets/annotated_fintech")
    # Path to an existing HF checkpoint directory, e.g. checkpoints/hf/checkpoint-309.
    ap.add_argument("--checkpoint-dir", default="checkpoints/hf/checkpoint-309")
    # Confidence threshold for positive predictions.
    ap.add_argument("--confidence-threshold", type=float, default=0.5)
    # Tokeniser/model max length to match training.
    ap.add_argument("--max-length", type=int, default=384)
    # Optional cap for quick runs.
    ap.add_argument("--max-samples", type=int, default=None)
    # Output folders for artefacts.
    ap.add_argument("--export-dir", default="artifacts")
    # Overlays directory.
    ap.add_argument("--overlays-dir", default="artifacts/overlays")
    # Dashboard directory.
    ap.add_argument("--dashboard-dir", default="artifacts/dashboard")
    # TensorBoard logging directory for eval scalars.
    ap.add_argument("--tensorboard-dir", default="runs/eval")
    # Optional HF Hub repo id 'user/repo' to push.
    ap.add_argument("--hub-repo", default=None)
    # Environment variable name that holds the HF token.
    ap.add_argument("--hub-token-env", default="HUGGINGFACE_HUB_TOKEN")
    # Parse the arguments.
    args = ap.parse_args()
    # Ensure directories exist.
    os.makedirs(args.export_dir, exist_ok=True)
    os.makedirs(args.overlays_dir, exist_ok=True)
    os.makedirs(args.dashboard_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    # Use the checkpoint dir as the tokenizer source (the prior conditional contained a typo and always chose the right-hand side; simplifying for clarity).
    records, _ = load_local_annotated_dataset(
        code_dir=args.code_dir,
        annotation_dir=args.annotation_dir,
        tokenizer_name=args.checkpoint_dir,
        max_samples=args.max_samples,
        max_length=args.max_length,
        confidence_threshold=args.confidence_threshold,
        stratify=False,
    )
    # Report number of examples discovered.
    print(f"✅ Loaded {len(records)} examples from local annotations")
    # Create the HF dataset wrapper.
    ds = HFDataset(records)
    # Before loading, ensure a valid config.json exists or reconstruct one on the fly.
    _ensure_config_json(args.checkpoint_dir, LABEL_MAP)
    # Load model and tokeniser from your checkpoint directory.
    config = AutoConfig.from_pretrained(args.checkpoint_dir)
    # Force multi-label just in case; respects saved num_labels.
    config.problem_type = "multi_label_classification"
    # Load model weights.
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir, config=config)
    # Load tokeniser (merges/vocab present in checkpoint root).
    tok = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    # Set up a Trainer purely for prediction, with TensorBoard logging enabled for eval scalars.
    targs = TrainingArguments(
        output_dir=args.checkpoint_dir,
        per_device_eval_batch_size=8,
        report_to=["tensorboard"],
        logging_dir=args.tensorboard_dir,
    )
    # Build Trainer.
    trainer = Trainer(model=model, args=targs, data_collator=default_data_collator, tokenizer=tok)
    # Run prediction to obtain logits.
    print("🔎 Running inference on evaluation split...")
    out = trainer.predict(ds)
    # Convert logits to probabilities via sigmoid.
    probs = _sigmoid(out.predictions)
    # Threshold to binary predictions.
    thr = args.confidence_threshold
    preds = (probs >= thr).astype(np.int32)
    # Stack ground truths from records.
    gts = np.stack([np.array(r["labels"], dtype=np.int32) for r in records], axis=0)
    # Compute per-label metrics.
    per_label = {}
    for lbl, idx in LABEL_MAP.items():
        # Extract truth and predictions for this label.
        y_true = gts[:, idx]
        y_pred = preds[:, idx]
        # Compute P/R/F1 (binary).
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        # Store metrics.
        per_label[lbl] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "support": int(y_true.sum()),
        }
    # Compute overall micro and macro F1.
    micro_f1 = float(f1_score(gts.flatten(), preds.flatten(), average="micro", zero_division=0))
    # Macro over labels.
    macro_f1 = float(f1_score(gts, preds, average="macro", zero_division=0))
    # Exact-match accuracy (all labels correct for a sample).
    exact_match = float((preds == gts).all(axis=1).mean())
    # Print summary.
    print(
        json.dumps(
            {
                "accuracy_exact_match": exact_match,
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "per_label": per_label,
            },
            indent=2,
        )
    )
    # Compute calibration per label and save reliability diagrams.
    calib_dir = os.path.join(args.export_dir, "calibration")
    os.makedirs(calib_dir, exist_ok=True)
    # Collect calibration results.
    calibration = {}
    for lbl, idx in LABEL_MAP.items():
        # ECE and curves.
        ece, bins, acc, conf = expected_calibration_error(
            gts[:, idx].astype(np.float32), probs[:, idx].astype(np.float32)
        )
        # Path for plot.
        plot_path = os.path.join(calib_dir, f"reliability_{lbl}.png")
        # Save plot.
        save_reliability_plot(lbl, bins, acc, conf, plot_path)
        # Record relative path in report.
        calibration[lbl] = {
            "ece": float(ece),
            "plot": os.path.relpath(plot_path, args.export_dir),
        }
    # Write metrics JSON.
    with open(os.path.join(args.export_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy_exact_match": exact_match,
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "per_label": per_label,
                "calibration": calibration,
            },
            f,
            indent=2,
        )
    # Create overlays (JSON + HTML) and dashboard index.
    rows = []
    # Ensure overlays directory exists.
    os.makedirs(args.overlays_dir, exist_ok=True)
    # Iterate predictions to write files.
    for i, rec in enumerate(records):
        # Current filename stem used throughout the pipeline.
        fname = rec["filename"]
        # Probability map for UI.
        pmap = {lbl: float(probs[i, idx]) for lbl, idx in LABEL_MAP.items()}
        # Prediction map for UI.
        predmap = {lbl: bool(preds[i, idx]) for lbl, idx in LABEL_MAP.items()}
        # JSON overlay content.
        overlay = {"filename": fname, "probs": pmap, "preds": predmap, "threshold": thr}
        # Write JSON overlay.
        with open(os.path.join(args.overlays_dir, f"{fname}.json"), "w", encoding="utf-8") as jf:
            json.dump(overlay, jf, indent=2)
        # Try to read source code for the HTML overlay.
        code_path = find_code_path(args.code_dir, fname)
        code_txt = ""
        # Read file if available.
        if code_path and os.path.exists(code_path):
            with open(code_path, "r", encoding="utf-8") as cf:
                code_txt = cf.read()
        # Render HTML overlay.
        html = render_overlay_html(fname, code_txt, pmap, thr)
        # Save HTML overlay.
        html_path = os.path.join(args.overlays_dir, f"{fname}.html")
        with open(html_path, "w", encoding="utf-8") as hf:
            hf.write(html)
        # Record row for dashboard table.
        rows.append(
            {
                "filename": fname,
                "probs": pmap,
                "preds": predmap,
                "overlay_rel": os.path.relpath(html_path, args.dashboard_dir),
            }
        )
    # Build a very small static dashboard HTML.
    thead = "<tr><th>file</th>" + "".join(f"<th>{lbl}</th>" for lbl in LABEL_MAP.keys()) + "</tr>"
    # Body rows with ticks for positives.
    tbody = "".join(
        f"<tr><td><a href='{r['overlay_rel']}'>{html_escape(r['filename'])}</a></td>"
        + "".join(
            f"<td>{r['probs'][lbl]:.2f}" + (" ✅" if r["preds"][lbl] else "") + "</td>"
            for lbl in LABEL_MAP.keys()
        )
        + "</tr>"
        for r in rows
    )
    # Minimal page with some default styling.
    dash = (
        "<!doctype html><meta charset='utf-8'><title>Code Analyser – Dashboard</title><style>body{font-family:Inter,system-ui;margin:24px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px;font-size:14px}th{background:#f2f2f2;text-align:left}a{text-decoration:none}</style><h2>Predictions</h2><table><thead>"
        + thead
        + "</thead><tbody>"
        + tbody
        + "</tbody></table>"
    )
    # Ensure dashboard directory and write index.html.
    os.makedirs(args.dashboard_dir, exist_ok=True)
    # Write the dashboard HTML.
    with open(os.path.join(args.dashboard_dir, "index.html"), "w", encoding="utf-8") as df:
        df.write(dash)
    # Save a quick text pointer for convenience.
    with open(os.path.join(args.dashboard_dir, "README.txt"), "w", encoding="utf-8") as rf:
        rf.write(
            "Open index.html in a browser. Overlays are in ../overlays. Metrics and calibration in ../calibration."
        )
    # Optionally push the checkpoint directory (model + tokeniser) to the Hugging Face Hub.
    if args.hub_repo:
        token = os.environ.get(args.hub_token_env)
        if not token:
            raise RuntimeError(f"Set an access token in env var {args.hub_token_env}")

        # Ensure model is saved locally
        local_model_dir = Path(args.export_dir) / "trained_model"
        model.save_pretrained(local_model_dir)
        tok.save_pretrained(local_model_dir)

        # Upload entire folder to dataset repo
        upload_folder(
            folder_path=str(local_model_dir),
            repo_id=args.hub_repo,
            repo_type="dataset",
            token=token,
            path_in_repo=local_model_dir.name,
        )
        print(f"☁️ Uploaded trained model to: https://huggingface.co/datasets/{args.hub_repo}")
    # Final message.
    print("✅ Evaluation and export complete.")

# Standard entry point.
if __name__ == "__main__":
    # Run the main function.
    main()
