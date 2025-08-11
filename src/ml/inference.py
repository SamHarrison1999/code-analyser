# --- file: code_analyser/src/ml/inference.py ---
# --- file: src/ml/inference.py ---
"""
inference.py — Supervised AI annotation engine with overlay metric extraction.

Includes:
- AnnotationEngine: loads model + tokeniser and annotates code lines
- gather_all_metric_names_with_ai(): static + AI metric names
- Uses overlay_utils.gather_ai_overlays(): confidence-annotated token heatmap overlays
"""
# Torch is optional in tests; provide a tiny shim if unavailable.
try:
    import torch
except Exception:
    class _Dev:
        def __init__(self,*a,**k): pass
    class _Torch:
        def device(self, *a, **k): return _Dev()
        def no_grad(self):
            class _C: __enter__=lambda s: None; __exit__=lambda s,*a: False
            return _C()
    torch = _Torch()

# JSON and CSV are used for exporting annotations when requested.
import json
import csv

# Path is used for file-system paths and cache file construction.
from pathlib import Path

# Try the real AutoTokenizer; fall back to a tiny stub for tests.
try:
    from transformers import AutoTokenizer
except Exception:
    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *a, **k):
            class _T:
                model_max_length = 128
                def __call__(self, text, **kw):
                    if isinstance(text, str): text=[text]
                    return {"input_ids": [[1,2,3] for _ in text], "attention_mask": [[1,1,1] for _ in text]}
            return _T()


# List typing is used in signatures below.
from typing import List

# We import the local TF-based classifier wrapper for the existing local path.
from .model_tf import AnnotationClassifier

# Config paths for checkpoints and AI cache storage.
from .config import CHECKPOINT_DIR, AI_CACHE_DIR

# hashlib is used to create cache file names based on content hashes.
import hashlib

# The HTTP client lets us query the FastAPI model service (added for GUI integration).
from .model_client import AnalyserModelClient

# os is used to read environment flags and configuration.
import os

# We import the rule engine to collect multi-tool reasons per line.
from .rule_engine import collect_reasons

# ✅ Label mapping for annotation types
LABELS = {0: "⚠️ SAST Risk", 1: "🧠 ML Signal", 2: "✅ Best Practice"}
# Remote service returns canonical keys; we map them to the pretty labels used in the GUI.
REMOTE_LABEL_TO_PRETTY = {
    "sast_risk": "⚠️ SAST Risk",
    "ml_signal": "🧠 ML Signal",
    "best_practice": "✅ Best Practice",
}

DEFAULT_MODEL_PATH = f"{CHECKPOINT_DIR}/supervised_epoch_3.pt"
DEFAULT_MODEL_NAME = "microsoft/codebert-base"
DISTILLED_MODEL_NAME = "Salesforce/codet5-small"

# Internal keywords to categorise rule messages to labels.
_SAST_HINTS = (
    "unsafe",
    "verify=false",
    "exec",
    "eval",
    "unpickl",
    "yaml.load",
    "shell=true",
    "credential",
    "injection",
    "deserial",
    "sql",
    "cwe",
    "bandit",
)
# Style/readability hints to bias towards best practice.
_BEST_PRACTICE_HINTS = (
    "pep8",
    "docstring",
    "unused",
    "import",
    "naming",
    "format",
    "style",
    "convention",
    "readability",
)


# === 🧠 Supervised Annotation Engine ===
# This is the existing local model path; kept intact so the GUI continues to work without the HTTP service if desired.
class AnnotationEngine:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
        use_distilled: bool = False,
        dropout: float = 0.1,
    ):
        # Choose base model depending on distilled flag.
        self.model_name = DISTILLED_MODEL_NAME if use_distilled else model_name
        # Load the tokeniser matching the chosen base model.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Build the classifier head and load the trained checkpoint weights.
        self.model = AnnotationClassifier(model_name=self.model_name, dropout=dropout)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        # Switch to eval mode to disable dropout etc.
        self.model.eval()

    def annotate_line(self, line: str, min_confidence: float = 0.0, line_num: int = None) -> dict:
        # Tokenise the single line and limit sequence length.
        inputs = self.tokenizer(
            line, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        # Record the number of tokens for overlay metrics.
        token_count = len(inputs["input_ids"][0])
        # Disable gradients for inference efficiency.
        with torch.no_grad():
            # Forward pass through the classifier head.
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Use argmax for a single best label and softmax for confidence.
            pred = torch.argmax(logits, dim=-1).item()
            confidence = torch.softmax(logits, dim=-1).max().item()

        # If confidence is high enough, prefix the line with an annotation marker; otherwise leave unchanged.
        if confidence >= min_confidence:
            annotation = LABELS[pred]
            annotated_line = f"# {annotation} (confidence: {confidence:.2f})\n{line}"
        else:
            annotation = None
            annotated_line = line

        # Return a structured record used by the GUI and export paths.
        return {
            "line_num": line_num,
            "line": line,
            "tokens": token_count,
            "annotation": annotation,
            "confidence": confidence,
            "annotated": annotated_line,
        }

    def annotate_file(
        self,
        filepath: str,
        min_confidence: float = 0.0,
        as_dict: bool = False,
        export_path: str = None,
    ):
        # Read all lines from the target file (UTF-8).
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Prepare containers for per-line results and final output text.
        annotated = []
        output_lines = []

        # Iterate through lines and annotate those that are not comments/blank.
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                result = self.annotate_line(line, min_confidence=min_confidence, line_num=i + 1)
                annotated.append(result)
                output_lines.append(result["annotated"])
            else:
                output_lines.append(line)
                annotated.append(
                    {
                        "line_num": i + 1,
                        "line": line,
                        "tokens": 0,
                        "annotation": None,
                        "confidence": 0.0,
                        "annotated": line,
                    }
                )

        # Optionally export results to JSON or CSV for offline analysis.
        if export_path:
            if export_path.endswith(".json"):
                with open(export_path, "w", encoding="utf-8") as jf:
                    json.dump(annotated, jf, indent=2)
            elif export_path.endswith(".csv"):
                with open(export_path, "w", encoding="utf-8", newline="") as cf:
                    writer = csv.DictWriter(
                        cf,
                        fieldnames=[
                            "line_num",
                            "annotation",
                            "confidence",
                            "tokens",
                            "line",
                        ],
                    )
                    writer.writeheader()
                    for row in annotated:
                        writer.writerow({k: row[k] for k in writer.fieldnames})

        # Return structured data or reconstituted text according to caller preference.
        return annotated if as_dict else "".join(output_lines)


# === 🛰️ Remote (FastAPI) Annotation Engine for GUI integration ===
# This engine calls the running FastAPI model service so the GUI can use your fine-tuned HF model without loading it in-process.
class RemoteAnnotationEngine:
    def __init__(
        self,
        service_url: str | None = None,
        request_timeout: float = 30.0,
    ):
        # Allow overriding via env var MODEL_SERVICE_URL; default to localhost if unset.
        base_url = service_url or os.environ.get("MODEL_SERVICE_URL", "http://127.0.0.1:8111")
        # Create the HTTP client with a sensible timeout for interactive use.
        self.client = AnalyserModelClient(base_url=base_url, timeout=request_timeout)

    def _label_for_reason(self, reason: str) -> str:
        # Decide the canonical label key for a given textual reason.
        low = reason.lower()
        if any(k in low for k in _SAST_HINTS):
            return "sast_risk"
        if any(k in low for k in _BEST_PRACTICE_HINTS):
            return "best_practice"
        return "ml_signal"

    def annotate_line(self, line: str, min_confidence: float = 0.0, line_num: int = None) -> dict:
        # Ask the service to score just this one line; pass min_confidence as the activation threshold.
        res = self.client.predict([line], threshold=min_confidence)
        # Extract the first result (there is only one).
        item = res["results"][0]
        # Pull the per-label probabilities.
        scores = item["scores"]
        # Compute the best label by maximum probability.
        best_label_key = max(scores, key=lambda k: scores[k]) if scores else None
        # Use the pretty label mapping for GUI display.
        annotation = REMOTE_LABEL_TO_PRETTY.get(best_label_key) if best_label_key else None
        # The confidence we report is the top probability.
        confidence = float(scores[best_label_key]) if best_label_key else 0.0
        # Count tokens crudely by splitting on whitespace (service does not return token counts).
        token_count = len(line.split())

        # Respect the min_confidence threshold for whether to prefix the line with an annotation.
        if annotation and confidence >= min_confidence:
            annotated_line = f"# {annotation} (confidence: {confidence:.2f})\n{line}"
        else:
            annotation = None
            annotated_line = line

        # Return a structure aligned with the local engine so the GUI does not need to change.
        return {
            "line_num": line_num,
            "line": line,
            "tokens": token_count,
            "annotation": annotation,
            "confidence": confidence,
            "annotated": annotated_line,
        }

    def annotate_file(
        self,
        filepath: str,
        min_confidence: float = 0.0,
        as_dict: bool = False,
        export_path: str = None,
    ):
        # Read all lines from the file.
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Build full source once to feed the rule engine.
        source_text = "".join(lines)
        # Gather reasons per line from AST/regex + external tools (flake8, pydocstyle, pylint, bandit, SonarLint).
        reasons_by_line = collect_reasons(source_text)

        # Identify the indices and texts of lines we actually want to score (non-blank, not comments).
        idx_and_texts = [
            (i, line)
            for i, line in enumerate(lines)
            if line.strip() and not line.lstrip().startswith("#")
        ]
        # Extract the raw texts in order to send as a batch to the service for efficiency.
        batch_texts = [t for _, t in idx_and_texts]

        # Call the service once for all lines in the batch; threshold controls label activation (GUI still uses top score).
        res = (
            self.client.predict(batch_texts or [""], threshold=min_confidence)
            if idx_and_texts
            else {"results": []}
        )
        # Prepare an iterator over the returned results.
        results_iter = iter(res["results"]) if idx_and_texts else iter([])

        # Prepare containers for final outputs.
        annotated = []
        output_lines = []

        # Walk through all original lines, splicing in service results for code lines and passing others through unchanged.
        for i, line in enumerate(lines):
            if idx_and_texts and i == idx_and_texts[0][0]:
                # Pop the head of idx_and_texts when we consume a scored line.
                idx_and_texts.pop(0)
                # Next result corresponds to this line.
                item = next(results_iter)
                # Pull per-label probabilities for this specific line.
                scores = item["scores"]
                # Prepare comment lines from all rule-based reasons (prioritise SAST).
                rule_reasons = reasons_by_line.get(i + 1, [])
                # Sort reasons so SAST-like reasons appear first.
                rule_reasons_sorted = sorted(
                    rule_reasons,
                    key=lambda r: 0 if any(k in r.lower() for k in _SAST_HINTS) else 1,
                )
                # Build one '# …' line per reason, using the model score for that reason's label.
                comment_lines: List[str] = []
                for reason in rule_reasons_sorted:
                    label_key = self._label_for_reason(reason)
                    pretty = REMOTE_LABEL_TO_PRETTY[label_key]
                    conf = float(scores.get(label_key, max(scores.values(), default=0.0)))
                    if conf >= min_confidence:
                        comment_lines.append(f"# {pretty}: {reason} (confidence: {conf:.2f})")
                # If no rule reasons triggered, fall back to a single model-only line.
                if not comment_lines:
                    continue
                # Combine comment lines with the original line (no extra blank lines).
                if comment_lines:
                    annotated_line = "\n".join(comment_lines) + f"\n{line}"
                else:
                    annotated_line = line
                # For the structured record, store the first reason (legacy field) and the top confidence.
                top_conf = 0.0
                if scores:
                    top_conf = float(max(scores.values()))
                first_anno = comment_lines[0] if comment_lines else None
                annotated.append(
                    {
                        "line_num": i + 1,
                        "line": line,
                        "tokens": len(line.split()),
                        "annotation": first_anno,
                        "confidence": top_conf,
                        "annotated": annotated_line,
                    }
                )
                output_lines.append(annotated_line)
            else:
                output_lines.append(line)
                annotated.append(
                    {
                        "line_num": i + 1,
                        "line": line,
                        "tokens": 0,
                        "annotation": None,
                        "confidence": 0.0,
                        "annotated": line,
                    }
                )

        # Optionally export the full annotation set to JSON or CSV for analysis.
        if export_path:
            if export_path.endswith(".json"):
                with open(export_path, "w", encoding="utf-8") as jf:
                    json.dump(annotated, jf, indent=2)
            elif export_path.endswith(".csv"):
                with open(export_path, "w", encoding="utf-8", newline="") as cf:
                    writer = csv.DictWriter(
                        cf,
                        fieldnames=[
                            "line_num",
                            "annotation",
                            "confidence",
                            "tokens",
                            "line",
                        ],
                    )
                    writer.writeheader()
                    for row in annotated:
                        writer.writerow({k: row[k] for k in writer.fieldnames})

        # Return either the structured records or the reconstructed annotated text.
        return annotated if as_dict else "".join(output_lines)


# === Static + AI Metric Integration ===
# The remaining functions are used by the GUI/metrics panels; left intact other than the remote additions above.
def _static_metric_names_stub() -> List[str]:
    return [
        "ast_node_count",
        "ast_function_count",
        "security_warnings",
        "comment_lines",
        "flake8_errors",
        "cyclomatic_complexity",
        "docstring_coverage",
        "pyflakes_warnings",
        "pylint_score",
        "halstead_volume",
        "unused_imports",
    ]


def _map_severity(conf: float) -> str:
    """
    Map confidence scores to severity levels.

    Args:
        conf (float): Confidence value

    Returns:
        str: Severity level ("low", "medium", "high")
    """
    if conf > 0.85:
        return "high"
    if conf > 0.65:
        return "medium"
    return "low"


def compute_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# Name the parameter 'annotation' and persist that value.
def save_annotation_to_cache(file_path: str, dict) -> None:
    file_hash = compute_file_hash(file_path)
    cache_path = Path(AI_CACHE_DIR) / f"{file_hash}.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2)
