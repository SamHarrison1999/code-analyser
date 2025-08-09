"""
inference.py — Supervised AI annotation engine with overlay metric extraction.

Includes:
- AnnotationEngine: loads model + tokeniser and annotates code lines
- gather_ai_metric_names(): returns AI-enhanced metric keys
- gather_all_metric_names_with_ai(): static + AI metric names
- Uses overlay_utils.gather_ai_overlays(): confidence-annotated token heatmap overlays
"""

import torch
import json
import csv
from pathlib import Path
from transformers import AutoTokenizer
from typing import List

from .model_tf import AnnotationClassifier
from .config import CHECKPOINT_DIR, AI_CACHE_DIR
from ml.ai_metric_names import gather_ai_metric_names
import hashlib


# ✅ Label mapping for annotation types
LABELS = {0: "⚠️ SAST Risk", 1: "🧠 ML Signal", 2: "✅ Best Practice"}

DEFAULT_MODEL_PATH = f"{CHECKPOINT_DIR}/supervised_epoch_3.pt"
DEFAULT_MODEL_NAME = "microsoft/codebert-base"
DISTILLED_MODEL_NAME = "Salesforce/codet5-small"


# === 🧠 Supervised Annotation Engine ===
class AnnotationEngine:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
        use_distilled: bool = False,
        dropout: float = 0.1,
    ):
        self.model_name = DISTILLED_MODEL_NAME if use_distilled else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AnnotationClassifier(model_name=self.model_name, dropout=dropout)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.model.eval()

    def annotate_line(self, line: str, min_confidence: float = 0.0, line_num: int = None) -> dict:
        inputs = self.tokenizer(
            line, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        token_count = len(inputs["input_ids"][0])
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()
            confidence = torch.softmax(logits, dim=-1).max().item()

        if confidence >= min_confidence:
            annotation = LABELS[pred]
            annotated_line = f"# {annotation} (confidence: {confidence:.2f})\n{line}"
        else:
            annotation = None
            annotated_line = line

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
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        annotated = []
        output_lines = []

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

        return annotated if as_dict else "".join(output_lines)


# === Static + AI Metric Integration ===


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


def gather_all_metric_names_with_ai() -> List[str]:
    """
    Return all metric names including static and AI-derived overlays.

    Returns:
        list[str]: Unified metric name list.
    """
    return _static_metric_names_stub() + gather_ai_metric_names()


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


def save_annotation_to_cache(file_path: str, annotation: dict) -> None:
    file_hash = compute_file_hash(file_path)
    cache_path = Path(AI_CACHE_DIR) / f"{file_hash}.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2)
