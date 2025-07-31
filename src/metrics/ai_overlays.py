# File: code_analyser/src/metrics/ai_overlays.py

"""
AI Overlay Metric Integration

This module provides:
- Aggregation of AI-generated annotations into confidence/severity overlays
- Token-level heatmap bundle extraction
- Support for severity filtering and scope-based annotation grouping
- Exportable structures for GUI overlays, CSV/JSON output, and TensorBoard
"""

import logging
from pathlib import Path
from typing import List, Dict, Union, Literal, Optional, Tuple
from metrics.metric_types import (
    AIAnnotationOverlay,
    AISummaryBundle,
    TokenExplanationMap,
)
from metrics.utils import load_annotations_from_file

logger = logging.getLogger(__name__)

SeverityLevel = Literal["low", "medium", "high"]

# 🧠 ML Signal: Confidence-weighted overlays enable risk-aware visualisation
# ✅ Best Practice: Token span heatmaps allow precise GUI feedback


def gather_ai_metric_overlays(
    file_path: Union[str, Path],
    annotation_file: Optional[Union[str, Path]] = None,
    severity_filter: Optional[List[SeverityLevel]] = None,
    scope_filter: Optional[str] = None,
) -> Tuple[List[AIAnnotationOverlay], AISummaryBundle]:
    """
    Load AI overlays and filter by severity/scope.

    Returns:
        - List[AIAnnotationOverlay]: Overlays for GUI/token view
        - AISummaryBundle: Summary metrics for dashboard
    """
    file_path = Path(file_path)
    annotation_file = annotation_file or _default_annotation_path(file_path)

    try:
        annotations = load_annotations_from_file(annotation_file)
    except Exception as e:
        logger.warning(f"[AIOverlay] Failed to load annotations for {file_path}: {e}")
        return [], {
            "file": str(file_path),
            "avg_confidence": 0.0,
            "high_risk_count": 0,
            "total_annotations": 0,
            "timestamp": "",
        }

    filtered: List[AIAnnotationOverlay] = []
    confidences: List[float] = []
    high_risk_count = 0

    for ann in annotations:
        if severity_filter and ann.get("severity") not in severity_filter:
            continue
        if scope_filter and ann.get("scope") != scope_filter:
            continue

        filtered.append(ann)
        confidence = ann.get("confidence", 0.0)
        confidences.append(confidence)
        if ann.get("severity") == "high":
            high_risk_count += 1

    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    summary: AISummaryBundle = {
        "file": str(file_path),
        "avg_confidence": avg_conf,
        "high_risk_count": high_risk_count,
        "total_annotations": len(filtered),
        "timestamp": annotations[0].get("timestamp", "") if annotations else "",
    }

    return filtered, summary


def _default_annotation_path(file_path: Union[str, Path]) -> Path:
    return Path(".ai_cache") / Path(file_path).with_suffix(".ann.json").name


def extract_token_heatmap(
    overlays: List[AIAnnotationOverlay],
) -> Dict[int, List[TokenExplanationMap]]:
    """
    Convert overlays to heatmap-style token maps per line.
    """
    heatmap: Dict[int, List[TokenExplanationMap]] = {}

    for overlay in overlays:
        line = overlay.get("line")
        token = overlay.get("token_span", None)
        label = overlay.get("label", "")
        confidence = overlay.get("confidence", 0.0)
        severity = overlay.get("severity", "low")

        if not isinstance(line, int) or token is None:
            continue

        heatmap.setdefault(line, []).append(
            {
                "line": line,
                "token": label,
                "confidence": confidence,
                "severity": severity,
            }
        )

    return heatmap


def get_ai_metric_names(
    file_path: Union[str, Path], annotation_file: Optional[Union[str, Path]] = None
) -> List[str]:
    """
    Returns the unique metric labels extracted from the annotation overlays
    of the given file.

    Returns:
        List[str]: Sorted list of annotation labels.
    """
    file_path = Path(file_path)
    annotation_file = annotation_file or _default_annotation_path(file_path)

    try:
        annotations = load_annotations_from_file(annotation_file)
    except Exception as e:
        logger.warning(f"[get_ai_metric_names] Failed to load annotation file: {e}")
        return []

    labels = sorted({ann.get("label") for ann in annotations if ann.get("label")})
    return labels


def gather_all_metric_names_with_ai(
    file_path: Union[str, Path], annotation_file: Optional[Union[str, Path]] = None
) -> List[str]:
    """
    Combines static metric names with unique AI annotation labels.

    Args:
        file_path (str): Target source file.
        annotation_file (str | Path | None): Custom annotation path (optional).

    Returns:
        List[str]: Combined static + AI metric names.
    """
    from metrics.gather import (
        get_all_metric_names,
    )  # avoid circular import at top level

    static = get_all_metric_names()
    ai_labels = get_ai_metric_names(file_path, annotation_file)
    return static + ai_labels
