from typing import List, Dict
from collections import Counter, defaultdict
import numpy as np

# ✅ Mapping emoji to label types
LABEL_PREFIXES = {"⚠️": "SAST Risk", "🧠": "ML Signal", "✅": "Best Practice"}

# ✅ Standardised severity levels
SEVERITY_LEVELS = ["Low", "Medium", "High"]


def count_annotations_by_type(annotations: List[Dict]) -> Dict[str, int]:
    """
    Counts the number of annotations for each signal type (e.g., SAST Risk).
    """
    counts = Counter()
    for ann in annotations:
        if "type" in ann:
            counts[ann["type"]] += 1
        elif "annotation" in ann:
            label = ann["annotation"].split()[0]
            label_name = LABEL_PREFIXES.get(label)
            if label_name:
                counts[label_name] += 1
    return dict(counts)


def compute_label_distribution(dataset: List[Dict]) -> Dict[str, int]:
    """
    Aggregates label occurrences across the dataset (multi-hot labels).
    """
    label_keys = list(LABEL_PREFIXES.values())
    totals = Counter()
    for sample in dataset:
        for i, val in enumerate(sample.get("labels", [])):
            if val:
                totals[label_keys[i]] += 1
    return dict(totals)


def extract_signals_from_code(code: str) -> List[str]:
    """
    Extracts all signal types from inline-annotated code comments.
    """
    lines = code.splitlines()
    signals = []
    for line in lines:
        for symbol, label in LABEL_PREFIXES.items():
            if line.strip().startswith(f"# {symbol}"):
                signals.append(label)
    return signals


def label_to_score(label: str) -> float:
    """
    Converts a signal label to a numeric score.
    """
    # 🧠 ML Signal: Used for quantifying label importance
    label = label.strip(" #🔍🧠⚠️✅:")
    mapping = {"SAST Risk": 1.0, "ML Signal": 0.7, "Best Practice": 0.5}
    return mapping.get(label, 0.0)


def compute_confidence_weighted_signals(annotations: List[Dict]) -> Dict[str, float]:
    """
    Computes total confidence-weighted score per signal type.
    """
    weights = defaultdict(float)
    for ann in annotations:
        if "type" in ann and "confidence" in ann:
            weights[ann["type"]] += float(ann["confidence"]) * label_to_score(
                ann["type"]
            )
    return dict(weights)


def severity_breakdown_by_type(annotations: List[Dict]) -> Dict[str, Dict[str, int]]:
    """
    Computes nested counts by type and severity.
    Returns:
        { "SAST Risk": {"Low": X, "Medium": Y, "High": Z}, ... }
    """
    nested = {
        label: {s: 0 for s in SEVERITY_LEVELS} for label in LABEL_PREFIXES.values()
    }
    for ann in annotations:
        label = ann.get("type")
        severity = ann.get("severity", "Medium").capitalize()
        if label in nested and severity in SEVERITY_LEVELS:
            nested[label][severity] += 1
    return nested


def prepare_gui_overlay_data(annotations: List[Dict]) -> Dict[str, Dict]:
    """
    Bundles all signal metrics into a structure suitable for live GUI overlays.

    Returns:
        {
            "type_counts": {label: count},
            "weighted_signals": {label: float},
            "severity_by_type": {label: {Low, Medium, High}},
            "total": int
        }
    """
    return {
        "type_counts": count_annotations_by_type(annotations),
        "weighted_signals": compute_confidence_weighted_signals(annotations),
        "severity_by_type": severity_breakdown_by_type(annotations),
        "total": len(annotations),
    }


def compute_summary_stats(annotations: List[Dict]) -> Dict[str, object]:
    """
    Computes aggregate statistics for a list of structured annotations.

    Returns:
        {
            "total_annotations": int,
            "average_confidence": float,
            "label_distribution": dict,
            "weighted_signals": dict,
            "severity_by_type": dict,
            "confidence_percentiles": dict
        }
    """
    if not annotations:
        return {
            "total_annotations": 0,
            "average_confidence": 0.0,
            "label_distribution": {},
            "weighted_signals": {},
            "severity_by_type": {},
            "confidence_percentiles": {},
        }

    confidences = [
        float(a.get("confidence", 0)) for a in annotations if "confidence" in a
    ]

    return {
        "total_annotations": len(annotations),
        "average_confidence": (
            sum(confidences) / len(confidences) if confidences else 0.0
        ),
        "label_distribution": count_annotations_by_type(annotations),
        "weighted_signals": compute_confidence_weighted_signals(annotations),
        "severity_by_type": severity_breakdown_by_type(annotations),
        "confidence_percentiles": (
            {
                "min": float(np.percentile(confidences, 0)),
                "25%": float(np.percentile(confidences, 25)),
                "50%": float(np.percentile(confidences, 50)),
                "75%": float(np.percentile(confidences, 75)),
                "max": float(np.percentile(confidences, 100)),
            }
            if confidences
            else {}
        ),
    }
