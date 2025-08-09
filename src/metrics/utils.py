"""
utils.py — Shared utilities for metric processing and annotation overlays.

Includes:
- Path helpers
- Annotation parsing
- Confidence/severity utilities
"""

import json
from pathlib import Path
from typing import List, Dict, Union
from metrics.metric_types import AIAnnotationOverlay


def resolve_cache_path(base: Union[str, Path], filename: str) -> Path:
    """
    Resolves a path inside the AI cache directory.

    Args:
        base (str | Path): Base cache dir (e.g., .ai_cache/)
        filename (str): Target filename (e.g., annotations.json)

    Returns:
        Path: Resolved full path
    """
    return Path(base).expanduser().resolve() / filename


def load_annotations_from_file(file_path: Union[str, Path]) -> List[AIAnnotationOverlay]:
    """
    Load annotation overlays from a .json file.

    Args:
        file_path (str | Path): Path to annotation file

    Returns:
        List[AIAnnotationOverlay]: Parsed list of overlays
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list of overlays in {path}, got {type(data)}")
        return data
    except Exception as e:
        print(f"[utils] Failed to load annotations from {file_path}: {e}")
        return []


def filter_overlays_by_severity(
    overlays: List[AIAnnotationOverlay], severity: str
) -> List[AIAnnotationOverlay]:
    """
    Filters overlays to only include a specific severity.

    Args:
        overlays (List[AIAnnotationOverlay]): Full overlay list
        severity (str): One of ['low', 'medium', 'high']

    Returns:
        List[AIAnnotationOverlay]: Filtered list
    """
    return [o for o in overlays if o.get("severity") == severity]


def filter_overlays_by_scope(
    overlays: List[AIAnnotationOverlay], scope: str
) -> List[AIAnnotationOverlay]:
    """
    Filters overlays by their plugin/scope identifier.

    Args:
        overlays (List[AIAnnotationOverlay]): Full overlay list
        scope (str): Scope string (e.g., 'together_ai', 'rl_agent')

    Returns:
        List[AIAnnotationOverlay]: Filtered overlays
    """
    return [o for o in overlays if o.get("scope") == scope]


def get_overlay_stats(
    overlays: List[AIAnnotationOverlay],
) -> Dict[str, Union[int, float]]:
    """
    Compute summary statistics for overlays.

    Args:
        overlays (List[AIAnnotationOverlay]): Input overlays

    Returns:
        Dict[str, Union[int, float]]: Stats including count, avg confidence, high-risk count
    """
    if not overlays:
        return {"total": 0, "high_risk": 0, "avg_confidence": 0.0}

    total = len(overlays)
    high_risk = sum(1 for o in overlays if o.get("severity") == "high")
    avg_conf = sum(o.get("confidence", 0.0) for o in overlays) / total
    return {
        "total": total,
        "high_risk": high_risk,
        "avg_confidence": round(avg_conf, 3),
    }
