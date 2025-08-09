# File: src/ml/overlay_gatherer.py
from pathlib import Path
from typing import Dict, List, Union, Tuple
from metrics.metric_types import AIAnnotationOverlay
from ml.ai_cache import load_cached_annotation


def gather_ai_overlays(
    file_path: Union[str, Path], min_confidence: float = 0.5
) -> Tuple[Dict[str, Union[int, float]], List[AIAnnotationOverlay]]:
    data = load_cached_annotation(file_path)
    if not data:
        return {}, []

    overlays = data.get("overlays", [])
    filtered = [
        o
        for o in overlays
        if isinstance(o, dict) and o.get("confidence", 0) >= min_confidence
    ]
    grouped: Dict[str, List[float]] = {}
    for overlay in filtered:
        label = overlay.get("type", "").lower()
        conf = overlay.get("confidence", 0)
        grouped.setdefault(label, []).append(conf)

    result = {}
    for label, values in grouped.items():
        result[f"{label}_count"] = len(values)
        result[f"{label}_avg_conf"] = round(sum(values) / len(values), 4)

    result["ai_overlay_total"] = len(filtered)

    overlay_objs = [
        AIAnnotationOverlay(**o) for o in filtered if "line" in o and "type" in o
    ]

    return result, overlay_objs