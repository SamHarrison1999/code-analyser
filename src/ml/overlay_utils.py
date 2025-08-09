# File: src/ml/overlay_utils.py

import csv
import json
from pathlib import Path
from typing import List, Dict, Any


def save_overlay_csv(annotations: List[Dict[str, Any]], csv_path: Path) -> None:
    """
    Save structured annotation data to CSV format.

    Args:
        annotations (List[Dict]): List of AI overlay annotation dictionaries.
        csv_path (Path): Path to save the CSV file.
    """
    if not annotations:
        return

    # 🔧 Automatically determine all field names, including "text"
    fieldnames = sorted(set().union(*(a.keys() for a in annotations)))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotations)


def save_overlay_heatmap(heatmap: List[Dict[str, Any]], heatmap_path: Path) -> None:
    """
    Save token-level confidence/severity overlay heatmap to a .json file.

    Args:
        heatmap (List[Dict]): List of heatmap entries with token metadata.
        heatmap_path (Path): Path to save the .json file.
    """
    if not heatmap:
        return

    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    with open(heatmap_path, "w", encoding="utf-8") as f:
        json.dump(heatmap, f, indent=2, ensure_ascii=False)