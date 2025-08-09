# src/ml/overlay_export.py

import csv
from pathlib import Path
from typing import List, Dict


def write_overlay_csv(spans: List[Dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["line", "type", "severity", "reason", "confidence"])
        writer.writeheader()
        for span in spans:
            writer.writerow(
                {
                    "line": span.get("line", ""),
                    "type": span.get("type", ""),
                    "severity": span.get("severity", ""),
                    "reason": span.get("reason", ""),
                    "confidence": span.get("confidence", ""),
                }
            )
