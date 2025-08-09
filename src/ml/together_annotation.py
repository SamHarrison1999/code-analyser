# ✅ Best Practice: Extract type, severity, and context from inline Together.ai annotations
# 🧠 ML Signal: Enables supervised learning from parsed natural annotations

import os
import re
import json
from typing import List, Dict


LABEL_MAP = {"⚠️": "SAST Risk", "🧠": "ML Signal", "✅": "Best Practice"}


SEVERITY_KEYWORDS = {
    "critical": "High",
    "dangerous": "High",
    "severe": "High",
    "unsafe": "High",
    "risky": "Medium",
    "moderate": "Medium",
    "costly": "Medium",
    "slow": "Low",
    "minor": "Low",
    "style": "Low",
    "inefficient": "Low",
}


def infer_severity_from_text(text: str) -> str:
    """Heuristic mapping of natural text to a severity label."""
    text_lower = text.lower()
    for keyword, level in SEVERITY_KEYWORDS.items():
        if keyword in text_lower:
            return level
    return "Medium"  # default fallback


def parse_annotated_code(raw_code: str) -> List[Dict]:
    """
    Parses inline-annotated Python code into structured Together.ai annotation format.
    """
    lines = raw_code.splitlines()
    annotations = []
    pending = None

    pattern = re.compile(
        r"#\s*(⚠️|🧠|✅)\s*(SAST Risk|ML Signal|Best Practice)?(?:\s*\((.*?)\))?:?\s*(.*)"
    )

    for idx, line in enumerate(lines):
        match = pattern.match(line.strip())
        if match:
            symbol, category, severity, content = match.groups()
            annotation_type = LABEL_MAP.get(symbol, category)
            content = content.strip()
            severity = severity or infer_severity_from_text(content)

            pending = {
                "line": idx + 2,  # annotation is assumed to be directly above target line
                "type": annotation_type,
                "severity": severity,
                "content": content,
                "confidence": 1.0,
            }
        elif pending:
            annotations.append(pending)
            pending = None

    return annotations


def generate_annotation_json(raw_code: str, filename: str = "") -> Dict:
    """
    Creates a JSON-serialisable annotation object for saving.
    """
    return {"filename": filename, "annotations": parse_annotated_code(raw_code)}


def save_annotation_json(
    code: str, source_path: str, output_dir: str = "datasets/annotated_fintech"
) -> str:
    """
    Parses and saves annotations to a .json file matching source filename.

    Returns:
        str: Path to saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(source_path)
    json_filename = f"{basename}.json"
    save_path = os.path.join(output_dir, json_filename)

    data = generate_annotation_json(code, filename=basename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return save_path
