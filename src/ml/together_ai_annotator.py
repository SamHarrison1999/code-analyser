# File: together_ai_annotator.py

import os
import re
import json
import time
import ast
import csv
import logging
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv
from transformers import AutoTokenizer
import torch
from datasets import Dataset

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOGETHER_API_KEYS = [k.strip() for k in os.getenv("TOGETHER_API_KEYS", "").split(",") if k.strip()]
TOGETHER_API_URL = "https://api.together.xyz/v1/completions"
DEFAULT_MODEL = "meta-llama/Llama-2-70b-hf"
MAX_TOKENS = 1024

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

INSTRUCTION = """Annotate the following Python code with inline comments. Use only these formats:
- ML Signal: "# 🧠 ML Signal: <reason>"
- SAST Risk: "# ⚠️ SAST Risk (High|Medium|Low): <reason>"
- Best Practice: "# ✅ Best Practice: <reason>"

If 'open()' is used, recommend a context manager.
Avoid hallucinated examples or unrelated code blocks.

Code:
"""

def annotate_code_with_together_ai(code: str, output_dir: Path = None, filename: str = "annotated") -> Tuple[str, float, List[Dict]]:
    code = strip_inline_comments(code)
    code_lines = code.splitlines()

    if not any(line.strip() and not line.strip().startswith("#") for line in code_lines):
        logger.warning(f"⚠️ Skipping file '{filename}' — empty or comment-only")
        return code, 1.0, []

    if len(code_lines) < 4:
        code_lines += [""] * (4 - len(code_lines))
    code = "\n".join(code_lines)

    chunks = split_code_by_function_ast(code)
    all_annotations, confidences = [], []

    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        logger.info(f"✨ Annotating chunk {idx + 1}/{len(chunks)} ({len(chunk.splitlines())} lines)")
        annotated_text, confidence, logprobs = annotate_chunk_with_retry(chunk)

        if "--- BEGIN ANNOTATED PYTHON CODE ---" not in annotated_text:
            logger.warning("⚠️ Skipped hallucinated markdown or quiz output.")
            continue

        annotations = parse_annotations_from_blocks(annotated_text, chunk)
        all_annotations.extend(annotations)
        confidences.append(confidence)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{filename}.heatmap.json").write_text(json.dumps(logprobs, indent=2))

    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 1.0

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        supervised = extend_supervised_annotations(all_annotations, code_lines)
        (output_dir / f"{filename}.annotations.json").write_text(json.dumps(all_annotations, indent=2))
        (output_dir / f"{filename}.supervised.json").write_text(json.dumps(supervised, indent=2))
        (output_dir / f"{filename}.source.py").write_text(code)
        (output_dir / f"{filename}.annotated.py").write_text("\n".join(insert_annotations_into_code(code_lines, all_annotations)))
        export_csv_overlay(all_annotations, output_dir / f"{filename}.overlay.csv")
        export_hf_dataset(supervised, output_dir / f"{filename}.arrow")

    return code, avg_conf, all_annotations

def annotate_chunk_with_retry(chunk: str, retries: int = 3) -> Tuple[str, float, List[Dict]]:
    prompt = INSTRUCTION + chunk

    for attempt in range(1, retries + 2):
        for key in TOGETHER_API_KEYS:
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {
                "model": DEFAULT_MODEL,
                "prompt": prompt,
                "temperature": 0.2,
                "max_tokens": MAX_TOKENS,
                "logprobs": 5,
                "stream": False,
            }
            try:
                resp = requests.post(TOGETHER_API_URL, headers=headers, json=payload, timeout=60)
                if resp.status_code == 429:
                    logger.warning(f"🔁 [Attempt {attempt}] 429 Too Many Requests — key: {key}")
                    time.sleep(5)
                    continue
                resp.raise_for_status()
                result = resp.json()
                text = result["choices"][0]["text"]
                if not text.strip():
                    logger.warning(f"⚠️ Empty response text received from Together.ai (Attempt {attempt})")
                    continue
                logger.info(f"📥 Raw model output:\n{text}")
                probs = result["choices"][0].get("logprobs", {}).get("token_logprobs", [])
                conf = round(float(torch.exp(torch.tensor(probs)).mean()), 4) if probs else 0.8
                return text, conf, probs
            except Exception as e:
                logger.warning(f"⚠️ Error during annotation (Attempt {attempt}): {e}")
                time.sleep(3 * attempt)

    return "", 0.0, []

def parse_annotations_from_blocks(raw_text: str, code: str) -> List[Dict]:
    lines = code.splitlines()
    annotations = []
    current_code_block = []
    current_annotation_block = []

    # Parse line-by-line chunks of annotated code
    for line in raw_text.splitlines():
        if line.strip().startswith("# 🧠") or line.strip().startswith("# ⚠️") or line.strip().startswith("# ✅"):
            current_annotation_block.append(line.strip())
        elif line.strip().startswith("--- BEGIN PYTHON CODE ---"):
            current_code_block = []
        elif line.strip().startswith("--- END PYTHON CODE ---"):
            pass
        elif line.strip().startswith("--- BEGIN ANNOTATED PYTHON CODE ---"):
            current_annotation_block = []
        elif line.strip().startswith("--- END ANNOTATED PYTHON CODE ---"):
            # Try to match annotations with original lines
            for idx, line in enumerate(lines):
                for ann in current_annotation_block:
                    if ("eval(" in lines[idx] and "eval" in ann) or \
                       ("open(" in lines[idx] and "open" in ann):
                        severity = "high" if "⚠️" in ann else "low"
                        annotations.append({
                            "line": idx + 1,
                            "text": ann,
                            "type": "sast_risk" if "⚠️" in ann else ("ml_signal" if "🧠" in ann else "best_practice"),
                            "severity": severity,
                            "confidence": 0.9
                        })
        else:
            current_code_block.append(line.strip())

    return annotations

def insert_annotations_into_code(original_lines: List[str], annotations: List[Dict]) -> List[str]:
    annotations = sorted(annotations, key=lambda x: x["line"])
    output, offset = original_lines[:], 0
    for ann in annotations:
        insert_at = ann["line"] + offset - 1
        indent = re.match(r"^(\s*)", output[insert_at]).group(1)
        output.insert(insert_at, f"{indent}{ann['text']}")
        offset += 1
    return output

def export_csv_overlay(annotations: List[Dict], path: Path):
    if not annotations:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["line", "type", "reason", "severity", "confidence"])
        writer.writeheader()
        for ann in annotations:
            writer.writerow({
                "line": ann["line"],
                "type": ann["type"],
                "reason": ann.get("text", ""),
                "severity": ann.get("severity", ""),
                "confidence": ann.get("confidence", 1.0)
            })

def export_hf_dataset(annotations: List[Dict], output_path: Path):
    if not annotations:
        return
    ds = Dataset.from_list(annotations)
    ds.save_to_disk(str(output_path))
    logger.info(f"✅ Saved HuggingFace dataset to: {output_path}")

def extend_supervised_annotations(annotations: List[Dict], lines: List[str]) -> List[Dict]:
    result = []
    for ann in annotations:
        entry = {
            "line": ann["line"],
            "label": ann["type"],
            "text": lines[ann["line"] - 1],
            "annotation": ann["text"],
            "confidence": ann.get("confidence", 1.0),
            "severity": ann.get("severity", ""),
            "tokens": tokenizer.encode(lines[ann["line"] - 1])
        }
        result.append(entry)
    return result

def split_code_by_function_ast(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [code]
    lines = code.splitlines()
    func_starts = sorted({n.lineno - 1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))})
    if not func_starts or func_starts[0] != 0:
        func_starts = [0] + func_starts
    chunks, current, count = [], [], 0
    for i, start in enumerate(func_starts):
        end = func_starts[i + 1] if i + 1 < len(func_starts) else len(lines)
        segment = lines[start:end]
        tokens = sum(len(tokenizer.encode(line)) for line in segment)
        if count + tokens > MAX_TOKENS and current:
            chunks.append("\n".join(current))
            current = segment
            count = tokens
        else:
            current += segment
            count += tokens
    if current:
        chunks.append("\n".join(current))
    return chunks

def strip_inline_comments(code: str) -> str:
    lines = code.splitlines()
    stripped = []
    for line in lines:
        code_only = re.split(r'\s+#', line, maxsplit=1)[0].rstrip()
        if code_only:
            stripped.append(code_only)
    return "\n".join(stripped)
