# together_ai_annotator.py

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

INSTRUCTION = """You are a static analysis assistant trained to detect security issues, best practices, and ML signals in Python code.

Annotate the following Python code using ONLY these comment formats:
- "# 🧠 ML Signal: <reason>"
- "# ⚠️ SAST Risk (High|Medium|Low): <reason>"
- "# ✅ Best Practice: <reason>"

Do NOT use markdown, bullet points, questions, or summaries.
Do NOT return anything except annotated Python code.

--- BEGIN PYTHON CODE ---
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
        chunk_lines = chunk.splitlines()
        if len(chunk_lines) < 5:
            chunk_lines += [""] * (5 - len(chunk_lines))  # ✅ Pad short chunks
        chunk_clean = "\n".join(line.strip() for line in chunk_lines if line.strip())
        if not chunk_clean:
            continue
        logger.info(f"✨ Annotating chunk {idx + 1}/{len(chunks)} ({len(chunk_clean.splitlines())} lines)")
        logger.debug(f"📤 Final chunk sent to model:\n{chunk_clean}")
        annotated_text, confidence, logprobs = annotate_chunk_with_retry(chunk_clean)
        if not annotated_text or "# 🧠" not in annotated_text and "# ⚠️" not in annotated_text and "# ✅" not in annotated_text:
            logger.warning(f"⚠️ Skipped hallucinated markdown or quiz output.")
            continue
        annotations = parse_annotations_from_response(annotated_text, code)
        all_annotations.extend(annotations)
        confidences.append(confidence)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{filename}.heatmap.json").write_text(json.dumps(logprobs, indent=2))

    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 1.0

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{filename}.annotations.json").write_text(json.dumps(all_annotations, indent=2))
        (output_dir / f"{filename}.supervised.json").write_text(json.dumps(all_annotations, indent=2))
        (output_dir / f"{filename}.source.py").write_text(code)
        (output_dir / f"{filename}.annotated.py").write_text("\n".join(insert_annotations_into_code(code_lines, all_annotations)))
        export_csv_overlay(all_annotations, output_dir / f"{filename}.overlay.csv")
        export_hf_dataset(all_annotations, output_dir / f"{filename}.arrow")

    return code, avg_conf, all_annotations

def annotate_chunk_with_retry(chunk: str, retries: int = 3) -> Tuple[str, float, List[Dict]]:
    prompt = INSTRUCTION + chunk + "\n--- END PYTHON CODE ---"
    fallback_prompt = "# 🧠 ML Signal: ...\n# ⚠️ SAST Risk (High|Medium|Low): ...\n# ✅ Best Practice: ...\n\n" + chunk

    for attempt in range(1, retries + 2):  # One fallback
        use_fallback = (attempt == retries + 1)
        current_prompt = fallback_prompt if use_fallback else prompt

        for key in TOGETHER_API_KEYS:
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {
                "model": DEFAULT_MODEL,
                "prompt": current_prompt,
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


def parse_annotations_from_response(llm_response: str, original_code: str) -> List[Dict]:
    parsed = []
    lines = original_code.splitlines()
    for i, line in enumerate(lines):
        block = []
        if "# 🧠 ML Signal:" in line:
            block.append(("ml_signal", line.strip()))
        if "# ⚠️ SAST Risk" in line:
            block.append(("sast_risk", line.strip()))
        if "# ✅ Best Practice:" in line:
            block.append(("best_practice", line.strip()))
        for label, text in block:
            parsed.append({
                "line": i + 1,
                "span": [i + 1, i + 1],
                "label": label,
                "text": text,
                "confidence": 1.0,
                "severity": "High" if "High" in text else ("Medium" if "Medium" in text else ("Low" if "Low" in text else "")),
            })
    return parsed

def insert_annotations_into_code(original_lines: List[str], annotations: List[Dict]) -> List[str]:
    annotations = sorted(annotations, key=lambda x: x["line"])
    output, offset = original_lines[:], 0
    for ann in annotations:
        insert_at = ann["line"] + offset - 1
        indent = ""
        for i in range(insert_at, len(output)):
            line = output[i]
            if line.strip() and not line.strip().startswith("#"):
                indent = line[: len(line) - len(line.lstrip())]
                break
        output.insert(insert_at, f"{indent}{ann['text']}")
        offset += 1
    return output

def export_csv_overlay(annotations: List[Dict], path: Path):
    if not annotations:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["line", "label", "reason", "severity"])
        writer.writeheader()
        for ann in annotations:
            writer.writerow({
                "line": ann["line"],
                "label": ann["label"],
                "reason": ann["text"],
                "severity": ann.get("severity", "")
            })

def export_hf_dataset(annotations: List[Dict], output_path: Path):
    if not annotations:
        return
    ds = Dataset.from_list(annotations)
    ds.save_to_disk(str(output_path))
    logger.info(f"✅ Saved HuggingFace dataset to: {output_path}")

def export_supervised_json(annotations: List[Dict], path: Path, avg_conf: float):
    for ann in annotations:
        ann["confidence"] = round(avg_conf, 4)
        ann["label"] = ann["label"]
        ann["tokens"] = tokenizer.tokenize(ann["text"])
    path.write_text(json.dumps(annotations, indent=2))
    logger.info(f"✅ Saved .supervised.json to: {path}")

def strip_inline_comments(code: str) -> str:
    lines = code.splitlines()
    stripped = []
    for line in lines:
        code_only = re.split(r'\s+#', line, maxsplit=1)[0].rstrip()
        stripped.append(code_only)
    return "\n".join(stripped)

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
