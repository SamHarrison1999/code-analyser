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
from torch.utils.tensorboard import SummaryWriter

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

Do not annotate lines that do not contain a clear risk, signal, or best practice issue.
Avoid placeholder comments or hallucinations like 'use a context manager' unless there is actual file I/O.

Code:
"""

TENSORBOARD_LOGDIR = "runs/annotation_logs"
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)

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
        chunk_clean = "\n".join(line.strip() for line in chunk.splitlines() if line.strip())
        if not chunk_clean:
            continue
        logger.info(f"✨ Annotating chunk {idx + 1}/{len(chunks)} ({len(chunk_clean.splitlines())} lines)")
        annotated_text, confidence, logprobs = annotate_chunk_with_retry(chunk_clean)
        if not annotated_text:
            logger.warning(f"⚠️ Empty response for chunk {idx + 1}")
            continue
        annotations = parse_annotations_from_response(annotated_text, code)
        for ann in annotations:
            ann["label"] = ann["type"]
            ann["confidence"] = confidence
            ann["tokens"] = tokenizer.tokenize(code_lines[ann["start_line"] - 1])
            ann["start_token"] = 0
            ann["end_token"] = len(ann["tokens"])
            ann["reason"] = ann.get("text", "")
        all_annotations.extend(annotations)
        confidences.append(confidence)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{filename}.heatmap.json").write_text(json.dumps(logprobs, indent=2))

    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 1.0

    # ✅ TensorBoard logging
    writer.add_scalar("confidence/average", avg_conf, global_step=0)
    writer.add_scalar("annotations/count", len(all_annotations), global_step=0)

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
    prompt = INSTRUCTION + chunk
    fallback_prompt = "# 🧠 ML Signal: ...\n# ⚠️ SAST Risk (High|Medium|Low): ...\n# ✅ Best Practice: ...\n\n" + chunk

    for attempt in range(1, retries + 2):
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
                logger.info(f"📥 Raw model output:\n{text}")
                probs = result["choices"][0].get("logprobs", {}).get("token_logprobs", [])
                conf = round(float(torch.exp(torch.tensor(probs)).mean()), 4) if probs else 0.8
                return text, conf, probs
            except Exception as e:
                logger.warning(f"⚠️ Error during annotation (Attempt {attempt}): {e}")
                time.sleep(3 * attempt)

    return "", 0.0, []

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

def parse_annotations_from_response(llm_response: str, original_code: str) -> List[dict]:
    code_lines = original_code.splitlines()
    parsed_annotations = []
    pending_block = {"ml": None, "sast": None, "best": None}
    annotation_blocks = []

    for line in llm_response.splitlines():
        line = line.strip()
        if line.startswith("# 🧠 ML Signal:"):
            pending_block["ml"] = line
        elif line.startswith("# ⚠️ SAST Risk"):
            pending_block["sast"] = line
        elif line.startswith("# ✅ Best Practice:"):
            pending_block["best"] = line
        elif line == "" and any(pending_block.values()):
            annotation_blocks.append(pending_block.copy())
            pending_block = {"ml": None, "sast": None, "best": None}

    match_targets = ["eval(", "exec(", "pickle.loads", "os.system", "subprocess", "input(", "__import__"]
    assigned_lines = set()
    for block in annotation_blocks:
        for idx, code_line in enumerate(code_lines):
            if idx in assigned_lines:
                continue
            if any(target in code_line for target in match_targets):
                ann = {"start_line": idx + 1, "end_line": idx + 1, "span_lines": 1}
                if block["ml"]:
                    parsed_annotations.append({**ann, "type": "ml_signal", "text": block["ml"]})
                if block["sast"]:
                    parsed_annotations.append({**ann, "type": "sast_risk", "text": block["sast"]})
                if block["best"]:
                    parsed_annotations.append({**ann, "type": "best_practice", "text": block["best"]})
                assigned_lines.add(idx)
                break

    return parsed_annotations

def insert_annotations_into_code(original_lines: List[str], annotations: List[Dict]) -> List[str]:
    annotations = sorted(annotations, key=lambda x: x["start_line"])
    output, offset = original_lines[:], 0
    for ann in annotations:
        insert_at = ann["start_line"] + offset - 1
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
        writer = csv.DictWriter(f, fieldnames=["start_line", "end_line", "span_lines", "type", "text"])
        writer.writeheader()
        for ann in annotations:
            writer.writerow({
                "start_line": ann.get("start_line", ann.get("line", "")),
                "end_line": ann.get("end_line", ann.get("line", "")),
                "span_lines": ann.get("span_lines", 1),
                "type": ann["type"],
                "text": ann.get("text", "")
            })

def export_hf_dataset(annotations: List[Dict], output_path: Path):
    if not annotations:
        return
    ds = Dataset.from_list(annotations)
    ds.save_to_disk(str(output_path))
    logger.info(f"✅ Saved HuggingFace dataset to: {output_path}")

def strip_inline_comments(code: str) -> str:
    lines = code.splitlines()
    stripped = []
    for line in lines:
        code_only = re.split(r'\s+#', line, maxsplit=1)[0].rstrip()
        if code_only:
            stripped.append(code_only)
    return "\n".join(stripped)
