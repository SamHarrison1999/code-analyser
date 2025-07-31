# together_ai_annotator.py

import os
import re
import json
import time
import ast
import logging
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import datasets

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOGETHER_API_KEYS = [
    k.strip() for k in os.getenv("TOGETHER_API_KEYS", "").split(",") if k.strip()
]
TOGETHER_API_URL = "https://api.together.xyz/v1/completions"
DEFAULT_MODEL = "meta-llama/Llama-2-70b-hf"
MAX_TOKENS = 1024

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
model.eval()

PROMPT_TEMPLATE = """
Annotate the following Python code with inline comments. Use only these formats:
- ML Signal: "# 🧠 ML Signal: <reason>"
- SAST Risk: "# ⚠️ SAST Risk (High|Medium|Low): <reason>"
- Best Practice: "# ✅ Best Practice: <reason>"

Do not annotate lines that do not contain a clear risk, signal, or best practice issue.
Avoid placeholder comments or hallucinations like 'use a context manager' unless there is actual file I/O.

🌟 Examples:
# ⚠️ SAST Risk (High): This line evaluates user input without sanitisation.
eval(user_input)

# ✅ Best Practice: Use list comprehensions for performance and readability.
result = [x for x in values if x > 0]

# 🧠 ML Signal: This model training call is critical for learning behaviours.
model.fit(X_train, y_train)

Code:
{code}
"""


def annotate_code_with_together_ai(
    code: str, output_dir: Path = None, filename: str = "annotated"
) -> Tuple[str, float, List[Dict]]:
    code_lines = code.splitlines()
    if not any(
        line.strip() and not line.strip().startswith("#") for line in code_lines
    ):
        logger.warning(f"⚠️ Skipping file '{filename}' — empty or comment-only")
        return code, 1.0, []

    chunks = split_code_by_function_ast(code)
    all_annotations, confidences = [], []
    total_offset = 0

    for idx, chunk in enumerate(chunks):
        logger.info(
            f"✨ Annotating chunk {idx + 1}/{len(chunks)} ({len(chunk.splitlines())} lines)"
        )
        annotated_text, confidence, logprobs = annotate_chunk_with_retry(chunk)
        annotations = parse_annotations_from_response(annotated_text, total_offset)
        all_annotations.extend(annotations)
        confidences.append(confidence)
        total_offset += len(chunk.splitlines())

        # Optional token logprob heatmap export
        if output_dir:
            heatmap_path = output_dir / f"{filename}.heatmap.json"
            heatmap_path.write_text(json.dumps(logprobs, indent=2))

    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 1.0

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{filename}.annotations.json").write_text(
            json.dumps(all_annotations, indent=2)
        )
        (output_dir / f"{filename}.source.py").write_text(code)
        (output_dir / f"{filename}.supervised.json").write_text(
            json.dumps(all_annotations, indent=2)
        )
        output_path = output_dir / f"{filename}.annotated.py"
        output_path.write_text(
            "\\n".join(insert_annotations_into_code(code_lines, all_annotations))
        )

    return code, avg_conf, all_annotations


def annotate_chunk_with_retry(
    chunk: str, retries: int = 3
) -> Tuple[str, float, List[Dict]]:
    prompt = PROMPT_TEMPLATE.format(code=chunk)
    for _ in range(retries):
        for key in TOGETHER_API_KEYS:
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": DEFAULT_MODEL,
                "prompt": prompt,
                "temperature": 0.2,
                "max_tokens": MAX_TOKENS,
                "logprobs": 5,
                "stream": False,
            }
            resp = requests.post(
                TOGETHER_API_URL, headers=headers, json=payload, timeout=60
            )
            if resp.status_code == 429:
                time.sleep(5)
                continue
            resp.raise_for_status()
            result = resp.json()
            text = result["choices"][0]["text"]
            probs = result["choices"][0].get("logprobs", {}).get("token_logprobs", [])
            conf = (
                round(sum(torch.exp(torch.tensor(probs))) / len(probs), 4)
                if probs
                else 0.8
            )
            return text, conf, probs
    return "", 0.0, []


def split_code_by_function_ast(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [code]
    lines = code.splitlines()
    func_starts = sorted(
        {
            n.lineno - 1
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
    )
    if not func_starts or func_starts[0] != 0:
        func_starts = [0] + func_starts
    chunks, current, count = [], [], 0
    for i, start in enumerate(func_starts):
        end = func_starts[i + 1] if i + 1 < len(func_starts) else len(lines)
        segment = lines[start:end]
        tokens = sum(len(tokenizer.encode(line)) for line in segment)
        if count + tokens > MAX_TOKENS and current:
            chunks.append("\\n".join(current))
            current = segment
            count = tokens
        else:
            current += segment
            count += tokens
    if current:
        chunks.append("\\n".join(current))
    return chunks


def parse_annotations_from_response(
    text: str, line_offset: int = 0
) -> List[Dict[str, Any]]:
    annotations, lines, seen = [], text.splitlines(), set()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        match_sast = re.match(
            r"^# ⚠️ SAST Risk(?: \\((High|Medium|Low)\\))?: (.+)", stripped
        )
        if match_sast:
            severity = match_sast.group(1) or "Medium"
            reason = match_sast.group(2).strip()
            key = (line_offset + idx + 1, "sast_risk")
            if key not in seen:
                annotations.append(
                    {
                        "type": "sast_risk",
                        "severity": severity,
                        "reason": reason,
                        "line": line_offset + idx + 1,
                    }
                )
                seen.add(key)
            continue
        match_ml = re.match(r"^# 🧠 ML Signal: (.+)", stripped)
        if match_ml:
            reason = match_ml.group(1).strip()
            key = (line_offset + idx + 1, "ml_signal")
            if key not in seen:
                annotations.append(
                    {
                        "type": "ml_signal",
                        "reason": reason,
                        "line": line_offset + idx + 1,
                    }
                )
                seen.add(key)
            continue
        match_best = re.match(r"^# ✅ Best Practice: (.+)", stripped)
        if match_best:
            reason = match_best.group(1).strip()
            key = (line_offset + idx + 1, "best_practice")
            if key not in seen:
                annotations.append(
                    {
                        "type": "best_practice",
                        "reason": reason,
                        "line": line_offset + idx + 1,
                    }
                )
                seen.add(key)
    return annotations


def insert_annotations_into_code(
    original_lines: List[str], annotations: List[Dict]
) -> List[str]:
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
        if ann["type"] == "sast_risk":
            comment = f"{indent}# ⚠️ SAST Risk ({ann['severity']}): {ann['reason']}"
        elif ann["type"] == "ml_signal":
            comment = f"{indent}# 🧠 ML Signal: {ann['reason']}"
        else:
            comment = f"{indent}# ✅ Best Practice: {ann['reason']}"
        output.insert(insert_at, comment)
        offset += 1
    return output


def export_model_to_onnx(output_path: str = "model.onnx"):
    dummy = torch.ones((1, 1), dtype=torch.long)
    torch.onnx.export(
        model, dummy, output_path, input_names=["input_ids"], output_names=["logits"]
    )
    logging.info(f"✅ Exported ONNX model to {output_path}")


def generate_hf_dataset_from_json(json_file: Path, output_path: Path):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = datasets.Dataset.from_list(data)
    dataset.save_to_disk(str(output_path))
    logging.info(f"📦 Saved HuggingFace dataset to {output_path}")
