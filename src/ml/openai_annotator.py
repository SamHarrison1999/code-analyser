# --- file: code_analyser/src/ml/openai_annotator.py ---
# ===== openai_annotator.py =====
# Core libraries for parsing, I/O, logging, environment access, text processing, and timing.
import ast
import csv
import json
import logging
import os
import re
import time
# Containers for efficient grouping and similarity; pathlib for robust path handling.
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
# Explicit typing aids readability and static checking.
from typing import List, Dict, Tuple, Any

# Load .env early so import-time defaults (e.g. model name) are available.
from dotenv import load_dotenv
# OpenAI client and errors; a stub is provided by tests to keep imports safe.
from openai import OpenAI, OpenAIError, RateLimitError
# TensorBoard writer for confidence traces; stubbed in tests.
from torch.utils.tensorboard import SummaryWriter
# Tokeniser to estimate token spans; stubbed in tests.
# Use GPT2TokenizerFast when available; provide a tiny stub in test envs.
try:
    from transformers import GPT2TokenizerFast
except Exception:
    class GPT2TokenizerFast:
        @classmethod
        def from_pretrained(cls, *a, **k): return cls()
        def __call__(self, text, **kw): return {'input_ids':[1,2,3]}
        def encode(self, text, **kw): return [1,2,3]
        def decode(self, ids, **kw): return ' '.join(str(i) for i in (ids or []))

# Bring environment variables into process for defaults and credentials.
load_dotenv()
# Configure a simple, global logger for the module.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Instantiate an OpenAI client (safe under the test stubs).
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Allow model override via environment; provide a sensible fallback.
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
FALLBACK_MODEL = "gpt-4o-mini"
# Control sampling fan-out and output bound.
NUM_SAMPLES = 2
MAX_TOKENS = 3000

# Build a tokeniser and output directories up front; tests supply cheap stubs.
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
DEBUG_DIR = Path("debug_model_outputs")
DEBUG_DIR.mkdir(exist_ok=True)
MISMATCH_DIR = Path("debug_mismatches")
MISMATCH_DIR.mkdir(exist_ok=True)
# Single SummaryWriter for this module.
writer = SummaryWriter("runs/annotation_confidences")

# System prompt guiding the LLM to produce inline comments only.
INSTRUCTION = """
You are a code reviewer and static analysis expert. Your task is to annotate the following Python code with precise inline comments to identify:

1. 🧠 ML Signal — behavioural or usage patterns that could be used to train ML models
2. ⚠️ SAST Risk (High/Medium/Low) — security vulnerabilities, unsafe patterns, or misuse of APIs
3. ✅ Best Practice — code quality, readability, or maintainability suggestions

🛑 Do not modify, delete, or reorder any code.
🛑 Do not omit or skip any lines.

✅ Insert comments only **above** the relevant lines using one of the exact formats below:

# 🧠 ML Signal: <reason>
# ⚠️ SAST Risk (High|Medium|Low): <reason>
# ✅ Best Practice: <reason>

Do not include any explanations, markdown, summaries, headings, or extra formatting. Annotate all relevant lines in the given code.

Ensure consistency: always annotate the same lines across runs, even if the comments may seem repetitive. Do not vary which lines you annotate or how you phrase the annotations.

Only return the annotated code block — no introductory text.

Now annotate the following code:
"""

# Map from emoji to canonical label used by JSON parsers.
LABEL_MAP = {"⚠️": "SAST Risk", "🧠": "ML Signal", "✅": "Best Practice"}
# Reverse map used to reconstruct an inline comment from a JSON record.
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Key phrases hinting at severity; used when the level is not provided explicitly.
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


# Guess a severity from free-text when one was not supplied.
def infer_severity_from_text(text: str) -> str:
    text_lower = text.lower()
    for keyword, level in SEVERITY_KEYWORDS.items():
        if keyword in text_lower:
            return level
    return "Medium"


# Normalise a heterogeneous annotations container (list or {'annotations': [...]}) into a list.
def _normalise_annotations_container(data: Any) -> List[Any]:
    if isinstance(data, dict) and "annotations" in data and isinstance(data["annotations"], list):
        return data["annotations"]
    if isinstance(data, list):
        return data
    return []


# Turn a record (dict/list/tuple) into a unified dict with 'line' and 'annotation' keys.
def _normalise_record(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        if "annotation" in raw and "line" in raw:
            return {
                "line": int(raw["line"]),
                "annotation": str(raw["annotation"]),
                "confidence": float(raw.get("confidence", 1.0)),
            }
        if "comment" in raw and "line" in raw:
            return {
                "line": int(raw["line"]),
                "annotation": str(raw["comment"]),
                "confidence": float(raw.get("confidence", 1.0)),
            }
        if "type" in raw and "content" in raw:
            symbol = REVERSE_LABEL_MAP.get(str(raw["type"]), "")
            sev = raw.get("severity")
            sev_part = f" ({sev})" if sev and str(raw["type"]) == "SAST Risk" else ""
            return {
                "line": int(raw.get("line") or raw.get("line_num") or 1),
                "annotation": f"{symbol} {raw['type']}{sev_part}: {raw['content']}",
                "confidence": float(raw.get("confidence", 1.0)),
            }
        if "line_num" in raw and "annotation" in raw:
            return {
                "line": int(raw["line_num"]),
                "annotation": str(raw["annotation"]),
                "confidence": float(raw.get("confidence", 1.0)),
            }
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        line = int(raw[0])
        text = str(raw[1])
        conf = float(raw[2]) if len(raw) >= 3 else 1.0
        return {"line": line, "annotation": text, "confidence": conf}
    return {}


# Validate that annotations align with the annotated code; tolerant to varied input shapes.
def validate_annotation_sync(annotated_code: str, annotations: Any, filename: str = "") -> bool:
    lines = annotated_code.splitlines()
    dynamic_window = max(10, int(len(lines) * 0.05))
    mismatches = []
    records = _normalise_annotations_container(annotations)
    for raw in records:
        rec = _normalise_record(raw)
        if not rec:
            continue
        expected_comment = re.sub(r"\s+", " ", f"# {rec['annotation']}".strip())
        target = rec["line"] - 1
        nearby = lines[
            max(0, target - dynamic_window) : min(len(lines), target + dynamic_window + 1)
        ]
        if not any(expected_comment in re.sub(r"\s+", " ", l.strip()) for l in nearby):
            mismatches.append(
                {
                    "line": rec["line"],
                    "expected": expected_comment,
                    "reason": "",
                    "text": "",
                }
            )
    if mismatches:
        csv_path = MISMATCH_DIR / f"{filename}_mismatches.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["line", "expected", "reason", "text"])
            writer.writeheader()
            writer.writerows(mismatches)
        logger.warning(f"❌ Found {len(mismatches)} mismatches. CSV written to {csv_path}")
        return False
    return True


# Infer minimum indentation for placing a comment above the next non-comment code line.
def infer_indentation(lines: List[str], start: int) -> int:
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if stripped and not stripped.startswith("#"):
            return len(lines[i]) - len(stripped)
    return 0


# Remove trailing inline comments whilst keeping code layout intact.
def strip_inline_comments(code: str) -> str:
    return "\n".join(
        re.split(r"\s+#", line, maxsplit=1)[0].rstrip()
        for line in code.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


# Compute (start, end) line bounds for functions and classes to guide chunking.
def extract_function_class_bounds(code: str) -> List[Tuple[int, int]]:
    tree = ast.parse(code)
    lines = code.splitlines()
    bounds = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            indent = len(lines[start]) - len(lines[start].lstrip())
            for i in range(start + 1, len(lines)):
                if lines[i].strip() == "":
                    continue
                current_indent = len(lines[i]) - len(lines[i].lstrip())
                if current_indent <= indent:
                    bounds.append((start, i - 1))
                    break
            else:
                bounds.append((start, len(lines) - 1))
    return bounds


# Thin wrapper with retries and fallback model selection.
def call_openai(messages, temperature=0.0, model=DEFAULT_MODEL, attempt=1, max_attempts=5):
    try:
        return client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=MAX_TOKENS
        )
    except RateLimitError:
        wait = min(10, 2**attempt)
        logger.warning(f"⏳ Rate limit hit — retrying in {wait:.1f}s...")
        time.sleep(wait)
        next_model = FALLBACK_MODEL if attempt >= 3 and model != FALLBACK_MODEL else model
        if attempt < max_attempts:
            return call_openai(messages, temperature, model=next_model, attempt=attempt + 1)
        raise RuntimeError("❌ Exceeded retries.")
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI error: {e}")


# Insert annotations back into code using a previously saved JSON list (dict- or list-shapes accepted).
def backfill_annotations_into_code(code_path: Path, annotations_path: Path, output_path: Path):
    code_lines = code_path.read_text().splitlines()
    raw = json.loads(annotations_path.read_text())
    annotations = _normalise_annotations_container(raw)
    inserted = 0
    matched_lines = set()
    for ann in sorted((_normalise_record(a) for a in annotations), key=lambda x: x.get("line", 1)):
        if not ann:
            continue
        expected_text = ""
        annotation = " " * infer_indentation(code_lines, ann["line"] - 1) + f"# {ann['annotation']}"
        insert_idx = ann["line"] - 1 + inserted
        if (
            insert_idx < len(code_lines)
            and expected_text
            and code_lines[insert_idx].strip() == expected_text
        ):
            code_lines.insert(insert_idx, annotation)
            inserted += 1
            matched_lines.add(insert_idx)
        else:
            if expected_text:
                for idx, line in enumerate(code_lines):
                    if idx in matched_lines:
                        continue
                    if SequenceMatcher(None, expected_text, line.strip()).ratio() > 0.95:
                        code_lines.insert(idx, annotation)
                        matched_lines.add(idx)
                        inserted += 1
                        break
            else:
                if 0 <= insert_idx <= len(code_lines):
                    code_lines.insert(insert_idx, annotation)
                    inserted += 1
                else:
                    logger.warning(f"❌ Could not backfill: {annotation}")
    output_path.write_text("\n".join(code_lines), encoding="utf-8")
    logger.info(f"📎 Backfilled {inserted} annotations to: {output_path}")


# Validate a saved JSON against code; if out of sync, attempt a best-effort backfill.
def maybe_backfill_if_mismatch(code_path: Path, annotations_path: Path, output_path: Path):
    if not code_path.exists() or not annotations_path.exists():
        logger.error("❌ Cannot backfill — missing .py or .json file")
        return
    code = code_path.read_text()
    raw = json.loads(annotations_path.read_text())
    annotations = _normalise_annotations_container(raw)
    logger.info(f"🔍 Validating annotation sync for: {code_path.name}")
    if validate_annotation_sync(code, annotations, code_path.stem):
        logger.info("✅ No mismatch — skipping backfill")
    else:
        logger.warning("🔁 Mismatch detected — backfilling...")
        backfill_annotations_into_code(code_path, annotations_path, output_path)


# Annotate a single chunk; returns inline-annotated text, raw per-line dicts, and a confidence.
def annotate_chunk_with_openai(
    chunk: str, samples: int = NUM_SAMPLES, temperature: float = 0.0
) -> Tuple[str, List[Dict], float]:
    completions = []
    for _ in range(samples):
        response = call_openai(
            [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": chunk}],
            temperature,
        )
        completions.append(response.choices[0].message.content)

    def extract_comments(text: str) -> List[str]:
        return [line.strip() for line in text.splitlines() if line.strip().startswith("#")]

    all_comments = [extract_comments(c) for c in completions]
    annotated_lines = [
        line for line in completions[0].splitlines() if not line.strip().startswith("```")
    ]
    annotations = []
    for i, line in enumerate(annotated_lines):
        if not line.strip().startswith("#"):
            continue
        comment = line.strip("# ").strip()
        match_count = sum(
            1
            for other in all_comments
            if any(SequenceMatcher(None, comment, cand).ratio() > 0.75 for cand in other)
        )
        confidence = round(match_count / samples, 3)
        annotations.append({"line": i + 1, "comment": comment, "confidence": confidence})
    chunk_conf = (
        round(sum(a["confidence"] for a in annotations) / len(annotations), 3)
        if annotations
        else 0.0
    )
    return "\n".join(annotated_lines), annotations, chunk_conf


# Annotate an entire file; returns the fully inline-annotated code, a file confidence, and the per-line records.
def annotate_code_with_openai(
    code: str, output_dir: Path = None, filename: str = "annotated"
) -> Tuple[str, float, List[Dict]]:
    original_lines = code.splitlines()
    stripped_code = strip_inline_comments(code)
    code_lines = stripped_code.splitlines()
    if not any(line.strip() and not line.strip().startswith("#") for line in code_lines):
        logger.warning(f"⚠️ Skipping '{filename}' — empty or comment-only")
        return code, 1.0, []
    if len(code_lines) < 8:
        code_lines += [""] * (8 - len(code_lines))
    stripped_code = "\n".join(code_lines)
    try:
        tree = ast.parse(stripped_code)
        lines = stripped_code.splitlines()
        stmt_lines = sorted({n.lineno for n in ast.walk(tree) if isinstance(n, ast.stmt)})

        def closest_stmt_lineno(t):
            return min(stmt_lines, key=lambda x: abs(x - t)) if stmt_lines else t

        func_bounds = extract_function_class_bounds(stripped_code)
        func_starts = sorted(
            n.lineno - 1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        )
        if not func_starts or func_starts[0] != 0:
            func_starts = [0] + func_starts
        chunks = [
            (
                start,
                "\n".join(
                    lines[start : func_starts[i + 1] if i + 1 < len(func_starts) else len(lines)]
                ),
            )
            for i, start in enumerate(func_starts)
        ]
    except SyntaxError:
        logger.warning(f"⚠️ Syntax error in {filename}; falling back to whole file")
        chunks = [(0, stripped_code)]
        closest_stmt_lineno = lambda x: x
        func_bounds = []
    all_annotations, confidences = [], []
    token_offset = 0
    for idx, (chunk_start, chunk_text) in enumerate(chunks):
        if not chunk_text.strip():
            continue
        logger.info(f"✨ Annotating chunk {idx + 1}/{len(chunks)}")
        try:
            _, chunk_annotations, confidence = annotate_chunk_with_openai(chunk_text)
        except Exception as e:
            logger.warning(f"❌ Chunk {idx + 1} failed: {e}")
            continue
        confidences.append(confidence)
        writer.add_scalar("confidence_per_chunk", confidence, idx)
        for annotation in chunk_annotations:
            rel_line = annotation["line"]
            full_line = min(chunk_start + rel_line - 1, len(original_lines) - 1)
            stmt_line = closest_stmt_lineno(full_line + 1) - 1
            for start, end in func_bounds:
                if start <= full_line <= end and not (start <= stmt_line <= end):
                    stmt_line = full_line
                    break
            comment = annotation["comment"]
            conf = annotation["confidence"]
            tokens = tokenizer.encode(original_lines[stmt_line])
            entry = {
                "line": stmt_line + 1,
                "text": original_lines[stmt_line],
                "annotation": comment,
                "confidence": conf,
                "tokens": tokens,
                "start_token": token_offset,
                "end_token": token_offset + len(tokens),
                "annotation_tokens": tokenizer.encode(comment),
            }
            if comment.startswith("🧠"):
                entry["label"] = "ml_signal"
                entry["reason"] = comment.split(":", 1)[1].strip()
            elif comment.startswith("⚠️"):
                entry["label"] = "sast_risk"
                match = re.search(r"SAST Risk \((.*?)\):", comment)
                entry["severity"] = match.group(1) if match else "Unknown"
                entry["reason"] = comment.split(":", 1)[1].strip()
            elif comment.startswith("✅"):
                entry["label"] = "best_practice"
                entry["reason"] = comment.split(":", 1)[1].strip()
            else:
                entry["label"] = "unknown"
                entry["reason"] = comment
            token_offset += len(tokens)
            all_annotations.append(entry)
    annotation_map = defaultdict(list)
    for ann in all_annotations:
        idx = ann["line"] - 1
        indent = infer_indentation(original_lines, idx)
        annotation_map[idx].append((" " * indent + f"# {ann['annotation']}", ann["confidence"]))
    adjusted_code_lines = []
    last_annotation = None
    in_docstring = False
    doc_delim = None
    for idx, line in enumerate(original_lines):
        stripped = line.strip()
        if stripped.startswith(("'''", '"""')):
            delim = stripped[:3]
            if stripped.count(delim) == 2:
                pass
            elif not in_docstring:
                in_docstring = True
                doc_delim = delim
            elif in_docstring and doc_delim == delim:
                in_docstring = False
                doc_delim = None
        if idx in annotation_map and not in_docstring:
            for c, _ in deduplicate_comments(annotation_map[idx]):
                if c != last_annotation:
                    adjusted_code_lines.append(c)
                    last_annotation = c
        adjusted_code_lines.append(line)
        last_annotation = None if not line.strip().startswith("#") else line.strip()
    file_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    writer.add_scalar("confidence_per_file", file_conf, 0)
    writer.flush()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{filename}_annotations.json", "w", encoding="utf-8") as f:
            json.dump(all_annotations, f, indent=2)
    logger.info(
        f"✅ Done: {len(all_annotations)} annotations inserted with average confidence {file_conf}"
    )
    return "\n".join(adjusted_code_lines).strip(), file_conf, all_annotations


# Parse inline-annotated code back into a structured list for post-processing.
def parse_annotated_code(raw_code: str) -> List[Dict]:
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
                "line": idx + 2,
                "type": annotation_type,
                "severity": severity,
                "content": content,
                "confidence": 1.0,
            }
        elif pending:
            annotations.append(pending)
            pending = None
    return annotations


# Wrap a file’s parsed annotations with a filename key for saving.
def generate_annotation_json(raw_code: str, filename: str = "") -> Dict:
    return {"filename": filename, "annotations": parse_annotated_code(raw_code)}


# Save parsed annotations next to their source; resilient directory creation.
def save_annotation_json(
    code: str, source_path: str, output_dir: str = "datasets/annotated_fintech"
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(source_path)
    json_filename = f"{basename}.json"
    save_path = os.path.join(output_dir, json_filename)
    data = generate_annotation_json(code, filename=basename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return save_path


# Deduplicate similar comments whilst preserving the highest-confidence variant.
def deduplicate_comments(
    comments: List[Tuple[str, float]], threshold: float = 0.85
) -> List[Tuple[str, float]]:
    deduped = []
    last_comment = ""
    for comment, conf in sorted(comments, key=lambda x: -x[1]):
        if SequenceMatcher(None, comment, last_comment).ratio() >= threshold:
            continue
        if all(
            SequenceMatcher(None, comment, existing).ratio() < threshold for existing, _ in deduped
        ):
            deduped.append((comment, conf))
            last_comment = comment
    return deduped


# Exported public surface of this module.
__all__ = [
    "annotate_code_with_openai",
    "annotate_chunk_with_openai",
    "backfill_annotations_into_code",
    "maybe_backfill_if_mismatch",
    "save_annotation_json",
    "parse_annotated_code",
    "generate_annotation_json",
]
