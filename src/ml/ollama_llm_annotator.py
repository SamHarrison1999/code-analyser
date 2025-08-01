import argparse
import json
import logging
import multiprocessing
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

import psutil
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer


# Estimate RAM and auto-scale max tokens
def estimate_max_tokens(memory_gb: float) -> int:
    if memory_gb >= 32:
        return 1000
    elif memory_gb >= 16:
        return 800
    elif memory_gb >= 8:
        return 600
    else:
        return 400


TOTAL_RAM_GB = psutil.virtual_memory().total / (1024**3)
ADAPTIVE_MAX_TOKENS = estimate_max_tokens(TOTAL_RAM_GB)
ADAPTIVE_MIN_TOKENS = max(int(TOTAL_RAM_GB * 10), 100)  # adaptive floor, min 100

# ✅ Load environment and constants
load_dotenv()
USE_LOCAL_LLM = True
LOCAL_LLM_URL = "http://localhost:11434/api/chat"
LOCAL_LLM_MODEL = "mistral"
MAX_MODEL_TOKENS = 30000
MAX_CALL_TIMEOUT = 600  # 10 minutes
AI_CACHE_DIR = Path(".ai_cache")
AI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TOOL_VERSION = "1.0.0"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

failed_chunks = []
chunk_semaphore = None


def get_tokenizer():
    return AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")


def get_cache_path(source: Path, input_dir: Path) -> Path:
    return AI_CACHE_DIR / source.relative_to(input_dir).with_suffix(".py.json")


def chunk_code(code: str, max_tokens: int = 800, overlap_lines: int = 5) -> List[str]:
    tokenizer = get_tokenizer()
    lines = code.splitlines()
    chunks, current_chunk, token_count = [], [], 0
    for i, line in enumerate(lines):
        token_count += len(tokenizer.tokenize(line))
        current_chunk.append(line)
        if token_count >= max_tokens or i == len(lines) - 1:
            chunks.append("\n".join(current_chunk))
            current_chunk = (
                lines[i - overlap_lines + 1 : i + 1] if i < len(lines) - 1 else []
            )
            token_count = sum(len(tokenizer.tokenize(l)) for l in current_chunk)
    return chunks


def annotate_with_local_llm(
    code: str, max_tokens=30000, min_tokens=1000, retries=3, backoff_delay=2.0
) -> Optional[str]:
    if min_tokens <= 50:
        logger.warning("⚠️ min_tokens is very low — setting safe floor at 100")
        min_tokens = 100
    min_tokens = max(min_tokens, len(code.splitlines()), ADAPTIVE_MIN_TOKENS)

    prompt = (
        "You are a secure code reviewer and machine learning expert.\\n"
        "Insert inline comments:\\n\\n"
        "# ⚠️ SAST Risk: <reason>\\n# 🧠 ML Signal: <reason>\\n# ✅ Best Practice: <reason>\\n\\n"
        "Do not change the code. No explanations. Only return the full annotated Python code."
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceH4/zephyr-7b-beta", local_files_only=True
        )
    except Exception as e:
        logger.error(f"❌ Failed to load tokenizer locally: {e}")
        return None

    for attempt in range(retries):
        try:
            code_tokens = tokenizer.tokenize(code)
            if len(code_tokens) < min_tokens:
                logger.warning(
                    f"⚠️ Padding code: {len(code_tokens)} tokens < {min_tokens} minimum required"
                )
                pad_lines = []
                while len(code_tokens) < min_tokens:
                    pad_lines.append("# pad\\n")
                    code_tokens = tokenizer.tokenize(code + "".join(pad_lines))
                code += "".join(pad_lines)

            payload = {
                "model": LOCAL_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": code},
                ],
                "stream": True,
                "temperature": 0.2,
            }

            with requests.post(
                LOCAL_LLM_URL, json=payload, timeout=MAX_CALL_TIMEOUT, stream=True
            ) as response:
                response.raise_for_status()
                full = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        full += chunk.get("message", {}).get("content", "")
                full = full.strip()
                # ✅ Clean fences and markdown from LLM
                if full.startswith("Here's the annotated Python code:"):
                    full = full.split("```python", 1)[-1].strip()
                if full.startswith("```python"):
                    full = full.removeprefix("```python").strip()
                if full.endswith("```"):
                    full = full.rsplit("```", 1)[0].strip()
                # ✅ Clean trailing whitespace and empty lines
                annotated = "\\n".join(
                    [line.rstrip() for line in full.splitlines() if line.strip()]
                )
                logger.info(
                    f"✅ Annotated chunk successfully [{len(code.splitlines())} lines]"
                )
                return annotated

        except requests.exceptions.ReadTimeout as e:
            logger.warning(
                f"🔁 Retry {attempt + 1}/{retries} — Ollama timeout: {e}. Reducing chunk size..."
            )
            time.sleep(backoff_delay)
            if max_tokens > min_tokens * 2:
                reduced_tokens = max(int(max_tokens * 0.75), min_tokens)
                logger.warning(f"⚠️ Reducing max_tokens to {reduced_tokens}")
                return annotate_with_local_llm(
                    code,
                    max_tokens=reduced_tokens,
                    min_tokens=min_tokens,
                    retries=retries - 1,
                )
            else:
                logger.error(
                    "❌ Chunk size too small to continue — final token count too low."
                )
                return None

        except Exception as e:
            logger.error(f"❌ Local LLM failed: {e}")
            return None


def annotate_code(
    code: str,
    file: Path,
    input_dir: Path,
    force=False,
    show_progress=True,
    chunk_delay=0.5,
    skip_chunks_over=None,
    disable_chunks=False,
    max_retries=3,
    max_tokens=ADAPTIVE_MAX_TOKENS,
    min_tokens=200,
    max_concurrent_chunks=2,
) -> Optional[str]:
    global chunk_semaphore
    if chunk_semaphore is None:
        chunk_semaphore = threading.Semaphore(max_concurrent_chunks)

    cache_file = get_cache_path(file, input_dir)
    if not force and cache_file.exists():
        return json.load(open(cache_file, "r", encoding="utf-8"))

    tokenizer = get_tokenizer()
    token_count = len(tokenizer.tokenize(code))
    lines = code.splitlines()

    if disable_chunks or (token_count < max_tokens and len(lines) < 600):
        annotated = annotate_with_local_llm(
            code, max_tokens=max_tokens, min_tokens=min_tokens, retries=max_retries
        )
    else:
        chunk_max = max_tokens - 100
        chunk_min = min_tokens
        chunks = chunk_code(code, max_tokens=chunk_max)
        if skip_chunks_over and len(chunks) > skip_chunks_over:
            return None
        results = []
        for i, chunk in enumerate(
            tqdm(chunks, desc=f"📦 {file.name}", disable=not show_progress)
        ):
            with chunk_semaphore:
                logger.info(f"🧩 Annotating chunk {i + 1}/{len(chunks)}")
                result = annotate_with_local_llm(
                    chunk,
                    max_tokens=chunk_max,
                    min_tokens=chunk_min,
                    retries=max_retries,
                )
                if result:
                    results.append(result)
                else:
                    failed_chunks.append(f"{file.name}: chunk {i + 1}")
                time.sleep(chunk_delay)
        annotated = "\n\n".join(results)

    if not annotated:
        return None

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "annotated_code": annotated,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": LOCAL_LLM_MODEL,
            "tool_version": TOOL_VERSION,
            "source_path": str(file),
        },
        open(cache_file, "w", encoding="utf-8"),
        indent=2,
    )
    return annotated


def process_file(file: Path, input_dir: Path, output_dir: Path, args) -> Optional[dict]:
    try:
        code = file.read_text(encoding="utf-8")
        # ✅ Automatically disable chunking for small files
        line_count = len(code.splitlines())
        token_count = len(get_tokenizer().tokenize(code))
        if not args.no_chunks and line_count < 100 and token_count < 1000:
            logger.info(
                f"📄 Small file detected ({line_count} lines, {token_count} tokens) — disabling chunking for: {file.name}"
            )
            args.no_chunks = True
        result = annotate_code(
            code,
            file,
            input_dir,
            force=args.force,
            show_progress=args.progress,
            chunk_delay=args.chunk_delay,
            skip_chunks_over=args.skip_chunks_over,
            disable_chunks=args.no_chunks,
            max_retries=args.max_retries,
            max_tokens=ADAPTIVE_MAX_TOKENS,  # Ensure token limit adapts to system
            min_tokens=args.min_chunk_tokens,  # Pass min chunk size from CLI
            max_concurrent_chunks=args.max_concurrent_chunks,
        )
        if result is None:
            return {"skipped": file.name}

        lines = result.splitlines()
        pattern = re.compile(r"# (⚠️|🧠|✅) (SAST Risk|ML Signal|Best Practice):")
        annotations, last = [], None

        for i, line in enumerate(lines):
            if m := pattern.match(line.strip()):
                last = {
                    "line": i + 2,
                    "type": m.group(2),
                    "confidence": 0.85,
                    "annotation": line.strip(),
                }
            elif last:
                annotations.append(last)
                last = None

        annotated_path = output_dir / file.relative_to(input_dir)
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        annotated_path.write_text(result, encoding="utf-8")
        annotated_path.with_suffix(".json").write_text(
            json.dumps(annotations, indent=2), encoding="utf-8"
        )

        return {
            "file": str(file.relative_to(input_dir)),
            "count": len(annotations),
            "avg_conf": round(
                sum(a["confidence"] for a in annotations) / max(len(annotations), 1), 3
            ),
        }

    except Exception as e:
        logger.error(f"❌ Failed: {file}: {e}")
        return None


def process_file_mp_wrapper(params):
    file, input_dir, output_dir, args_dict = params

    class Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return process_file(file, input_dir, output_dir, Args(**args_dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-chunks-over", type=int, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--chunk-delay", type=float, default=0.5)
    parser.add_argument("--no-chunks", action="store_true")
    parser.add_argument("--use-mp", action="store_true")
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Max retries for Ollama LLM"
    )
    parser.add_argument("--min-chunk-tokens", type=int, default=200)
    parser.add_argument(
        "--max-concurrent-chunks",
        type=int,
        default=2,
        help="Max concurrent annotated chunks",
    )

    args = parser.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_files = list(input_dir.rglob("*.py"))

    logger.info(f"📁 Annotating {len(all_files)} files from: {input_dir}")
    results, skipped = [], []

    if args.use_mp:
        max_pool_size = max(1, multiprocessing.cpu_count() // 2)
        with multiprocessing.Pool(processes=max_pool_size) as pool:
            file_args = [(f, input_dir, output_dir, vars(args)) for f in all_files]
            for res in tqdm(
                pool.imap_unordered(process_file_mp_wrapper, file_args),
                total=len(all_files),
            ):
                if res:
                    if "skipped" in res:
                        skipped.append(res["skipped"])
                    elif "file" in res:
                        results.append(res)
    else:
        for f in all_files:
            res = process_file(f, input_dir, output_dir, args)
            if res:
                if "skipped" in res:
                    skipped.append(res["skipped"])
                elif "file" in res:
                    results.append(res)

    if skipped:
        Path("skipped_files.json").write_text(json.dumps(skipped, indent=2))
    if failed_chunks:
        Path("failed_chunks.json").write_text(json.dumps(failed_chunks, indent=2))

    logger.info("✅ All files processed.")
    for r in results:
        logger.info(
            f" - {r['file']}: {r['count']} annotations @ avg_conf {r['avg_conf']}"
        )
