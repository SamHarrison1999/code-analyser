# scripts/batch_annotate_fintech.py

import argparse
import json
import logging
import os
import sys
import time
import webbrowser
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import upload_folder

# ✅ Environment & sys.path setup
load_dotenv()

SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Imports
from ml.openai_annotator import annotate_code_with_openai, maybe_backfill_if_mismatch
from ml.overlay_utils import save_overlay_csv
from ml.html_dashboard import write_html_dashboard


def annotate_file(filepath: Path, output_dir: Path, args) -> None:
    rel_path = filepath.relative_to(args.input_dir)
    base_name = rel_path.stem
    output_base = output_dir / rel_path.with_suffix("")

    for attempt in range(args.max_retries):
        try:
            code = filepath.read_text(encoding="utf-8")
            annotated_code, confidence, annotations = annotate_code_with_openai(
                code, output_dir=output_base.parent, filename=base_name
            )

            logger.info(f"🧠 Annotations returned: {len(annotations)}")

            output_base.parent.mkdir(parents=True, exist_ok=True)
            output_py_path = output_base.with_suffix(".annotated.py")
            output_json_path = output_base.with_name(f"{base_name}_annotations.json")

            # Write annotated code
            output_py_path.write_text(annotated_code, encoding="utf-8")

            # ✅ Conditional backfill
            if args.allow_backfill and output_json_path.exists():
                maybe_backfill_if_mismatch(output_py_path, output_json_path)

            logger.info(f"✅ Annotated: {rel_path} ({confidence:.2f} confidence)")

            if args.export_overlays and annotations:
                overlay_path = output_base.with_suffix(".overlay.csv")
                save_overlay_csv(annotations, overlay_path)
                logger.info(f"🧠 Overlay saved: {overlay_path}")

            time.sleep(args.chunk_delay)
            return

        except Exception as e:
            logger.warning(f"🔁 Retry {attempt + 1}/{args.max_retries} for {filepath.name}: {e}")
            time.sleep(args.retry_delay)

    logger.error(f"❌ Failed to annotate after {args.max_retries} attempts: {filepath.name}")


def zip_directory(source_dir: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in source_dir.rglob("*"):
            if file.is_file():
                zipf.write(file, file.relative_to(source_dir))


def upload_to_huggingface(local_dir: Path, repo_id: str, token: str):
    logger.info(f"🚀 Uploading {local_dir} to HuggingFace repo: {repo_id}")
    try:
        upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            path_in_repo=local_dir.name,
        )
        logger.info("✅ Upload successful to HuggingFace Hub.")
    except Exception as e:
        logger.error(f"❌ Failed to upload to HuggingFace: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--export-dashboard", action="store_true")
    parser.add_argument("--export-overlays", action="store_true")
    parser.add_argument("--export-arrow", action="store_true")
    parser.add_argument("--export-heatmaps", action="store_true")
    parser.add_argument("--zip-dashboard", action="store_true")
    parser.add_argument("--launch-dashboard", action="store_true")
    parser.add_argument("--allow-backfill", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--chunk-delay", type=float, default=3.0)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    args.input_dir = Path(args.input_dir).resolve()
    args.output_dir = Path(args.output_dir).resolve()

    logger.info(f"📂 Annotating {args.input_dir}")
    py_files = list(args.input_dir.rglob("*.py"))
    for f in py_files:
        logger.info(f" - {f}")
    logger.info(f"🔎 Found {len(py_files)} Python files.")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(annotate_file, f, args.output_dir, args) for f in py_files]
        for f in futures:
            f.result()

    if args.export_overlays:
        overlay_files = list(args.output_dir.rglob("*.overlay.csv"))
        (
            logger.info(f"✅ {len(overlay_files)} overlay CSV files saved.")
            if overlay_files
            else logger.error(
                "❌ --export-overlays was set, but no overlay CSV files were saved."
            )
        )

    if args.export_dashboard:
        html_path = args.output_dir / "dashboard.html"
        write_html_dashboard(args.output_dir, html_path)
        logger.info(f"📊 Dashboard saved to {html_path}")

        if args.zip_dashboard:
            zip_path = args.output_dir / "dashboard_bundle.zip"
            zip_directory(args.output_dir, zip_path)
            logger.info(f"🗜️ Dashboard ZIP exported: {zip_path}")

        if args.launch_dashboard:
            webbrowser.open(f"file://{html_path.resolve()}", new=2)

    # ✅ Export Arrow Dataset
    if args.export_arrow:
        try:
            import pandas as pd
            from datasets import Dataset

            records = []
            for file in args.output_dir.rglob("*_annotations.json"):
                with open(file, encoding="utf-8") as f:
                    samples = json.load(f)
                    for item in samples:
                        records.append({
                            "file": file.name.replace("_annotations.json", ""),
                            "line": item["line"],
                            "text": item["text"],
                            "tokens": item["tokens"],
                            "start_token": item["start_token"],
                            "end_token": item["end_token"],
                            "label": item.get("label", "unknown"),
                            "severity": item.get("severity", None),
                            "confidence": item["confidence"],
                            "reason": item["reason"],
                        })
            if records:
                ds = Dataset.from_pandas(pd.DataFrame(records))
                arrow_path = args.output_dir / "annotated.arrow"
                ds.save_to_disk(str(arrow_path))
                logger.info(f"📦 HuggingFace Arrow dataset saved: {arrow_path}")
            else:
                logger.warning("⚠️ No valid records to export as Arrow.")
        except ImportError:
            logger.error(
                "❌ 'datasets' package not installed. Run: pip install datasets"
            )

    # ✅ Export Heatmaps
    if args.export_heatmaps:
        heatmap_count = 0
        for file in args.output_dir.rglob("*_annotations.json"):
            try:
                with open(file, encoding="utf-8") as f:
                    samples = json.load(f)
                heatmap = []
                for item in samples:
                    for i, token in enumerate(item.get("tokens", [])):
                        heatmap.append({
                            "line": item["line"],
                            "token": token,
                            "index": i,
                            "confidence": item["confidence"] if item["start_token"] <= i < item["end_token"] else 0.0,
                            "severity": item.get("severity", None) if item["start_token"] <= i < item[
                                "end_token"] else None,
                        })
                output_path = file.with_suffix(".heatmap.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(heatmap, f, indent=2)
                heatmap_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Heatmap export failed for {file.name}: {e}")
        logger.info(f"🧠 Token heatmaps exported: {heatmap_count}")

    # ✅ HuggingFace Upload (always runs if creds are present)
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    hf_repo = os.getenv("HUGGINGFACE_REPO")
    if hf_token and hf_repo:
        upload_to_huggingface(args.output_dir, hf_repo, hf_token)
    else:
        logger.warning("⚠️ HuggingFace credentials not set — skipping upload.")


if __name__ == "__main__":
    main()
