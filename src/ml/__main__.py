# code_analyser/src/ml/__main__.py

import argparse
import os
import json
import csv
from pathlib import Path

from .inference import AnnotationEngine
from .ai_cache import compute_file_hash, load_cached_annotation
from .config import AI_CACHE_DIR, DATA_PATHS
from .ollama_llm_annotator import annotate_code_with_together_ai
from .together_annotation import save_annotation_json

DEFAULT_EXPORT_DIR = "./ai_annotations/"


def export_csv(filepath, records):
    with open(filepath, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(
            cf, fieldnames=["line_num", "annotation", "confidence", "tokens", "line"]
        )
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})


def summarize_annotations(all_annotations):
    total = len(all_annotations)
    flagged = [a for a in all_annotations if a["annotation"]]
    counts = {"⚠️": 0, "🧠": 0, "✅": 0}
    confidence_sum = 0.0

    for a in flagged:
        label = a["annotation"].split()[0]
        counts[label] += 1
        confidence_sum += a["confidence"]

    annotated = len(flagged)
    avg_conf = confidence_sum / annotated if annotated else 0.0

    print("\n📊 Summary:")
    print(f"- Total lines processed: {total}")
    print(f"- Annotated lines: {annotated} ({annotated/total:.1%})")
    print(f"- Average confidence: {avg_conf:.2f}")
    for label, count in counts.items():
        print(f"- {label}: {count} ({count/total:.1%})")

    return {
        "total": total,
        "annotated": annotated,
        "avg_confidence": avg_conf,
        "counts": counts,
    }


def export_summary_csv(filepath, summary):
    with open(filepath, "w", encoding="utf-8", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Lines", summary["total"]])
        writer.writerow(["Annotated Lines", summary["annotated"]])
        writer.writerow(["Average Confidence", round(summary["avg_confidence"], 4)])
        for label, count in summary["counts"].items():
            writer.writerow([f"{label} Count", count])


# 🔁 Together.ai flow: single file
def annotate_and_parse_file(
    filepath: str, stream: bool = False, overwrite: bool = False, output_dir: str = None
):
    if not filepath.endswith(".py"):
        print(f"⚠️ Skipped non-Python file: {filepath}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    annotated_code = annotate_code_with_together_ai(code, stream=stream)

    if overwrite:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(annotated_code)
        print(f"✍️ Overwrote: {filepath}")

    output_dir = output_dir or DATA_PATHS["annotation_dir"]
    save_path = save_annotation_json(
        annotated_code, source_path=filepath, output_dir=output_dir
    )
    print(f"✅ Annotation saved to {save_path}")


# 🔁 Together.ai flow: batch folder
def batch_annotate(folder: str, stream: bool = False, overwrite: bool = False):
    all_py_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".py"):
                all_py_files.append(os.path.join(root, file))

    print(f"🔍 Found {len(all_py_files)} Python files in {folder}")
    for path in all_py_files:
        annotate_and_parse_file(path, stream=stream, overwrite=overwrite)


# 🧠 ML model-based annotation
def annotate_file(
    filepath: str,
    use_cache: bool = True,
    refresh: bool = False,
    use_distilled: bool = False,
    export: str = None,
):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    file_hash = compute_file_hash(filepath)
    cache_path = os.path.join(AI_CACHE_DIR, f"{file_hash}.json")

    if use_cache and not refresh:
        cached = load_cached_annotation(filepath)
        if cached:
            print(f"✅ Loaded from cache: {cache_path}")
            return cached["annotated_code"]

    print(f"🧠 Running AI annotation on: {filepath}")
    engine = AnnotationEngine(use_distilled=use_distilled)
    annotated_code = engine.annotate_file(filepath, export_path=export)

    save_annotation_to_cache(filepath, annotated_code)
    print(f"💾 Cached: {cache_path}")
    return annotated_code


def main():
    parser = argparse.ArgumentParser(
        description="Unified AI/Together.ai CLI for code annotation"
    )

    parser.add_argument("--file", help="Annotate a single file using model")
    parser.add_argument("--dir", help="Annotate all files in directory using model")
    parser.add_argument(
        "--refresh", action="store_true", help="Refresh cache (model mode)"
    )
    parser.add_argument(
        "--use-distilled", action="store_true", help="Use distilled transformer model"
    )
    parser.add_argument("--export", help="Export model output (.json or .csv)")
    parser.add_argument(
        "--export-dir",
        default=DEFAULT_EXPORT_DIR,
        help="Output folder for batch results",
    )

    parser.add_argument(
        "--annotate-and-parse", help="Annotate + parse a single file using Together.ai"
    )
    parser.add_argument("--batch", help="Batch annotate folder using Together.ai")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite original .py files"
    )
    parser.add_argument(
        "--stream", action="store_true", help="Enable Together.ai streaming"
    )

    args = parser.parse_args()

    try:
        # 🔁 Together.ai modes
        if args.annotate_and_parse:
            annotate_and_parse_file(
                args.annotate_and_parse, stream=args.stream, overwrite=args.overwrite
            )

        elif args.batch:
            batch_annotate(args.batch, stream=args.stream, overwrite=args.overwrite)

        # 🧠 Model-based modes
        elif args.dir:
            print(f"🧠 Annotating directory: {args.dir} (model)")
            export_dir = Path(args.export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            all_results = []

            for root, _, files in os.walk(args.dir):
                for fname in files:
                    if fname.endswith(".py"):
                        full_path = os.path.join(root, fname)
                        engine = AnnotationEngine(use_distilled=args.use_distilled)
                        annotated = engine.annotate_file(full_path, as_dict=True)

                        base = Path(full_path).stem
                        json_path = export_dir / f"{base}.annotated.json"
                        csv_path = export_dir / f"{base}.annotated.csv"

                        with open(json_path, "w", encoding="utf-8") as jf:
                            json.dump(annotated, jf, indent=2)
                        export_csv(csv_path, annotated)

                        all_results.extend(annotated)

            merged_summary = summarize_annotations(all_results)
            with open(
                export_dir / "annotations_summary.json", "w", encoding="utf-8"
            ) as f:
                json.dump(merged_summary, f, indent=2)
            export_summary_csv(export_dir / "annotations_summary.csv", merged_summary)

        elif args.file:
            annotated_code = annotate_file(
                args.file,
                use_cache=True,
                refresh=args.refresh,
                use_distilled=args.use_distilled,
                export=args.export,
            )
            print("\n🔍 Annotated Code:\n")
            print(annotated_code)

        else:
            parser.print_help()

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
