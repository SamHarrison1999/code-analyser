import os
import json
import csv
import argparse
from collections import Counter

from code_analyser.src.ml.together_ai_annotator import annotate_code_with_together_ai
from code_analyser.src.ml.together_annotation import (
    save_annotation_json,
    parse_annotated_code,
)
from code_analyser.src.ml.ai_signal_utils import (
    count_annotations_by_type,
    severity_breakdown_by_type,
)

INPUT_DIR = "datasets/github_fintech"
OUTPUT_DIR = "datasets/annotated_fintech"
SUMMARY_JSON = "annotations_summary.json"
SUMMARY_CSV = "per_file_stats.csv"


def generate_dataset(
    input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, overwrite=False, stream=False
):
    os.makedirs(output_dir, exist_ok=True)
    files = [
        f
        for f in os.listdir(input_dir)
        if f.endswith(".py") and os.path.isfile(os.path.join(input_dir, f))
    ]

    all_annotations = []
    per_file_stats = []

    print(f"🔍 Found {len(files)} Python files in {input_dir}")

    for file in files:
        source_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file + ".json")

        if not overwrite and os.path.exists(output_path):
            print(f"⏭️ Skipping (already annotated): {file}")
            continue

        try:
            with open(source_path, "r", encoding="utf-8") as f:
                code = f.read()

            print(f"🧠 Annotating: {file} (stream={stream})")
            annotated_code = annotate_code_with_together_ai(code, stream=stream)

            save_path = save_annotation_json(
                annotated_code, source_path=file, output_dir=output_dir
            )
            print(f"✅ Saved to: {save_path}")

            parsed = parse_annotated_code(annotated_code)
            all_annotations.extend(parsed)

            # ⬇️ Per-file stats
            types = count_annotations_by_type(parsed)
            severity = severity_breakdown_by_type(parsed)
            confidence_avg = (
                round(sum(a.get("confidence", 1.0) for a in parsed) / len(parsed), 4)
                if parsed
                else 0.0
            )
            per_file_stats.append(
                {
                    "file": file,
                    "count": len(parsed),
                    "confidence": confidence_avg,
                    **types,
                    **{
                        f"{k}_{s}": severity[k][s]
                        for k in severity
                        for s in severity[k]
                    },
                }
            )

        except Exception as e:
            print(f"❌ Error annotating {file}: {e}")

    # 📦 Write merged summary JSON
    summary_path = os.path.join(output_dir, SUMMARY_JSON)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, indent=2)
    print(f"📁 Merged annotations saved to: {summary_path}")

    # 🧮 Write per-file stats CSV
    if per_file_stats:
        csv_path = os.path.join(output_dir, SUMMARY_CSV)
        with open(csv_path, "w", encoding="utf-8", newline="") as cf:
            fieldnames = sorted(set(k for row in per_file_stats for k in row))
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_file_stats:
                writer.writerow(row)
        print(f"📊 Per-file stats saved to: {csv_path}")

    # 📢 Print global stats
    print("\n📊 Global Stats:")
    print(f"Total files annotated: {len(per_file_stats)}")
    print(f"Total annotations: {len(all_annotations)}")

    type_total = Counter()
    severity_total = Counter()
    for ann in all_annotations:
        type_total[ann["type"]] += 1
        sev = ann.get("severity", "Medium").capitalize()
        severity_total[sev] += 1

    for t, c in type_total.items():
        print(f" - {t}: {c}")
    for s, c in severity_total.items():
        print(f" - Severity {s}: {c}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate annotated dataset using Together.ai"
    )
    parser.add_argument(
        "--input-dir", default=INPUT_DIR, help="Directory of source .py files"
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Output folder for annotation .json files",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing annotation files"
    )
    parser.add_argument(
        "--stream", action="store_true", help="Use Together.ai streaming mode"
    )
    args = parser.parse_args()

    generate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        stream=args.stream,
    )


if __name__ == "__main__":
    main()
