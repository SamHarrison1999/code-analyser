
import json
from pathlib import Path

def convert_annotations_to_supervised(input_dir: Path):
    input_dir = Path(input_dir)
    annotation_files = list(input_dir.rglob("*_annotations.json"))
    if not annotation_files:
        print(f"❌ No _annotations.json files found in {input_dir}")
        return

    print(f"🔄 Converting {len(annotation_files)} files...")
    for file in annotation_files:
        try:
            with open(file, encoding="utf-8") as f:
                data = json.load(f)

            # Sanity check: should be a list of annotation dicts
            if not isinstance(data, list):
                print(f"⚠️ Skipped (not a list): {file.name}")
                continue

            out_path = file.with_name(file.name.replace("_annotations.json", ".supervised.json"))
            with open(out_path, "w", encoding="utf-8") as out_f:
                json.dump(data, out_f, indent=2)

            print(f"✅ Saved: {out_path.name}")
        except Exception as e:
            print(f"❌ Error processing {file.name}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert *_annotations.json to *.supervised.json")
    parser.add_argument("--input-dir", required=True, help="Directory containing *_annotations.json files")
    args = parser.parse_args()

    convert_annotations_to_supervised(Path(args.input_dir))
