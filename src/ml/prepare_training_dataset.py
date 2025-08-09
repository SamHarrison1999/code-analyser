import argparse
import json
import os
from collections import Counter
from typing import List, Dict, Tuple

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

LABEL_MAP = {"sast_risk": 0, "ml_signal": 1, "best_practice": 2}


def extract_labels(
        annotations: List[Dict], confidence_threshold: float = 0.7
) -> Tuple[List[int], List[str], List[Tuple[int, int]]]:
    label_flags = [0] * len(LABEL_MAP)
    severities = []
    token_spans = []

    for entry in annotations:
        print(f"🔎 ENTRY: {entry}")
        conf = entry.get("confidence", 1.0)
        label = entry.get("label", "").lower()

        print(f"      → Label: {label}, Confidence: {conf}")

        if conf < confidence_threshold:
            print("        ⛔ Below threshold, skipped.")
            continue

        if label not in LABEL_MAP:
            print("        ⚠️ Unknown label, skipping.")
        else:
            label_flags[LABEL_MAP[label]] = 1
            print(f"        ✅ Mapped label: {label} → {LABEL_MAP[label]}")

        if "severity" in entry:
            severities.append(entry["severity"].lower())

        if isinstance(entry.get("span"), list) and len(entry["span"]) == 2:
            token_spans.append(tuple(entry["span"]))

    return label_flags, severities, token_spans


def load_local_annotated_dataset(
        code_dir: str = "datasets/github_fintech",
        annotation_dir: str = "datasets/annotated_fintech",
        tokenizer_name: str = "microsoft/codebert-base",
        max_samples: int = None,
        max_length: int = 512,
        confidence_threshold: float = 0.7,
        stratify: bool = True,
        seed: int = 42,
) -> Tuple[List[Dict], Dict]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = []
    stats = {
        "total_files": 0,
        "label_counts": Counter(),
        "severity_counts": Counter(),
        "span_count": 0,
    }

    py_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(code_dir)
        for file in files if file.endswith(".py")
    ]
    print(f"🧪 Found {len(py_paths)} .py files in: {code_dir}")

    if max_samples:
        py_paths = py_paths[:max_samples]

    for py_path in py_paths:
        rel_path = os.path.relpath(py_path, code_dir)
        filename = os.path.splitext(os.path.basename(py_path))[0]
        annot_path = os.path.join(annotation_dir, rel_path).replace(".py", "_annotations.json")

        print(f"\n🔍 Checking: {py_path}")
        print(f"    ↪ rel_path: {rel_path}")
        print(f"    ↪ annot_path: {annot_path}")

        if not os.path.exists(annot_path):
            print(f"    ❌ Missing annotation file: {annot_path}")
            continue

        try:
            with open(py_path, "r", encoding="utf-8") as f:
                code = f.read()
            with open(annot_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)
        except Exception as e:
            print(f"    ❌ Error reading files: {e}")
            continue

        print(f"    📦 Loaded {len(annotations)} annotations")

        labels, severities, spans = extract_labels(annotations, confidence_threshold)
        if sum(labels) == 0:
            print("    ⚠️ No high-confidence labels, skipping.")
            continue

        tokenised = tokenizer(code, truncation=True, padding="max_length", max_length=max_length)

        dataset.append({
            "input_ids": tokenised["input_ids"],
            "attention_mask": tokenised["attention_mask"],
            "labels": labels,
            "spans": spans,
            "filename": filename,
        })

        stats["total_files"] += 1
        for i, v in enumerate(labels):
            if v:
                stats["label_counts"][list(LABEL_MAP.keys())[i]] += 1
        for s in severities:
            stats["severity_counts"][s] += 1
        stats["span_count"] += len(spans)

    print(f"\n✅ Loaded {len(dataset)} annotated examples")
    print(f"📊 Stats: {stats}")

    if stratify and len(dataset) >= 10:
        X, y = [], []
        for d in dataset:
            X.append(d)
            y.append(tuple(d["labels"]))
        try:
            _, stratified = train_test_split(
                X, test_size=0.9, stratify=y, random_state=seed
            )
            return stratified, stats
        except ValueError as e:
            print(f"⚠️ Stratified split failed: {e}. Returning full dataset.")
            return dataset, stats
    else:
        return dataset, stats


def save_dataset(train_data, val_data, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "val": val_dataset
    })

    dataset_dict.save_to_disk(output_dir)
    print(f"✅ Saved dataset to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", type=str, default="datasets/github_fintech")
    parser.add_argument("--annotation-dir", type=str, default="datasets/annotated_fintech")
    parser.add_argument("--output-dir", type=str, default="datasets/processed")
    parser.add_argument("--tokenizer", type=str, default="microsoft/codebert-base")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset, stats = load_local_annotated_dataset(
        code_dir=args.code_dir,
        annotation_dir=args.annotation_dir,
        tokenizer_name=args.tokenizer,
        max_samples=args.max_samples,
        max_length=args.max_length,
        confidence_threshold=args.confidence_threshold,
        stratify=args.stratify,
        seed=args.seed
    )

    if len(dataset) < 2:
        print("❌ Not enough examples to split or save.")
    else:
        train, val = train_test_split(dataset, test_size=0.2, random_state=args.seed)
        save_dataset(train, val, args.output_dir)
