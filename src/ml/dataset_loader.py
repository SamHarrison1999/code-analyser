# src/ml/dataset_loader.py
import os
import json
from typing import List, Dict, Tuple
from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Align label map with the pipeline used in prepare_training_dataset.py so that labels from JSON like "ml_signal" map correctly.
LABEL_MAP = {"sast_risk": 0, "ml_signal": 1, "best_practice": 2}

# Provide a legacy phrase map for backward compatibility when annotations only include human-readable strings.
LEGACY_PHRASE_MAP = {
    "sast_risk": ["sast risk", "⚠️ sast risk"],
    "ml_signal": ["ml signal", "🧠 ml signal"],
    "best_practice": ["best practice", "✅ best practice"],
}

def extract_labels(
    annotations: List[Dict], confidence_threshold: float = 0.7
) -> Tuple[List[int], List[str], List[Tuple[int, int]]]:
    label_flags = [0] * len(LABEL_MAP)
    severities = []
    token_spans = []
    # Iterate each annotation entry, respecting confidence where provided.
    for entry in annotations:
        conf = entry.get("confidence", 1.0)
        if conf < confidence_threshold:
            continue
        # Primary path: read normalised label from the 'label' field (e.g. 'ml_signal').
        label_key = None
        if isinstance(entry.get("label"), str):
            candidate = entry["label"].strip().lower()
            if candidate in LABEL_MAP:
                label_key = candidate
        # Fallback path: scan 'annotation' or 'content' text for legacy phrases (case-insensitive).
        if label_key is None:
            hay = " ".join(
                str(entry.get(k, "")) for k in ("annotation", "content", "reason")
            ).lower()
            for k, phrases in LEGACY_PHRASE_MAP.items():
                if any(p in hay for p in phrases):
                    label_key = k
                    break
        # If we found a label by either method, set the corresponding flag.
        if label_key is not None:
            label_flags[LABEL_MAP[label_key]] = 1
        # Track severity in lower-case if present for summary stats.
        if "severity" in entry and isinstance(entry["severity"], str):
            severities.append(entry["severity"].lower())
        # Collect token spans if a 2-tuple list is provided.
        if (
            "span" in entry
            and isinstance(entry["span"], list)
            and len(entry["span"]) == 2
        ):
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
    # Gather Python files under the code directory.
    py_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(code_dir)
        for file in files if file.endswith(".py")
    ]
    # Respect max_samples if provided for faster iteration.
    if max_samples:
        py_paths = py_paths[:max_samples]
    # Iterate files and pair them with their annotation JSON using robust path handling.
    for py_path in py_paths:
        rel_path = os.path.relpath(py_path, code_dir)
        base_no_ext, _ = os.path.splitext(rel_path)
        annot_path = os.path.join(annotation_dir, base_no_ext + "_annotations.json")
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
            print(f"    ⚠️ Error loading files: {e}")
            continue
        # Extract labels using the unified logic that supports both new and legacy formats.
        labels, severities, spans = extract_labels(annotations, confidence_threshold)
        if sum(labels) == 0:
            print("    ⚠️ No high-confidence labels, skipping.")
            continue
        # Tokenise code to fixed length for model input.
        tokenised = tokenizer(
            code, truncation=True, padding="max_length", max_length=max_length
        )
        dataset.append(
            {
                "input_ids": tokenised["input_ids"],
                "attention_mask": tokenised["attention_mask"],
                "labels": labels,
                "spans": spans,
                "filename": os.path.splitext(os.path.basename(py_path))[0],
            }
        )
        print(f"    ✅ Accepted: {len(annotations)} annotations")
        stats["total_files"] += 1
        for i, v in enumerate(labels):
            if v:
                stats["label_counts"][list(LABEL_MAP.keys())[i]] += 1
        for s in severities:
            stats["severity_counts"][s] += 1
        stats["span_count"] += len(spans)
    print(f"\n📊 Total accepted examples: {len(dataset)}")
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
