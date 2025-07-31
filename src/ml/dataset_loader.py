import os
import json
from typing import List, Dict, Tuple
from collections import Counter

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# ✅ Define consistent labels for multi-label classification
LABEL_MAP = {"SAST Risk": 0, "ML Signal": 1, "Best Practice": 2}


def extract_labels(
    annotations: List[Dict], confidence_threshold: float = 0.7
) -> Tuple[List[int], List[str], List[Tuple[int, int]]]:
    """
    Extract multi-label binary vector, severity levels, and token spans from annotations.
    """
    label_flags = [0] * len(LABEL_MAP)
    severities = []
    token_spans = []

    for entry in annotations:
        conf = entry.get("confidence", 1.0)
        if conf < confidence_threshold:
            continue

        for label in LABEL_MAP:
            if label in entry.get("content", ""):
                label_flags[LABEL_MAP[label]] = 1

        if "severity" in entry:
            severities.append(entry["severity"].lower())

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
    """
    Load annotated examples, return dataset + stats (labels, severity, etc).
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = []
    stats = {
        "total_files": 0,
        "label_counts": Counter(),
        "severity_counts": Counter(),
        "span_count": 0,
    }

    filenames = [
        f
        for f in os.listdir(code_dir)
        if f.endswith(".py")
        and os.path.exists(os.path.join(annotation_dir, f + ".json"))
    ]

    if max_samples:
        filenames = filenames[:max_samples]

    for filename in filenames:
        with open(os.path.join(code_dir, filename), "r", encoding="utf-8") as f:
            code = f.read()

        with open(
            os.path.join(annotation_dir, filename + ".json"), "r", encoding="utf-8"
        ) as f:
            annotations = json.load(f).get("annotations", [])

        labels, severities, spans = extract_labels(annotations, confidence_threshold)
        if sum(labels) == 0:
            continue

        tokenised = tokenizer(
            code, truncation=True, padding="max_length", max_length=max_length
        )

        dataset.append(
            {
                "input_ids": tokenised["input_ids"],
                "attention_mask": tokenised["attention_mask"],
                "labels": labels,
                "spans": spans,
                "filename": filename,
            }
        )

        stats["total_files"] += 1
        for i, v in enumerate(labels):
            if v:
                stats["label_counts"][list(LABEL_MAP.keys())[i]] += 1
        for s in severities:
            stats["severity_counts"][s] += 1
        stats["span_count"] += len(spans)

    # ✅ Only apply stratification if possible
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
