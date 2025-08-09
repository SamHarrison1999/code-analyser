import os
from collections import Counter

from datasets import load_from_disk
from ml.config import DATA_PATHS, TRAINING_CONFIG, MODEL_CONFIG
from ml.dataset_loader import load_local_annotated_dataset
from ml.train_model import train_model


def main():
    data_dir = DATA_PATHS.get("processed_dataset", "datasets/processed")

    if os.path.exists(os.path.join(data_dir, "train")):
        print(f"📦 Loading dataset from disk: {data_dir}")
        dataset_dict = load_from_disk(data_dir)
        train_dataset = dataset_dict["train"]
        val_dataset = dataset_dict["val"]
        stats = {
            "total_files": len(train_dataset),
            "label_counts": Counter(),
            "severity_counts": Counter(),
            "span_count": 0,
        }

        for item in train_dataset:
            for i, v in enumerate(item["labels"]):
                if v:
                    label_name = list(DATA_PATHS["label_map"].keys())[i]
                    stats["label_counts"][label_name] += 1
            stats["span_count"] += len(item.get("spans", []))

    else:
        print("⚠️ No preprocessed dataset found, falling back to raw annotation loading.")
        entries, stats = load_local_annotated_dataset(
            code_dir=DATA_PATHS["code_dir"],
            annotation_dir=DATA_PATHS["annotation_dir"],
            tokenizer_name=MODEL_CONFIG["model_name"],
            max_samples=TRAINING_CONFIG["max_train_samples"],
            confidence_threshold=TRAINING_CONFIG["confidence_threshold"],
            stratify=TRAINING_CONFIG["stratify"],
            seed=TRAINING_CONFIG["seed"],
        )
        train_dataset = entries
        val_dataset = []

    print(f"📊 Total accepted examples: {stats['total_files']}")
    print(f"📈 Training set label distribution: {stats['label_counts']}")
    print(f"📌 Span count: {stats['span_count']}")

    if not train_dataset:
        print("❌ No annotated training data found.")
        return

    train_model(train_dataset, val_dataset)


if __name__ == "__main__":
    main()
