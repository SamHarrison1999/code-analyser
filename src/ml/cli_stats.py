# ✅ CLI tool to print label + severity distributions
# 🧠 ML Signal: Helps verify annotation coverage and label diversity

from code_analyser.src.ml.dataset_loader import load_local_annotated_dataset
from code_analyser.src.ml.config import DATA_PATHS, MODEL_CONFIG


def print_stats():
    _, stats = load_local_annotated_dataset(
        code_dir=DATA_PATHS["code_dir"],
        annotation_dir=DATA_PATHS["annotation_dir"],
        tokenizer_name=MODEL_CONFIG["model_name"],
        confidence_threshold=0.7,
    )

    print("\n📊 Annotated Dataset Statistics")
    print("-------------------------------")
    print(f"Files Loaded: {stats['total_files']}")
    print("Label Distribution:")
    for label, count in stats["label_counts"].items():
        print(f"  - {label}: {count}")
    print("\nSeverity Distribution:")
    for sev, count in stats["severity_counts"].items():
        print(f"  - {sev.capitalize()}: {count}")
    print(f"\nToken Spans Collected: {stats['span_count']}\n")


if __name__ == "__main__":
    print_stats()
