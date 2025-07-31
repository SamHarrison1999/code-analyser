import json
import argparse
from pathlib import Path
from collections import Counter
import pandas as pd

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def generate_summary(output_dir: Path, use_tensorboard: bool):
    summary_rows = []
    log_dir = output_dir / "logs"
    writer = (
        SummaryWriter(log_dir=str(log_dir))
        if use_tensorboard and TENSORBOARD_AVAILABLE
        else None
    )

    supervised_files = list(output_dir.rglob("*.supervised.json"))
    for file in supervised_files:
        with open(file, encoding="utf-8") as f:
            data = json.load(f)

        file_name = file.stem.replace(".supervised", "")
        count_by_label = Counter(d["label"] for d in data)
        count_by_severity = Counter(d["severity"] for d in data if d.get("severity"))
        confidences = [d["confidence"] for d in data]

        summary = {
            "file": file_name,
            "annotations": len(data),
            "avg_confidence": (
                round(sum(confidences) / len(confidences), 4) if confidences else 0.0
            ),
            "sast_risks": count_by_label.get("SAST_RISK", 0),
            "ml_signals": count_by_label.get("ML_SIGNAL", 0),
            "best_practices": count_by_label.get("BEST_PRACTICE", 0),
            "high_severity": count_by_severity.get("High", 0),
            "medium_severity": count_by_severity.get("Medium", 0),
            "low_severity": count_by_severity.get("Low", 0),
        }

        summary_rows.append(summary)

        if writer:
            writer.add_scalar(f"{file_name}/avg_confidence", summary["avg_confidence"])
            writer.add_scalar(f"{file_name}/sast_risks", summary["sast_risks"])
            writer.add_scalar(f"{file_name}/ml_signals", summary["ml_signals"])
            writer.add_scalar(f"{file_name}/best_practices", summary["best_practices"])

    df = pd.DataFrame(summary_rows)
    csv_path = output_dir / "annotation_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Summary CSV saved: {csv_path}")

    if writer:
        writer.flush()
        writer.close()
        print(f"📈 TensorBoard logs saved to: {log_dir}")
    elif use_tensorboard:
        print("⚠️ TensorBoard not available. Run: pip install torch tensorboard")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--tensorboard", action="store_true", help="Enable TensorBoard logging"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return

    generate_summary(output_dir, use_tensorboard=args.tensorboard)


if __name__ == "__main__":
    main()
