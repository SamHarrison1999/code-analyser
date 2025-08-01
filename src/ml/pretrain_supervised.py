import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    classification_report,
)
import csv
import json
from huggingface_hub import create_repo, upload_folder

from ml.model_torch import load_model
from ml.dataset_loader import load_local_annotated_dataset
from ml.config import MODEL_CONFIG, TRAINING_CONFIG, DATA_PATHS


class AnnotatedDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __getitem__(self, idx):
        item = self.entries[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["labels"], dtype=torch.float),
        }

    def __len__(self):
        return len(self.entries)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids)
            preds = (outputs > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(
        all_labels, all_preds, output_dict=True, zero_division=0
    )
    macro = report["macro avg"]
    return {
        "accuracy": report["accuracy"],
        "precision": macro["precision"],
        "recall": macro["recall"],
        "f1": macro["f1-score"],
        "report": report,
    }


def upload_to_huggingface(output_dir: str):
    token = os.getenv("HF_TOKEN")
    repo_id = os.getenv("HF_REPO")
    if not token or not repo_id:
        print("⚠️ HF_TOKEN or HF_REPO missing from environment.")
        return

    try:
        create_repo(repo_id, token=token, repo_type="model", exist_ok=True)
        upload_folder(
            repo_id=repo_id, folder_path=output_dir, token=token, repo_type="model"
        )
        print(f"🚀 Model uploaded to HuggingFace: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ HF upload failed: {e}")


def train_supervised():
    entries, stats = load_local_annotated_dataset(
        code_dir=DATA_PATHS["code_dir"],
        annotation_dir=DATA_PATHS["annotation_dir"],
        tokenizer_name=MODEL_CONFIG["model_name"],
        max_samples=TRAINING_CONFIG["max_train_samples"],
        confidence_threshold=TRAINING_CONFIG["confidence_threshold"],
        stratify=TRAINING_CONFIG["stratify"],
        seed=TRAINING_CONFIG["seed"],
    )

    print(f"📊 Loaded {len(entries)} training samples")
    print(f"📈 Training set label distribution: {stats['label_counts']}")
    print(f"⚠️ Severity distribution: {stats['severity_counts']}")
    print(f"📌 Span count: {stats['span_count']}")

    if not entries:
        print("❌ No annotated training data found.")
        return

    dataset = AnnotatedDataset(entries)
    dataloader = DataLoader(
        dataset, batch_size=TRAINING_CONFIG["batch_size"], shuffle=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        distilled=MODEL_CONFIG["use_distilled"],
        use_hf=MODEL_CONFIG["use_hf"],
        use_attention=MODEL_CONFIG["use_attention"],
        vocab_size=MODEL_CONFIG["vocab_size"],
        embed_dim=MODEL_CONFIG["embed_dim"],
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        output_dim=MODEL_CONFIG["output_dim"],
        dropout=MODEL_CONFIG["dropout"],
        hf_model_name=MODEL_CONFIG["model_name"],
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])

    writer = SummaryWriter(log_dir="runs/supervised")

    log_dir = TRAINING_CONFIG["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    json_path = os.path.join(log_dir, "training_metrics.json")
    csv_path = os.path.join(log_dir, "training_metrics.csv")

    metrics_log = []
    with open(csv_path, "w", newline="") as csvfile:
        writer_csv = csv.DictWriter(
            csvfile,
            fieldnames=["epoch", "loss", "accuracy", "precision", "recall", "f1"],
        )
        writer_csv.writeheader()

        for epoch in range(TRAINING_CONFIG["epochs"]):
            model.train()
            total_loss = 0.0
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            metrics = evaluate(model, dataloader, device)
            result = {
                "epoch": epoch + 1,
                "loss": round(total_loss, 4),
                **{k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)},
            }

            print(
                f"🔁 Epoch {result['epoch']} - Loss: {result['loss']} | F1: {result['f1']:.3f}"
            )
            writer.add_scalar("Loss", result["loss"], epoch)
            writer.add_scalar("F1", result["f1"], epoch)
            writer.add_scalar("Accuracy", result["accuracy"], epoch)

            metrics_log.append(result)
            writer_csv.writerow(result)

    writer.close()

    # ✅ Save model checkpoint
    save_path = f"{TRAINING_CONFIG['output_dir']}/supervised_epoch_{TRAINING_CONFIG['epochs']}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")

    with open(json_path, "w") as jf:
        json.dump(metrics_log, jf, indent=2)
    print(f"📁 Metrics written to {json_path} and {csv_path}")

    # ✅ Optional HuggingFace upload
    upload_to_huggingface(TRAINING_CONFIG["output_dir"])


if __name__ == "__main__":
    train_supervised()
