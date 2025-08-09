from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from ml.config import TRAINING_CONFIG, MODEL_CONFIG
from ml.model_torch import load_model


def to_tensor_batch(data):
    def pad_and_stack(key, dtype):
        return torch.stack([torch.tensor(item, dtype=dtype) for item in data[key]])

    input_ids = pad_and_stack("input_ids", torch.long)
    attention_mask = pad_and_stack("attention_mask", torch.long)
    labels = pad_and_stack("labels", torch.float)

    if labels.ndim == 2 and labels.shape[0] != input_ids.shape[0]:
        labels = labels.T  # transpose if shape is [num_labels, batch_size]

    print(f"Fixed labels shape: {labels.shape}")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def train_model(train_dataset, val_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Load model
    model = load_model(
        model_name=MODEL_CONFIG["model_name"],
        use_hf=TRAINING_CONFIG["use_hf"],
        hidden_size=MODEL_CONFIG["hidden_size"],
        num_labels=MODEL_CONFIG["num_labels"],
    ).to(device)

    # Load config
    batch_size = TRAINING_CONFIG["batch_size"]
    learning_rate = TRAINING_CONFIG["learning_rate"]
    epochs = TRAINING_CONFIG["epochs"]
    use_tensorboard = TRAINING_CONFIG.get("use_tensorboard", False)
    output_dir = Path(TRAINING_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    writer = SummaryWriter(log_dir=str(output_dir / "runs")) if use_tensorboard else None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in train_loader:
            batch = to_tensor_batch(batch)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model({"input_ids": input_ids, "attention_mask": attention_mask})

            # Debug shapes (optional)
            print("logits:", logits.shape, "labels:", labels.shape)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits).detach().cpu().numpy() > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average="micro")

        print(
            f"🟢 Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}"
        )
        if writer:
            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("F1/train", train_f1, epoch)

        # Evaluation
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = to_tensor_batch(batch)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model({"input_ids": input_ids, "attention_mask": attention_mask})
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(int)
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="micro")

        print(
            f"✅ Epoch {epoch+1}/{epochs} | Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}"
        )
        if writer:
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("F1/val", val_f1, epoch)

        checkpoint_path = output_dir / f"model_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Saved model checkpoint to: {checkpoint_path}")

    if writer:
        writer.close()
