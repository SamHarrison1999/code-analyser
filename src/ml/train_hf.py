from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
import torch

from code_analyser.src.ml.dataset_loader import load_local_annotated_dataset


class LocalFintechDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        return {
            "input_ids": torch.tensor(e["input_ids"]),
            "attention_mask": torch.tensor(e["attention_mask"]),
            "labels": torch.tensor(e["labels"], dtype=torch.float),
        }


def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int().numpy()
    labels = labels.astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def train_on_local_data(
    output_dir="./checkpoints/local_finetuned",
    model_name="microsoft/codebert-base",
    log_dir="./logs/tensorboard/",
    epochs=3,
    batch_size=8,
    lr=2e-5,
    confidence_threshold=0.7,
    max_samples=None,
):
    entries = load_local_annotated_dataset(
        tokenizer_name=model_name,
        confidence_threshold=confidence_threshold,
        max_samples=max_samples,
    )
    dataset = LocalFintechDataset(entries)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=log_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        save_steps=50,
        save_total_limit=2,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=model.config._name_or_path),
    )

    trainer.train()
    trainer.save_model(output_dir)
