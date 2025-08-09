import argparse
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from ml.model_tf import AnnotationClassifier
from ml.dataset_loader import load_local_annotated_dataset


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
    logits = pred.predictions
    labels = pred.label_ids
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int().numpy()
    labels = labels.astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def train_on_local_data(args):
    entries, _ = load_local_annotated_dataset(
        code_dir=args.code_dir,
        annotation_dir=args.annotation_dir,
        tokenizer_name=args.model_name,
        confidence_threshold=args.confidence_threshold,
        max_samples=args.max_samples,
        stratify=False,
    )
    dataset = LocalFintechDataset(entries)
    print(f"dataset = {dataset}")

    model = AnnotationClassifier(model_name=args.model_name, num_labels=args.num_labels)
    if args.pretrained_path:
        state_dict = torch.load(args.pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir="./logs/tensorboard/",
        per_device_train_batch_size=args.batch_size,
        eval_strategy="epoch",
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
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
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", default="datasets/github_fintech")
    parser.add_argument("--annotation-dir", default="datasets/annotated_fintech")
    parser.add_argument("--output-dir", default="./checkpoints/hf")
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--num-labels", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--pretrained-path", type=str, default=None)

    args = parser.parse_args()
    train_on_local_data(args)
