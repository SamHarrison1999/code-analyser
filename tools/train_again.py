# Annotation: Standard imports for argument parsing, CSV IO, filesystem paths, reproducibility and light maths.
import argparse, csv, os, random

# Annotation: NumPy is used for simple array maths (class weights).
import numpy as np

# Annotation: Torch powers tensors, loss functions and the training loop via the HF Trainer.
import torch

# Annotation: Transformers provides the tokenizer, model and Trainer scaffolding.
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# Annotation: Torch Dataset is used for a tiny explicit dataset wrapper.
from torch.utils.data import Dataset

# Annotation: dataclass keeps the custom collator tidy.
from dataclasses import dataclass

# Annotation: Fixed canonical order must match your service/config and training labels.
LABEL_ORDER = ["sast_risk", "ml_signal", "best_practice"]


# Annotation: Tiny deterministic seed helper so runs are comparable.
def set_seed(seed: int = 42) -> None:
    # Annotation: Seed Python RNG.
    random.seed(seed)
    # Annotation: Seed NumPy RNG.
    np.random.seed(seed)
    # Annotation: Seed Torch CPU RNG (and CUDA if available).
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Annotation: Convert a semicolon-separated label string into a 3-long float multi-hot vector.
def to_multi_hot(label_str: str) -> list[float]:
    # Annotation: Split and normalise tags; ignore empties.
    tags = {t.strip() for t in (label_str or "").split(";") if t.strip()}
    # Annotation: Generate in canonical order.
    return [1.0 if lab in tags else 0.0 for lab in LABEL_ORDER]


# Annotation: Load a CSV with columns: id,text,labels and return list[{"text":..., "labels":[...]}].
def load_csv(path: str) -> list[dict]:
    # Annotation: Read rows as dictionaries with UTF-8.
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    # Annotation: Materialise multi-hot targets.
    data = [{"text": r["text"], "labels": to_multi_hot(r.get("labels", ""))} for r in rows]
    return data


# Annotation: Minimal dataset wrapper that tokenises on the fly and attaches float label vectors.
class SnippetDataset(Dataset):
    # Annotation: Store rows, tokenizer and maximum length.
    def __init__(self, rows: list[dict], tokenizer, max_length: int = 256):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length

    # Annotation: Dataset length equals number of rows.
    def __len__(self) -> int:
        return len(self.rows)

    # Annotation: Build one example with truncation; padding is deferred to the collator.
    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        enc = self.tok(r["text"], truncation=True, max_length=self.max_length)
        enc["labels"] = torch.tensor(r["labels"], dtype=torch.float32)
        return enc


# Annotation: Compute BCE 'pos_weight' = negatives/positives per class to counter label imbalance.
def compute_pos_weights(rows: list[dict], n_labels: int) -> torch.Tensor:
    pos = np.zeros(n_labels, dtype=np.float64)
    n = len(rows)
    for r in rows:
        pos += np.array(r["labels"], dtype=np.float64)
    pos = np.clip(pos, 1.0, None)
    neg = n - pos
    w = neg / pos
    return torch.tensor(w, dtype=torch.float32)


# Annotation: Custom collator that pads inputs and stacks float32 multi-label targets safely.
@dataclass
class MultiLabelCollator:
    tokenizer: any
    pad_to_multiple_of: int | None = None

    def __call__(self, features):
        # Annotation: Avoid the warning by only re-wrapping when needed.
        def to_f32_tensor(x):
            return (
                x.float() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
            )

        labels = [to_f32_tensor(f["labels"]) for f in features]
        feats = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = self.tokenizer.pad(
            feats, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
        )
        batch["labels"] = torch.stack(labels)
        return batch


# Annotation: Trainer subclass that swaps in BCEWithLogitsLoss (multi-label) with optional class weights.
class MultiLabelTrainer(Trainer):
    # Annotation: Accept a precomputed pos_weight tensor.
    def __init__(self, *args, pos_weight: torch.Tensor | None = None, **kwargs):
        self.pos_weight = pos_weight
        super().__init__(*args, **kwargs)

    # Annotation: Accept extra kwargs for compatibility with newer Trainer (e.g., num_items_in_batch).
    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        # Annotation: Pull labels and move to the same device as the model.
        labels = inputs.pop("labels").to(model.device)
        # Annotation: Forward pass; models return a tuple or a ModelOutput with 'logits'.
        outputs = model(**inputs)
        # Annotation: Extract logits.
        logits = outputs.logits
        # Annotation: Use BCEWithLogitsLoss; include pos_weight when available.
        if self.pos_weight is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(model.device))
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()
        # Annotation: Compute loss against float labels.
        loss = loss_fct(logits, labels)
        # Annotation: Respect the Trainer API for returning outputs optionally.
        return (loss, outputs) if return_outputs else loss


# Annotation: Freeze the encoder so only the classifier head trains (warm-up phase).
def freeze_encoder(model) -> None:
    for name, p in model.named_parameters():
        if any(k in name for k in ["classifier", "score", "lm_head"]):
            p.requires_grad = True
        else:
            p.requires_grad = False


# Annotation: Unfreeze everything for full fine-tuning.
def unfreeze_all(model) -> None:
    for _, p in model.named_parameters():
        p.requires_grad = True


# Annotation: Build TrainingArguments with compatibility for older Transformers (no evaluation_strategy).
def build_training_args(**kw) -> TrainingArguments:
    # Annotation: Common arguments across phases.
    common = dict(
        output_dir=kw["output_dir"],
        learning_rate=kw["lr"],
        num_train_epochs=kw["epochs"],
        per_device_train_batch_size=kw["bsz"],
        per_device_eval_batch_size=kw["bsz"],
        logging_strategy="steps",
        logging_steps=50,
        warmup_ratio=kw["warmup"],
        report_to=[],
        seed=kw["seed"],
    )
    # Annotation: Prefer modern API; fall back gracefully if unavailable.
    try:
        _ = TrainingArguments(
            output_dir="tmp_probe", evaluation_strategy="epoch", save_strategy="epoch"
        )
        extra = dict(
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    except TypeError:
        extra = dict(
            do_eval=True,
            save_steps=500,
        )
    return TrainingArguments(**common, **extra)


# Annotation: Main entry: two-phase schedule (head-only → full model) with class weighting and proper collator.
def main():
    # Annotation: CLI flags for data, model paths and hyper-parameters.
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="datasets/train/supervised.csv")
    ap.add_argument("--val_csv", type=str, default="datasets/val/val.csv")
    ap.add_argument("--model_dir", type=str, default="models/trained_model")
    ap.add_argument("--out_dir", type=str, default="checkpoints/hf_retrained")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--head_epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    # Annotation: Make runs deterministic.
    set_seed(args.seed)
    # Annotation: Load data.
    train_rows = load_csv(args.train_csv)
    val_rows = load_csv(args.val_csv)
    # Annotation: Load tokenizer/model from the HF artefact you already exported (config has problem_type=multi_label_classification).
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    # Annotation: Build datasets (tokenisation per example, padding in collator).
    ds_train = SnippetDataset(train_rows, tok, max_length=args.max_length)
    ds_val = SnippetDataset(val_rows, tok, max_length=args.max_length)
    # Annotation: Class imbalance correction.
    pos_weight = compute_pos_weights(train_rows, n_labels=len(LABEL_ORDER))
    # Annotation: Use the custom multi-label collator for both phases.
    collator = MultiLabelCollator(tok, pad_to_multiple_of=8)
    # Annotation: Phase 1 — train head only with a slightly larger LR.
    freeze_encoder(model)
    head_out = os.path.join(args.out_dir, "phase1_head_only")
    os.makedirs(head_out, exist_ok=True)
    targs1 = build_training_args(
        output_dir=head_out,
        lr=args.head_lr,
        epochs=max(1, args.head_epochs),
        bsz=args.bsz,
        warmup=args.warmup_ratio,
        seed=args.seed,
    )
    trainer1 = MultiLabelTrainer(
        model=model,
        args=targs1,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tok,
        data_collator=collator,
        pos_weight=pos_weight,
    )
    # Annotation: Train and save phase 1.
    trainer1.train()
    trainer1.save_model(head_out)
    tok.save_pretrained(head_out)
    # Annotation: Phase 2 — unfreeze all layers and fine-tune at a conservative LR.
    unfreeze_all(model)
    full_out = os.path.join(args.out_dir, "phase2_full_ft")
    os.makedirs(full_out, exist_ok=True)
    remaining_epochs = max(1, args.epochs - args.head_epochs)
    targs2 = build_training_args(
        output_dir=full_out,
        lr=args.lr,
        epochs=remaining_epochs,
        bsz=args.bsz,
        warmup=args.warmup_ratio,
        seed=args.seed,
    )
    trainer2 = MultiLabelTrainer(
        model=model,
        args=targs2,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tok,
        data_collator=collator,
        pos_weight=pos_weight,
    )
    # Annotation: Train and save phase 2 (complete HF folder; you can set MODEL_DIR to this path to serve it).
    trainer2.train()
    trainer2.save_model(full_out)
    tok.save_pretrained(full_out)


# Annotation: Script entry point.
if __name__ == "__main__":
    main()
