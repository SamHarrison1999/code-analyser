# model_torch.py
import torch
import torch.nn as nn
from transformers import AutoModel


class CodeAnnotationModel(nn.Module):
    def __init__(self, model_name: str, hidden_size: int, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        hidden = self.encoder(**x).last_hidden_state  # [seq_len, batch_size, hidden_size]
        print(f"Hidden shape: {hidden.shape}")

        hidden = hidden.transpose(0, 1)  # ✅ [batch_size, seq_len, hidden_size]
        cls_token = hidden[:, 0, :]  # ✅ [batch_size, hidden_size]
        print(f"cls token shape: {cls_token.shape}")

        logits = self.classifier(self.dropout(cls_token))  # [batch_size, num_labels]
        print(f"Logits shape: {logits.shape}")

        return logits


def load_model(model_name: str, use_hf: bool, hidden_size: int, num_labels: int):
    return CodeAnnotationModel(model_name, hidden_size, num_labels)
