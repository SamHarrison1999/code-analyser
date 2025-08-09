# code_analyser/src/ml/model_tf.py
import sys

print("✅ sys.path:", sys.path)

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoConfig
import torch

print("✅ torch loaded from:", torch.__file__)
print("✅ hasattr(torch, '__version__'):", hasattr(torch, "__version__"))
print("✅ torch.__version__:", getattr(torch, "__version__", "MISSING"))
import torch.nn as nn


class AnnotationClassifier(nn.Module):
    """
    Code annotation classifier based on a HuggingFace transformer.
    Predicts:
    - ⚠️ SAST Risk
    - 🧠 ML Signal
    - ✅ Best Practice
    """

    def __init__(self, model_name="microsoft/codebert-base", num_labels=3, dropout: float = 0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        config.hidden_dropout_prob = dropout
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def predict(self, input_ids, attention_mask):
        """
        Runs a forward pass and returns predicted class and confidence.

        Returns:
            dict with keys: logits, pred, confidence
        """
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred].item()
        return {"logits": logits, "pred": pred, "confidence": confidence}


def load_tokenizer(model_name="microsoft/codebert-base"):
    """
    Loads tokenizer matching model.

    Args:
        model_name (str): HF model name.
    Returns:
        Tokenizer instance.
    """
    return AutoTokenizer.from_pretrained(model_name)
