# ✅ Best Practice: Tokeniser abstraction compatible with both HuggingFace and custom embedding pipelines

import torch

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


class CodeTokenizer:
    def __init__(
        self,
        use_hf: bool = False,
        hf_model_name: str = "microsoft/codebert-base",
        vocab_size: int = 10000,
    ):
        self.use_hf = use_hf
        if self.use_hf:
            if AutoTokenizer is None:
                raise ImportError(
                    "Transformers not installed. Install with `pip install transformers`."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        else:
            self.vocab_size = vocab_size
            self.token_to_id = {"<PAD>": 0, "<UNK>": 1}
            self.id_to_token = {0: "<PAD>", 1: "<UNK>"}

    def encode(self, text: str, max_length: int = 128):
        if self.use_hf:
            return torch.tensor(
                self.tokenizer.encode(
                    text, padding="max_length", truncation=True, max_length=max_length
                )
            ).unsqueeze(0)
        else:
            tokens = text.strip().split()
            ids = [self.token_to_id.get(tok, 1) for tok in tokens]
            padded = ids[:max_length] + [0] * max(0, max_length - len(ids))
            return torch.tensor(padded).unsqueeze(0)

    def vocab_size(self):
        return self.tokenizer.vocab_size if self.use_hf else self.vocab_size
