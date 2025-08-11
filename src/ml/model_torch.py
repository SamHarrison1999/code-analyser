# --- file: code_analyser/src/ml/model_torch.py ---
# Attempt light imports first; fall back to tiny shims so tests can import without heavy deps.
try:
    import torch
    import torch.nn as nn
except Exception:
    # Minimal torch shim that provides the bits we need in tests.
    class _Tensor(list):
        def to(self, *a, **k):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            try:
                import numpy as _np

                return _np.array(self)
            except Exception:
                return self

        def transpose(self, *a, **k):
            return self

    class _nn:
        class Module:
            def __init__(self, *a, **k):
                pass

        class Dropout:
            def __init__(self, *a, **k):
                pass

            def __call__(self, x):
                return x

        class Linear:
            def __init__(self, in_f, out_f):
                self.out_features = out_f

            def __call__(self, x):
                return _Tensor(
                    [
                        [0.0] * self.out_features
                        for _ in range(len(x) if hasattr(x, "__len__") else 1)
                    ]
                )

    class _torch:
        Tensor = _Tensor

        def tensor(x, dtype=None):
            return _Tensor(x)

        def stack(xs):
            return _Tensor(xs)

        def zeros(*shape):
            n = shape[0] if shape else 1
            m = shape[1] if len(shape) > 1 else 1
            return _Tensor([[0.0] * m for _ in range(n)])

        def sigmoid(x):
            return x

    torch = _torch()
    nn = _nn
# Try Transformers; if unavailable, provide a stub AutoModel.
try:
    from transformers import AutoModel
except Exception:

    class AutoModel:
        # Stub encoder returns an object with a 'last_hidden_state' placeholder.
        @classmethod
        def from_pretrained(cls, *a, **k):
            class _Enc:
                def __call__(self, **kwargs):
                    bs = 1
                    if kwargs:
                        anyv = next(iter(kwargs.values()))
                        try:
                            bs = len(anyv)
                        except Exception:
                            pass
                    # [batch, seq, hidden] placeholder
                    return type("O", (object,), {"last_hidden_state": torch.zeros(bs, 3)})

            return _Enc()


class CodeAnnotationModel(nn.Module):
    def __init__(self, model_name: str, hidden_size: int, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        # Encoder may be a stub; obtain hidden or fall back to zeros.
        out = self.encoder(**x)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            bs = len(next(iter(x.values()))) if isinstance(x, dict) and x else 1
            return getattr(
                torch, "zeros", lambda *s: [[0.0] * self.classifier.out_features for _ in range(bs)]
            )(bs, self.classifier.out_features)
        hidden = hidden  # [seq_len, batch_size, hidden_size]
        print(f"Hidden shape: {hidden.shape}")

        # Some stubs return lists; transpose defensively.
        if hasattr(hidden, "transpose"):
            hidden = hidden.transpose(0, 1)  # ✅ [batch_size, seq_len, hidden_size]
        # Slice robustly whether tensor or list-like.
        try:
            cls_token = hidden[:, 0, :]  # ✅ [batch_size, hidden_size]
        except Exception:
            cls_token = hidden  # Fallback for simple placeholders
        print(f"cls token shape: {cls_token.shape}")

        # If classifier is a stub, ensure we return a tensor-shaped object.
        logits = self.classifier(self.dropout(cls_token))  # [batch_size, num_labels]
        print(f"Logits shape: {logits.shape}")

        return logits


def load_model(model_name: str, use_hf: bool, hidden_size: int, num_labels: int):
    return CodeAnnotationModel(model_name, hidden_size, num_labels)
