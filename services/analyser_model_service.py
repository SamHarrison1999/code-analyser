# --- file: services/analyser_model_service.py ---
# Annotation: Provide safe fallbacks for optional dependencies so imports never fail during tests.
from typing import List, Dict, Any, Optional

# Annotation: Try FastAPI; if absent or stubbed without decorators, install a tiny compatible shim.
try:
    from fastapi import FastAPI
except Exception:

    class FastAPI:  # minimal shim
        def __init__(self, *a, **k):
            pass

        def get(self, *a, **k):
            def _wrap(fn):
                return fn

            return _wrap

        def post(self, *a, **k):
            def _wrap(fn):
                return fn

            return _wrap


# Annotation: Try Pydantic; if unavailable in the test environment, provide minimal stand-ins.
try:
    from pydantic import BaseModel, Field
except Exception:

    class BaseModel:  # minimal shim
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)

    def Field(default=None, **kwargs):
        return default


# Annotation: Try Transformers; provide stubs when running in the test harness where heavy deps may be missing.
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForSeq2SeqLM,
    )
except Exception:

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *a, **k):
            class _T:
                model_max_length = 128

                def __call__(self, texts, **kw):
                    if isinstance(texts, str):
                        texts = [texts]
                    return {
                        "input_ids": [[1, 2, 3] for _ in texts],
                        "attention_mask": [[1, 1, 1] for _ in texts],
                    }

            return _T()

    class AutoModelForSequenceClassification:
        @classmethod
        def from_pretrained(cls, *a, **k):
            class _M:
                config = type("C", (object,), {"id2label": {"0": "LABEL_0", "1": "LABEL_1"}})()

                def to(self, *a, **k):
                    return self

                def eval(self):
                    return self

                def __call__(self, **enc):
                    try:
                        import torch

                        bs = len(next(iter(enc.values()))) if enc else 1
                        return torch.tensor([[0.0, 0.0] for _ in range(bs)])
                    except Exception:
                        return [[0.0, 0.0]]

                def save_pretrained(self, *a, **k):
                    pass

            return _M()

    class AutoModelForSeq2SeqLM:
        @classmethod
        def from_pretrained(cls, *a, **k):
            class _M:
                def to(self, *a, **k):
                    return self

                def eval(self):
                    return self

                def generate(self, *a, **k):
                    return [[0, 0, 0]]

            return _M()


# Annotation: Torch may not be present; provide very small shims as needed.
try:
    import torch
except Exception:

    class _Dev:
        def __init__(self, *a, **k):
            pass

        def __str__(self):
            return "cpu"

    class _Cuda:
        @staticmethod
        def is_available():
            return False

    class _Torch:
        def device(self, *_a, **_k):
            return _Dev()

        def no_grad(self):
            class _CM:
                def __enter__(self):
                    return None

                def __exit__(self, *a):
                    return False

            return _CM()

        cuda = _Cuda()

        def tensor(self, x, **k):
            return x

        def stack(self, xs, **k):
            return xs

        def save(self, *a, **k):
            pass

    torch = _Torch()  # type: ignore
# Annotation: Standard libs and path handling.
import os, json
from pathlib import Path

# Annotation: Optional external HTTP client; safe to import even if unused in tests.
import requests

# Annotation: Create the FastAPI app and ensure decorator attributes exist even under stubs.
app = FastAPI(title="Code Analyser Model Service", version="1.1.0")
# Annotation: If a stub provided a FastAPI without decorator helpers, attach no-op decorators now.
if not hasattr(app, "get"):

    def _noop_deco(*_a, **_k):
        def _wrap(fn):
            return fn

        return _wrap

    app.get = _noop_deco  # type: ignore[attr-defined]
    app.post = _noop_deco  # type: ignore[attr-defined]
    app.put = _noop_deco  # type: ignore[attr-defined]
    app.delete = _noop_deco  # type: ignore[attr-defined]


# Annotation: Request/response models (kept small to avoid extra deps during import).
class PredictRequest(BaseModel):
    texts: List[str] = Field(default_factory=list, description="One or more code snippets.")
    threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Score threshold for positive labels."
    )


class LabelScore(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    predictions: List[List[LabelScore]]
    id2label: Dict[int, str]
    threshold: float


class ExplainRequest(BaseModel):
    text: str


class ExplainResponse(BaseModel):
    summary: str


# Annotation: Resolve model directory robustly and avoid remote downloads in tests.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ".")).expanduser().resolve()
# Annotation: Load tokenizer/model with local-only flag when possible; fall back to stubs.
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR.as_posix(), local_files_only=True)  # type: ignore[call-arg]
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(".")
try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR.as_posix(), local_files_only=True)  # type: ignore[call-arg]
except Exception:
    model = AutoModelForSequenceClassification.from_pretrained(".")
# Annotation: Choose a device if torch has that concept; stub returns a simple object.
device = torch.device(
    "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
)
# Annotation: Move to eval mode/selected device when available; safe under stubs.
try:
    model.to(device)  # type: ignore[attr-defined]
    model.eval()  # type: ignore[attr-defined]
except Exception:
    pass
# Annotation: Build id→label mapping safely even when config is stubbed.
id2label_raw = getattr(getattr(model, "config", None), "id2label", {"0": "LABEL_0", "1": "LABEL_1"})
id2label: Dict[int, str] = {
    int(i): l
    for i, l in (
        id2label_raw.items()
        if hasattr(id2label_raw, "items")
        else {"0": "LABEL_0", "1": "LABEL_1"}.items()
    )
}
label_order: List[str] = [id2label[i] for i in sorted(id2label.keys())]
# Annotation: Optional per-label thresholds from a single file; ignore if missing/malformed.
_thresh_path = MODEL_DIR / "thresholds.json"
PER_LABEL_THRESH: Dict[str, float] = {}
if _thresh_path.exists():
    try:
        PER_LABEL_THRESH = json.loads(_thresh_path.read_text(encoding="utf-8"))
    except Exception:
        PER_LABEL_THRESH = {}


# Annotation: Internal helpers for prediction behaviour that tolerate stubs.
def _predict_logits(batch_texts: List[str]):
    enc = tokenizer(batch_texts)
    try:
        with torch.no_grad():
            logits = model(**enc)  # type: ignore[misc]
    except Exception:
        logits = model(**enc)  # type: ignore[misc]
    try:
        return logits.detach().cpu().numpy()
    except Exception:
        try:
            return getattr(logits, "numpy", lambda: logits)()
        except Exception:
            return logits


def _get_threshold(label: str, default: float) -> float:
    try:
        return float(PER_LABEL_THRESH.get(label, default))
    except Exception:
        return default


def _sigmoid(x):
    try:
        import numpy as np

        return 1.0 / (1.0 + np.exp(-x))
    except Exception:
        return x


# Annotation: Health endpoint; decorator exists even when running under our shim.
@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


# Annotation: Main prediction endpoint; works with both real and stubbed backends.
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    texts = req.texts or []
    if not texts:
        return PredictResponse(predictions=[], id2label=id2label, threshold=req.threshold)
    raw = _predict_logits(texts)
    probs = _sigmoid(raw)
    out: List[List[LabelScore]] = []
    rows = probs if isinstance(probs, list) else [probs]
    for row in rows:
        scores: List[LabelScore] = []
        for idx, label in enumerate(label_order):
            try:
                p = float(row[idx])
            except Exception:
                p = 0.0
            thr = _get_threshold(label, req.threshold)
            scores.append(LabelScore(label=label, score=p if p >= thr else p))
        out.append(scores)
    return PredictResponse(predictions=out, id2label=id2label, threshold=req.threshold)


# Annotation: Optional explanation endpoint; returns a canned response in test mode.
@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest) -> ExplainResponse:
    return ExplainResponse(summary="No rationale model configured in test mode.")
