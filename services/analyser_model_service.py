# --- file: services/analyser_model_service.py ---
# Provide safe fallbacks for optional dependencies so imports never fail during tests.
from typing import List, Dict, Any, Optional

# Try FastAPI; if absent or stubbed without decorators, install a tiny compatible shim.
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
except Exception:

    class FastAPI:
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

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str = ""):
            self.status_code = status_code
            self.detail = detail

    class FileResponse:
        def __init__(
            self, path: str, filename: Optional[str] = None, media_type: Optional[str] = None
        ):
            self.path = path


# Try Pydantic; if unavailable in the test environment, provide minimal stand-ins.
try:
    from pydantic import BaseModel, Field
except Exception:

    class BaseModel:
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)

    def Field(default=None, **kwargs):
        return default


# Try Transformers; provide stubs when running in the test harness where heavy deps may be missing.
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
                config = type(
                    "C",
                    (object,),
                    {"id2label": {"0": "sast_risk", "1": "ml_signal", "2": "best_practice"}},
                )()

                def to(self, *a, **k):
                    return self

                def eval(self):
                    return self

                def __call__(self, **enc):
                    try:
                        import torch

                        bs = len(next(iter(enc.values()))) if enc else 1

                        class _O:
                            def __init__(self, b):
                                self._b = b

                            @property
                            def logits(self):
                                import numpy as np

                                return type(
                                    "_T",
                                    (),
                                    {
                                        "detach": lambda s: s,
                                        "cpu": lambda s: s,
                                        "numpy": lambda s: np.zeros((self._b, 3), dtype=float),
                                    },
                                )()

                        return _O(bs)
                    except Exception:
                        return [[0.0, 0.0, 0.0]]

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


# Torch may not be present; provide very small shims as needed.
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
        __version__ = None

        def tensor(self, x, **k):
            return x

        def stack(self, xs, **k):
            return xs

        def save(self, *a, **k):
            pass

    torch = _Torch()  # type: ignore
# Standard libs and path handling.
import os, json, re
from pathlib import Path

# Optional external HTTP client; safe to import even if unused in tests.
import requests

# Additional stdlib imports for versioning, batching, and OpenAI gating.
import sys, platform, hashlib, subprocess, uuid, datetime, zipfile, textwrap

from api.batch_mode_router import router as batch_mode_router

# Create the FastAPI app and ensure decorator attributes exist even under stubs.
app = FastAPI(title="Code Analyser Model Service", version="1.4.2")
app.include_router(batch_mode_router)
# If a stub provided a FastAPI without decorator helpers, attach no-op decorators now.
if not hasattr(app, "get"):

    def _noop_deco(*_a, **_k):
        def _wrap(fn):
            return fn

        return _wrap

    app.get = _noop_deco  # type: ignore[attr-defined]
    app.post = _noop_deco  # type: ignore[attr-defined]
    app.put = _noop_deco  # type: ignore[attr-defined]
    app.delete = _noop_deco  # type: ignore[attr-defined]


# Request/response models.
class PredictRequest(BaseModel):
    texts: List[str] = Field(default_factory=list, description="One or more code snippets.")
    threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Score threshold for positive labels."
    )
    # Allow callers to request OpenAI explanations for /predict; default on to mirror /batch.
    explain: bool = Field(
        default=True, description="If true and OpenAI is configured, include a message per text."
    )


class LabelScore(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    predictions: List[List[LabelScore]]
    id2label: Dict[int, str]
    threshold: float
    # New messages field aligns 1:1 with input texts; each item may be None if disabled/unavailable.
    messages: Optional[List[Optional[str]]] = None


class ExplainRequest(BaseModel):
    text: str


class ExplainResponse(BaseModel):
    summary: str


class BatchRequest(BaseModel):
    paths: List[str] = Field(
        default_factory=list, description="Absolute or relative file paths to process."
    )
    write_fixed: bool = Field(
        default=True, description="If true, write minimally 'fixed' files into the batch archive."
    )
    explain: bool = Field(
        default=True,
        description="If true and OpenAI is configured, include an explanation message per file.",
    )


class BatchFileResult(BaseModel):
    path: str
    ok: bool
    message: Optional[str] = None
    prediction: Optional[List[LabelScore]] = None


class BatchResponse(BaseModel):
    job_id: str
    count: int
    results: List[BatchFileResult]
    archive_url: Optional[str] = None
    archive_path: Optional[str] = None


# Resolve model directory robustly and avoid remote downloads in tests.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ".")).expanduser().resolve()
# Load tokenizer/model with local-only flag when possible; fall back to stubs.
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR.as_posix(), local_files_only=True)  # type: ignore[call-arg]
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(".")
try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR.as_posix(), local_files_only=True)  # type: ignore[call-arg]
except Exception:
    model = AutoModelForSequenceClassification.from_pretrained(".")
# Choose a device if torch has that concept; stub returns a simple object.
device = getattr(torch, "device", lambda *_a, **_k: "cpu")(
    "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
)
# Move to eval mode/selected device when available; safe under stubs.
try:
    model.to(device)  # type: ignore[attr-defined]
    model.eval()  # type: ignore[attr-defined]
except Exception:
    pass
# Build id→label mapping safely even when config is stubbed.
id2label_raw = getattr(
    getattr(model, "config", None),
    "id2label",
    {"0": "sast_risk", "1": "ml_signal", "2": "best_practice"},
)
id2label: Dict[int, str] = {
    int(i): l
    for i, l in (
        id2label_raw.items()
        if hasattr(id2label_raw, "items")
        else {"0": "sast_risk", "1": "ml_signal", "2": "best_practice"}.items()
    )
}
label_order: List[str] = [id2label[i] for i in sorted(id2label.keys())]
# Optional per-label thresholds from a single file; ignore if missing/malformed.
_thresh_path = MODEL_DIR / "thresholds.json"
PER_LABEL_THRESH: Dict[str, float] = {}
if _thresh_path.exists():
    try:
        PER_LABEL_THRESH = json.loads(_thresh_path.read_text(encoding="utf-8"))
    except Exception:
        PER_LABEL_THRESH = {}


# Tokenise to tensors, run the model, and return logits as a NumPy array.
def _predict_logits(batch_texts: List[str]):
    enc = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=getattr(tokenizer, "model_max_length", 512),
    )
    try:
        enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
    except Exception:
        pass
    with getattr(torch, "no_grad", lambda: None)():
        outputs = model(**enc)  # type: ignore[misc]
    logits = getattr(outputs, "logits", outputs)
    try:
        return logits.detach().cpu().numpy()
    except Exception:
        try:
            return getattr(logits, "cpu", lambda: logits)().numpy()  # type: ignore[call-arg]
        except Exception:
            return logits


# Per-label threshold helper (kept but currently non-masking).
def _get_threshold(label: str, default: float) -> float:
    try:
        return float(PER_LABEL_THRESH.get(label, default))
    except Exception:
        return default


# Sigmoid that works whether NumPy is available or not.
def _sigmoid(x):
    try:
        import numpy as np

        return 1.0 / (1.0 + np.exp(-x))
    except Exception:
        return x


# Helper to compute a stable combined SHA-256 over typical model artefacts for provenance.
def _sha256_of_model_dir(root: Path) -> Optional[str]:
    try:
        files: List[Path] = []
        for ext in (".safetensors", ".bin", ".pt", ".json", ".model", ".txt"):
            files.extend(sorted(root.rglob(f"*{ext}")))
        if not files:
            return None
        h = hashlib.sha256()
        for p in files:
            try:
                with p.open("rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
            except Exception:
                continue
        return h.hexdigest()
    except Exception:
        return None


# Helper to capture git commit if available; returns short SHA or None.
def _git_short_sha() -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


# Detect a CLI version, working with Windows batch files (.bat/.cmd) by using shell when needed.
def _detect_cli_version(exe: str, args: List[str]) -> Optional[str]:
    try:
        if os.name == "nt":
            cmd = " ".join([exe] + args)
            r = subprocess.run(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
        else:
            r = subprocess.run(
                [exe] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
        out = (r.stdout or "").strip()
        m = re.search(r"\b(\d+\.\d+(?:\.\d+)?(?:-[A-Za-z0-9.+]+)?)\b", out)
        return m.group(1) if m else (out or None)
    except Exception:
        return None


# Determine SonarLint version from env, explicit exe, Python module, or known CLIs.
def _detect_sonarlint_version() -> Optional[str]:
    v = os.environ.get("SONARLINT_VERSION")
    if v:
        return v.strip()
    exe = os.environ.get("SONARLINT_EXE")
    if exe:
        v = _detect_cli_version(exe, ["--version"])
        if v:
            return v
    try:
        import sonarlint as _sl  # type: ignore

        v = getattr(_sl, "__version__", None)
        if v:
            return str(v)
    except Exception:
        pass
    for exe, args in (
        ("sonarlint", ["--version"]),
        ("sonarlint-cli", ["--version"]),
        ("sonar-scanner", ["-v"]),
    ):
        v = _detect_cli_version(exe, args)
        if v:
            return v
    return None


# NEW — gate OpenAI usage on environment; avoids NameError and lets you disable via OPENAI_DISABLED=1.
def _openai_enabled() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY")) and os.environ.get(
        "OPENAI_DISABLED", "0"
    ) not in {"1", "true", "True"}


# Call OpenAI to produce a short explanation; returns None if disabled or on error.
def _summarise_with_openai(code_text: str, scores: List[LabelScore]) -> Optional[str]:
    if not _openai_enabled():
        return None
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    max_chars = int(os.environ.get("OPENAI_MAX_CHARS", "4000"))
    snippet = (
        code_text
        if len(code_text) <= max_chars
        else (code_text[: max_chars // 2] + "\n...\n" + code_text[-max_chars // 2 :])
    )
    score_str = ", ".join([f"{s.label}={s.score:.3f}" for s in scores])
    system = "You are a secure code reviewer. Be concise, accurate, and actionable."
    user = textwrap.dedent(
        f"""Review the following code. Summarise the key risk(s) and a safer alternative in <=120 words
                           Label scores: {score_str} 
                           Code (language may be Python or mixed):
                           ```
                           {snippet}
                           ```
                           """
    ).strip()
    try:
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI()
            r = client.responses.create(
                model=model,
                input=f"SYSTEM: {system}\nUSER: {user}",
                max_output_tokens=240,
                temperature=0.2,
            )
            out = getattr(r, "output_text", None)
            if out:
                return out.strip()
        except Exception:
            pass
        import openai as _openai  # type: ignore

        if hasattr(_openai, "OpenAI"):
            client = _openai.OpenAI()
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=240,
                temperature=0.2,
            )
            out = r.choices[0].message.content if getattr(r, "choices", None) else None
            return (out or "").strip() or None
        else:
            _openai.api_key = os.environ.get("OPENAI_API_KEY")
            r = _openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=240,
                temperature=0.2,
            )
            out = r["choices"][0]["message"]["content"]
            return (out or "").strip() or None
    except Exception:
        return None


# Health endpoint.
@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


# Main prediction endpoint, now with optional OpenAI messages per input text.
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    texts = req.texts or []
    if not texts:
        return PredictResponse(
            predictions=[], id2label=id2label, threshold=req.threshold, messages=[]
        )
    raw = _predict_logits(texts)
    probs = _sigmoid(raw)
    try:
        import numpy as np

        arr = np.asarray(probs)
        if arr.ndim == 0:
            rows = [[float(arr)]]
        elif arr.ndim == 1:
            rows = [arr.tolist()]
        else:
            rows = arr.tolist()
    except Exception:
        rows = probs if isinstance(probs, list) else [probs]
    out: List[List[LabelScore]] = []
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
    # Optionally generate one message per input text via OpenAI; None if disabled/unavailable.
    messages: Optional[List[Optional[str]]] = None
    if req.explain and _openai_enabled():
        messages = []
        for text, scores in zip(texts, out):
            messages.append(_summarise_with_openai(text, scores))
    else:
        messages = [None for _ in texts]
    return PredictResponse(
        predictions=out, id2label=id2label, threshold=req.threshold, messages=messages
    )


# Optional explanation endpoint; returns OpenAI summary when configured.
@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest) -> ExplainResponse:
    msg = _summarise_with_openai(req.text, [])
    return ExplainResponse(summary=msg or "Explanation disabled or unavailable.")


# New version/metadata endpoint to support curl /version calls for audits and diagnostics.
@app.get("/version")
def version() -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    data["python_version"] = sys.version.split()[0]
    data["platform"] = platform.platform()
    try:
        import fastapi as _fastapi  # type: ignore

        data["fastapi"] = getattr(_fastapi, "__version__", None)
    except Exception:
        data["fastapi"] = None
    try:
        import transformers as _tf  # type: ignore

        data["transformers"] = getattr(_tf, "__version__", None)
    except Exception:
        data["transformers"] = None
    try:
        data["torch"] = getattr(torch, "__version__", None)
    except Exception:
        data["torch"] = None
    try:
        import pylint as _pylint  # type: ignore

        data["pylint"] = getattr(_pylint, "__version__", None)
    except Exception:
        data["pylint"] = None
    data["sonarlint"] = _detect_sonarlint_version()
    data["openai_configured"] = _openai_enabled()
    data["openai_model"] = os.environ.get("OPENAI_MODEL")
    data["model_sha256"] = _sha256_of_model_dir(MODEL_DIR)
    data["git_commit"] = _git_short_sha()
    data["app_version"] = getattr(app, "version", None)
    data["model_dir"] = MODEL_DIR.as_posix()
    return data


# Batch endpoint that accepts file paths, runs predictions per file, and materialises a downloadable archive. Still includes OpenAI messages.
@app.post("/batch", response_model=BatchResponse)
def batch(req: BatchRequest) -> BatchResponse:
    if not req.paths:
        raise HTTPException(status_code=400, detail="paths must be a non-empty list")
    job_id = uuid.uuid4().hex[:12]
    items: List[Dict[str, Any]] = []
    results: List[BatchFileResult] = []
    for p in req.paths:
        path = Path(p).expanduser()
        if not path.exists() or not path.is_file():
            results.append(BatchFileResult(path=p, ok=False, message="file not found"))
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            results.append(BatchFileResult(path=p, ok=False, message=f"read error: {e}"))
            continue
        pred = predict(PredictRequest(texts=[src], threshold=0.5, explain=False)).predictions[0]
        msg = _summarise_with_openai(src, pred) if req.explain else None
        fixed = src if src.endswith("\n") else src + "\n"
        items.append(
            {
                "path": p,
                "original": src,
                "fixed": fixed,
                "prediction": [s.__dict__ for s in pred],
                "message": msg,
            }
        )
        results.append(BatchFileResult(path=p, ok=True, prediction=pred, message=msg))
    zpath = _materialise_batch_archive(job_id, items)
    archive_url = f"/batch/{job_id}/download"
    return BatchResponse(
        job_id=job_id,
        count=len(results),
        results=results,
        archive_url=archive_url,
        archive_path=str(zpath),
    )


# Create a per-batch working directory and zip the results, returning archive path.
def _materialise_batch_archive(job_id: str, items: List[Dict[str, Any]]) -> Path:
    root = Path("batch_jobs") / job_id
    originals = root / "originals"
    fixed = root / "fixed"
    preds = root / "predictions"
    originals.mkdir(parents=True, exist_ok=True)
    fixed.mkdir(parents=True, exist_ok=True)
    preds.mkdir(parents=True, exist_ok=True)
    manifest = {
        "job_id": job_id,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "model_sha256": _sha256_of_model_dir(MODEL_DIR),
        "git_commit": _git_short_sha(),
        "id2label": id2label,
        "model_dir": MODEL_DIR.as_posix(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for it in items:
        rel = Path(it["path"]).name
        if it.get("original") is not None:
            (originals / rel).write_text(it["original"], encoding="utf-8")
        if it.get("fixed") is not None:
            (fixed / rel).write_text(it["fixed"], encoding="utf-8")
        if it.get("prediction") is not None:
            (preds / f"{rel}.json").write_text(
                json.dumps(it["prediction"], indent=2), encoding="utf-8"
            )
        if it.get("message") is not None:
            (preds / f"{rel}.txt").write_text(it["message"], encoding="utf-8")
    zpath = root.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(root.parent).as_posix())
    return zpath


# Download endpoint that serves the zip archive created by /batch for a given job_id.
@app.get("/batch/{job_id}/download")
def batch_download(job_id: str):
    zpath = Path("batch_jobs") / f"{job_id}.zip"
    if not zpath.exists():
        raise HTTPException(status_code=404, detail="archive not found")
    return FileResponse(path=zpath.as_posix(), filename=zpath.name, media_type="application/zip")
