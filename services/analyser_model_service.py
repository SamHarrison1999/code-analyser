# --- file: services/analyser_model_service.py ---
# Annotation: This service exposes your multi-label classifier over HTTP and adds an optional rationale generator and a separate /explain endpoint.
# This file exposes your fine-tuned RoBERTa model (multi-label) over HTTP for the Code Analyser to call.
from typing import List, Dict, Any, Optional

# Annotation: FastAPI provides a minimal, fast HTTP layer for inference.
# We use FastAPI for a lightweight REST API.
from fastapi import FastAPI

# Annotation: Pydantic validates request/response schemas for a stable contract.
# Pydantic models define stable request/response shapes.
from pydantic import BaseModel, Field

# Annotation: Auto* loaders pick the correct classes from the on-disk config.
# We rely on Hugging Face Transformers to load the exported artefacts.
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

# Annotation: Torch provides device placement and sigmoid for multi-label probabilities.
# Torch handles tensors, device placement, and the sigmoid we need for multi-label output.
import torch

# Annotation: We use os/pathlib to resolve paths and read optional environment configuration.
# OS/env lets us configure the model directory without touching code.
import os

# Annotation: Pathlib ensures cross-platform path handling.
from pathlib import Path

# Annotation: requests allows optional LLM API calls for /explain when no local rationale model is configured.
import requests

import json

# Annotation: Helpful metadata for /docs and debugging.
# We construct the FastAPI application instance.
app = FastAPI(title="Code Analyser Model Service", version="1.1.0")


# Annotation: Batch input and configurable threshold are sufficient for most uses.
# This Pydantic model defines the request schema for predictions.
class PredictRequest(BaseModel):
    # A batch of texts/snippets to classify.
    texts: List[str] = Field(..., description="List of source code or text snippets.")
    # Optional decision threshold for multi-label; defaults to a sensible 0.5.
    threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Sigmoid threshold for label activation."
    )
    # Annotation: If true and a rationale model is available, the service will attach a single-sentence rationale per item.
    include_rationale: bool = Field(
        False, description="Attach a rationale string when supported (optional)."
    )


# Annotation: Each item returns both activated labels and the raw per-label probabilities; rationale is optional.
# This Pydantic model defines each item in the batch response.
class PredictItem(BaseModel):
    # The labels that cleared the threshold for this input.
    labels: List[str]
    # Raw scores for all labels so you can render probabilities/heatmaps in the UI.
    scores: Dict[str, float]
    # Annotation: Optional, short explanation either produced by a local seq2seq model or empty when not requested/unavailable.
    rationale: Optional[str] = None


# Annotation: The response wraps results and a canonical label order for downstream stability.
# This Pydantic model wraps the batch response.
class PredictResponse(BaseModel):
    # One entry per input text.
    results: List[PredictItem]
    # Echo the model's label order to make downstream mapping explicit.
    label_order: List[str]


# Annotation: Request schema for the /explain endpoint used to generate a one-sentence rationale with an LLM/API.
class ExplainRequest(BaseModel):
    # Annotation: The code snippet to explain (single item).
    code: str = Field(..., description="Single code snippet the user is inspecting.")
    # Annotation: The top label key (e.g., 'sast_risk', 'ml_signal', 'best_practice') to steer the rationale.
    top_label: str = Field(..., description="Top label key to condition the rationale.")


# Annotation: Response schema returned by /explain.
class ExplainResponse(BaseModel):
    # Annotation: A concise, single-sentence explanation.
    rationale: str


# Annotation: Resolve MODEL_DIR to an absolute path; default to models/trained_model under the repo.
# We resolve the model directory; default to a local folder named 'trained_model'.
_default_dir = Path(__file__).resolve().parent.parent / "models" / "trained_model"
# Annotation: Read from environment, falling back to the default; then normalise to an absolute Path.
_model_env = os.environ.get("MODEL_DIR", str(_default_dir))
MODEL_DIR = Path(_model_env).expanduser().resolve()
# Annotation: Fail fast if the directory does not exist to avoid accidental remote downloads.
if not MODEL_DIR.is_dir():
    raise FileNotFoundError(f"MODEL_DIR does not exist locally: {MODEL_DIR}")

# Annotation: Optional local seq2seq rationale generator directory; when set and present, we will load it.
_rationale_env = os.environ.get("RATIONALE_MODEL_DIR", "").strip()
RATIONALE_MODEL_DIR: Optional[Path] = (
    Path(_rationale_env).expanduser().resolve() if _rationale_env else None
)
# Annotation: Choose CUDA if available, otherwise CPU; the model is loaded once at startup.
# We pick the most capable device available (CUDA if present, else CPU).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Annotation: Force local file loading so we never hit Hugging Face by mistake.
# We load the tokenizer directly from the exported folder; this includes merges/vocab and special tokens.
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR.as_posix(), local_files_only=True)
# We load the classifier; Transformers will pick up model.safetensors and config.json automatically.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR.as_posix(), local_files_only=True
)
# Annotation: Keep the model on the selected device.
# We place the model on the chosen device for efficient inference.
model.to(device)
# Annotation: Evaluation mode disables dropout for deterministic inference.
# We switch to evaluation mode to disable dropout etc.
model.eval()

# Annotation: Load an optional local rationale model (seq2seq) if configured; otherwise keep None and rely on /explain if desired.
# We pull the id→label mapping from the model config so responses use your trained names.
id2label: Dict[int, str] = {int(i): l for i, l in model.config.id2label.items()}
# Annotation: Cache label order once for predictable indexing in the UI.
# We also capture the label order once, so clients can rely on a stable ordering.
label_order: List[str] = [id2label[i] for i in range(len(id2label))]
_thresh_path = MODEL_DIR / "thresholds/json"
PER_LABEL_THRESH = {}
if _thresh_path.exists():
    try:
        PER_LABEL_THRESH = json.loads(_thresh_path.read_text())
    except Exception:
        PER_LABEL_THRESH = {}
# Annotation: Prepare optional rationale generator components.
rationale_tokenizer = None
rationale_model = None
# Annotation: Only attempt to load a local rationale model if a directory was provided and exists.
if RATIONALE_MODEL_DIR and RATIONALE_MODEL_DIR.is_dir():
    # Annotation: Load a small seq2seq model (e.g., T5) for one-sentence rationales; strictly local.
    rationale_tokenizer = AutoTokenizer.from_pretrained(
        RATIONALE_MODEL_DIR.as_posix(), local_files_only=True
    )
    # Annotation: Use a generic seq2seq head capable of conditional generation.
    rationale_model = AutoModelForSeq2SeqLM.from_pretrained(
        RATIONALE_MODEL_DIR.as_posix(), local_files_only=True
    )
    # Annotation: Place rationale model on the same device for efficiency.
    rationale_model.to(device)
    # Annotation: Ensure eval mode.
    rationale_model.eval()


# Annotation: Stateless helper to classify a batch with sigmoid + threshold per label.
# This helper runs a batch through the model and returns activated labels and per-label scores.
def classify_batch(texts: List[str], threshold: float) -> List[PredictItem]:
    # Annotation: Truncate/pad to model max length (commonly 512 for RoBERTa).
    # We tokenize with truncation/padding to the model's maximum sequence length.
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=min(getattr(tokenizer, "model_max_length", 512), 512),
        return_tensors="pt",
    )
    # Annotation: Ensure tensors reside on the same device as the model.
    # We move tensors to the same device as the model.
    enc = {k: v.to(device) for k, v in enc.items()}
    # Annotation: Inference runs without gradients for speed and lower memory.
    # Inference should not build gradients; this keeps it memory- and latency-friendly.
    with torch.no_grad():
        # Annotation: Get raw logits and convert to per-label probabilities.
        # We obtain raw logits for each label.
        logits = model(**enc).logits
        # Because the model is configured for multi-label, we apply sigmoid per label.
        probs = torch.sigmoid(logits)
    # Annotation: Convert to plain Python numbers for JSON serialisation.
    # We move probabilities back to CPU and convert to plain Python floats.
    probs_np = probs.cpu().numpy().tolist()
    # Annotation: Build one PredictItem per input, preserving input order.
    # We build a PredictItem for each input, thresholding per label.
    items: List[PredictItem] = []
    # Iterate over each probability vector in the batch.
    for p in probs_np:
        # Annotation: Provide a full score map for confidence bars and later calibration.
        # Map label names to their scores for transparency in the UI.
        score_map = {label_order[i]: float(p[i]) for i in range(len(p))}
        # Annotation: Activate labels with probability meeting or exceeding the threshold.
        # Activate labels whose probability meets or exceeds the threshold.
        active = [label_order[i] for i in range(len(p)) if p[i] >= float(PER_LABEL_THRESH.get(label_order[i], threshold))]
        if not active:
            top = max(range(len(p)), key=lambda i: p[i])
            active = [label_order[top]]
        # Annotation: Append the result for this sample.
        # Append the structured result for this text.
        items.append(PredictItem(labels=active, scores=score_map))
    # Annotation: Return the full batch of structured predictions.
    # Return the full batch of structured predictions.
    return items


# Annotation: Generate a single-sentence rationale using a local seq2seq model when available.
def _local_rationale(code: str, top_label: str) -> Optional[str]:
    # Annotation: If no local model is configured, indicate absence.
    if not (rationale_model and rationale_tokenizer):
        return None
    # Annotation: Build a compact prompt that conditions on the top label and asks for one sentence.
    prompt = f"Explain in one concise British English sentence why the following code may be categorised as '{top_label}':\n{code}\nReason:"
    # Annotation: Tokenise and generate with modest limits to keep latency low.
    enc = rationale_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(
        device
    )
    # Annotation: Use deterministic decoding with a small beam for quality.
    with torch.no_grad():
        out = rationale_model.generate(**enc, max_length=64, num_beams=4, early_stopping=True)
    # Annotation: Decode and normalise whitespace.
    return rationale_tokenizer.decode(out[0], skip_special_tokens=True).strip()


# Annotation: Fallback to an external LLM provider via HTTP APIs (OpenAI or Together) based on environment configuration.
def _external_rationale(code: str, top_label: str) -> Optional[str]:
    # Annotation: Read provider configuration from environment variables.
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    # Annotation: Return None if no provider configured.
    if not provider:
        return None
    # Annotation: Normalise a minimal prompt for the API.
    prompt = f"One-sentence reason (British English): why is this code categorised as '{top_label}'?\n\n{code}\n\nReturn only one sentence."
    # Annotation: OpenAI path expects OPENAI_API_KEY and model name (default: gpt-4o-mini).
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        if not api_key:
            return None
        # Annotation: Use the Chat Completions HTTP API directly to avoid extra dependencies.
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 64,
        }
        r = requests.post(url, headers=headers, json=body, timeout=30)
        if r.status_code >= 400:
            return None
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    # Annotation: Together path expects TOGETHER_API_KEY and model name (default: meta-llama/Meta-Llama-3-8B-Instruct-Turbo).
    if provider == "together":
        api_key = os.environ.get("TOGETHER_API_KEY", "")
        model = os.environ.get("TOGETHER_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct-Turbo")
        if not api_key:
            return None
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 64,
        }
        r = requests.post(url, headers=headers, json=body, timeout=30)
        if r.status_code >= 400:
            return None
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    # Annotation: Unknown provider; return None to indicate no rationale.
    return None


# Annotation: Health endpoint assists readiness checks and quick model introspection.
# A simple health endpoint so orchestrators can probe readiness.
@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    # Annotation: Device and labels help confirm you are talking to the expected model.
    # Report device and label metadata to aid debugging.
    return {
        "status": "ok",
        "device": str(device),
        "num_labels": len(label_order),
        "labels": label_order,
        "model_dir": str(MODEL_DIR),
        "rationale_local": bool(rationale_model is not None),
    }


# Annotation: Main prediction endpoint used by the Code Analyser; can optionally attach a rationale per item if a local generator is set and requested.
# The main prediction endpoint used by the Code Analyser pipeline.
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    # Annotation: Run the batch classification with the caller-supplied threshold.
    # We run the batch classification with the caller-supplied threshold.
    results = classify_batch(req.texts, req.threshold)
    # Annotation: If rationale was requested and a local generator is available, attach a single-sentence explanation for each item using its strongest label.
    if req.include_rationale and rationale_model and rationale_tokenizer:
        for text, item in zip(req.texts, results):
            if not item.scores:
                continue
            top_key = max(item.scores, key=lambda k: item.scores[k])
            item.rationale = _local_rationale(text, top_key)
    # Annotation: Return predictions and canonical order.
    # We return both results and the canonical label order.
    return PredictResponse(results=results, label_order=label_order)


# Annotation: Secondary endpoint to obtain a one-sentence rationale from an LLM or local seq2seq; used by the GUI when richer explanations are desired.
@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest) -> ExplainResponse:
    # Annotation: Try local generator first for privacy/latency.
    reason = _local_rationale(req.code, req.top_label)
    # Annotation: Fall back to external provider if local generation is unavailable.
    if not reason:
        reason = _external_rationale(req.code, req.top_label)
    # Annotation: Final fallback to a safe default if nothing could be generated.
    if not reason:
        reason = ""
    # Annotation: Wrap and return the rationale.
    return ExplainResponse(rationale=reason)


# Annotation: Allow running as a script for local testing; otherwise start via `python -m uvicorn ...`.
# This allows 'python analyser_model_service.py' for local testing without uvicorn CLI.
if __name__ == "__main__":
    # Annotation: Use localhost by default for development.
    # Deferred import keeps uvicorn optional until explicitly run as a script.
    import uvicorn

    # Annotation: Start the server on 127.0.0.1:8111; adjust as needed.
    # Start a single-worker development server bound to localhost:8111.
    uvicorn.run("services.analyser_model_service:app", host="127.0.0.1", port=8111, reload=False)
