# This module provides a /batch endpoint that properly honours 'mode' (rules, model, fusion) and wires concrete backends for each mode.
from typing import Dict, List, Optional, Any, Tuple
# FastAPI imports for routing, request parsing and HTTP error handling.
from fastapi import APIRouter, Header, Query, HTTPException
# Pydantic BaseModel to describe the /batch request body.
from pydantic import BaseModel
# Standard library imports for paths, regexes, environment configuration and safety.
from pathlib import Path
# Regular expressions are used by the rule engine to detect risky patterns.
import re
# OS provides access to environment variables for thresholds and runtime toggles.
import os
# JSON is used only for small serialisation tasks in warnings; kept minimal.
import json
# Create an APIRouter to plug into your FastAPI app in main.py.
router = APIRouter()
# The request model accepted by this router; aligns with the existing service contract.
class BatchRequest(BaseModel):
    # List of local file paths to analyse.
    paths: List[str]
    # Optional language hint to guide rules; defaults to 'python'.
    language: Optional[str] = "python"
    # Optional flag to request LLM explanations; kept for schema symmetry.
    explain: Optional[bool] = False
# Resolve the effective mode from header, query, or environment default.
def _resolve_mode(x_mode: Optional[str], q_mode: Optional[str]) -> str:
    # Prefer the header, then the query parameter, then ENGINE_MODE, else 'fusion'.
    raw = (x_mode or q_mode or os.getenv("ENGINE_MODE") or "fusion").strip().lower()
    # Normalise short-hands and validate.
    if raw in ("rules", "rule", "r"):
        return "rules"
    if raw in ("model", "m"):
        return "model"
    if raw in ("fusion", "fuse", "f"):
        return "fusion"
    # Reject anything else clearly, so clients can correct their calls.
    raise HTTPException(status_code=400, detail=f"Unknown mode '{raw}', expected one of rules|model|fusion")
# Read a UTF-8 text file, returning (ok, text_or_error) so callers can branch cleanly.
def _read_text(path: Path) -> Tuple[bool, str]:
    # Fail fast if the path is missing or not a regular file.
    if not path.exists() or not path.is_file():
        return False, "file not found"
    # Try to read as UTF-8 with replacement to be robust to odd encodings.
    try:
        return True, path.read_text(encoding="utf-8", errors="replace")
    # On any exception, surface a concise message.
    except Exception as e:
        return False, f"read error: {e}"
# Heuristic rule patterns keyed by canonical labels; tuned for common Python pitfalls.
_RULES: Dict[str, List[re.Pattern]] = {
    # sast_risk flagged by dangerous sinks and insecure primitives.
    "sast_risk": [
        re.compile(r"\bsubprocess\.\w+\s*\(", re.I),
        re.compile(r"shell\s*=\s*True", re.I),
        re.compile(r"\beval\s*\(", re.I),
        re.compile(r"\bexec\s*\(", re.I),
        re.compile(r"\bos\.system\s*\(", re.I),
        re.compile(r"\bpickle\.(load|loads)\s*\(", re.I),
        re.compile(r"\byaml\.load\s*\(", re.I),
        re.compile(r"\bhashlib\.(md5|sha1)\s*\(", re.I),
        re.compile(r"\.execute\s*\(\s*f?['\"].*[%{]\s*", re.I),
    ],
    # ml_signal flagged by ML/AI library usage and training/parsing idioms.
    "ml_signal": [
        re.compile(r"\b(sklearn|scikit|tensorflow|tf\.|torch|pytorch|xgboost|lightgbm)\b", re.I),
        re.compile(r"\.fit\s*\(", re.I),
        re.compile(r"\.predict\s*\(", re.I),
        re.compile(r"\bmodel\s*=", re.I),
    ],
    # best_practice flagged by common quality issues that are not necessarily vulnerabilities.
    "best_practice": [
        re.compile(r"\brequests\.(get|post|put|delete)\s*\([^)]*(?<!timeout=)[^)]*\)", re.I),
        re.compile(r"\bexcept\s*:\s*pass\b", re.I),
        re.compile(r"\bexcept\s+Exception\s*:\s*pass\b", re.I),
        re.compile(r"\bprint\s*\(", re.I),
        re.compile(r"#\s*TODO\b", re.I),
    ],
}
# Compute a bounded score in [0,1] from matched pattern counts with light weighting.
def _score_from_matches(text: str, patterns: List[re.Pattern], base: float = 0.0, step: float = 0.25) -> float:
    # Count how many distinct patterns match at least once.
    hits = sum(1 for rx in patterns if rx.search(text) is not None)
    # Convert count into a score using a small staircase with saturation.
    score = min(1.0, base + hits * step)
    # Return the bounded score.
    return float(score)
# Produce a concise human-readable message listing the strongest issues found.
def _rules_message(path: Path, text: str) -> str:
    # Collect bullet points by label to form a short advisory.
    bullets: List[str] = []
    # Add a command injection warning when shell=True or subprocess is present.
    if re.search(r"shell\s*=\s*True", text, re.I) or re.search(r"\bsubprocess\.\w+\s*\(", text, re.I):
        bullets.append("Command Injection risk: avoid shell=True; pass an argument list to subprocess.")
    # Add a code injection warning when eval/exec appears.
    if re.search(r"\beval\s*\(|\bexec\s*\(", text, re.I):
        bullets.append("Code Injection risk: avoid eval/exec; prefer safe parsers or direct computation.")
    # Add a YAML unsafe load warning.
    if re.search(r"\byaml\.load\s*\(", text, re.I):
        bullets.append("Unsafe YAML load: use yaml.safe_load instead of yaml.load.")
    # Add legacy hash warning for MD5/SHA1.
    if re.search(r"\bhashlib\.(md5|sha1)\s*\(", text, re.I):
        bullets.append("Weak hash: prefer SHA-256 or stronger instead of MD5/SHA-1.")
    # Add requests timeout advice when HTTP calls lack a timeout.
    if re.search(r"\brequests\.(get|post|put|delete)\s*\([^)]*(?<!timeout=)[^)]*\)", text, re.I):
        bullets.append("HTTP call without timeout: specify timeout= to avoid hangs.")
    # Add broad except advice.
    if re.search(r"\bexcept\s*:\s*pass\b|\bexcept\s+Exception\s*:\s*pass\b", text, re.I):
        bullets.append("Broad exception handler: catch specific exceptions and log appropriately.")
    # If nothing matched, return a generic OK message.
    if not bullets:
        return "No major issues detected by heuristic rules."
    # Otherwise, join the bullets into a compact message.
    return "Heuristic findings:\n- " + "\n- ".join(bullets)
# Convert a label→score map into the service’s list form [{label,score}].
def _to_prediction_list(scores: Dict[str, float]) -> List[Dict[str, float]]:
    # Preserve canonical label order for stability downstream.
    order = ["sast_risk", "ml_signal", "best_practice"]
    # Build and return the list in order.
    return [{"label": lbl, "score": float(scores.get(lbl, 0.0))} for lbl in order]
# RULES backend — apply lightweight, deterministic heuristics over file contents.
def run_rules(paths: List[str], language: str, ret: Optional[str]) -> Dict[str, Any]:
    # Read a default threshold from env (fallback 0.5) for clients that use it.
    default_thr = float(os.getenv("THRESHOLD_DEFAULT", "0.5"))
    # Prepare the result items list.
    results: List[Dict[str, Any]] = []
    # Iterate over each requested path in order.
    for p in paths:
        # Normalise and expand the path for robustness.
        path = Path(p).expanduser()
        # Read the file text or capture an error message.
        ok, text_or_err = _read_text(path)
        # Populate an error item when the read failed.
        if not ok:
            results.append({"path": p, "ok": False, "message": text_or_err})
            continue
        # Compute per-label scores using the pattern families.
        scores = {
            "sast_risk": _score_from_matches(text_or_err, _RULES["sast_risk"], base=0.25, step=0.25),
            "ml_signal": _score_from_matches(text_or_err, _RULES["ml_signal"], base=0.10, step=0.20),
            "best_practice": _score_from_matches(text_or_err, _RULES["best_practice"], base=0.10, step=0.20),
        }
        # Build a short advisory message describing key findings.
        msg = _rules_message(path, text_or_err)
        # Append the successful item with a standardised prediction list.
        results.append({"path": p, "ok": True, "message": msg, "prediction": _to_prediction_list(scores)})
    # Return the payload with a top-level threshold for clients that use it.
    return {"count": len(results), "results": results, "threshold": default_thr}
# MODEL backend — reuse the existing classifier from analyser_model_service over file contents.
def run_model(paths: List[str], language: str, ret: Optional[str]) -> Dict[str, Any]:
    # Import lazily to avoid circular imports at start-up.
    from services.analyser_model_service import predict as svc_predict  # :contentReference[oaicite:0]{index=0}
    from services.analyser_model_service import PredictRequest  # :contentReference[oaicite:1]{index=1}
    # Default threshold mirrors the service default so downstream tools behave consistently.
    default_thr = float(os.getenv("THRESHOLD_DEFAULT", "0.5"))
    # Prepare the result items list.
    results: List[Dict[str, Any]] = []
    # Iterate paths, read text and call the in-process /predict function.
    for p in paths:
        # Expand and normalise the file path.
        path = Path(p).expanduser()
        # Read the file text, tolerating encoding issues.
        ok, text_or_err = _read_text(path)
        # If reading failed, record an error item and continue.
        if not ok:
            results.append({"path": p, "ok": False, "message": text_or_err})
            continue
        # Call the service’s predictor with one-text batch, disabling explanations for speed.
        resp = svc_predict(PredictRequest(texts=[text_or_err], threshold=default_thr, explain=False))
        # Extract the first prediction list from the response model.
        pred_list = [{"label": s.label, "score": float(s.score)} for s in (resp.predictions[0] if resp.predictions else [])]
        # Append the successful result item with the prediction list and no extra message.
        results.append({"path": p, "ok": True, "message": None, "prediction": pred_list})
    # Return the payload; include the threshold for clients that gate on it.
    return {"count": len(results), "results": results, "threshold": default_thr}
# FUSION backend — combine rules and model scores per label (max-pool by default; configurable weights via env).
def run_fusion(paths: List[str], language: str, ret: Optional[str]) -> Dict[str, Any]:
    # Compute rules output for all paths first.
    rules_payload = run_rules(paths, language, ret)
    # Compute model output for all paths next.
    model_payload = run_model(paths, language, ret)
    # Read optional fusion weights from environment (defaults favour max-pool behaviour when equal).
    w_rules = float(os.getenv("FUSE_W_RULES", "0.5"))
    w_model = float(os.getenv("FUSE_W_MODEL", "0.5"))
    # Pick a default threshold to echo in the response.
    default_thr = float(os.getenv("THRESHOLD_DEFAULT", "0.5"))
    # Index model results by basename for robust matching.
    model_by_name: Dict[str, Dict[str, float]] = {}
    # Convert the model prediction lists into label→score maps for quick access.
    for item in model_payload.get("results", []):
        if not item.get("ok"):
            continue
        name = Path(item["path"]).name
        model_by_name[name] = {d["label"]: float(d["score"]) for d in item.get("prediction", [])}
    # Build fused results by walking the rules list and combining with model scores.
    fused_results: List[Dict[str, Any]] = []
    # For each rules result, compute the fused label scores and join messages.
    for item in rules_payload.get("results", []):
        if not item.get("ok"):
            fused_results.append(item)
            continue
        name = Path(item["path"]).name
        rules_map = {d["label"]: float(d["score"]) for d in item.get("prediction", [])}
        model_map = model_by_name.get(name, {})
        fused_scores: Dict[str, float] = {}
        for lbl in ("sast_risk", "ml_signal", "best_practice"):
            rs = rules_map.get(lbl, 0.0)
            ms = model_map.get(lbl, 0.0)
            # Weighted max-like fusion: if weights equal, equivalent to simple max; otherwise convex combination with emphasis on the stronger.
            fused = max(rs, ms, w_rules * rs + w_model * ms)
            fused_scores[lbl] = float(min(1.0, max(0.0, fused)))
        # Combine advisory messages, preferring the rules message when available.
        msg = item.get("message") or None
        # Append the fused item with a standardised prediction list.
        fused_results.append({"path": item["path"], "ok": True, "message": msg, "prediction": _to_prediction_list(fused_scores)})
    # Return the fused payload with the same top-level fields used by the other backends.
    return {"count": len(fused_results), "results": fused_results, "threshold": default_thr}
# Public /batch route that honours mode via header or query and dispatches to the appropriate backend.
@router.post("/batch")
async def batch_mode(req: BatchRequest, ret: Optional[str] = Query(default="scores", alias="return"), mode_param: Optional[str] = Query(default=None, alias="mode"), x_mode: Optional[str] = Header(default=None, alias="X-Mode")):
    # Resolve the effective mode using header→query→env precedence.
    mode = _resolve_mode(x_mode, mode_param)
    # Dispatch to the requested backend and capture the payload.
    if mode == "rules":
        payload = run_rules(req.paths, req.language or "python", ret)
    elif mode == "model":
        payload = run_model(req.paths, req.language or "python", ret)
    else:
        payload = run_fusion(req.paths, req.language or "python", ret)
    # Validate the backend result to ensure it is a dict with a results field.
    if not isinstance(payload, dict) or "results" not in payload:
        raise HTTPException(status_code=500, detail="Backend did not return a valid payload")
    # Always include the resolved mode for transparent debugging from clients.
    payload.setdefault("meta", {})
    payload["meta"]["mode"] = mode
    # Return the final payload as the HTTP response body.
    return payload
