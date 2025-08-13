# src/api/batch_mode_router.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os, re, json
from pathlib import Path

# Import only the FastAPI names guaranteed by the test stub; add shims for others.
from fastapi import APIRouter, HTTPException  # stub provides these
try:
    from fastapi import Header, Query  # not provided by stub -> guarded
except Exception:  # pragma: no cover
    def Header(default=None, **kwargs):  # minimal stand-in for annotation time
        return default
    def Query(default=None, **kwargs):   # minimal stand-in for annotation time
        return default

# Pydantic is stubbed by the harness; BaseModel exists.
from pydantic import BaseModel

router = APIRouter()

# Polyfill HTTP verb decorators when the stub lacks .post (uses add_api_route instead).
if not hasattr(router, "post"):  # pragma: no cover
    def _verb(method: str):
        def _decorator(path: str, *args, **kwargs):
            def _wrap(fn):
                try:
                    router.add_api_route(path, fn, methods=[method.upper()])
                except Exception:
                    pass
                return fn
            return _wrap
        return _decorator
    router.post = _verb("post")  # type: ignore[attr-defined]

class BatchRequest(BaseModel):
    paths: List[str]
    language: Optional[str] = "python"
    explain: Optional[bool] = False

def _resolve_mode(x_mode: Optional[str], q_mode: Optional[str]) -> str:
    raw = (x_mode or q_mode or os.getenv("ENGINE_MODE") or "fusion").strip().lower()
    if raw in ("rules", "rule", "r"): return "rules"
    if raw in ("model", "m"): return "model"
    if raw in ("fusion", "fuse", "f"): return "fusion"
    raise HTTPException(status_code=400, detail=f"Unknown mode '{raw}', expected one of rules|model|fusion")

def _read_text(path: Path) -> Tuple[bool, str]:
    if not path.exists() or not path.is_file():
        return False, "file not found"
    try:
        return True, path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"read error: {e}"

_RULES: Dict[str, List[re.Pattern]] = {
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
    "ml_signal": [
        re.compile(r"\b(sklearn|scikit|tensorflow|tf\.|torch|pytorch|xgboost|lightgbm)\b", re.I),
        re.compile(r"\.fit\s*\(", re.I),
        re.compile(r"\.predict\s*\(", re.I),
        re.compile(r"\bmodel\s*=", re.I),
    ],
    "best_practice": [
        re.compile(r"\brequests\.(get|post|put|delete)\s*\([^)]*(?<!timeout=)[^)]*\)", re.I),
        re.compile(r"\bexcept\s*:\s*pass\b", re.I),
        re.compile(r"\bexcept\s+Exception\s*:\s*pass\b", re.I),
        re.compile(r"\bprint\s*\(", re.I),
        re.compile(r"#\s*TODO\b", re.I),
    ],
}

def _score_from_matches(text: str, patterns: List[re.Pattern], base: float = 0.0, step: float = 0.25) -> float:
    hits = sum(1 for rx in patterns if rx.search(text) is not None)
    return float(min(1.0, base + hits * step))

def _rules_message(path: Path, text: str) -> str:
    bullets: List[str] = []
    if re.search(r"shell\s*=\s*True", text, re.I) or re.search(r"\bsubprocess\.\w+\s*\(", text, re.I):
        bullets.append("Command Injection risk: avoid shell=True; pass an argument list to subprocess.")
    if re.search(r"\beval\s*\(|\bexec\s*\(", text, re.I):
        bullets.append("Code Injection risk: avoid eval/exec; prefer safe parsers or direct computation.")
    if re.search(r"\byaml\.load\s*\(", text, re.I):
        bullets.append("Unsafe YAML load: use yaml.safe_load instead of yaml.load.")
    if re.search(r"\bhashlib\.(md5|sha1)\s*\(", text, re.I):
        bullets.append("Weak hash: prefer SHA-256 or stronger instead of MD5/SHA-1.")
    if re.search(r"\brequests\.(get|post|put|delete)\s*\([^)]*(?<!timeout=)[^)]*\)", text, re.I):
        bullets.append("HTTP call without timeout: specify timeout= to avoid hangs.")
    if re.search(r"\bexcept\s*:\s*pass\b|\bexcept\s+Exception\s*:\s*pass\b", text, re.I):
        bullets.append("Broad exception handler: catch specific exceptions and log appropriately.")
    return "No major issues detected by heuristic rules." if not bullets else "Heuristic findings:\n- " + "\n- ".join(bullets)

def _to_prediction_list(scores: Dict[str, float]) -> List[Dict[str, float]]:
    order = ["sast_risk", "ml_signal", "best_practice"]
    return [{"label": lbl, "score": float(scores.get(lbl, 0.0))} for lbl in order]

def run_rules(paths: List[str], language: str, ret: Optional[str]) -> Dict[str, Any]:
    default_thr = float(os.getenv("THRESHOLD_DEFAULT", "0.5"))
    results: List[Dict[str, Any]] = []
    for p in paths:
        path = Path(p).expanduser()
        ok, text_or_err = _read_text(path)
        if not ok:
            results.append({"path": p, "ok": False, "message": text_or_err})
            continue
        scores = {
            "sast_risk": _score_from_matches(text_or_err, _RULES["sast_risk"], base=0.25, step=0.25),
            "ml_signal": _score_from_matches(text_or_err, _RULES["ml_signal"], base=0.10, step=0.20),
            "best_practice": _score_from_matches(text_or_err, _RULES["best_practice"], base=0.10, step=0.20),
        }
        msg = _rules_message(path, text_or_err)
        results.append({"path": p, "ok": True, "message": msg, "prediction": _to_prediction_list(scores)})
    return {"count": len(results), "results": results, "threshold": default_thr}

def run_model(paths: List[str], language: str, ret: Optional[str]) -> Dict[str, Any]:
    # Lazy import so the module remains importable under the test harness.
    from services.analyser_model_service import predict as svc_predict
    from services.analyser_model_service import PredictRequest
    default_thr = float(os.getenv("THRESHOLD_DEFAULT", "0.5"))
    results: List[Dict[str, Any]] = []
    for p in paths:
        path = Path(p).expanduser()
        ok, text_or_err = _read_text(path)
        if not ok:
            results.append({"path": p, "ok": False, "message": text_or_err})
            continue
        resp = svc_predict(PredictRequest(texts=[text_or_err], threshold=default_thr, explain=False))
        pred_list = [{"label": s.label, "score": float(s.score)} for s in (resp.predictions[0] if resp.predictions else [])]
        results.append({"path": p, "ok": True, "message": None, "prediction": pred_list})
    return {"count": len(results), "results": results, "threshold": default_thr}

def run_fusion(paths: List[str], language: str, ret: Optional[str]) -> Dict[str, Any]:
    rules_payload = run_rules(paths, language, ret)
    model_payload = run_model(paths, language, ret)
    w_rules = float(os.getenv("FUSE_W_RULES", "0.5"))
    w_model = float(os.getenv("FUSE_W_MODEL", "0.5"))
    default_thr = float(os.getenv("THRESHOLD_DEFAULT", "0.5"))
    model_by_name: Dict[str, Dict[str, float]] = {}
    for item in model_payload.get("results", []):
        if not item.get("ok"):
            continue
        name = Path(item["path"]).name
        model_by_name[name] = {d["label"]: float(d["score"]) for d in item.get("prediction", [])}
    fused_results: List[Dict[str, Any]] = []
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
            fused = max(rs, ms, w_rules * rs + w_model * ms)
            fused_scores[lbl] = float(min(1.0, max(0.0, fused)))
        msg = item.get("message") or None
        fused_results.append({"path": item["path"], "ok": True, "message": msg, "prediction": _to_prediction_list(fused_scores)})
    return {"count": len(fused_results), "results": fused_results, "threshold": default_thr}

@router.post("/batch")
async def batch_mode(
    req: BatchRequest,
    ret: Optional[str] = Query(default="scores", alias="return"),
    mode_param: Optional[str] = Query(default=None, alias="mode"),
    x_mode: Optional[str] = Header(default=None, alias="X-Mode"),
):
    mode = _resolve_mode(x_mode, mode_param)
    if mode == "rules":
        payload = run_rules(req.paths, req.language or "python", ret)
    elif mode == "model":
        payload = run_model(req.paths, req.language or "python", ret)
    else:
        payload = run_fusion(req.paths, req.language or "python", ret)
    if not isinstance(payload, dict) or "results" not in payload:
        raise HTTPException(status_code=500, detail="Backend did not return a valid payload")
    payload.setdefault("meta", {})
    payload["meta"]["mode"] = mode
    return payload
