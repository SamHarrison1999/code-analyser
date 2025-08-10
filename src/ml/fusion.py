# --- file: src/ml/fusion.py ---
import os, json
# Annotation: Canonical label order used across your project; keep in sync with the model service.
LABELS = ["sast_risk","ml_signal","best_practice"]
# Annotation: Default model thresholds picked conservatively; you can overwrite at call-time.
DEFAULT_MODEL_TH = {
    "sast_risk": float(os.getenv("TH_SAST", "0.35")),
    "ml_signal": float(os.getenv("TH_ML", "0.65")),
    "best_practice": float(os.getenv("TH_BEST", "0.61")),
}
# Annotation: Default rule+model thresholds; start equal to the model thresholds and tune later.
DEFAULT_RULE_TH  = {"sast_risk":0.50,"ml_signal":0.55,"best_practice":0.60}

# Annotation: This helper pulls a threshold for a label, falling back to a default if missing.
def _th(d:dict, k:str, fallback:float)->float:
    # Annotation: Return the threshold for the given key or fallback.
    return float(d.get(k, fallback))

# Annotation: Implement gated fusion: accept if (model ≥ TH_model) OR (rule hit AND model ≥ TH_rule).
def gated_fuse_one(scores:dict[str,float], rule_hits:set[str], th_model:dict|None=None, th_rule:dict|None=None)->dict[str,bool]:
    # Annotation: Use provided thresholds or defaults.
    tm = th_model or DEFAULT_MODEL_TH
    tr = th_rule  or DEFAULT_RULE_TH
    # Annotation: Build the fused boolean decisions per label.
    out:dict[str,bool] = {}
    # Annotation: Iterate labels in canonical order.
    for lab in LABELS:
        # Annotation: Fetch model score for this label (absent → 0.0).
        s = float(scores.get(lab,0.0))
        # Annotation: Positive if score clears the model threshold.
        model_pos = s >= _th(tm, lab, 0.5)
        # Annotation: Gated rule positive requires a rule hit AND score clears the rule threshold.
        rule_pos  = (lab in rule_hits) and (s >= _th(tr, lab, 0.5))
        # Annotation: Final decision is model_pos OR rule_pos (gated).
        out[lab] = bool(model_pos or rule_pos)
    # Annotation: Return fused decisions.
    return out

# Annotation: Convenience wrapper to process a batch; keeps GUI/eval code simple.
def gated_fuse_batch(scores_list:list[dict[str,float]], rule_hits_list:list[set[str]], th_model:dict|None=None, th_rule:dict|None=None)->list[dict[str,bool]]:
    # Annotation: Zip scores and rule hits per example and fuse each with the same thresholds.
    return [gated_fuse_one(sc, rh, th_model=th_model, th_rule=th_rule) for sc, rh in zip(scores_list, rule_hits_list)]
