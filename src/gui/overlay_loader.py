import json
from pathlib import Path
import logging
from typing import List, Dict, Any, Union

# === Types ===
Overlay = Dict[
    str, Union[int, float, str]
]  # e.g., {"line": 4, "token": "eval", "confidence": 0.91, "severity": "High"}
Summary = Dict[
    str, Dict[str, Union[str, float]]
]  # e.g., { "Maintainability": {"AI Signal": "High"} }


def load_together_ai_overlay(json_path: Union[str, Path]) -> Dict[str, Any]:
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"No annotation overlay found at: {json_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary: Summary = data.get("summary", {})
        overlays: List[Overlay] = data.get("overlays", [])
        logging.debug(
            f"✅ Loaded Together.ai overlay: {json_path.name} with {len(overlays)} tokens"
        )
        return {"summary": summary, "overlays": overlays}
    except Exception as e:
        logging.error(f"❌ Failed to parse Together.ai overlay: {e}")
        raise


def load_rl_overlay_from_json(json_path: Union[str, Path]) -> Summary:
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"No RL overlay log found at: {json_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        summary: Summary = {
            k: {
                "Total": round(v.get("value", 0.0), 2),
                "Average": round(v.get("value", 0.0), 2),
                "AI Signal": v.get("severity", "Unknown"),
            }
            for k, v in metrics.items()
        }
        logging.debug(f"✅ Loaded RL overlay summary: {json_path.name}")
        return summary
    except Exception as e:
        logging.error(f"❌ Failed to parse RL overlay JSON: {e}")
        raise


def merge_overlay_summaries(*summaries: Summary) -> Summary:
    merged = {}
    for summary in summaries:
        for key, val in summary.items():
            if key not in merged:
                merged[key] = val
    return merged


def extract_rl_tokens(path: Union[str, Path]) -> List[Overlay]:
    """
    Extract token-level overlays from a reinforcement learning log JSON.

    Each token should include:
    - line: int
    - token: str
    - confidence: float
    - severity: str (e.g. 'High', 'Medium', 'Low')

    Returns:
        List of token overlays or empty list if unavailable.
    """
    path = Path(path)
    if not path.exists():
        logging.warning(f"⚠️ RL token file not found: {path}")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokens = data.get("tokens", [])
        if not isinstance(tokens, list):
            logging.warning(f"⚠️ No 'tokens' list in RL file: {path}")
            return []

        valid_tokens = [
            {
                "line": int(tok.get("line", 0)),
                "token": str(tok.get("token", "")),
                "confidence": float(tok.get("confidence", 0.0)),
                "severity": str(tok.get("severity", "Unknown")),
            }
            for tok in tokens
            if "token" in tok and "line" in tok
        ]

        logging.debug(f"✅ Extracted {len(valid_tokens)} RL tokens from: {path}")
        return valid_tokens

    except Exception as e:
        logging.error(f"❌ Failed to extract RL tokens: {type(e).__name__}: {e}")
        return []
