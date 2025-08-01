# File: src/ml/ai_metric_names.py


def gather_ai_metric_names() -> list[str]:
    """
    Returns a list of AI-specific metric names used by the annotation system.
    """
    return [
        "ai_confidence",
        "ai_severity",
        "ai_token_density",
        "ai_annotation_density",
        "ai_risk_score",
    ]
