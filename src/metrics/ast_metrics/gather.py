# File: code_analyser/src/metrics/ast_metrics/gather.py

"""
Gather AST Metrics

Provides helper functions to extract and return fixed-length, ordered
lists of AST-based metric values, severity levels, and confidence scores
for downstream use in ML models, GUI overlays, or CSV/JSON export.
"""

import ast
import json
import csv
from typing import List, Dict
from metrics.ast_metrics.extractor import ASTMetricExtractor
from metrics.ast_metrics.plugins import load_plugins

# ✅ Best Practice: Maintain strict order for model features and GUI overlays
_METRIC_ORDER: List[str] = [
    "functions",
    "classes",
    "function_docstrings",
    "class_docstrings",
    "module_docstring",
    "todo_comments",
    "assert_statements",
    "exceptions",
    "loops_conditionals",
    "nested_functions",
    "global_variables",
    "chained_methods",
    "lambda_functions",
    "magic_methods",
]


def gather_ast_metrics(file_path: str) -> List[int]:
    """
    Extract AST metrics from a Python file and return them in fixed order.

    Args:
        file_path (str): Absolute or relative path to the Python source file.

    Returns:
        List[int]: Ordered list of AST metric values aligned with _METRIC_ORDER.
    """
    extractor = ASTMetricExtractor(file_path)
    metrics = extractor.extract()
    return [int(metrics.get(key, 0)) for key in _METRIC_ORDER]


def gather_ast_confidence_scores(file_path: str) -> List[float]:
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    tree = compile(
        code, file_path, mode="exec", flags=ast.PyCF_ONLY_AST, dont_inherit=True
    )
    scores = {p.name(): p.confidence_score(tree, code) for p in load_plugins()}
    return [float(scores.get(key, 0.0)) for key in _METRIC_ORDER]


def gather_ast_severity_levels(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    tree = compile(
        code, file_path, mode="exec", flags=ast.PyCF_ONLY_AST, dont_inherit=True
    )
    levels = {p.name(): p.severity_level(tree, code) for p in load_plugins()}
    return [str(levels.get(key, "low")) for key in _METRIC_ORDER]


def gather_ast_metrics_bundle(file_path: str) -> List[Dict[str, object]]:
    """
    Returns a bundled list of metrics including value, confidence, and severity for each metric.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        List[Dict]: [
            {'metric': 'functions', 'value': 7, 'confidence': 0.7, 'severity': 'medium'},
            ...
        ]
    """
    try:
        values = gather_ast_metrics(file_path)
        confidences = gather_ast_confidence_scores(file_path)
        severities = gather_ast_severity_levels(file_path)
    except Exception as e:
        logging.exception(f"❌ Failed to extract AST metrics for bundle: {e}")
        values = [0] * len(_METRIC_ORDER)
        confidences = [0.0] * len(_METRIC_ORDER)
        severities = ["low"] * len(_METRIC_ORDER)

    # Ensure safe fallback lengths
    if not (len(values) == len(confidences) == len(severities) == len(_METRIC_ORDER)):
        logging.warning(
            "⚠️ Mismatch in AST metric vector lengths; filling with defaults."
        )
        values = (
            values if len(values) == len(_METRIC_ORDER) else [0] * len(_METRIC_ORDER)
        )
        confidences = (
            confidences
            if len(confidences) == len(_METRIC_ORDER)
            else [0.0] * len(_METRIC_ORDER)
        )
        severities = (
            severities
            if len(severities) == len(_METRIC_ORDER)
            else ["low"] * len(_METRIC_ORDER)
        )

    return [
        {
            "metric": name,
            "value": values[i],
            "confidence": round(confidences[i], 2),
            "severity": severities[i],
        }
        for i, name in enumerate(_METRIC_ORDER)
    ]


def export_ast_metrics_to_json(file_path: str, output_path: str):
    """
    Export gathered AST metric bundle to a JSON file.
    """
    bundle = gather_ast_metrics_bundle(file_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)


def export_ast_metrics_to_csv(file_path: str, output_path: str):
    """
    Export gathered AST metric bundle to a CSV file.
    """
    bundle = gather_ast_metrics_bundle(file_path)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["metric", "value", "confidence", "severity"]
        )
        writer.writeheader()
        writer.writerows(bundle)


def get_ast_metric_names() -> List[str]:
    """
    Return the list of metric names used in gather_ast_metrics, in strict order.

    Returns:
        List[str]: Ordered metric name list for consistent model or CSV output.
    """
    return _METRIC_ORDER
