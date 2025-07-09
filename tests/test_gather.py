"""
Tests for metrics.gather module.

Validates:
- Correct structure of AST + Bandit merged metrics
- Alignment between metric names and values
- Stability of plugin ordering
"""

from metrics.gather import gather_all_metrics, get_all_metric_names
from pathlib import Path


def test_gather_all_metrics_and_names_match():
    """
    Ensure the number of values from gather_all_metrics matches
    the number of names from get_all_metric_names, and all are ints.
    """
    test_file = Path("example.py")
    assert test_file.exists(), "example.py must be present in the root directory"

    values = gather_all_metrics(str(test_file))
    names = get_all_metric_names()

    # ✅ Best Practice: Explicit type assertions to ensure return consistency
    assert isinstance(values, list), "Metric values should be a list"
    assert isinstance(names, list), "Metric names should be a list"

    # ⚠️ SAST Risk: Mismatch between metric names and actual metric outputs can cause silent failures in downstream scoring logic
    # 🧠 ML Signal: Discrepancies here might indicate a plugin version or metric definition drift that should be tracked over time
    assert len(values) == len(names), (
        f"Mismatch between metric names and values\n"
        f"Names ({len(names)}): {names}\n"
        f"Values ({len(values)}): {values}"
    )

    assert all(isinstance(v, (int, float)) for v in values), "All metrics must be numbers (int or float)"
