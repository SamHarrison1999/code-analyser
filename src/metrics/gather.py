"""
Aggregates metrics from various extractors into a single unified list.
"""

from metrics.ast_metrics.extractor import ASTMetricExtractor
from metrics.bandit_metrics.extractor import BanditExtractor
import tempfile


def gather_all_metrics(file_path: str) -> list[int]:
    """
    Gathers all metric values from AST and Bandit extractors for a given file.
    Returns:
        A list of integers representing the combined metrics.
    """
    ast_metrics = ASTMetricExtractor(file_path).extract()
    bandit_metrics = BanditExtractor(file_path).extract()

    # ✅ Best Practice: Maintain ordering between values and declared names
    # 🧠 ML Signal: Order drift in feature columns is a common cause of prediction error
    return list(ast_metrics.values()) + list(bandit_metrics.values())


def get_all_metric_names() -> list[str]:
    """
    Returns the list of metric names in the order used by gather_all_metrics.
    This ensures alignment between feature names and values.
    """
    # ✅ Best Practice: Use real file contents to guarantee extractor behaviour
    # ⚠️ SAST Risk: Calling extract() on dummy input may cause silent errors or misalignment
    # 🧠 ML Signal: Tracking extractor schema from real input ensures stable training labels

    with tempfile.NamedTemporaryFile("w+", suffix=".py") as f:
        f.write("def foo(): pass")
        f.flush()
        ast_keys = list(ASTMetricExtractor(f.name).extract().keys())
        bandit_keys = list(BanditExtractor(f.name).extract().keys())

    return ast_keys + bandit_keys
