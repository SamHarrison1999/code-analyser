"""
Aggregates metrics from AST, Bandit, and CLOC extractors into a single unified list.
Used for CLI, GUI, and ML metric pipelines.
"""

from metrics.ast_metrics.extractor import ASTMetricExtractor
from metrics.bandit_metrics.extractor import BanditExtractor
from metrics.cloc_metrics.extractor import ClocExtractor
import tempfile


def gather_all_metrics(file_path: str) -> list[int | float]:
    """
    Gathers all metric values from AST, Bandit, and CLOC extractors.

    Args:
        file_path (str): Path to the Python file to analyse.

    Returns:
        list[int | float]: Combined metric values in consistent order.
    """
    ast_metrics = ASTMetricExtractor(file_path).extract()
    bandit_metrics = BanditExtractor(file_path).extract()
    cloc_metrics = ClocExtractor(file_path).extract()

    # ✅ Best Practice: Combine all metrics in extractor order
    return (
        list(ast_metrics.values())
        + list(bandit_metrics.values())
        + list(cloc_metrics.values())
    )


def get_all_metric_names() -> list[str]:
    """
    Returns all metric names in the order used by gather_all_metrics().

    Returns:
        list[str]: Metric names in same order as extracted values.
    """
    with tempfile.NamedTemporaryFile("w+", suffix=".py") as f:
        f.write("def foo(): pass")
        f.flush()

        ast_keys = list(ASTMetricExtractor(f.name).extract().keys())
        bandit_keys = list(BanditExtractor(f.name).extract().keys())
        cloc_keys = list(ClocExtractor(f.name).extract().keys())

    # ✅ Matches order used in gather_all_metrics()
    return ast_keys + bandit_keys + cloc_keys
