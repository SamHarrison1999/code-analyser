import tempfile
from typing import List, Union

from metrics.ast_metrics.extractor import ASTMetricExtractor
from metrics.bandit_metrics.extractor import BanditExtractor
from metrics.cloc_metrics.extractor import ClocExtractor
from metrics.flake8_metrics.extractor import Flake8Extractor
from metrics.lizard_metrics.extractor import extract_lizard_metrics
from metrics.pydocstyle_metrics.extractor import PydocstyleExtractor
from metrics.pyflakes_metrics.extractor import extract_pyflakes_metrics
from metrics.pylint_metrics.gather import gather_pylint_metrics


def safe_number(val):
    if isinstance(val, (int, float)):
        return val
    try:
        return float(val)
    except Exception:
        return 0


def gather_all_metrics(file_path: str) -> List[Union[int, float]]:
    """
    Gathers all metric values and normalizes to numeric values.
    """
    ast = ASTMetricExtractor(file_path).extract()
    bandit = BanditExtractor(file_path).extract()
    cloc = ClocExtractor(file_path).extract()
    flake8 = Flake8Extractor(file_path).extract()
    lizard = extract_lizard_metrics(file_path)
    pydocstyle = PydocstyleExtractor(file_path).extract()
    pyflakes = extract_pyflakes_metrics(file_path)
    pylint = gather_pylint_metrics(file_path)

    all_metrics = []
    for metric_set in [ast, bandit, cloc, flake8, lizard, pydocstyle, pyflakes, pylint]:
        all_metrics.extend(safe_number(v) for v in metric_set.values())

    return all_metrics


def get_all_metric_names() -> List[str]:
    """
    Returns the names of all metrics in the same order as gather_all_metrics.

    Returns:
        list[str]: List of all metric names.
    """
    with tempfile.NamedTemporaryFile("w+", suffix=".py") as f:
        f.write("def foo(): pass")
        f.flush()

        ast_keys = list(ASTMetricExtractor(f.name).extract().keys())
        bandit_keys = list(BanditExtractor(f.name).extract().keys())
        cloc_keys = list(ClocExtractor(f.name).extract().keys())
        flake8_keys = list(Flake8Extractor(f.name).extract().keys())
        lizard_keys = list(extract_lizard_metrics(f.name).keys())
        pydocstyle_keys = list(PydocstyleExtractor(f.name).extract().keys())
        pyflakes_keys = list(extract_pyflakes_metrics(f.name).keys())
        pylint_keys = list(gather_pylint_metrics(f.name).keys())

    return (
        ast_keys +
        bandit_keys +
        cloc_keys +
        flake8_keys +
        lizard_keys +
        pydocstyle_keys +
        pyflakes_keys +
        pylint_keys
    )
