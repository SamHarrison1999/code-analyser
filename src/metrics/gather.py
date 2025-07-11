import tempfile
from typing import List, Union, Callable

from metrics.ast_metrics.gather import gather_ast_metrics, get_ast_metric_names
from metrics.bandit_metrics.gather import gather_bandit_metrics, get_bandit_metric_names
from metrics.cloc_metrics.gather import gather_cloc_metrics, get_cloc_metric_names
from metrics.flake8_metrics.gather import gather_flake8_metrics, get_flake8_metric_names
from metrics.lizard_metrics.gather import gather_lizard_metrics, get_lizard_metric_names
from metrics.pydocstyle_metrics.gather import gather_pydocstyle_metrics, get_pydocstyle_metric_names
from metrics.pyflakes_metrics.gather import gather_pyflakes_metrics, get_pyflakes_metric_names
from metrics.pylint_metrics.gather import gather_pylint_metrics, get_pylint_metric_names
from metrics.radon_metrics.gather import gather_radon_metrics, get_radon_metric_names


def get_metric_names_from_gatherer(
    gather_func: Callable[[str], Union[dict, list]],
    get_names_func: Callable[[], List[str]],
    file_path: str
) -> List[str]:
    """
    Determine metric names from a gatherer function's output.

    Args:
        gather_func (Callable): Function that extracts metrics and returns dict or list.
        get_names_func (Callable): Function returning list of metric names if gather_func returns a list.
        file_path (str): Path to file to pass to gather_func.

    Returns:
        List[str]: Ordered list of metric names.
    """
    result = gather_func(file_path)
    if isinstance(result, dict):
        return list(result.keys())
    elif isinstance(result, list):
        return get_names_func()
    return []


def get_all_metric_names() -> List[str]:
    """
    Aggregate and return all metric names from all metric gatherers
    in a consistent, ordered manner for CSV export or ML input features.

    Returns:
        List[str]: Combined ordered list of all metric names.
    """
    with tempfile.NamedTemporaryFile("w+", suffix=".py") as f:
        f.write("def foo(): pass\n")
        f.flush()
        file_path = f.name

        ast_keys = get_metric_names_from_gatherer(gather_ast_metrics, get_ast_metric_names, file_path)
        bandit_keys = get_metric_names_from_gatherer(gather_bandit_metrics, get_bandit_metric_names, file_path)
        cloc_keys = get_metric_names_from_gatherer(gather_cloc_metrics, get_cloc_metric_names, file_path)
        flake8_keys = get_metric_names_from_gatherer(gather_flake8_metrics, get_flake8_metric_names, file_path)
        lizard_keys = get_metric_names_from_gatherer(gather_lizard_metrics, get_lizard_metric_names, file_path)
        pydocstyle_keys = get_metric_names_from_gatherer(gather_pydocstyle_metrics, get_pydocstyle_metric_names, file_path)
        pyflakes_keys = get_metric_names_from_gatherer(gather_pyflakes_metrics, get_pyflakes_metric_names, file_path)
        pylint_keys = get_metric_names_from_gatherer(gather_pylint_metrics, get_pylint_metric_names, file_path)
        radon_keys = get_metric_names_from_gatherer(gather_radon_metrics, get_radon_metric_names, file_path)

    return (
        ast_keys +
        bandit_keys +
        cloc_keys +
        flake8_keys +
        lizard_keys +
        pydocstyle_keys +
        pyflakes_keys +
        pylint_keys +
        radon_keys
    )


def gather_all_metrics(file_path: str) -> List[Union[int, float]]:
    """
    Gathers all metrics from all gatherers into a single ordered list.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        List[int | float]: Combined list of metric values in consistent order.
    """
    ast_values = gather_ast_metrics(file_path)
    bandit_values = gather_bandit_metrics(file_path)
    cloc_values = gather_cloc_metrics(file_path)
    flake8_values = gather_flake8_metrics(file_path)
    lizard_values = gather_lizard_metrics(file_path)
    pydocstyle_values = gather_pydocstyle_metrics(file_path)
    pyflakes_values = gather_pyflakes_metrics(file_path)
    pylint_dict = gather_pylint_metrics(file_path)
    pylint_values = list(pylint_dict.values()) if isinstance(pylint_dict, dict) else []
    radon_values = gather_radon_metrics(file_path)

    return (
        ast_values +
        bandit_values +
        cloc_values +
        flake8_values +
        lizard_values +
        pydocstyle_values +
        pyflakes_values +
        pylint_values +
        radon_values
    )


__all__ = [
    "get_metric_names_from_gatherer",
    "get_all_metric_names",
    "gather_all_metrics",
]
