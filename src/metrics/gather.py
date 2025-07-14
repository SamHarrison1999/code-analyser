"""
Unified metric gathering entry point.

This module provides:
- `gather_all_metrics(file_path: str)`: Runs all enabled metric sources and returns combined results.
- `get_all_metric_names()`: Returns the consistent list of metric names used in CLI/CSV outputs.
"""

import logging
from typing import Union

from metrics.ast_metrics.gather import gather_ast_metrics, get_ast_metric_names
from metrics.bandit_metrics.gather import gather_bandit_metrics, get_bandit_metric_names
from metrics.cloc_metrics.gather import gather_cloc_metrics, get_cloc_metric_names
from metrics.flake8_metrics.gather import gather_flake8_metrics, get_flake8_metric_names
from metrics.lizard_metrics.gather import gather_lizard_metrics, get_lizard_metric_names
from metrics.pydocstyle_metrics.gather import gather_pydocstyle_metrics, get_pydocstyle_metric_names
from metrics.pyflakes_metrics.gather import gather_pyflakes_metrics, get_pyflakes_metric_names
from metrics.pylint_metrics.gather import gather_pylint_metrics, get_pylint_metric_names
from metrics.radon_metrics.gather import gather_radon_metrics, get_radon_metric_names
from metrics.vulture_metrics.gather import gather_vulture_metrics, get_vulture_metric_names

from metrics.sonar_metrics import get_metric_gatherer, get_metric_names as get_sonar_metric_names


def gather_all_metrics(file_path: str) -> dict[str, Union[int, float, str]]:
    """
    Runs all configured metric gatherers and merges their outputs.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        dict[str, int | float | str]: All collected metrics as a single flat dictionary.
    """
    all_metrics: dict[str, Union[int, float, str]] = {}

    metric_sources = [
        (gather_ast_metrics, get_ast_metric_names),
        (gather_bandit_metrics, get_bandit_metric_names),
        (gather_cloc_metrics, get_cloc_metric_names),
        (gather_flake8_metrics, get_flake8_metric_names),
        (gather_lizard_metrics, get_lizard_metric_names),
        (gather_pydocstyle_metrics, get_pydocstyle_metric_names),
        (gather_pyflakes_metrics, get_pyflakes_metric_names),
        (gather_pylint_metrics, get_pylint_metric_names),
        (gather_radon_metrics, get_radon_metric_names),
        (gather_vulture_metrics, get_vulture_metric_names),
    ]

    for gather_func, name_func in metric_sources:
        try:
            values = gather_func(file_path)
            names = name_func()
            if len(values) != len(names):
                logging.warning(
                    f"[gather_all_metrics] Metric count mismatch for {gather_func.__name__}: "
                    f"{len(names)} names vs {len(values)} values"
                )
            all_metrics.update(dict(zip(names, values)))
        except Exception as e:
            logging.error(f"[gather_all_metrics] Failed to gather from {gather_func.__name__}: {e}")
            all_metrics[gather_func.__name__] = f"error: {e}"

    # ✅ Handle SonarQube metrics separately with correct structure
    try:
        sonar_gatherer = get_metric_gatherer()
        sonar_metrics = sonar_gatherer(file_path)
        if not isinstance(sonar_metrics, dict):
            raise ValueError("Sonar gatherer did not return a dictionary")
        all_metrics.update(sonar_metrics)
    except Exception as e:
        logging.error(f"[gather_all_metrics] Failed to gather SonarQube metrics: {e}")
        all_metrics["sonar"] = f"error: {e}"

    return all_metrics


def get_all_metric_names() -> list[str]:
    """
    Returns consistent metric name list for use in output and headers.

    Returns:
        list[str]: Known metrics used for ordering and fallback.
    """
    return (
        get_ast_metric_names() +
        get_bandit_metric_names() +
        get_cloc_metric_names() +
        get_flake8_metric_names() +
        get_lizard_metric_names() +
        get_pydocstyle_metric_names() +
        get_pyflakes_metric_names() +
        get_pylint_metric_names() +
        get_radon_metric_names() +
        get_vulture_metric_names() +
        get_sonar_metric_names()  # ✅ Properly loaded from plugin registry
    )
