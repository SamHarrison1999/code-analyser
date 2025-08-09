# File: src/metrics/gather.py

from typing import Dict, Any

from ml.ai_metric_names import gather_ai_metric_names
from ml.overlay_gatherer import gather_ai_overlays
from metrics.ast_metrics.gather import gather_ast_metrics_bundle
from metrics.bandit_metrics.gather import gather_bandit_metrics_bundle
from metrics.flake8_metrics.gather import gather_flake8_metrics_bundle
from metrics.cloc_metrics.gather import gather_cloc_metrics_bundle
from metrics.pylint_metrics.gather import gather_pylint_metrics_bundle
from metrics.radon_metrics.gather import gather_radon_metrics_bundle
from metrics.vulture_metrics.gather import gather_vulture_metrics_bundle
from metrics.lizard_metrics.gather import gather_lizard_metrics_bundle
from metrics.pyflakes_metrics.gather import gather_pyflakes_metrics_bundle
from metrics.pydocstyle_metrics.gather import gather_pydocstyle_metrics_bundle
from metrics.sonar_metrics.gather import gather_sonar_metrics_bundle


# ✅ Helper to load AI overlays safely
def get_ai_overlay_metrics(file_path: str) -> dict:
    metrics, _ = gather_ai_overlays(file_path)
    return metrics or {}


# ✅ Helper to expose metric names without recursion
def get_ai_overlay_metric_names() -> list[str]:
    return gather_ai_metric_names()


def gather_all_metrics(file_path: str, include_ai: bool = True) -> Dict[str, Any]:
    all_metrics: Dict[str, Any] = {}

    for gatherer in [
        gather_ast_metrics_bundle,
        gather_bandit_metrics_bundle,
        gather_flake8_metrics_bundle,
        gather_cloc_metrics_bundle,
        gather_pylint_metrics_bundle,
        gather_radon_metrics_bundle,
        gather_vulture_metrics_bundle,
        gather_lizard_metrics_bundle,
        gather_pyflakes_metrics_bundle,
        gather_pydocstyle_metrics_bundle,
        gather_sonar_metrics_bundle,
    ]:
        try:
            bundle = gatherer(file_path)
            all_metrics.update(bundle)
        except Exception:
            continue

    if include_ai:
        try:
            all_metrics.update(get_ai_overlay_metrics(file_path))
        except Exception:
            pass

    return all_metrics


def get_all_metric_names() -> list[str]:
    names = []
    for gatherer in [
        gather_ast_metrics_bundle,
        gather_bandit_metrics_bundle,
        gather_flake8_metrics_bundle,
        gather_cloc_metrics_bundle,
        gather_pylint_metrics_bundle,
        gather_radon_metrics_bundle,
        gather_vulture_metrics_bundle,
        gather_lizard_metrics_bundle,
        gather_pyflakes_metrics_bundle,
        gather_pydocstyle_metrics_bundle,
        gather_sonar_metrics_bundle,
    ]:
        try:
            bundle = gatherer("dummy.py")
            names.extend(bundle.keys())
        except Exception:
            continue

    names.extend(get_ai_overlay_metric_names())
    return sorted(set(names))