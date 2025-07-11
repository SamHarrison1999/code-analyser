"""
Top-level package initializer for the `metrics` module.

Exposes core extractors and metric gatherers for:
- AST (structure + constructs)
- Bandit (security analysis)
- Cloc (line and comment statistics)
- Flake8 (style and lint errors)
- Lizard (complexity + maintainability)
- Pydocstyle (docstring compliance)
- Pyflakes (syntax + import warnings)
- Pylint (multi-category quality issues)
- Pylint Plugins (extendable metric rules)
- Radon (code complexity and Halstead metrics)
- Unified metric aggregation (gather_all_metrics)
"""

from metrics.ast_metrics.extractor import ASTMetricExtractor
from metrics.bandit_metrics.extractor import BanditExtractor
from metrics.cloc_metrics.extractor import ClocExtractor
from metrics.flake8_metrics.extractor import Flake8Extractor
from metrics.lizard_metrics.extractor import LizardExtractor, extract_lizard_metrics
from metrics.pydocstyle_metrics.extractor import PydocstyleExtractor
from metrics.pyflakes_metrics.extractor import PyflakesExtractor, extract_pyflakes_metrics
from metrics.pylint_metrics.extractor import PylintMetricExtractor
from metrics.radon_metrics.extractor import run_radon
from metrics.pylint_metrics.plugins.default_plugins import load_plugins

from metrics.ast_metrics.gather import gather_ast_metrics
from metrics.bandit_metrics.gather import gather_bandit_metrics
from metrics.cloc_metrics.gather import gather_cloc_metrics
from metrics.flake8_metrics.gather import gather_flake8_metrics
from metrics.lizard_metrics.gather import gather_lizard_metrics
from metrics.pydocstyle_metrics.gather import gather_pydocstyle_metrics
from metrics.pyflakes_metrics.gather import gather_pyflakes_metrics
from metrics.pylint_metrics.gather import gather_pylint_metrics
# ✅ Minimal and safe (let gui_logic or gather.py handle direct imports)
from .radon_metrics import gather


from metrics.gather import gather_all_metrics, get_all_metric_names

__all__ = [
    # Extractors
    "ASTMetricExtractor",
    "BanditExtractor",
    "ClocExtractor",
    "Flake8Extractor",
    "LizardExtractor",
    "PydocstyleExtractor",
    "PyflakesExtractor",
    "PylintMetricExtractor",

    # Utilities / Plugins
    "extract_lizard_metrics",
    "extract_pyflakes_metrics",
    "load_plugins",

    # Metric gatherers
    "gather_ast_metrics",
    "gather_bandit_metrics",
    "gather_cloc_metrics",
    "gather_flake8_metrics",
    "gather_lizard_metrics",
    "gather_pydocstyle_metrics",
    "gather_pyflakes_metrics",
    "gather_pylint_metrics",
    "gather_radon_metrics",
    "run_radon",
    "gather_all_metrics",
    "get_all_metric_names",
]
