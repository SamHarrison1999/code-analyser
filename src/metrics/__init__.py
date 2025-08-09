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
- Radon (code complexity and Halstead metrics)
- Vulture (unused code detection)
- SonarQube (coverage, bugs, smells, duplication, ratings)
- Unified metric aggregation (gather_all_metrics)
"""

# === Core Extractors ===
from metrics.ast_metrics.extractor import ASTMetricExtractor
from metrics.bandit_metrics.extractor import BanditExtractor
from metrics.cloc_metrics.extractor import ClocExtractor
from metrics.flake8_metrics.extractor import Flake8Extractor
from metrics.lizard_metrics.extractor import LizardExtractor, extract_lizard_metrics
from metrics.pydocstyle_metrics.extractor import PydocstyleExtractor
from metrics.pyflakes_metrics.extractor import (
    PyflakesExtractor,
    extract_pyflakes_metrics,
)
from metrics.pylint_metrics.extractor import PylintMetricExtractor
from metrics.radon_metrics.extractor import RadonMetricExtractor, extract_radon_metrics
from metrics.vulture_metrics.extractor import VultureMetricExtractor
from metrics.sonar_metrics.extractor import SonarMetricExtractor
from metrics.sonar_metrics.scanner import run_sonar

# === Plugin Loaders ===
from metrics.ast_metrics import load_plugins as load_ast_plugins
from metrics.bandit_metrics import load_plugins as load_bandit_plugins
from metrics.cloc_metrics import load_plugins as load_cloc_plugins
from metrics.flake8_metrics import load_plugins as load_flake8_plugins
from metrics.lizard_metrics import load_plugins as load_lizard_plugins
from metrics.pydocstyle_metrics import load_plugins as load_pydocstyle_plugins
from metrics.pyflakes_metrics import load_plugins as load_pyflakes_plugins
from metrics.pylint_metrics import load_plugins as load_pylint_plugins
from metrics.radon_metrics import load_plugins as load_radon_plugins
from metrics.vulture_metrics import load_plugins as load_vulture_plugins
from metrics.sonar_metrics import load_plugins as load_sonar_plugins

# === Metric Gather Functions ===
from metrics.ast_metrics.gather import gather_ast_metrics
from metrics.bandit_metrics.gather import gather_bandit_metrics
from metrics.cloc_metrics.gather import gather_cloc_metrics
from metrics.flake8_metrics.gather import gather_flake8_metrics
from metrics.lizard_metrics.gather import gather_lizard_metrics
from metrics.pydocstyle_metrics.gather import gather_pydocstyle_metrics
from metrics.pyflakes_metrics.gather import gather_pyflakes_metrics
from metrics.pylint_metrics.gather import gather_pylint_metrics
from metrics.radon_metrics.gather import (
    gather_radon_metrics,
    gather_radon_metrics_bundle,
    get_radon_metric_names,
)
from metrics.vulture_metrics.gather import (
    gather_vulture_metrics,
    gather_vulture_metrics_bundle,
    get_vulture_metric_names,
)
from metrics.sonar_metrics.gather import (
    gather_sonar_metrics,
    get_sonar_metric_names,
)

# === AI Overlays (Token-level, Confidence, Severity) ===
from metrics.ai_overlays import (
    gather_ai_metric_overlays,
    extract_token_heatmap,
    get_ai_metric_names,
    gather_all_metric_names_with_ai,  # ✅ include this now
)


# === Unified Aggregation ===
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
    "RadonMetricExtractor",
    "VultureMetricExtractor",
    "SonarMetricExtractor",
    # Plugin loaders
    "load_ast_plugins",
    "load_bandit_plugins",
    "load_cloc_plugins",
    "load_flake8_plugins",
    "load_lizard_plugins",
    "load_pydocstyle_plugins",
    "load_pyflakes_plugins",
    "load_pylint_plugins",
    "load_radon_plugins",
    "load_vulture_plugins",
    "load_sonar_plugins",
    # Utility extractors
    "extract_lizard_metrics",
    "extract_pyflakes_metrics",
    "extract_radon_metrics",
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
    "gather_radon_metrics_bundle",
    "gather_vulture_metrics",
    "gather_vulture_metrics_bundle",
    "gather_sonar_metrics",
    # AI overlays
    "gather_ai_metric_overlays",
    "extract_token_heatmap",
    "get_ai_metric_names",
    "gather_all_metric_names_with_ai",
    # Raw tool runners
    "run_sonar",
    # Unified API
    "gather_all_metrics",
    "get_all_metric_names",
]