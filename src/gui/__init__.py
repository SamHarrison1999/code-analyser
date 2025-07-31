# File: code_analyser\src\gui\__init__.py

"""
Code Analyser GUI Package

This package provides a unified graphical user interface for analysing Python source files
using static and AI-enhanced code metrics aggregated from dynamically discovered tools and models.
It supports interactive exploration through scrollable charts, filtering, overlays, and
file-level or directory-wide summaries.

Integrated Tools (dynamically discovered):
- AST: Analyses abstract syntax tree structure
- Bandit: Detects security vulnerabilities and unsafe code patterns
- Cloc: Counts lines of code, comments, and blank lines
- Flake8: Reports PEP8 violations and formatting inconsistencies
- Lizard: Measures cyclomatic complexity and maintainability index
- Pydocstyle: Validates docstring compliance with Python standards
- Pyflakes: Identifies undefined names, unused imports, and syntax errors
- Pylint: Scores code across convention, warning, error, and refactor categories
- Radon: Computes Halstead complexity and logical line counts
- Vulture: Detects unused code such as dead functions and imports
- SonarQube: Aggregates multi-dimensional metrics (bugs, coverage, code smells, etc.)
- Together AI: Generates annotation overlays for ML signals and SAST risks
- AI Agent: Applies supervised + RL models to generate confidence heatmaps and global scores

Modules:
- main.py: Bootstraps the GUI, splash screen, and shared state
- gui_components.py: Builds layout tabs, input widgets, chart containers, and AI toggles
- gui_logic.py: Updates the TreeView, footer summary, overlay signals, and dynamic scope
- file_ops.py: Runs metric extraction, batch AI annotation, and CSV/HTML export
- chart_utils.py: Renders scrollable Matplotlib pie/bar charts with hover and overlays
- shared_state.py: Manages Tkinter variables and cross-component shared state
- utils.py: Flattens nested metrics and resolves dynamic plugin metric scopes
- overlay_loader.py: Loads Together AI and RL overlays, merges signals and token heatmaps
- heatmap_renderer.py: Renders token-level overlays with severity/confidence filtering and exports

Fixes and Enhancements:
- Enables scrollable metric charts with per-tool scope switching
- Adds plugin auto-discovery across all metric types including SonarQube and Vulture
- Supports overlay toggles for AI signal confidence, severity filters, and per-line heatmaps
- Embeds TensorBoard visualisation hook for AI training logs and rewards
- Exports annotation summaries in CSV, JSON, and HTML dashboard formats
- Ensures live overlays auto-update on TreeView navigation and filtering
- Adds 'Export All' button with tooltip and multi-format support (CSV, PNG, JSON)
"""

__version__ = "1.2.1"
__author__ = "Samuel Harrison"
__email__ = "sh18784@essex.ac.uk"

# 🎯 Publicly exposed API for top-level imports
from .gui_components import launch_gui
from .file_ops import run_metric_extraction, run_directory_analysis
from .chart_utils import draw_chart
from .gui_logic import update_tree, update_footer_summary

__all__ = [
    "launch_gui",
    "run_metric_extraction",
    "run_directory_analysis",
    "draw_chart",
    "update_tree",
    "update_footer_summary",
]
