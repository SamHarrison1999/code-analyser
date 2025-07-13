"""
Code Analyser GUI Package

This package provides a unified graphical user interface for analysing Python source files
via static code metrics aggregated from dynamically discovered tools. It supports interactive exploration
through charts, filters, and file-level or directory-wide summaries.

Integrated Tools (discovered dynamically):
- AST: Analyses abstract syntax tree structure
- Bandit: Detects security vulnerabilities and unsafe code patterns
- Cloc: Counts lines of code, comments, and blank lines
- Flake8: Reports PEP8 violations and formatting inconsistencies
- Lizard: Measures cyclomatic complexity and token-level metrics
- Pydocstyle: Validates docstring compliance with Python standards
- Pyflakes: Identifies undefined names, unused imports, and syntax errors
- Pylint: Scores code across convention, warning, error, and refactor dimensions
- Radon: Computes Halstead complexity metrics and logical line counts
- Vulture: Finds unused code such as dead functions and imports

Modules:
- main.py: Bootstraps the GUI, splash screen, and shared state
- gui_components.py: Builds layout tabs, input widgets, and chart containers
- gui_logic.py: Updates the Treeview, summary view, and metric state logic
- file_ops.py: Runs metric collection, batch directory scans, and CSV exports
- chart_utils.py: Renders scrollable Matplotlib pie/bar charts with hover support
- shared_state.py: Manages all Tkinter variables and cross-component state sharing
- utils.py: Provides helpers for flattening metrics and discovering dynamic plugin scopes

Fixes:
- Ensures all extracted metrics are merged and flattened before filtering or charting
- Adds plugin discovery to auto-detect tools from the metrics directory
- Resolves metric loss caused by nested metric formats (e.g. pylint.refactor, lizard.total_function_count)
"""

__version__ = "1.1.0"
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
