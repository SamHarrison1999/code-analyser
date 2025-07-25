"""
Code Analyser GUI Package

This package provides a unified graphical user interface for analysing Python source files
using static code metrics aggregated from dynamically discovered tools. It supports interactive exploration
through charts, filters, and file-level or directory-wide summaries.

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

Modules:
- main.py: Bootstraps the GUI, splash screen, and shared state
- gui_components.py: Builds layout tabs, input widgets, and chart containers
- gui_logic.py: Updates the TreeView, summary view, and metric state logic
- file_ops.py: Runs metric extraction, directory-wide scans, and CSV export
- chart_utils.py: Renders scrollable Matplotlib pie/bar charts with hover support
- shared_state.py: Manages Tkinter variables and cross-component shared state
- utils.py: Flattens nested metrics and resolves dynamic plugin metric scopes

Fixes and Enhancements:
- Ensures all extracted metrics are flattened for charting and filtering
- Adds plugin auto-discovery across all metric types including SonarQube
- Prevents metric omissions caused by nested tool output formats (e.g., pylint.warning, sonar.bugs)
- Enables dynamic filtering of metric charts by tool scope via shared state
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
