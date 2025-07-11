"""
Code Analyser GUI Package

This package provides a unified graphical user interface for analysing Python source files
via static code metrics collected from multiple tools, including:

- AST: Analyses abstract syntax tree structure
- Bandit: Detects security issues
- Flake8: Linting and style checks
- Cloc: Counts lines of code, comments, and blank lines
- Lizard: Measures cyclomatic complexity and function length
- Pylint: Reports coding standard violations and maintainability issues
- Pydocstyle: Checks docstring conventions
- Pyflakes: Identifies syntax errors and undefined variables

Modules:
- main.py: Launches the application with a splash screen and full initialisation of all UI components.
- gui_components.py: Builds and manages the main Tkinter interface with tabbed navigation and layout.
- gui_logic.py: Updates the Treeview structure, visual summaries, and chart refresh logic.
- file_ops.py: Coordinates metric collection, file/directory analysis, and CSV/image export.
- chart_utils.py: Responsible for rendering and formatting interactive charts with Matplotlib.
- shared_state.py: Defines and shares Tkinter variables and GUI state across the application.
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
