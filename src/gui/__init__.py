"""
Code Analyser GUI Package

This package provides a graphical interface for analysing Python source files
using static metrics collected from AST, Bandit, Flake8, Cloc, Lizard,
Pylint, Pydocstyle, and Pyflakes.

Modules:
- main.py: Launches the GUI application with a splash screen and full initialisation.
- gui_components.py: Constructs and manages the main Tkinter layout and tabbed interface.
- gui_logic.py: Handles updates to the Treeview, metric charts, and summary footer.
- file_ops.py: Executes metric extraction and manages CSV/image export functionality.
- chart_utils.py: Draws, updates, and formats charts using Matplotlib.
- shared_state.py: Stores shared Tkinter variables and visual state across components.
"""

__version__ = "1.1.0"
__author__ = "Samuel Harrison"
__email__ = "samuel.harrison@example.com"

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
