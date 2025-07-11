"""
Code Analyser GUI Package

This package provides a graphical interface for analysing Python source files
using static metrics collected from AST, Bandit, Flake8, Cloc, Lizard,
Pylint, Pydocstyle, and Pyflakes.

Modules:
- main.py: Launches the application with splash screen and initialisation
- gui_components.py: Builds and manages the main Tkinter layout
- gui_logic.py: Handles updates to the Treeview, charts, and summary footer
- file_ops.py: Triggers metric extraction and handles CSV/image export
- chart_utils.py: Generates and redraws charts with Matplotlib
- shared_state.py: Stores shared Tkinter variables and visual state
"""

__version__ = "1.0.0"
__author__ = "Samuel Harrison"
__email__ = "samuel.harrison@example.com"

# 🎯 Publicly exposed API for top-level imports
from .gui_components import launch_gui
from .file_ops import run_metric_extraction, run_directory_analysis
from .chart_utils import draw_chart
from .gui_logic import update_tree, update_footer_summary
