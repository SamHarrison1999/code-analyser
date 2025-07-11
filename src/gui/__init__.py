# src/gui/__init__.py

"""
Code Analyser GUI Package

This package provides a graphical interface for analysing Python source files
using static metrics collected from AST, Bandit, Flake8, Cloc, Lizard, and other tools.

Modules:
- main.py: Launches the application with splash screen
- gui_components.py: Builds and manages the main Tkinter layout
- gui_logic.py: Handles data updates for trees and summaries
- file_ops.py: Triggers metric extraction and export operations
- chart_utils.py: Generates and redraws charts with Matplotlib
- shared_state.py: Holds shared state and Tkinter variables
"""

__version__ = "1.0.0"
__author__ = "Samuel Harrison"
__email__ = "samuel.harrison@example.com"

# 🎯 Publicly exposed API for top-level imports
from .gui_components import launch_gui
from .file_ops import run_metric_extraction, run_directory_analysis
from .chart_utils import draw_chart
from .gui_logic import update_tree, update_footer_summary
