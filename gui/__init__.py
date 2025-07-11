# gui/__init__.py

"""
Code Analyser GUI Package

This package provides the graphical interface for analysing Python source files
using metrics collected from tools like AST, Bandit, Flake8, Cloc, Lizard, and others.

Modules:
- main.py: Entry point with splash screen
- gui_components.py: GUI layout and widget configuration
- gui_logic.py: Handles GUI updates and summary logic
- file_ops.py: File and folder metric analysis, CSV export
- chart_utils.py: Matplotlib-based chart rendering
- shared_state.py: Shared tkinter state and global results
"""

__version__ = "1.0.0"
__author__ = "Samuel Harrison"
__email__ = "samuel.harrison@example.com"

# Expose commonly used entry points for ease of import
from .gui_components import launch_gui
from .file_ops import run_metric_extraction, run_directory_analysis
from .chart_utils import draw_chart
from .gui_logic import update_tree, update_footer_summary
