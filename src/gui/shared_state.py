# src/gui/shared_state.py

from typing import Optional
import tkinter as tk
from tkinter.ttk import Treeview

# 🔢 Collected metrics across all analysed files
# Format: { file_path: {metric_name: value, ...}, ... }
results: dict[str, dict] = {}

# 🖼️ Currently rendered Matplotlib chart canvas (used for redraw/export)
chart_canvas: Optional[object] = None  # Holds FigureCanvasTkAgg instance

# 📊 Frame where charts are drawn (created during GUI layout)
chart_frame: Optional[tk.Frame] = None

# 📋 TreeView showing all metrics by file
tree: Optional[Treeview] = None

# 🔍 Text filter variable bound to search box
filter_var: Optional[tk.StringVar] = None

# 📚 Metric scope radio toggle (e.g. 'ast', 'flake8', 'pylint', 'all')
metric_scope: Optional[tk.StringVar] = None

# 📈 Chart type toggle (e.g. 'bar', 'pie')
chart_type: Optional[tk.StringVar] = None

# 📊 TreeView used for summary totals/averages by metric
summary_tree: Optional[Treeview] = None
