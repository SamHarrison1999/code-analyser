"""
Shared GUI State Module

This module defines globally shared variables for coordinating state between
different GUI components (e.g., chart rendering, metric filters, and TreeViews).
"""

from typing import Optional
import tkinter as tk
from tkinter.ttk import Treeview
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 🔢 Collected metrics across all analysed files
# Format: { file_path: {metric_name: value, ...}, ... }
results: dict[str, dict] = {}

# 🖼️ Currently rendered Matplotlib chart canvas (used for redraw/export)
chart_canvas: Optional[FigureCanvasTkAgg] = None

# 📊 Frame widget where charts are drawn (set during GUI layout)
chart_frame: Optional[tk.Frame] = None

# 📋 TreeView widget displaying per-file metrics
tree: Optional[Treeview] = None

# 🔍 Search filter bound to Entry box to filter tree contents
filter_var: Optional[tk.StringVar] = None

# 📚 Metric scope selector (e.g. 'ast', 'flake8', 'pylint', 'all')
metric_scope: Optional[tk.StringVar] = None

# 📈 Chart type toggle (e.g. 'bar', 'pie')
chart_type: Optional[tk.StringVar] = None

# 📊 TreeView widget displaying totals and averages by metric
summary_tree: Optional[Treeview] = None
