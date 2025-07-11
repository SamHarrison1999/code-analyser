from typing import Optional
import tkinter as tk
from tkinter.ttk import Treeview

# 🔢 Collected metrics across all analysed files
# Format: { file_path: {metric_name: value, ...}, ... }
results: dict[str, dict] = {}

# 🖼️ Currently rendered Matplotlib chart canvas (used for redraw/export)
# Typically an instance of FigureCanvasTkAgg
chart_canvas: Optional[object] = None

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
