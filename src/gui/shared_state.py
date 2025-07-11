# src/gui/shared_state.py

from typing import Optional
import tkinter as tk
from tkinter.ttk import Treeview

# 🔢 Collected metrics across all files
results: dict[str, dict] = {}

# 🖼️ Currently displayed Matplotlib chart canvas (for redraw/export)
chart_canvas: Optional[object] = None

# 📊 Frame where charts are rendered (set during GUI init)
chart_frame: Optional[tk.Frame] = None

# 📋 TreeView widget for file-metric display
tree: Optional[Treeview] = None

# 🔍 User filter input (set after root window is created)
filter_var: Optional[tk.StringVar] = None

# 📚 What type of metric set to include (e.g. 'bandit', 'all') – dynamic via GUI
metric_scope: Optional[tk.StringVar] = None

# 📈 Chart style preference (e.g. 'bar', 'pie') – dynamic via GUI
chart_type: Optional[tk.StringVar] = None

# 📊 Summary statistics view widget (totals and averages)
summary_tree: Optional[Treeview] = None
