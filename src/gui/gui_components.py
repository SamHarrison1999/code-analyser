import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from gui.file_ops import run_metric_extraction, run_directory_analysis, export_to_csv
from gui.chart_utils import draw_chart, include_metric, redraw_last_chart
from gui.gui_logic import update_tree, update_footer_summary
from gui.utils import flatten_metrics  # ✅ utility import


def launch_gui() -> None:
    """Initialise and launch the Code Analyser Tkinter GUI."""
    from gui import shared_state  # ✅ defer to avoid circular import

    root = tk.Tk()
    root.title("🧠 Code Analyser GUI")
    root.geometry("1000x900")

    # Initialise shared state variables
    shared_state.filter_var = tk.StringVar()
    shared_state.metric_scope = tk.StringVar(value="all")
    shared_state.chart_type = tk.StringVar(value="bar")
    shared_state.results = {}
    shared_state.current_file_path = ""  # ✅ Track current file for charting

    root.bind("<Configure>", on_resize)

    # Top control buttons
    top_frame = tk.Frame(root)
    top_frame.pack(pady=10)

    tk.Button(top_frame, text="📂 File", command=prompt_and_extract_file).grid(row=0, column=0, padx=5)
    tk.Button(top_frame, text="📁 Folder", command=run_directory_analysis).grid(row=0, column=1, padx=5)
    tk.Button(top_frame, text="📊 File Chart", command=show_chart).grid(row=0, column=2, padx=5)
    tk.Button(top_frame, text="📊 Dir Chart", command=show_directory_summary_chart).grid(row=0, column=3, padx=5)
    tk.Button(top_frame, text="📄 Export CSV", command=export_to_csv).grid(row=0, column=4, padx=5)
    tk.Button(top_frame, text="Exit", command=root.destroy).grid(row=0, column=5, padx=5)

    # Chart options
    option_frame = tk.Frame(root)
    option_frame.pack(pady=5)

    tk.Label(option_frame, text="Chart Type:").pack(side=tk.LEFT)
    tk.Radiobutton(option_frame, text="Bar", variable=shared_state.chart_type, value="bar").pack(side=tk.LEFT)
    tk.Radiobutton(option_frame, text="Pie", variable=shared_state.chart_type, value="pie").pack(side=tk.LEFT)

    tk.Label(option_frame, text="Metric Scope:").pack(side=tk.LEFT, padx=(20, 5))
    for label, value in [
        ("AST", "ast"), ("Bandit", "bandit"), ("Cloc", "cloc"),
        ("Flake8", "flake8"), ("Lizard", "lizard"), ("Pydocstyle", "pydocstyle"),
        ("Pyflakes", "pyflakes"), ("Pylint", "pylint"), ("Radon", "radon"), ("All", "all")
    ]:
        tk.Radiobutton(option_frame, text=label, variable=shared_state.metric_scope, value=value).pack(side=tk.LEFT)

    # Filter entry bar
    filter_frame = tk.Frame(root)
    filter_frame.pack(fill=tk.X, padx=10, pady=5)
    shared_state.filter_var.trace_add("write", on_filter_change)
    tk.Label(filter_frame, text="Filter: ").pack(side=tk.LEFT)
    tk.Entry(filter_frame, textvariable=shared_state.filter_var, width=40).pack(side=tk.LEFT, expand=True, fill=tk.X)

    # Tabbed layout
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    # 📊 Charts tab
    chart_tab = tk.Frame(notebook)
    chart_canvas_widget = tk.Canvas(chart_tab)
    chart_scroll = ttk.Scrollbar(chart_tab, orient="vertical", command=chart_canvas_widget.yview)
    scrollable_chart = tk.Frame(chart_canvas_widget)
    scrollable_chart.bind("<Configure>", lambda e: chart_canvas_widget.configure(scrollregion=chart_canvas_widget.bbox("all")))
    chart_canvas_widget.create_window((0, 0), window=scrollable_chart, anchor="nw")
    chart_canvas_widget.configure(yscrollcommand=chart_scroll.set)
    chart_canvas_widget.pack(side="left", fill="both", expand=True)
    chart_scroll.pack(side="right", fill="y")
    shared_state.chart_frame = scrollable_chart
    shared_state.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    notebook.add(chart_tab, text="📊 Charts")

    # 📋 Metrics tab
    tree_tab = tk.Frame(notebook)
    tree_scroll = ttk.Scrollbar(tree_tab, orient="vertical")
    tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    shared_state.tree = ttk.Treeview(tree_tab, columns=("Metric", "Value"), show="headings", yscrollcommand=tree_scroll.set)
    for col in ("Metric", "Value"):
        shared_state.tree.heading(col, text=col)
        shared_state.tree.column(col, anchor="w")
    shared_state.tree.pack(fill=tk.BOTH, expand=True)
    tree_scroll.config(command=shared_state.tree.yview)
    notebook.add(tree_tab, text="📋 Metrics")

    # 📈 Summary tab
    summary_tab = tk.Frame(notebook)
    summary_scroll = ttk.Scrollbar(summary_tab, orient="vertical")
    summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    shared_state.summary_tree = ttk.Treeview(summary_tab, columns=("Metric", "Total", "Average"), show="headings", yscrollcommand=summary_scroll.set)
    for col in ("Metric", "Total", "Average"):
        shared_state.summary_tree.heading(col, text=col)
        shared_state.summary_tree.column(col, anchor="center")
    shared_state.summary_tree.column("Metric", anchor="w", width=300)
    shared_state.summary_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    summary_scroll.config(command=shared_state.summary_tree.yview)
    notebook.add(summary_tab, text="📈 Summary")

    root.mainloop()


def on_resize(event: tk.Event) -> None:
    """Redraw the chart when the main window is resized."""
    if event.widget == event.widget.winfo_toplevel():
        redraw_last_chart()


def prompt_and_extract_file() -> None:
    """Prompt the user to select a file and extract metrics."""
    from gui import shared_state
    path = filedialog.askopenfilename()
    if path:
        shared_state.current_file_path = path  # ✅ Save current path
        run_metric_extraction(path)
        update_tree(shared_state.tree, path)
        flat_metrics = flatten_metrics(shared_state.results.get(path, {}))
        update_footer_summary(shared_state.summary_tree, flat_metrics)


def on_filter_change(*args) -> None:
    """Update the Treeview and footer summary when the filter changes."""
    from gui import shared_state
    keys = list(shared_state.results.keys())
    file_path = keys[0] if keys else None
    if file_path:
        update_tree(shared_state.tree, file_path)
        flat_metrics = flatten_metrics(shared_state.results.get(file_path, {}))
        update_footer_summary(shared_state.summary_tree, flat_metrics)


def show_chart() -> None:
    """Display a chart of metrics for the currently selected file, filtered by scope."""
    from gui import shared_state

    if not shared_state.results or not shared_state.tree:
        return

    scope = shared_state.metric_scope.get()
    filename = shared_state.current_file_path
    file_metrics = shared_state.results.get(filename, {})

    flat = flatten_metrics(file_metrics)
    filtered = {
        k: float(v)
        for k, v in flat.items()
        if include_metric(k)
        and isinstance(v, (int, float, str))
        and str(v).replace(".", "", 1).isdigit()
    }

    if not filtered:
        messagebox.showinfo("No Metrics", f"No metrics found for scope: {scope}")
        return

    keys = list(filtered.keys())
    vals = [round(float(filtered[k]), 2) for k in keys]
    draw_chart(keys, vals, f"Metrics - Scope: {scope}", "scope_chart.png")


def show_directory_summary_chart() -> None:
    """Display a summary chart for all aggregated metrics across files based on selected metric scope."""
    from gui import shared_state

    if not shared_state.results:
        messagebox.showinfo("No Data", "No analysis has been run.")
        return

    scope = shared_state.metric_scope.get()
    combined = {}

    for file_data in shared_state.results.values():
        flat = flatten_metrics(file_data)
        for k, v in flat.items():
            if include_metric(k):
                try:
                    combined[k] = combined.get(k, 0) + float(v)
                except (TypeError, ValueError):
                    continue  # skip non-numeric values

    if not combined:
        messagebox.showinfo("No Metrics", f"No numeric metrics available for scope: {scope}")
        return

    keys = list(combined.keys())
    vals = [round(combined[k], 2) for k in keys]
    draw_chart(keys, vals, f"Metrics - Scope: {scope}", f"summary_scope_{scope}.png")
