# ⚠️ SAST Risk: GUI callback arguments were being evaluated immediately instead of being passed as callables
# ✅ Best Practice: Use lambda to defer function execution

import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import sys

# ✅ Explicit imports for GUI components and logic
from gui.file_ops import run_metric_extraction, run_directory_analysis, export_to_csv
from gui.chart_utils import draw_chart, redraw_last_chart, filter_metrics_by_scope
from gui.gui_logic import update_tree, update_footer_summary
from gui.utils import flatten_metrics
from gui.shared_state import get_shared_state  # Ensures shared state is accessible at top level


def launch_gui(root: tk.Tk) -> None:
    shared_state = get_shared_state()
    root.title("🧠 Code Analyser GUI")
    root.geometry("1000x900")
    root.bind("<Configure>", on_resize)

    def clean_exit(root: tk.Tk) -> None:
        """Safely shut down the application."""
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception as e:
            logging.warning(f"⚠️ Could not close matplotlib figures: {e}")
        try:
            root.quit()
            root.destroy()
        except Exception as e:
            logging.warning(f"⚠️ Error during GUI shutdown: {e}")
        logging.info("📤 Clean exit triggered")
        sys.exit(0)

    # ✅ Use lambda to defer execution
    root.protocol("WM_DELETE_WINDOW", lambda: clean_exit(root))

    top_frame = tk.Frame(root)
    top_frame.pack(pady=10)
    tk.Button(top_frame, text="📂 File", command=lambda: prompt_and_extract_file()).grid(row=0, column=0, padx=5)
    tk.Button(top_frame, text="📁 Folder", command=lambda: run_directory_analysis()).grid(row=0, column=1, padx=5)
    tk.Button(top_frame, text="📊 File Chart", command=lambda: show_chart()).grid(row=0, column=2, padx=5)
    tk.Button(top_frame, text="📊 Dir Chart", command=lambda: show_directory_summary_chart()).grid(row=0, column=3, padx=5)
    tk.Button(top_frame, text="📄 Export CSV", command=lambda: export_to_csv()).grid(row=0, column=4, padx=5)
    tk.Button(top_frame, text="Exit", command=lambda: clean_exit(root)).grid(row=0, column=5, padx=5)

    option_frame = tk.Frame(root)
    option_frame.pack(pady=5)
    tk.Label(option_frame, text="Chart Type:").pack(side=tk.LEFT)
    tk.Radiobutton(option_frame, text="Bar", variable=shared_state.chart_type, value="bar").pack(side=tk.LEFT)
    tk.Label(option_frame, text="Metric Scope:").pack(side=tk.LEFT, padx=(20, 5))

    for label, value in [
        ("AST", "ast"), ("Bandit", "bandit"), ("Cloc", "cloc"),
        ("Flake8", "flake8"), ("Lizard", "lizard"), ("Pydocstyle", "pydocstyle"),
        ("Pyflakes", "pyflakes"), ("Pylint", "pylint"), ("Radon", "radon"),
        ("Vulture", "vulture"), ("Sonar", "sonar"), ("All", "all")
    ]:
        tk.Radiobutton(
            option_frame, text=label, variable=shared_state.metric_scope,
            value=value, command=lambda: refresh_chart_on_scope_change()
        ).pack(side=tk.LEFT)

    filter_frame = tk.Frame(root)
    filter_frame.pack(fill=tk.X, padx=10, pady=5)
    tk.Label(filter_frame, text="Filter: ").pack(side=tk.LEFT)
    tk.Entry(filter_frame, textvariable=shared_state.filter_var, width=40).pack(side=tk.LEFT, expand=True, fill=tk.X)

    # ✅ Register trace_add *after* GUI setup to avoid premature callback
    logging.debug("📌 Calling trace_add from <launch_gui>")
    shared_state.filter_trace_id = shared_state.filter_var.trace_add("write", on_filter_change)

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

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
    try:
        shared_state = get_shared_state()
    except RuntimeError:
        return
    if event.widget == event.widget.winfo_toplevel():
        redraw_last_chart()


def prompt_and_extract_file() -> None:
    logging.debug("📌 prompt_and_extract_file() triggered")
    from gui.shared_state import get_shared_state
    shared_state = get_shared_state()

    # ✅ Temporarily block trace callback to avoid double-trigger
    try:
        shared_state.filter_var.trace_remove("write", shared_state.filter_trace_id)
    except Exception as e:
        logging.debug(f"⚠️ Could not remove trace temporarily: {e}")

    path = filedialog.askopenfilename()
    if path:
        shared_state.current_file_path = path
        run_metric_extraction(path)
        update_tree(shared_state.tree, path)
        flat_metrics = flatten_metrics(shared_state.results.get(path, {}))
        update_footer_summary(shared_state.summary_tree, flat_metrics)

    # ✅ Re-enable filter_var trace after update
    shared_state.filter_trace_id = shared_state.filter_var.trace_add("write", on_filter_change)


def on_filter_change(*args) -> None:
    logging.debug("📌 on_filter_change() triggered")
    shared_state = get_shared_state()
    keys = list(shared_state.results.keys())
    file_path = keys[0] if keys else None
    if file_path:
        update_tree(shared_state.tree, file_path)
        flat_metrics = flatten_metrics(shared_state.results.get(file_path, {}))
        update_footer_summary(shared_state.summary_tree, flat_metrics)


def refresh_chart_on_scope_change() -> None:
    shared_state = get_shared_state()
    if shared_state.current_file_path:
        show_chart()


def show_chart() -> None:
    shared_state = get_shared_state()
    filename = shared_state.current_file_path
    if not filename:
        return
    file_metrics = shared_state.results.get(filename, {})
    filtered = filter_metrics_by_scope(file_metrics)
    if not filtered:
        messagebox.showinfo("No Metrics", f"No metrics found for scope: {shared_state.metric_scope.get()}")
        return
    keys = list(filtered.keys())
    vals = [round(float(filtered[k]), 2) for k in keys]
    draw_chart(keys, vals, f"Metrics - Scope: {shared_state.metric_scope.get()}", "scope_chart.png")


def show_directory_summary_chart() -> None:
    shared_state = get_shared_state()
    if not shared_state.results:
        messagebox.showinfo("No Data", "No analysis has been run.")
        return
    scope = shared_state.metric_scope.get()
    combined = {}
    for file_data in shared_state.results.values():
        filtered = filter_metrics_by_scope(file_data)
        for k, v in filtered.items():
            try:
                combined[k] = combined.get(k, 0) + float(v)
            except (TypeError, ValueError):
                continue
    if not combined:
        messagebox.showinfo("No Metrics", f"No numeric metrics available for scope: {scope}")
        return
    keys = list(combined.keys())
    vals = [round(combined[k], 2) for k in keys]
    draw_chart(keys, vals, f"Metrics - Scope: {scope}", f"summary_scope_{scope}.png")
