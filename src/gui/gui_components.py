import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from gui import shared_state
from gui.file_ops import run_metric_extraction, run_directory_analysis, export_to_csv
from gui.chart_utils import draw_chart, include_metric, redraw_last_chart
from gui.gui_logic import update_tree, update_footer_summary


def launch_gui():
    root = tk.Tk()
    root.title("🧠 Code Analyser GUI")
    root.geometry("1000x900")

    # Initialise shared tkinter variables
    shared_state.filter_var = tk.StringVar()
    shared_state.metric_scope = tk.StringVar(value="all")
    shared_state.chart_type = tk.StringVar(value="bar")

    root.bind("<Configure>", lambda event: on_resize(event))

    # Control panel
    top_frame = tk.Frame(root)
    top_frame.pack(pady=10)

    def prompt_and_extract_file():
        path = filedialog.askopenfilename()
        if path:
            run_metric_extraction(path)

    tk.Button(top_frame, text="📂 File", command=prompt_and_extract_file).grid(row=0, column=0, padx=5)
    tk.Button(top_frame, text="📁 Folder", command=run_directory_analysis).grid(row=0, column=1, padx=5)
    tk.Button(top_frame, text="📊 File Chart", command=show_chart).grid(row=0, column=2, padx=5)
    tk.Button(top_frame, text="📊 Dir Chart", command=show_directory_summary_chart).grid(row=0, column=3, padx=5)
    tk.Button(top_frame, text="📄 Export CSV", command=export_to_csv).grid(row=0, column=4, padx=5)
    tk.Button(top_frame, text="Exit", command=root.destroy).grid(row=0, column=5, padx=5)

    # Chart controls
    option_frame = tk.Frame(root)
    option_frame.pack(pady=5)
    tk.Label(option_frame, text="Chart Type:").pack(side=tk.LEFT)
    tk.Radiobutton(option_frame, text="Bar", variable=shared_state.chart_type, value="bar").pack(side=tk.LEFT)
    tk.Radiobutton(option_frame, text="Pie", variable=shared_state.chart_type, value="pie").pack(side=tk.LEFT)
    tk.Label(option_frame, text="Metric Scope:").pack(side=tk.LEFT, padx=(20, 5))

    for label, value in [
        ("AST", "ast"),
        ("Bandit", "bandit"),
        ("Flake8", "flake8"),
        ("Cloc", "cloc"),
        ("Lizard", "lizard"),
        ("Pydocstyle", "pydocstyle"),
        ("Pyflakes", "pyflakes"),
        ("Pylint", "pylint"),
        ("All", "all"),
    ]:
        tk.Radiobutton(option_frame, text=label, variable=shared_state.metric_scope, value=value).pack(side=tk.LEFT)

    # Filter bar
    filter_frame = tk.Frame(root)
    filter_frame.pack(fill=tk.X, padx=10, pady=5)
    shared_state.filter_var.trace_add("write", lambda *args: update_tree(shared_state.results))
    tk.Label(filter_frame, text="Filter: ").pack(side=tk.LEFT)
    tk.Entry(filter_frame, textvariable=shared_state.filter_var, width=40).pack(side=tk.LEFT, expand=True, fill=tk.X)

    # Main tab layout
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    # 📊 Chart tab
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
    shared_state.tree = ttk.Treeview(tree_tab, columns=("File", "Metric", "Value"), show="headings", yscrollcommand=tree_scroll.set)
    for col in ("File", "Metric", "Value"):
        shared_state.tree.heading(col, text=col)
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

    # Initial population
    root.after(100, update_tree, shared_state.results)
    root.after(100, update_footer_summary)
    root.mainloop()


def on_resize(event):
    """Redraw chart when the main window is resized."""
    if event.widget == event.widget.winfo_toplevel():
        redraw_last_chart()


def show_chart():
    """Display chart for a selected file's metrics."""
    if not shared_state.results or not shared_state.tree:
        return

    selected = shared_state.tree.selection()
    if not selected:
        messagebox.showinfo("No Selection", "Please select a row.")
        return

    values = shared_state.tree.item(selected[0], "values")
    file_name = values[0]
    file_metrics = {}

    for f, data in shared_state.results.items():
        if Path(f).name == file_name:
            file_metrics = data
            break

    flat_metrics = flatten_metrics(file_metrics)
    keys, vals = [], []

    for k, v in flat_metrics.items():
        if include_metric(k):
            keys.append(k)
            vals.append(round(v, 2))

    if not keys:
        messagebox.showinfo("No Numeric Metrics", f"No numeric metrics available for {file_name}.")
        return

    draw_chart(keys, vals, f"Metrics for {file_name}", f"chart_{file_name.replace('.py', '')}.png")


def show_directory_summary_chart():
    """Display combined chart summarising all metric values across files."""
    if not shared_state.results:
        messagebox.showinfo("No Data", "No analysis has been run.")
        return

    combined = {}
    for data in shared_state.results.values():
        flat = flatten_metrics(data)
        for k, v in flat.items():
            if include_metric(k):
                combined[k] = combined.get(k, 0) + v

    if not combined:
        messagebox.showinfo("No Numeric Metrics", "No numeric metrics available to summarise.")
        return

    keys = list(combined.keys())
    vals = [round(combined[k], 2) for k in keys]
    draw_chart(keys, vals, "Directory-wide Metric Summary", "summary_chart.png")


def flatten_metrics(d, prefix=""):
    """Recursively flatten nested metric dictionaries for charting."""
    flat = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, (int, float)):
            flat[full_key] = v
        elif isinstance(v, dict):
            flat.update(flatten_metrics(v, full_key))
    return flat
