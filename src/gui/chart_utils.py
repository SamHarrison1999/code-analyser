import tkinter as tk
from typing import List
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors
import logging
import sys
from tkinter import messagebox

# Cache last chart data
_last_keys = []
_last_vals = []
_last_title = ""
_last_filename = ""

_cached_scope_names = {}

# Strict Radon metric list (prevent pydocstyle/pylint bleed-through)
KNOWN_RADON_METRICS = {
    "number_of_logical_lines",
    "number_of_blank_lines",
    "number_of_doc_strings",
    "average_halstead_volume",
    "average_halstead_difficulty",
    "average_halstead_effort"
}

def include_metric(metric: str, valid_names: List[str] = None) -> bool:
    """Determine if a given metric should be included based on selected analysis scope."""
    from gui import shared_state
    scope = shared_state.metric_scope.get().lower()
    metric = metric.lower()
    is_frozen = getattr(sys, 'frozen', False)

    if scope == "ast":
        return metric in {
            "module_docstring", "todo_comments", "nested_functions", "lambda_functions",
            "magic_methods", "assert_statements", "class_docstrings", "functions",
            "classes", "function_docstrings", "exceptions", "loops_conditionals",
            "global_variables", "chained_methods"
        }
    elif scope == "bandit":
        return "security" in metric or "vulnerability" in metric or "cwe" in metric
    elif scope == "cloc":
        return "line" in metric or "comment" in metric
    elif scope == "flake8":
        return "style" in metric or "whitespace" in metric or "indent" in metric or "line_length" in metric
    elif scope == "pydocstyle":
        return "docstring" in metric or "pydocstyle" in metric or "compliance" in metric
    elif scope == "pyflakes":
        return "undefined" in metric or "syntax" in metric or "import" in metric
    elif scope == "pylint":
        return metric in {"convention", "refactor", "warning", "error", "fatal"}
    elif scope == "radon":
        return metric in KNOWN_RADON_METRICS
    elif scope == "lizard":
        return any(k in metric for k in ["cyclomatic", "token", "parameter", "function_count"])
    return True  # fallback for "all"

def filter_metrics_by_scope(metrics: dict) -> dict:
    """Filter metrics based on the selected analysis scope from shared_state.

    Args:
        metrics (dict): Dictionary of metric names and their values.

    Returns:
        dict: Filtered dictionary containing only metrics relevant to the current scope.
    """
    from gui import shared_state
    scope = shared_state.metric_scope.get().lower()

    # Define allowed metric keys explicitly for strict scopes
    if scope == "pylint":
        # ✅ Only show these specific Pylint categories
        allowed = {"convention", "refactor", "warning", "error", "fatal"}
    elif scope == "radon":
        # ✅ Strict Radon metrics as returned by the Radon gatherer
        allowed = {
            "number_of_logical_lines", "number_of_blank_lines", "number_of_doc_strings",
            "average_halstead_volume", "average_halstead_difficulty", "average_halstead_effort"
        }
    else:
        # ✅ For other scopes, defer to include_metric() logic
        allowed = None

    filtered = {}
    for k, v in metrics.items():
        # ⚠️ Skip non-numeric values (e.g. strings like 'N/A')
        if not isinstance(v, (int, float, str)):
            continue

        # ⚠️ Skip non-numeric strings
        val_str = str(v).replace(".", "", 1)
        if not val_str.isdigit():
            continue

        if allowed is not None:
            # ✅ If strict scope list is defined, use it
            if k in allowed:
                filtered[k] = v
        else:
            # ✅ Fallback: use dynamic filter
            if include_metric(k):
                filtered[k] = v

    return filtered



def draw_chart(keys: List[str], vals: List[float], title: str, filename: str) -> None:
    from gui import shared_state

    global _last_keys, _last_vals, _last_title, _last_filename
    _last_keys = keys
    _last_vals = vals
    _last_title = title
    _last_filename = filename

    chart_frame = shared_state.chart_frame
    chart_type = shared_state.chart_type.get() if shared_state.chart_type else "bar"

    if not chart_frame:
        logging.warning("⚠️ chart_frame not initialised. Cannot draw chart.")
        return

    if not keys or not vals:
        logging.warning(f"⚠️ No metrics passed to draw_chart for scope: {shared_state.metric_scope.get()}")
        messagebox.showinfo("No Metrics", f"No metrics found for scope: {shared_state.metric_scope.get()}\n\n"
                                          f"Available metric keys:\n" +
                                          "\n".join(sorted(shared_state.results.get(shared_state.current_file_path, {}).keys())))
        return

    print(f"🧮 Drawing chart with {len(keys)} metrics. Type: {chart_type}. Title: {title}")
    logging.debug(f"[Chart] Drawing chart: {title} with metrics: {keys}")

    for widget in chart_frame.winfo_children():
        widget.destroy()

    chart_frame.update_idletasks()
    pixel_width = max(chart_frame.winfo_width(), 800)
    inch_width = pixel_width / 100

    def prettify(label: str) -> str:
        return label.replace("metrics.", "").replace("_", " ").strip().capitalize()

    pretty_keys = [prettify(k) for k in keys]

    if chart_type == "bar":
        height_per_metric = 0.4
        fig_height = max(4, len(pretty_keys) * height_per_metric)
        fig, ax = plt.subplots(figsize=(inch_width, fig_height))
        bars = ax.barh(pretty_keys, vals)
        ax.set_xlabel("Value")
        ax.set_ylabel("Metric")
        ax.tick_params(axis='y', labelsize=8)
        ax.set_title(title)
        ax.set_ylim(-0.6, len(pretty_keys) - 0.4)
        cursor = mplcursors.cursor(bars, hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(
            f"{pretty_keys[sel.index]}: {vals[sel.index]}"
        ))
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(inch_width, inch_width))
        wedges, _ = ax.pie(vals, labels=None, startangle=90)
        ax.set_title(title)
        ax.axis('equal')
        fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)

        tooltip = ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->")
        )
        tooltip.set_visible(False)

        def format_tooltip(event):
            if event.inaxes == ax:
                for i, wedge in enumerate(wedges):
                    if wedge.contains_point([event.x, event.y]):
                        tooltip.xy = (event.xdata, event.ydata)
                        tooltip.set_text(f"{pretty_keys[i]}: {vals[i]:.1f}")
                        tooltip.set_visible(True)
                        fig.canvas.draw_idle()
                        return
            tooltip.set_visible(False)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", format_tooltip)

    fig.savefig(filename)

    canvas_container = tk.Canvas(chart_frame)
    scrollbar = tk.Scrollbar(chart_frame, orient="vertical", command=canvas_container.yview)
    canvas_container.configure(yscrollcommand=scrollbar.set)

    chart_widget = tk.Frame(canvas_container)
    chart_canvas = FigureCanvasTkAgg(fig, master=chart_widget)
    chart_canvas.draw()
    chart_canvas.get_tk_widget().pack()

    canvas_container.create_window((0, 0), window=chart_widget, anchor="nw")
    canvas_container.update_idletasks()
    canvas_container.config(scrollregion=canvas_container.bbox("all"))
    canvas_container.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    shared_state.chart_canvas = chart_canvas

def redraw_last_chart() -> None:
    from gui import shared_state
    if not hasattr(shared_state, "chart_frame") or not shared_state.chart_frame:
        logging.debug("⚠️ shared_state.chart_frame is missing; skipping redraw.")
        return
    if not shared_state.chart_frame.winfo_exists():
        logging.debug("⚠️ Chart frame does not exist; skipping redraw.")
        return
    if _last_keys and _last_vals and _last_filename:
        try:
            draw_chart(_last_keys, _last_vals, _last_title, _last_filename)
        except Exception as e:
            logging.error(f"❌ Failed to redraw chart: {type(e).__name__}: {e}")
    elif not hasattr(shared_state, "_warned_empty_redraw"):
        logging.debug("⚠️ No previous chart data to redraw.")
        shared_state._warned_empty_redraw = True
