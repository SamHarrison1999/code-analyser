# gui/chart_utils.py

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors
import tkinter as tk
from gui import shared_state


# Store last chart config for resize redrawing
_last_keys = []
_last_vals = []
_last_title = ""
_last_filename = ""


def include_metric(metric: str) -> bool:
    """Filter metrics based on selected analysis scope."""
    scope = shared_state.metric_scope.get()
    if scope == "ast":
        return not metric.startswith("number_of_") and "docstring" not in metric
    elif scope == "bandit":
        return "security" in metric or "vulnerability" in metric
    elif scope == "flake8":
        return "styling" in metric or "line_length" in metric
    elif scope == "cloc":
        return "line" in metric or "comment" in metric
    elif scope == "lizard":
        return any(key in metric for key in ["complexity", "token", "parameter", "maintainability"])
    elif scope == "pydocstyle":
        return any(key in metric.lower() for key in ["docstring", "pydocstyle", "compliance"])
    elif scope == "pyflakes":
        return any(key in metric.lower() for key in ["undefined", "redefined", "syntax", "import"])
    return True


def draw_chart(keys, vals, title, filename):
    """Render either bar or pie chart for given metric values."""
    global _last_keys, _last_vals, _last_title, _last_filename
    _last_keys = keys
    _last_vals = vals
    _last_title = title
    _last_filename = filename

    chart_frame = shared_state.chart_frame
    chart_type = shared_state.chart_type

    if not chart_frame:
        print("⚠️ chart_frame not initialised. Cannot draw chart.")
        return

    print(f"🧮 Drawing chart with {len(keys)} metrics. Type: {chart_type.get()}. Title: {title}")

    # Clear old chart widgets
    for widget in chart_frame.winfo_children():
        widget.destroy()

    chart_frame.update_idletasks()  # Ensure dimensions are up-to-date
    pixel_width = max(chart_frame.winfo_width(), 800)
    inch_width = pixel_width / 100  # DPI = 100

    def prettify(label):
        return label.replace("metrics.", "").replace("_", " ").strip().capitalize()

    pretty_keys = [prettify(k) for k in keys]

    if chart_type.get() == "bar":
        height_per_metric = 0.4
        fig_height = max(4, len(pretty_keys) * height_per_metric)
        fig, ax = plt.subplots(figsize=(inch_width, fig_height))

        bars = ax.barh(pretty_keys, vals)
        ax.set_xlabel("Value")
        ax.set_ylabel("Metric")
        ax.tick_params(axis='y', labelsize=8)
        ax.set_title(title)

        # ✅ Fix top and bottom clipping
        ax.set_ylim(-0.6, len(pretty_keys) - 0.4)

        cursor = mplcursors.cursor(bars, hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(
            f"{pretty_keys[sel.index]}: {vals[sel.index]}"
        ))

        fig.tight_layout()

    else:
        # Pie chart with dynamic square size
        fig, ax = plt.subplots(figsize=(inch_width, inch_width))
        wedges, _ = ax.pie(vals, labels=None, startangle=90)
        ax.set_title(title)
        ax.axis('equal')

        # Compact layout
        fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)

        tooltip = ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->")
        )
        tooltip.set_visible(False)

        def format_tooltip(event):
            for i, wedge in enumerate(wedges):
                if wedge.contains_point([event.x, event.y]):
                    tooltip.xy = (event.xdata, event.ydata)
                    tooltip.set_text(f"{pretty_keys[i]}: {vals[i]:.1f}%")
                    tooltip.set_visible(True)
                    fig.canvas.draw_idle()
                    return
            tooltip.set_visible(False)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", format_tooltip)

    # Save chart to file
    fig.savefig(filename)

    # Embed chart in scrollable Tkinter canvas
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


def redraw_last_chart():
    """Redraw the most recently displayed chart to fit resized window."""
    if _last_keys and _last_vals and _last_filename:
        draw_chart(_last_keys, _last_vals, _last_title, _last_filename)
