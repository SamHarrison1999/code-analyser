# ✅ Best Practice: Use logging for diagnostics and debugging
import logging
from typing import Dict, Any
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path

from gui.shared_state import get_shared_state
from gui.utils import flatten_metrics, merge_nested_metrics

logger = logging.getLogger(__name__)

_last_keys: list[str] = []
_last_vals: list[float] = []
_last_title: str = ""
_last_filename: str = ""

# ✅ Best Practice: Group metrics by tool name for filtering
SCOPE_KEYWORDS = {
    "bandit": [
        "number_of_high_security_vulnerabilities",
        "number_of_medium_security_vulnerabilities",
        "number_of_low_security_vulnerabilities",
        "number_of_undefined_security_vulnerabilities",
        "number_of_distinct_cwes",
        "most_frequent_cwe",
        "number_of_distinct_cwe_names",
        "most_frequent_cwe_with_name"
    ],
    "cloc": [
        "number_of_comments",
        "comment_density",
        "number_of_source_lines_of_code",
        "number_of_lines"
    ],
    "flake8": [
        "number_of_unused_variables",
        "number_of_unused_imports",
        "number_of_inconsistent_indentations",
        "number_of_trailing_whitespaces",
        "number_of_long_lines",
        "number_of_doc_string_issues",
        "number_of_naming_issues",
        "number_of_whitespace_issues",
        "average_line_length",
        "number_of_styling_warnings",
        "number_of_styling_errors",
        "number_of_styling_issues"
    ],
    "lizard": [
        "average_cyclomatic_complexity",
        "average_token_count",
        "total_function_count",
        "max_cyclomatic_complexity",
        "average_parameters"
    ],
    "pydocstyle": [
        "number_of_pydocstyle_violations",
        "number_of_missing_doc_strings",
        "percentage_of_compliance_with_docstring_style"
    ],
    "pyflakes": [
        "number_of_undefined_names",
        "number_of_syntax_errors"
    ],
    "pylint": [
        "pylint.convention",
        "pylint.refactor",
        "pylint.warning",
        "pylint.error",
        "pylint.fatal"
    ],
    "radon": [
        "logical_lines",
        "blank_lines",
        "docstring_lines",
        "halstead_volume",
        "halstead_difficulty",
        "halstead_effort"
    ],
    "vulture": [
        "unused_functions",
        "unused_classes",
        "unused_variables",
        "unused_imports"
    ],
    "ast": [
        "functions",
        "classes",
        "function_docstrings",
        "class_docstrings",
        "module_docstring",
        "todo_comments",
        "assert_statements",
        "exceptions",
        "loops_conditionals",
        "nested_functions",
        "global_variables",
        "chained_methods",
        "lambda_functions",
        "magic_methods"
    ],
    "sonar": [
        "bugs",
        "code_smells",
        "cognitive_complexity",
        "comment_lines_density",
        "complexity",
        "coverage",
        "duplicated_blocks",
        "duplicated_lines",
        "duplicated_lines_density",
        "files",
        "ncloc",
        "tests",
        "reliability_rating",
        "security_rating",
        "sqale_index",
        "sqale_rating",
        "test_success_density",
        "vulnerabilities"
    ]
}



def filter_metrics_by_scope(metrics_dict: Dict[str, Any]) -> Dict[str, float]:
    """Filter the full metrics dictionary based on current metric scope selection."""
    shared_state = get_shared_state()
    scope = shared_state.metric_scope.get().lower()
    merged = merge_nested_metrics(metrics_dict)
    flat_metrics = flatten_metrics(merged)
    logger.debug(f"[Filter Scope] Flattened metrics: {len(flat_metrics)}")

    if scope == "all":
        return {k: float(v) for k, v in flat_metrics.items() if isinstance(v, (int, float))}

    if scope not in SCOPE_KEYWORDS:
        logger.warning(f"⚠️ Unknown scope '{scope}' specified")
        return {}

    wanted_keys = set(SCOPE_KEYWORDS[scope])
    filtered = {
        k: float(v)
        for k, v in flat_metrics.items()
        if isinstance(v, (int, float)) and k in wanted_keys
    }

    if not filtered:
        logger.info(f"No matching metrics found for scope: {scope}")
        messagebox.showinfo("No Metrics", f"No metrics found for scope: {scope}")
    return filtered


def draw_chart(keys: list[str], values: list[float], title: str, filename: str) -> None:
    """Draw a scrollable horizontal bar chart with consistent spacing and embedded scrollbars."""
    global _last_keys, _last_vals, _last_title, _last_filename
    _last_keys, _last_vals, _last_title, _last_filename = keys, values, title, filename

    shared_state = get_shared_state()
    bar_height = 1.4
    fig_height = max(6, len(keys) * bar_height)
    fig_width = 12

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    bars = ax.barh(keys, values, height=1.0)

    ax.set_xlabel("Value", fontsize=14)
    ax.set_ylabel("Metric", fontsize=14)
    ax.set_title(title, fontsize=18, pad=2)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(-0.5, len(keys) - 0.5)
    fig.subplots_adjust(left=0.35, right=0.98, top=0.95, bottom=0.05)

    # 🧠 Hover annotation
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 0), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(bar, idx):
        annot.xy = (bar.get_width(), bar.get_y() + bar.get_height() / 2)
        annot.set_text(f"{keys[idx]}: {bar.get_width():.2f}")
        annot.get_bbox_patch().set_alpha(0.9)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            for idx, bar in enumerate(bars):
                if bar.contains(event)[0]:
                    update_annot(bar, idx)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    fig.savefig(filename)
    logger.debug(f"[Chart] Chart saved to '{filename}'")

    if shared_state.chart_frame.winfo_exists():
        for widget in shared_state.chart_frame.winfo_children():
            widget.destroy()

        outer_canvas = tk.Canvas(shared_state.chart_frame, highlightthickness=0)
        scroll_y = ttk.Scrollbar(shared_state.chart_frame, orient="vertical", command=outer_canvas.yview)
        scroll_x = ttk.Scrollbar(shared_state.chart_frame, orient="horizontal", command=outer_canvas.xview)

        outer_canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        outer_canvas.pack(side="left", fill="both", expand=True)

        inner_frame = tk.Frame(outer_canvas)
        outer_canvas.create_window((0, 0), window=inner_frame, anchor="nw")

        def update_scroll_region(event):
            outer_canvas.configure(scrollregion=outer_canvas.bbox("all"))

        inner_frame.bind("<Configure>", update_scroll_region)

        chart_canvas = FigureCanvasTkAgg(fig, master=inner_frame)
        chart_canvas.draw()
        chart_canvas.get_tk_widget().pack(fill="both", expand=True)

        logger.debug("[Chart] Chart displayed with accurate bar layout and scrollbars.")

    # ✅ Prevent figure memory leak by closing it after embedding
    plt.close(fig)


def redraw_last_chart() -> None:
    """Redraw the last chart if one exists."""
    shared_state = get_shared_state()
    if not shared_state.chart_frame or not shared_state.chart_frame.winfo_exists():
        return
    if _last_keys and _last_vals and _last_filename:
        try:
            logger.debug("[Redraw] Redrawing previous chart...")
            draw_chart(_last_keys, _last_vals, _last_title, _last_filename)
        except Exception as e:
            logger.error(f"❌ Failed to redraw chart: {type(e).__name__}: {e}")


def save_chart_as_image(path: Path | str) -> None:
    """Save the last drawn chart to the specified path."""
    if not _last_keys or not _last_vals:
        logger.warning("⚠️ No chart data available to save.")
        return

    try:
        fig_height = max(6, len(_last_keys) * 1.4)
        fig_width = 12

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.barh(_last_keys, _last_vals, height=1.0)

        ax.set_xlabel("Value", fontsize=14)
        ax.set_ylabel("Metric", fontsize=14)
        ax.set_title(_last_title, fontsize=18, pad=2)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylim(-0.5, len(_last_keys) - 0.5)
        fig.subplots_adjust(left=0.35, right=0.98, top=0.95, bottom=0.05)

        fig.savefig(path)
        logger.info(f"🖼️ Chart image saved to {path}")

        # ✅ Always close the figure to avoid memory leaks
        plt.close(fig)

    except Exception as e:
        logger.exception(f"❌ Failed to save chart image: {e}")

