# File: code_analyser/src/gui/chart_utils.py

# ✅ Best Practice: Use logging for diagnostics and debugging
import logging
from typing import Dict, Any
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
import json
import csv
from PIL import ImageGrab
from unittest.mock import MagicMock

from gui.shared_state import get_shared_state
from gui.utils import flatten_metrics, merge_nested_metrics

logger = logging.getLogger(__name__)


# Defensive fallback to ensure valid float DPI under all environments
def _safe_dpi() -> float:
    try:
        import matplotlib

        dpi_val = matplotlib.rcParams.get("figure.dpi", 100)
        if not isinstance(dpi_val, (int, float)):
            raise TypeError(f"DPI is not numeric: {type(dpi_val)} = {dpi_val}")
        result = float(dpi_val)
        logger.debug(f"[DEBUG] _safe_dpi() returning: {result}")
        return result
    except Exception as e:
        logger.warning(f"⚠️ Failed to get valid dpi from rcParams: {e}")
        return 100.0


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
        "most_frequent_cwe_with_name",
    ],
    "cloc": [
        "number_of_comments",
        "comment_density",
        "number_of_source_lines_of_code",
        "number_of_lines",
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
        "number_of_styling_issues",
    ],
    "lizard": [
        "average_cyclomatic_complexity",
        "average_token_count",
        "total_function_count",
        "max_cyclomatic_complexity",
        "average_parameters",
    ],
    "pydocstyle": [
        "number_of_pydocstyle_violations",
        "number_of_missing_doc_strings",
        "percentage_of_compliance_with_docstring_style",
    ],
    "pyflakes": ["number_of_undefined_names", "number_of_syntax_errors"],
    "pylint": [
        "pylint.convention",
        "pylint.refactor",
        "pylint.warning",
        "pylint.error",
        "pylint.fatal",
    ],
    "radon": [
        "logical_lines",
        "blank_lines",
        "docstring_lines",
        "halstead_volume",
        "halstead_difficulty",
        "halstead_effort",
    ],
    "vulture": [
        "unused_functions",
        "unused_classes",
        "unused_variables",
        "unused_imports",
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
        "magic_methods",
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
        "vulnerabilities",
    ],
    # ✅ AI-enhanced scopes
    "together_ai": ["SAST Risk", "ML Signal", "Best Practice"],
    "rl_agent": [
        "confidence_weighted_sast_risk",
        "confidence_weighted_ml_signal",
        "confidence_weighted_best_practice",
    ],
    "ai": [  # Unified AI scope
        "SAST Risk",
        "ML Signal",
        "Best Practice",
        "confidence_weighted_sast_risk",
        "confidence_weighted_ml_signal",
        "confidence_weighted_best_practice",
    ],
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
    global _last_keys, _last_vals, _last_title, _last_filename
    _last_keys, _last_vals, _last_title, _last_filename = keys, values, title, filename

    shared_state = get_shared_state()
    dpi = _safe_dpi()
    bar_height = 1.4

    # ✅ Default figure size (fallback)
    fig_width, fig_height = 8, 6

    # ✅ Try using GUI dimensions if available
    if shared_state.chart_frame and shared_state.chart_frame.winfo_exists():
        try:
            width = shared_state.chart_frame.winfo_width()
            height = shared_state.chart_frame.winfo_height()
            logger.debug(f"chart_frame.winfo_width(): {width} ({type(width)})")
            logger.debug(f"chart_frame.winfo_height(): {height} ({type(height)})")

            if not isinstance(width, (int, float)):
                raise TypeError(f"❌ fig_width is not numeric: {width} ({type(width)})")
            if not isinstance(height, (int, float)):
                raise TypeError(f"❌ fig_height is not numeric: {height} ({type(height)})")

            logger.debug(f"[DPI] dpi: {dpi} ({type(dpi)})")
            fig_width = float(width) / dpi
            fig_height = float(height) / dpi
        except Exception as e:
            logger.warning(f"⚠️ Failed to get chart frame dimensions: {e}")

    overlay_bundle = getattr(shared_state, "ast_metric_bundle", {}).get(
        shared_state.current_file_path, []
    )
    overlay_lookup = {m["metric"]: m for m in overlay_bundle if isinstance(m, dict)}

    def get_severity_colour(sev):
        return {"low": "#B7E4C7", "medium": "#FFD966", "high": "#FF6F61"}.get(
            sev.lower(), "#B0BEC5"
        )

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    logger.debug(f"[DEBUG] fig.canvas: {fig.canvas} ({type(fig.canvas)})")

    bars = ax.barh(
        keys,
        values,
        height=1.0,
        color=[get_severity_colour(overlay_lookup.get(k, {}).get("severity", "low")) for k in keys],
    )

    ax.set_xlabel("Value", fontsize=14)
    ax.set_ylabel("Metric", fontsize=14)
    ax.set_title(title, fontsize=18, pad=2)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(-0.5, len(keys) - 0.5)
    fig.subplots_adjust(left=0.35, right=0.98, top=0.95, bottom=0.05)

    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(20, 0),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)

    def update_annot(bar, idx):
        annot.xy = (bar.get_width(), bar.get_y() + bar.get_height() / 2)
        annot.set_text(
            f"{keys[idx]}: {bar.get_width():.2f}\n"
            f"Confidence: {overlay_lookup.get(keys[idx], {}).get('confidence', 0.0):.2f}\n"
            f"Severity: {overlay_lookup.get(keys[idx], {}).get('severity', 'low').capitalize()}"
        )
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

    # ✅ Display chart inside GUI if available
    if (
        shared_state.chart_frame
        and shared_state.chart_frame.winfo_exists()
        and not isinstance(shared_state.chart_frame, MagicMock)
    ):
        for widget in shared_state.chart_frame.winfo_children():
            widget.destroy()

        outer_canvas = tk.Canvas(shared_state.chart_frame, highlightthickness=0)
        scroll_y = ttk.Scrollbar(
            shared_state.chart_frame, orient="vertical", command=outer_canvas.yview
        )
        scroll_x = ttk.Scrollbar(
            shared_state.chart_frame, orient="horizontal", command=outer_canvas.xview
        )
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
    dpi = _safe_dpi()
    if not _last_keys or not _last_vals:
        logger.warning("⚠️ No chart data available to save.")
        return

    try:
        shared_state = get_shared_state()
        fig_width, fig_height = 8, 6
        if shared_state.chart_frame and shared_state.chart_frame.winfo_exists():
            try:
                width = shared_state.chart_frame.winfo_width()
                height = shared_state.chart_frame.winfo_height()
                if isinstance(width, (int, float)) and isinstance(height, (int, float)):
                    fig_width = float(width) / dpi
                    fig_height = float(height) / dpi
                else:
                    logger.warning(
                        f"⚠️ Chart frame dimensions not numeric: ({type(width)}, {type(height)})"
                    )
            except Exception as e:
                logger.warning(f"⚠️ Failed to get chart frame dimensions: {e}")

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
        plt.close(fig)
    except Exception as e:
        logger.exception(f"❌ Failed to save chart image: {e}")


EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)


def export_last_chart_data(formats=["csv", "json"], filename_base="chart_export"):
    """
    Export the last chart's data in specified formats (CSV, JSON).
    """
    shared_state = get_shared_state()
    data = [{"metric": k, "value": v} for k, v in zip(_last_keys, _last_vals)]
    summary = {
        "title": _last_title,
        "scope": shared_state.metric_scope.get(),
        "entries": data,
    }

    if "csv" in formats:
        csv_path = EXPORT_DIR / f"{filename_base}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(data)

    if "json" in formats:
        json_path = EXPORT_DIR / f"{filename_base}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


def export_overlay_as_json(filename="overlay_export.json"):
    """
    Export the currently loaded overlay tokens and summary to JSON.
    """
    shared_state = get_shared_state()
    overlays = getattr(shared_state, "overlay_tokens", [])
    summary = getattr(shared_state, "overlay_summary", {})

    overlay_data = {"summary": summary, "overlays": overlays}
    out_path = EXPORT_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(overlay_data, f, indent=2)

    return out_path


def export_html_dashboard(filename="dashboard.html"):
    """
    Generate a simple HTML dashboard summarising chart metrics and overlay insights.
    """
    summary = export_last_chart_data(formats=["json"])
    overlay_path = export_overlay_as_json()

    html_path = EXPORT_DIR / filename
    from gui.shared_state import get_shared_state

    shared_state = get_shared_state()
    heatmap_path = EXPORT_DIR / "heatmap_overlay.png"
    if shared_state.heatmap_frame:
        try:
            widget = shared_state.heatmap_frame
            widget.update_idletasks()
            x = widget.winfo_rootx()
            y = widget.winfo_rooty()
            w = widget.winfo_width()
            h = widget.winfo_height()
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            img.save(heatmap_path)
        except Exception as e:
            logging.warning(f"⚠️ Could not capture heatmap image: {e}")
            heatmap_path = None

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><title>Code Analyser Dashboard</title></head><body>")
        f.write(f"<h1>{summary['title']}</h1>")
        f.write("<h2>Metrics</h2><table border='1'><tr><th>Metric</th><th>Value</th></tr>")
        for row in summary["entries"]:
            f.write(f"<tr><td>{row['metric']}</td><td>{row['value']}</td></tr>")
        f.write("</table>")

        f.write("<h2>AI Overlay Summary</h2>")
        with open(overlay_path, "r", encoding="utf-8") as overlay_file:
            overlay_data = json.load(overlay_file)
            for k, v in overlay_data.get("summary", {}).items():
                f.write(f"<p><strong>{k}</strong>: {v}</p>")

        f.write("<h2>Token-Level Overlay Heatmap</h2>")
        overlays = overlay_data.get("overlays", [])
        if overlays:
            f.write(
                "<table border='1'><tr><th>Line</th><th>Token</th><th>Confidence</th><th>Severity</th></tr>"
            )
            for item in overlays:
                line = item.get("line", "")
                token = item.get("token", "")
                conf = item.get("confidence", 0)
                severity = item.get("severity", "")
                f.write(
                    f"<tr><td>{line}</td><td>{token}</td><td>{conf:.2f}</td><td>{severity}</td></tr>"
                )
            f.write("</table>")
        else:
            f.write("<p>No overlay tokens available.</p>")

        if heatmap_path and heatmap_path.exists():
            f.write("<h2>Token Heatmap</h2>")
            f.write(
                f"<img src='{heatmap_path.name}' style='max-width:100%;border:1px solid black;'>"
            )

        f.write("</body></html>")

    return html_path


def export_all_assets():
    """
    Export all current charts (bar/pie), overlays, and heatmap as PNG, CSV, JSON.
    """
    formats = ["csv", "json"]
    export_last_chart_data(formats=formats)
    export_overlay_as_json()
    export_html_dashboard()

    # Also export PNG of last bar chart
    save_chart_as_image(EXPORT_DIR / "last_chart.png")

    # Export heatmap PNG if available
    from gui.shared_state import get_shared_state

    shared_state = get_shared_state()
    if shared_state.heatmap_frame:
        try:
            widget = shared_state.heatmap_frame
            widget.update_idletasks()
            x = widget.winfo_rootx()
            y = widget.winfo_rooty()
            w = widget.winfo_width()
            h = widget.winfo_height()
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            img.save(EXPORT_DIR / "heatmap_overlay.png")
        except Exception as e:
            logging.warning(f"⚠️ Could not export heatmap PNG: {e}")

    logging.info("✅ Export All completed")
    return EXPORT_DIR
