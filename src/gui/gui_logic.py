
from typing import Dict, Any
import logging
from tkinter import ttk

from metrics.gather import gather_all_metrics
from metrics.radon_metrics.gather import gather_radon_metrics
from gui.chart_utils import draw_chart, filter_metrics_by_scope
from gui.utils import merge_nested_metrics, flatten_metrics

def update_tree(tree: ttk.Treeview, file_path: str) -> None:
    """Update the Treeview with metrics from the specified file."""
    if not file_path or not tree:
        logging.warning("⚠️ update_tree() called with missing file_path or tree reference.")
        return

    try:
        from gui.shared_state import get_shared_state
        shared_state = get_shared_state()

        all_metrics = gather_all_metrics(file_path)
        radon_metrics = gather_radon_metrics(file_path)

        combined_dict = all_metrics.copy()
        combined_dict.update(dict(zip([
            "number_of_logical_lines",
            "number_of_blank_lines",
            "number_of_doc_strings",
            "average_halstead_volume",
            "average_halstead_difficulty",
            "average_halstead_effort",
        ], radon_metrics)))

        shared_state.results[file_path] = combined_dict

        tree.delete(*tree.get_children())
        merged = merge_nested_metrics(combined_dict)
        flat_metrics = flatten_metrics(merged)

        for name, value in flat_metrics.items():
            tree.insert("", "end", values=(name, value))

        update_chart(flat_metrics)

    except Exception as e:
        logging.error(f"❌ Failed to update tree for {file_path}: {type(e).__name__}: {e}")

def update_chart(metrics_dict: Dict[str, Any]) -> None:
    """Filter metrics by scope and draw the corresponding chart."""
    try:
        from gui.shared_state import get_shared_state
        shared_state = get_shared_state()

        filtered = filter_metrics_by_scope(metrics_dict)
        keys = list(filtered.keys())
        vals = [round(float(filtered[k]), 2) for k in keys]

        chart_title = f"Metrics - Scope: {shared_state.metric_scope.get()}"
        chart_filename = "last_metric_chart.png"

        draw_chart(keys, vals, chart_title, chart_filename)

    except Exception as e:
        logging.warning(f"⚠️ Could not draw chart: {type(e).__name__}: {e}")

def update_footer_summary(tree: ttk.Treeview, metrics_dict: Dict[str, Any]) -> None:
    """
    Populate the summary Treeview with total and average values of each metric.

    Args:
        tree (ttk.Treeview): The summary Treeview widget.
        metrics_dict (dict): Flattened metric dictionary for one file.
    """
    try:
        from gui.shared_state import get_shared_state
        get_shared_state()  # ✅ Best Practice: trigger initialisation if needed

        tree.delete(*tree.get_children())
        if not metrics_dict:
            return

        total_metrics = {}
        count = 1  # Currently fixed at 1 since it's per-file summary

        for key, value in metrics_dict.items():
            try:
                numeric_val = float(value)
                total_metrics[key] = total_metrics.get(key, 0) + numeric_val
            except Exception:
                continue

        for key, total in total_metrics.items():
            avg = round(total / count, 2)
            tree.insert("", "end", values=(key, round(total, 2), avg))

    except Exception as e:
        logging.error(f"❌ Failed to update summary footer: {type(e).__name__}: {e}")
