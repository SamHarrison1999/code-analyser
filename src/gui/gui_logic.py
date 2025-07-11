import logging
from tkinter import ttk
from typing import List, Dict, Any

from metrics.gather import gather_all_metrics, get_all_metric_names
from metrics.radon_metrics.gather import gather_radon_metrics
from gui.chart_utils import draw_chart, include_metric


def update_tree(tree: ttk.Treeview, file_path: str) -> None:
    """Update the Treeview with metrics from the specified file."""
    if not file_path or not tree:
        logging.warning("⚠️ update_tree() called with missing file_path or tree reference.")
        return

    try:
        # Gather metrics from core and Radon sources
        all_metrics = gather_all_metrics(file_path)
        metric_names = get_all_metric_names()

        radon_metrics = gather_radon_metrics(file_path)
        radon_metric_names = [
            "number_of_logical_lines",
            "number_of_blank_lines",
            "number_of_doc_strings",
            "average_halstead_volume",
            "average_halstead_difficulty",
            "average_halstead_effort",
        ]

        # Combine both sets
        combined_dict = dict(zip(metric_names, all_metrics))
        combined_dict.update(dict(zip(radon_metric_names, radon_metrics)))

        # Clear and populate Treeview
        tree.delete(*tree.get_children())
        for name, value in combined_dict.items():
            tree.insert("", "end", values=(name, value))

        update_chart(list(combined_dict.keys()), list(combined_dict.values()))

    except Exception as e:
        logging.error(f"❌ Failed to update tree for {file_path}: {type(e).__name__}: {e}")


def update_chart(metric_names: List[str], metric_values: List[float]) -> None:
    """Filter metrics by scope and draw the corresponding chart."""
    from gui import shared_state  # 🔁 deferred to avoid circular import

    filtered_names = [name for name in metric_names if include_metric(name)]
    filtered_values = [metric_values[i] for i, name in enumerate(metric_names) if include_metric(name)]

    chart_title = f"Metrics - Scope: {shared_state.metric_scope.get()}"
    chart_filename = "last_metric_chart.png"

    draw_chart(filtered_names, filtered_values, chart_title, chart_filename)


def update_footer_summary(tree: ttk.Treeview, metrics_dict: Dict[str, Any]) -> None:
    """
    Populate the summary Treeview with total and average values of each metric.

    Args:
        tree (ttk.Treeview): The summary Treeview widget.
        metrics_dict (dict): Flattened metric dictionary for one file.
    """
    tree.delete(*tree.get_children())

    if not metrics_dict:
        return

    # Compute total and average
    total_metrics = {}
    count = 1  # Single file view context
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            total_metrics[key] = total_metrics.get(key, 0) + value

    for key, total in total_metrics.items():
        average = round(total / count, 2)
        tree.insert("", "end", values=(key, round(total, 2), average))
