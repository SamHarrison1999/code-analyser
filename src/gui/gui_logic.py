from pathlib import Path
from typing import Any

from gui import shared_state


def update_tree(data: dict[str, dict[str, Any]]) -> None:
    """
    Refresh the metric tree view with new results, applying any active text filter.

    Args:
        data (dict): A mapping of file paths to metric dictionaries.
    """
    tree = shared_state.tree
    filter_var = shared_state.filter_var

    if tree is None or filter_var is None:
        print("⚠️ Tree or filter_var not initialised")
        return

    tree.delete(*tree.get_children())
    filter_text = filter_var.get().lower().strip() if filter_var.get() else ""

    print(f"🔄 Refreshing tree view (filter: '{filter_text}')")

    for file, metrics in data.items():
        base_name = Path(file).name
        flat = flatten_metrics(metrics)
        for metric_key, metric_val in flat.items():
            if filter_text in metric_key.lower() or filter_text in base_name.lower():
                val = round(metric_val, 2) if isinstance(metric_val, (float, int)) else metric_val
                tree.insert("", "end", values=(base_name, metric_key, val))


def update_footer_summary() -> None:
    """
    Update the summary tree view with total and average values for each metric.
    """
    summary_tree = shared_state.summary_tree
    results = shared_state.results

    if summary_tree is None:
        print("⚠️ Summary tree not initialised")
        return

    summary_tree.delete(*summary_tree.get_children())

    if not results:
        print("⚠️ No results available to summarise")
        return

    flattened_results = {
        file: flatten_metrics(metrics)
        for file, metrics in results.items()
    }

    all_keys = sorted({k for metrics in flattened_results.values() for k in metrics})

    def safe_numeric(val: Any) -> float:
        try:
            return float(val)
        except Exception:
            return 0.0

    totals = {
        key: round(sum(safe_numeric(flattened_results[file].get(key, 0)) for file in flattened_results), 2)
        for key in all_keys
    }

    avgs = {
        key: round(totals[key] / len(flattened_results), 2)
        for key in all_keys
    }

    print(f"📊 Summary totals: {totals}")
    print(f"📊 Summary averages: {avgs}")

    for key in all_keys:
        summary_tree.insert("", "end", values=(key, totals[key], avgs[key]))


def flatten_metrics(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """
    Recursively flatten a nested dictionary of metrics.

    Args:
        d (dict): The (possibly nested) dictionary of metric values.
        prefix (str): The current path prefix for recursive keys.

    Returns:
        dict[str, float]: Flattened dictionary with full dotted paths as keys.
    """
    flat: dict[str, float] = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, (int, float)):
            flat[full_key] = v
        elif isinstance(v, dict):
            flat.update(flatten_metrics(v, full_key))
    return flat
