# gui/gui_logic.py

from pathlib import Path
from gui import shared_state



def update_tree(data: dict):
    """Refresh the metric tree view with new results, applying any active filter."""
    tree = shared_state.tree
    filter_var = shared_state.filter_var

    if not tree or not filter_var:
        print("⚠️ Tree or filter_var not initialised")
        return

    tree.delete(*tree.get_children())
    filter_text = filter_var.get().lower()

    print("🔄 Updating tree with data for filtering:", filter_text)
    for file, top_level in data.items():
        base_name = Path(file).name
        for key, value in top_level.items():
            if key == "metrics" and isinstance(value, dict):
                for metric_key, metric_val in value.items():
                    if filter_text in metric_key.lower() or filter_text in base_name.lower():
                        rounded_value = round(metric_val, 2) if isinstance(metric_val, (float, int)) else metric_val
                        tree.insert("", "end", values=(base_name, metric_key, rounded_value))
            else:
                if filter_text in key.lower() or filter_text in base_name.lower():
                    tree.insert("", "end", values=(base_name, key, value))


def update_footer_summary():
    """Update the summary tree view with totals and averages of all collected metrics."""
    summary_tree = shared_state.summary_tree
    results = shared_state.results

    if not summary_tree:
        print("⚠️ Summary tree not initialised")
        return

    if not results:
        summary_tree.delete(*summary_tree.get_children())
        print("⚠️ No results available to summarise")
        return

    all_keys = sorted({k for r in results.values() for k in r.get("metrics", {})})

    def safe_numeric(val):
        try:
            return float(val)
        except Exception:
            return 0.0

    totals = {
        k: sum(safe_numeric(r.get("metrics", {}).get(k, 0)) for r in results.values())
        for k in all_keys
    }

    avgs = {
        k: round(totals[k] / len(results), 2)
        for k in all_keys
    }

    print(f"📊 Summary totals: {totals}")
    print(f"📊 Summary averages: {avgs}")

    summary_tree.delete(*summary_tree.get_children())

    for k in all_keys:
        summary_tree.insert("", "end", values=(k, totals[k], avgs[k]))
