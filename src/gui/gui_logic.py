# File: code_analyser/src/gui/gui_logic.py

from typing import Dict, Any
import logging
from tkinter import ttk

from gui.chart_utils import draw_chart, filter_metrics_by_scope
from gui.utils import merge_nested_metrics
from metrics.ast_metrics.gather import gather_ast_metrics_bundle

logger = logging.getLogger(__name__)


def update_tree(tree: ttk.Treeview, file_path: str) -> None:
    """Update the Treeview with metrics from the specified file."""
    if not file_path or not tree:
        logger.warning("⚠️ update_tree() called with missing file_path or tree reference.")
        return

    try:
        from gui.shared_state import get_shared_state

        shared_state = get_shared_state()

        # ✅ Use existing CLI results if available, avoid rerunning gather_all_metrics()
        if file_path in shared_state.results:
            logger.debug(f"📦 Using cached metrics for {file_path}")
            all_metrics = shared_state.results[file_path]
        else:
            logger.debug(f"📡 Calling gather_ast_metrics_bundle() for {file_path}")
            all_metrics = gather_ast_metrics_bundle(file_path)
            shared_state.results[file_path] = all_metrics

        if (
            isinstance(all_metrics, list)
            and isinstance(all_metrics[0], dict)
            and "metric" in all_metrics[0]
        ):
            shared_state.ast_metric_bundle[file_path] = all_metrics

        tree.delete(*tree.get_children())
        flat_metrics = {m["metric"]: m["value"] for m in all_metrics}
        for name, value in flat_metrics.items():
            tree.insert("", "end", values=(name, value))

        update_chart(flat_metrics)

    except Exception as e:
        logger.error(f"❌ Failed to update tree for {file_path}: {type(e).__name__}: {e}")


def update_chart(metrics_dict: Dict[str, Any]) -> None:
    """Filter metrics by scope and draw the corresponding chart."""
    try:
        from gui.shared_state import get_shared_state

        shared_state = get_shared_state()

        filtered = filter_metrics_by_scope(metrics_dict)
        if not filtered:
            logger.info("[Chart] No matching metrics for current scope.")
            return

        keys = list(filtered.keys())
        vals = [round(float(filtered[k]), 2) for k in keys]

        chart_title = f"Metrics - Scope: {shared_state.metric_scope.get().capitalize()}"
        chart_filename = "last_metric_chart.png"
        draw_chart(keys, vals, chart_title, chart_filename)

    except Exception as e:
        logger.warning(f"⚠️ Could not draw chart: {type(e).__name__}: {e}")


def update_footer_summary(tree: ttk.Treeview, file_or_merged_metrics: Dict[str, Any]) -> None:
    try:
        from gui.shared_state import get_shared_state

        shared_state = get_shared_state()

        tree.delete(*tree.get_children())

        if not shared_state.results:
            return

        all_flattened: list[Dict[str, float]] = []
        for file_data in shared_state.results.values():
            merged = merge_nested_metrics(file_data)
            filtered = filter_metrics_by_scope(merged)
            numeric_only = {k: float(v) for k, v in filtered.items() if isinstance(v, (int, float))}
            all_flattened.append(numeric_only)

        if not all_flattened:
            logger.info("📉 No numeric metrics found to summarise.")
            return

        keys = sorted({k for f in all_flattened for k in f})
        totals = {k: 0.0 for k in keys}
        for file_metrics in all_flattened:
            for k in keys:
                try:
                    totals[k] += float(file_metrics.get(k, 0.0))
                except Exception:
                    continue

        averages = {k: round(totals[k] / len(all_flattened), 2) for k in keys}

        bundle = shared_state.ast_metric_bundle.get(shared_state.current_file_path, [])
        sev_map = {m["metric"]: m.get("severity", "low").lower() for m in bundle}

        for k in keys:
            severity = sev_map.get(k, "")
            if severity == "high":
                icon = "🔴"
            elif severity == "medium":
                icon = "🟡"
            elif severity == "low":
                icon = "🟢"
            else:
                icon = ""
            tree.insert("", "end", values=(k, round(totals[k], 2), averages[k], icon))

    except Exception as e:
        logger.error(f"❌ Failed to update summary footer: {type(e).__name__}: {e}")
