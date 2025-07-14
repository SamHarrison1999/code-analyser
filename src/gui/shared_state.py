import tkinter as tk
import logging
from typing import Optional

_shared_state = None  # 🔒 Private module-level shared instance


class SharedState:
    """Container for global GUI variables and components."""
    def __init__(self, root: tk.Tk):
        self.metric_scope = tk.StringVar(master=root, value="all")
        self.chart_type = tk.StringVar(master=root, value="bar")
        self.filter_var = tk.StringVar(master=root, value="")

        self.results: dict[str, dict] = {}
        self.current_file_path: str = ""
        self.chart_frame: Optional[tk.Frame] = None
        self.chart_canvas: Optional[tk.Canvas] = None
        self.tree: Optional[tk.ttk.Treeview] = None
        self.summary_tree: Optional[tk.ttk.Treeview] = None

        # 🧩 Store trace_add ID so it can be removed/re-applied
        self.filter_trace_id: Optional[str] = None

        # 🔄 Automatically redraw chart when the metric scope changes
        logging.debug("📌 Calling trace_add from <SharedState>")
        self.metric_scope.trace_add("write", self._on_metric_scope_change)

    def _on_metric_scope_change(self, *args):
        """Callback when the metric scope changes, triggers chart redraw."""
        try:
            from gui.chart_utils import redraw_last_chart
            redraw_last_chart()
        except Exception as e:
            logging.warning(f"⚠️ Failed to redraw chart on metric scope change: {type(e).__name__}: {e}")


def setup_shared_gui_state(root: tk.Tk) -> SharedState:
    """
    Initialise and store the shared GUI state after the Tk root is created.

    Must be called before accessing any shared GUI state.
    """
    global _shared_state
    _shared_state = SharedState(root)
    logging.debug("✅ Shared GUI state initialised.")
    return _shared_state


def get_shared_state() -> SharedState:
    """
    Return the current shared GUI state.

    Raises:
        RuntimeError: If setup_shared_gui_state() was not called before access.
    """
    if _shared_state is None:
        raise RuntimeError("Shared state not initialised. Call setup_shared_gui_state(root) first.")
    return _shared_state
