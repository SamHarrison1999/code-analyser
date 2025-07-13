
import tkinter as tk
import logging

_shared_state = None  # 🔒 Module-private shared instance

def setup_shared_gui_state(root: tk.Tk):
    """Initialise and store the shared GUI state after the root window is created."""
    global _shared_state
    shared_state = type("SharedState", (), {})()

    shared_state.metric_scope = tk.StringVar(master=root, value="all")
    shared_state.chart_type = tk.StringVar(master=root, value="bar")
    shared_state.filter_var = tk.StringVar(master=root, value="")

    shared_state.results = {}
    shared_state.current_file_path = ""
    shared_state.chart_frame = None
    shared_state.chart_canvas = None
    shared_state.tree = None
    shared_state.summary_tree = None

    def on_metric_scope_change(*args):
        try:
            from gui.gui_components import show_chart
            if callable(show_chart):
                show_chart()
        except Exception as e:
            logging.warning(
                f"⚠️ Failed to redraw chart on metric scope change: {type(e).__name__}: {e}"
            )

    shared_state.metric_scope.trace_add("write", on_metric_scope_change)

    _shared_state = shared_state
    return _shared_state

def get_shared_state():
    """Safely access the global shared state object."""
    if _shared_state is None:
        raise RuntimeError("Shared state not initialised. Call setup_shared_gui_state(root) first.")
    return _shared_state
