import tkinter as tk
import logging

def setup_shared_gui_state():
    shared_state = type("SharedState", (), {})()
    shared_state.metric_scope = tk.StringVar(value="all")

    # Trigger redraw when metric scope changes
    def on_metric_scope_change(*args):
        try:
            from gui_components import show_chart
            show_chart()
        except Exception as e:
            logging.warning(f"⚠️ Failed to trigger chart update on scope change: {type(e).__name__}: {e}")
    shared_state.metric_scope.trace_add("write", on_metric_scope_change)

    shared_state.results = {}
    shared_state.current_file_path = ""
    shared_state.chart_frame = None
    shared_state.chart_type = tk.StringVar(value="bar")
    shared_state.chart_canvas = None
    return shared_state
