import tkinter as tk
import pytest
from gui.shared_state import setup_shared_gui_state

def test_setup_and_get_shared_state_triggers_callback(monkeypatch):
    # ✅ Patch the correct function
    triggered = {"called": False}
    monkeypatch.setattr("gui.chart_utils.redraw_last_chart", lambda: triggered.update(called=True))

    root = tk.Tk()
    root.withdraw()

    shared = setup_shared_gui_state(root)
    shared.metric_scope.set("initial")
    triggered["called"] = False
    shared.metric_scope.set("changed")  # Should call the patched function

    assert triggered["called"], "Expected redraw_last_chart() to be triggered on metric_scope change"
