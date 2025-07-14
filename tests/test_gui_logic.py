# File: tests/test_gui_logic.py

import os
import tkinter as tk
import pytest
from tkinter import ttk
from gui.gui_logic import update_tree, update_chart, update_footer_summary
from gui.shared_state import setup_shared_gui_state, get_shared_state

# ✅ Ensure Tcl/Tk libraries are configured for headless environments
os.environ["TCL_LIBRARY"] = r"C:\Python312\tcl\tcl8.6"
os.environ["TK_LIBRARY"] = r"C:\Python312\tcl\tk8.6"

@pytest.fixture
def root():
    """Create and tear down a hidden Tk root widget."""
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()

def test_update_tree_basic(monkeypatch, root):
    shared = setup_shared_gui_state(root)
    tree = ttk.Treeview(root, columns=("Metric", "Value"))
    tree.pack()
    shared.tree = tree

    sample_metrics = {"flake8": {"unused_imports": 2}, "bandit": {"issues": 1}}

    def fake_gather(file_path):
        return sample_metrics

    monkeypatch.setattr("gui.gui_logic.gather_all_metrics", fake_gather)

    update_tree(tree, "test_file.py")

    assert "test_file.py" in shared.results
    expected_total = sum(len(d) for d in sample_metrics.values())
    assert len(tree.get_children()) == expected_total

def test_update_tree_handles_missing_file(monkeypatch, root, caplog):
    tree = ttk.Treeview(root)
    update_tree(tree, "")  # Should log a warning
    assert "missing file_path or tree reference" in caplog.text.lower()

def test_update_chart_with_valid_data(monkeypatch, root):
    shared = setup_shared_gui_state(root)
    shared.metric_scope.set("all")

    metrics = {
        "flake8.unused_imports": 3,
        "bandit.issues": 1,
        "pylint.error_count": 4,
    }

    monkeypatch.setattr("gui.gui_logic.draw_chart", lambda *a, **k: None)

    update_chart(metrics)  # Should not raise

def test_update_chart_with_empty_scope(monkeypatch, root, caplog):
    shared = setup_shared_gui_state(root)
    shared.metric_scope.set("none")

    monkeypatch.setattr("gui.gui_logic.draw_chart", lambda *a, **k: None)
    monkeypatch.setattr("gui.gui_logic.filter_metrics_by_scope", lambda x: {})

    update_chart({"flake8.unused_imports": 2})
    assert "no matching metrics" in caplog.text.lower()

def test_update_footer_summary_shows_totals_and_avg(root):
    tree = ttk.Treeview(root, columns=("Metric", "Total", "Average"))
    tree.pack()
    setup_shared_gui_state(root)

    flat_metrics = {
        "flake8.unused_imports": 3,
        "bandit.issues": 2,
        "not_numeric": "N/A"
    }

    update_footer_summary(tree, flat_metrics)

    items = tree.get_children()
    assert len(items) == 2  # Only numeric metrics are summarised
    values = [list(map(str, tree.item(i)["values"])) for i in items]

    assert ["flake8.unused_imports", "3.0", "3.0"] in values
    assert ["bandit.issues", "2.0", "2.0"] in values
