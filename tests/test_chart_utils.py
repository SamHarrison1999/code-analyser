import matplotlib
matplotlib.use("Agg")  # ✅ Use headless backend for testing

import os
os.environ["TCL_LIBRARY"] = r"C:\Python312\tcl\tcl8.6"
os.environ["TK_LIBRARY"] = r"C:\Python312\tcl\tk8.6"


import tkinter as tk
import pytest
from pathlib import Path

from gui.chart_utils import draw_chart, save_chart_as_image
from gui.shared_state import setup_shared_gui_state, get_shared_state


@pytest.fixture
def setup_gui_state():
    """Fixture to create and destroy a hidden root Tk window with shared GUI state."""
    root = tk.Tk()
    root.withdraw()
    setup_shared_gui_state(root)
    yield root
    root.destroy()


def test_draw_chart_basic(tmp_path, setup_gui_state):
    """Test that a basic chart image is drawn and saved to the filesystem."""
    shared = get_shared_state()
    shared.chart_frame = tk.Frame(setup_gui_state)
    shared.chart_frame.pack()

    keys = ["metric1", "metric2"]
    values = [1.0, 2.5]

    output_path = tmp_path / "test_chart.png"
    draw_chart(keys, values, "Test Title", str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_draw_chart_and_save(tmp_path, setup_gui_state):
    """Test draw_chart followed by explicit save_chart_as_image output."""
    shared = get_shared_state()
    shared.chart_frame = tk.Frame(setup_gui_state)
    shared.chart_frame.pack()

    output_path = tmp_path / "chart.png"
    draw_chart(["A", "B"], [1, 2], "Title", str(output_path))

    save_chart_as_image(output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
