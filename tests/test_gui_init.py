# test_gui_init.py
import importlib
import gui


def test_metadata_fields():
    assert gui.__version__ == "1.2.1"
    assert gui.__author__ == "Samuel Harrison"
    assert gui.__email__ == "sh18784@essex.ac.uk"


def test_exported_symbols_exist():
    assert callable(gui.launch_gui)
    assert callable(gui.run_metric_extraction)
    assert callable(gui.run_directory_analysis)
    assert callable(gui.draw_chart)
    assert callable(gui.update_tree)
    assert callable(gui.update_footer_summary)


def test___all__contents():
    expected = {
        "launch_gui",
        "run_metric_extraction",
        "run_directory_analysis",
        "draw_chart",
        "update_tree",
        "update_footer_summary",
    }
    assert set(gui.__all__) == expected


def test_import_gui_components():
    assert importlib.import_module("gui.gui_components")


def test_import_file_ops():
    assert importlib.import_module("gui.file_ops")


def test_import_chart_utils():
    assert importlib.import_module("gui.chart_utils")


def test_import_gui_logic():
    assert importlib.import_module("gui.gui_logic")
