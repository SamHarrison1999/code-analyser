# File: tests/test_file_ops.py

import pytest
import csv
import tkinter as tk
from unittest import mock
from pathlib import Path
from gui import file_ops
from gui.shared_state import setup_shared_gui_state, get_shared_state


@pytest.fixture(scope="module", autouse=True)
def gui_env():
    """Initialise and destroy the shared Tkinter root once per test module."""
    root = tk.Tk()
    root.withdraw()
    setup_shared_gui_state(root)
    yield
    root.destroy()


@mock.patch("tkinter.filedialog.askdirectory")
@mock.patch("gui.file_ops.run_metric_extraction")
def test_run_directory_analysis(mock_extract, mock_dialog, tmp_path):
    """Simulate analysis of multiple files and check results are populated."""
    test_dir = tmp_path / "sample_dir"
    test_dir.mkdir()
    (test_dir / "a.py").write_text("print('a')\n")
    (test_dir / "b.py").write_text("print('b')\n")
    mock_dialog.return_value = str(test_dir)

    shared = get_shared_state()
    shared.results.clear()

    # Simulate CLI extraction side effect
    def fake_extract(filepath, show_result=False):
        shared.results[str(filepath)] = {"flake8": {"unused_imports": 1}}

    mock_extract.side_effect = fake_extract

    file_ops.run_directory_analysis()

    assert mock_extract.call_count == 2
    assert any("a.py" in Path(f).name for f in shared.results)
    assert any("flake8" in metrics for metrics in shared.results.values())


@mock.patch("tkinter.filedialog.asksaveasfilename")
@mock.patch("tkinter.messagebox.showinfo")
@mock.patch("tkinter.messagebox.showerror")
def test_export_to_csv(mock_error, mock_info, mock_save, tmp_path):
    """Verify CSV export writes raw tool.metric keys in headers."""
    shared = get_shared_state()
    shared.results = {
        "file1.py": {
            "flake8": {"unused_imports": 2.0},
            "bandit": {"security_issues": 1.0}
        },
        "file2.py": {
            "flake8": {"unused_imports": 3.0},
            "bandit": {"security_issues": 0.0}
        }
    }

    save_path = tmp_path / "out.csv"
    mock_save.return_value = str(save_path)

    file_ops.export_to_csv()
    assert save_path.exists()

    with open(save_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)

    # ✅ Match actual raw keys written
    assert "flake8.unused_imports" in headers
    assert "bandit.security_issues" in headers

    # No error popup should have occurred
    mock_error.assert_not_called()


@mock.patch("tkinter.messagebox.showinfo")
@mock.patch("tkinter.messagebox.showerror")
@mock.patch("tkinter.filedialog.asksaveasfilename", return_value="ignored.csv")
def test_export_to_csv_no_data(mock_save, mock_error, mock_info):
    """Test that exporting with no results shows a messagebox, not an error."""
    shared = get_shared_state()
    shared.results = {}

    file_ops.export_to_csv()

    mock_info.assert_called_once()
    mock_error.assert_not_called()


@mock.patch("tkinter.filedialog.askdirectory", return_value="")
def test_run_directory_analysis_cancel(mock_dialog):
    """Ensure cancelling directory selection does not raise error."""
    file_ops.run_directory_analysis()
