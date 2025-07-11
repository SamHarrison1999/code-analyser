import csv
import json
import subprocess
import sys
import logging
from pathlib import Path
from tkinter import filedialog, messagebox

from gui.gui_logic import update_tree, update_footer_summary
from gui.utils import flatten_metrics


def run_metric_extraction(file_path: str, show_result: bool = True) -> None:
    """Run static metric analysis on a single Python file and store the result."""
    from gui import shared_state  # 🔁 defer to prevent circular import

    if not file_path:
        return

    out_file = Path("metrics.json")

    try:
        if getattr(sys, 'frozen', False):
            script_args = [
                sys.executable, "-m", "metrics.main",
                "--file", file_path,
                "--out", str(out_file),
                "--format", "json"
            ]
        else:
            script_path = Path(__file__).resolve().parent.parent / "metrics" / "main.py"
            script_args = [
                sys.executable, str(script_path),
                "--file", file_path,
                "--out", str(out_file),
                "--format", "json"
            ]

        subprocess.run(script_args, capture_output=True, text=True, check=True)

        if not out_file.exists():
            raise FileNotFoundError("metrics.json not created.")

        with out_file.open("r", encoding="utf-8") as f:
            parsed = json.load(f)

        if not isinstance(parsed, dict) or "metrics" not in parsed:
            raise ValueError("Invalid metrics.json format. Expected top-level 'metrics' key.")

        shared_state.results[file_path] = parsed["metrics"]

        if show_result:
            update_tree(shared_state.tree, file_path)
            flat_metrics = flatten_metrics(shared_state.results[file_path])
            update_footer_summary(shared_state.summary_tree, flat_metrics)

    except subprocess.CalledProcessError as e:
        messagebox.showerror("CLI Error", f"Error analysing: {file_path}\n\nExit Code: {e.returncode}\n\n{e.stderr}")
    except Exception as e:
        messagebox.showerror("Unexpected Error", f"{type(e).__name__}: {str(e)}")


def run_directory_analysis() -> None:
    """Prompt the user to select a folder and analyse all .py files inside recursively."""
    from gui import shared_state  # 🔁 defer
    folder = filedialog.askdirectory()
    if not folder:
        return

    py_files = list(Path(folder).rglob("*.py"))
    if not py_files:
        messagebox.showinfo("No Files", "No .py files found in the selected directory.")
        return

    for file in py_files:
        run_metric_extraction(str(file), show_result=False)

    update_tree(shared_state.tree, list(shared_state.results.keys())[0])
    flat_metrics = flatten_metrics(shared_state.results[list(shared_state.results.keys())[0]])
    update_footer_summary(shared_state.summary_tree, flat_metrics)


def export_to_csv() -> None:
    """Export the collected metrics to a CSV file, including total and average summary rows."""
    from gui import shared_state  # 🔁 defer

    if not shared_state.results:
        messagebox.showinfo("No Data", "No metrics to export.")
        return

    save_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Export Metrics as CSV"
    )
    if not save_path:
        return

    metric_keys = sorted({key for metrics in shared_state.results.values() for key in metrics})

    with open(save_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["File"] + metric_keys)

        for file, metrics in shared_state.results.items():
            row = [file]
            for key in metric_keys:
                val = metrics.get(key, 0)
                if isinstance(val, float):
                    val = round(val, 2)
                row.append(val)
            writer.writerow(row)

        writer.writerow([])
        writer.writerow(["Summary"])

        def safe_numeric(val):
            try:
                return float(val)
            except Exception:
                return 0.0

        totals = [round(sum(safe_numeric(shared_state.results[f].get(k, 0)) for f in shared_state.results), 2)
                  for k in metric_keys]
        avgs = [round(t / len(shared_state.results), 2) for t in totals]

        writer.writerow(["Total"] + totals)
        writer.writerow(["Average"] + avgs)

    messagebox.showinfo("Exported", "CSV successfully saved.")
