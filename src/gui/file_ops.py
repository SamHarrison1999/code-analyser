import subprocess
import sys
import json
import csv
from pathlib import Path
from tkinter import filedialog, messagebox

from gui.shared_state import results
from gui.gui_logic import update_tree, update_footer_summary


def run_metric_extraction(file_path: str, show_result: bool = True) -> None:
    """
    Run static metric analysis on a single Python file and store the result.

    Args:
        file_path (str): Full path to the file to analyse.
        show_result (bool): Whether to immediately update the GUI with the result.
    """
    if not file_path:
        return

    out_file = Path("metrics.json")
    try:
        # ✅ Use module invocation in frozen mode to avoid relaunching GUI
        if getattr(sys, 'frozen', False):
            script_args = [
                sys.executable,
                "-m", "metrics.main",
                "--file", file_path,
                "--out", str(out_file),
                "--format", "json"
            ]
        else:
            script_path = Path(__file__).resolve().parent.parent / "metrics" / "main.py"
            script_args = [
                sys.executable,
                str(script_path),
                "--file", file_path,
                "--out", str(out_file),
                "--format", "json"
            ]

        result = subprocess.run(script_args, capture_output=True, text=True, check=True)

        if not out_file.exists():
            raise FileNotFoundError("metrics.json not created.")

        with out_file.open("r", encoding="utf-8") as f:
            parsed = json.load(f)

        if not isinstance(parsed, dict) or "metrics" not in parsed:
            raise ValueError("Invalid metrics.json format. Expected top-level 'metrics' key.")

        results[file_path] = parsed["metrics"]

        if show_result:
            update_tree({file_path: parsed["metrics"]})
            update_footer_summary()

    except subprocess.CalledProcessError as e:
        messagebox.showerror(
            "CLI Error",
            f"Error analysing: {file_path}\n\nExit Code: {e.returncode}\n\n{e.stderr}"
        )
    except Exception as e:
        messagebox.showerror("Unexpected Error", f"{type(e).__name__}: {str(e)}")


def run_directory_analysis() -> None:
    """
    Prompt the user to select a folder and analyse all .py files inside recursively.
    """
    folder = filedialog.askdirectory()
    if not folder:
        return

    py_files = list(Path(folder).rglob("*.py"))
    if not py_files:
        messagebox.showinfo("No Files", "No .py files found in the selected directory.")
        return

    for file in py_files:
        run_metric_extraction(str(file), show_result=False)

    update_tree(results)
    update_footer_summary()


def export_to_csv() -> None:
    """
    Export the collected metrics to a CSV file, including total and average summary rows.
    """
    if not results:
        messagebox.showinfo("No Data", "No metrics to export.")
        return

    save_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Export Metrics as CSV"
    )
    if not save_path:
        return

    # Collect all unique top-level metric keys across all results
    metric_keys = sorted({key for data in results.values() for key in data})

    with open(save_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["File"] + metric_keys)

        for file, metrics in results.items():
            row = [file]
            for key in metric_keys:
                val = metrics.get(key, 0)
                if isinstance(val, float):
                    val = round(val, 2)
                row.append(val)
            writer.writerow(row)

        # Add blank line and summary rows
        writer.writerow([])
        writer.writerow(["Summary"])

        def safe_numeric(val):
            try:
                return float(val)
            except Exception:
                return 0.0

        totals = [round(sum(safe_numeric(results[f].get(k, 0)) for f in results), 2) for k in metric_keys]
        avgs = [round(t / len(results), 2) for t in totals]

        writer.writerow(["Total"] + totals)
        writer.writerow(["Average"] + avgs)

    messagebox.showinfo("Exported", "CSV successfully saved.")
