# gui/file_ops.py

import subprocess
import json
import csv
from pathlib import Path
from tkinter import filedialog, messagebox
from gui.shared_state import results
from gui.gui_logic import update_tree, update_footer_summary


def run_metric_extraction(file_path, show_result=True):
    """
    Run metric analysis on a single file and store the result.
    """
    if not file_path:
        return

    out_file = Path("metrics.json")
    try:
        # Use correct script path depending on whether frozen or not
        if hasattr(__import__('sys'), "_MEIPASS"):
            script_path = Path(__import__('sys')._MEIPASS) / "src" / "metrics" / "main.py"
        else:
            script_path = Path(__file__).resolve().parent.parent / "src" / "metrics" / "main.py"

        cmd = [
            "python", str(script_path),
            "--file", file_path,
            "--out", str(out_file),
            "--format", "json"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        if not out_file.exists():
            raise FileNotFoundError("metrics.json not created.")

        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        results[file_path] = data

        if show_result:
            update_tree({file_path: data})
            update_footer_summary()

    except subprocess.CalledProcessError as e:
        messagebox.showerror(
            "CLI Error",
            f"Error analysing: {file_path}\n\nExit Code: {e.returncode}\n\n{e.stderr}"
        )
    except Exception as e:
        messagebox.showerror("Unexpected Error", str(e))


def run_directory_analysis():
    """
    Prompt the user to select a folder and run analysis on all .py files inside.
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


def export_to_csv():
    """
    Export collected metrics to a CSV file, including summary rows.
    """
    if not results:
        messagebox.showinfo("No Data", "No metrics to export.")
        return

    save_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
    )
    if not save_path:
        return

    # Collect all unique top-level metric keys
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

        # Add total and average summary rows
        writer.writerow([])
        writer.writerow(["Summary"])

        def safe_numeric(val):
            try:
                return float(val)
            except Exception:
                return 0.0

        totals = [sum(safe_numeric(results[f].get(k, 0)) for f in results) for k in metric_keys]
        avgs = [round(t / len(results), 2) for t in totals]

        writer.writerow(["Total"] + totals)
        writer.writerow(["Average"] + avgs)

    messagebox.showinfo("Exported", "CSV successfully saved.")
