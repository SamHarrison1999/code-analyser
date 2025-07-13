import csv
import json
import subprocess
import sys
import logging
from pathlib import Path
from tkinter import filedialog, messagebox

from gui.gui_logic import update_tree, update_footer_summary
from gui.utils import flatten_metrics, merge_nested_metrics


def run_metric_extraction(file_path: str, show_result: bool = True) -> None:
    """Run static metric analysis on a single Python file and store the result."""
    if not file_path:
        return

    from gui.shared_state import get_shared_state
    shared_state = get_shared_state()

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

        result = subprocess.run(script_args, capture_output=True, text=True, check=True)
        logging.debug(f"🛠️ CLI Output for {file_path}:{result.stdout}{result.stderr}")

        if not out_file.exists():
            raise FileNotFoundError("metrics.json not created.")

        with out_file.open("r", encoding="utf-8") as f:
            parsed = json.load(f)

        if not isinstance(parsed, dict) or "metrics" not in parsed:
            raise ValueError("Invalid metrics.json format. Expected top-level 'metrics' key.")

        shared_state.results[file_path] = parsed["metrics"]

        if show_result:
            update_tree(shared_state.tree, file_path)
            merged = merge_nested_metrics(shared_state.results[file_path])
            flat_metrics = flatten_metrics(merged)
            update_footer_summary(shared_state.summary_tree, flat_metrics)

    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Subprocess error for {file_path}: {e.stderr}")
        messagebox.showerror(
            "CLI Error",
            f"Error analysing: {file_path}\n\nExit Code: {e.returncode}\n\n{e.stderr}"
        )
    except Exception as e:
        logging.exception(f"❌ Unexpected error analysing {file_path}: {e}")
        messagebox.showerror("Unexpected Error", f"{type(e).__name__}: {str(e)}")


def run_directory_analysis() -> None:
    """Prompt the user to select a folder and analyse all .py files inside recursively."""
    from gui.shared_state import get_shared_state
    shared_state = get_shared_state()

    folder = filedialog.askdirectory()
    if not folder:
        return

    py_files = list(Path(folder).rglob("*.py"))
    if not py_files:
        messagebox.showinfo("No Files", "No .py files found in the selected directory.")
        return

    for file in py_files:
        run_metric_extraction(str(file), show_result=False)

    first_file = list(shared_state.results.keys())[0]
    update_tree(shared_state.tree, first_file)
    merged = merge_nested_metrics(shared_state.results[first_file])
    flat_metrics = flatten_metrics(merged)
    update_footer_summary(shared_state.summary_tree, flat_metrics)


def export_to_csv() -> None:
    """Export the collected metrics to a CSV file, preserving nested tool.metric keys and summary rows."""
    from gui.shared_state import get_shared_state
    shared_state = get_shared_state()

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

    # ✅ Preserve tool.metric structure in keys
    structured_results = {
        file: merge_nested_metrics(metrics)
        for file, metrics in shared_state.results.items()
    }

    # ✅ Collect all full tool.metric keys
    all_keys = sorted({
        f"{tool}.{metric}"
        for metrics in structured_results.values()
        for tool, group in metrics.items()
        if isinstance(group, dict)
        for metric in group
    })

    try:
        with open(save_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["File"] + all_keys)

            for file, metrics in structured_results.items():
                row = [file]
                for full_key in all_keys:
                    tool, metric = full_key.split(".", 1)
                    val = metrics.get(tool, {}).get(metric, 0)
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

            totals = [
                round(sum(safe_numeric(structured_results[f].get(tool, {}).get(metric, 0))
                          for f in structured_results), 2)
                for tool, metric in [key.split(".", 1) for key in all_keys]
            ]
            avgs = [round(t / len(structured_results), 2) for t in totals]

            writer.writerow(["Total"] + totals)
            writer.writerow(["Average"] + avgs)

        logging.info(f"📄 CSV exported with full tool.metric keys: {save_path}")
        messagebox.showinfo("Exported", "CSV successfully saved.")

    except Exception as e:
        logging.exception(f"❌ Failed to export CSV: {e}")
        messagebox.showerror("Export Error", f"{type(e).__name__}: {str(e)}")
