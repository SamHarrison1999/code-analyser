# File: code_analyser/src/gui/file_ops.py

import os
import subprocess
import csv
import json
import sys
import logging
import tempfile
from pathlib import Path
from tkinter import filedialog, messagebox

from gui.gui_logic import update_tree, update_footer_summary
from gui.utils import flatten_metrics, merge_nested_metrics
from gui.chart_utils import _last_keys, _last_vals, _last_title

logger = logging.getLogger(__name__)

EXPORT_DIR = Path("exports").resolve()
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def run_metric_extraction(file_path: str, show_result: bool = True) -> None:
    """Run static metric analysis on a single Python file and store the result."""
    from gui.shared_state import get_shared_state

    shared_state = get_shared_state()

    if not file_path:
        return

    if file_path in shared_state.results:
        logger.debug(f"⏩ Skipping already-analysed file: {file_path}")
        return

    logger.debug(f"🚨 run_metric_extraction() called for: {file_path}")

    out_file = Path(tempfile.gettempdir()) / "metrics.json"

    try:
        if getattr(sys, "frozen", False):
            script_args = [
                sys.executable,
                "-m",
                "metrics.main",
                "--file",
                file_path,
                "--out",
                str(out_file),
                "--format",
                "json",
            ]
        else:
            script_path = Path(__file__).resolve().parents[1] / "metrics" / "main.py"
            script_args = [
                sys.executable,
                str(script_path),
                "--file",
                file_path,
                "--out",
                str(out_file),
                "--format",
                "json",
            ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

        result = subprocess.run(script_args, capture_output=True, text=True, check=True, env=env)
        logger.debug(f"🛠️ CLI Output for {file_path}:\n{result.stdout}\n{result.stderr}")

        if not out_file.exists():
            out_file.write_text(json.dumps({"metrics": {"error": "subprocess failed"}}))
            logger.warning("🛑 Wrote fallback metrics.json due to missing file.")

        with out_file.open("r", encoding="utf-8") as f:
            parsed = json.load(f)

        if not isinstance(parsed, dict) or "metrics" not in parsed:
            raise ValueError("Invalid format: 'metrics' key not found in metrics.json")

        shared_state.results[file_path] = parsed["metrics"]

        if show_result:
            update_tree(shared_state.tree, file_path)
            merged = merge_nested_metrics(shared_state.results[file_path])
            flat_metrics = flatten_metrics(merged)
            update_footer_summary(shared_state.summary_tree, flat_metrics)

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Subprocess error for {file_path}: {e.stderr}")
        messagebox.showerror(
            "CLI Error",
            f"Error analysing file: {file_path}\n\nExit Code: {e.returncode}\n\n{e.stderr}",
        )
    except Exception as e:
        logger.exception(f"❌ Unexpected error analysing {file_path}: {e}")
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
        file_path = str(file)
        if file_path in shared_state.results:
            logger.debug(f"⏩ Skipping already-analysed file: {file_path}")
            continue
        logger.debug(f"📌 Calling run_metric_extraction from <run_directory_analysis>: {file_path}")
        run_metric_extraction(file_path, show_result=False)

    if not shared_state.results:
        messagebox.showinfo("No Results", "No metrics were collected.")
        return

    first_file = list(shared_state.results.keys())[0]
    update_tree(shared_state.tree, first_file)
    merged = merge_nested_metrics(shared_state.results[first_file])
    flat_metrics = flatten_metrics(merged)
    update_footer_summary(shared_state.summary_tree, flat_metrics)


def export_to_csv(default_path: Path = None) -> None:
    """Export the collected metrics to a CSV file, preserving nested tool.metric keys and summary rows."""
    from gui.shared_state import get_shared_state
    from gui.utils import merge_nested_metrics

    shared_state = get_shared_state()

    if not shared_state.results:
        messagebox.showinfo("No Data", "No metrics to export.")
        return

    save_path = default_path or filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Export Metrics as CSV",
    )
    if not save_path:
        return

    try:
        flat_results = {
            file: merge_nested_metrics(metrics) for file, metrics in shared_state.results.items()
        }

        all_keys = sorted({key for metrics in flat_results.values() for key in metrics})

        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["File"] + all_keys)

            for file, metrics in flat_results.items():
                row = [file] + [round(metrics.get(key, 0), 2) for key in all_keys]
                writer.writerow(row)

            writer.writerow([])
            writer.writerow(["Summary"])

            def safe_numeric(val):
                try:
                    return float(val)
                except Exception:
                    return 0.0

            totals = [
                round(
                    sum(safe_numeric(flat_results[f].get(k, 0)) for f in flat_results),
                    2,
                )
                for k in all_keys
            ]
            averages = [round(t / len(flat_results), 2) for t in totals]

            writer.writerow(["Total"] + totals)
            writer.writerow(["Average"] + averages)

        logger.info(f"📄 CSV exported with full tool.metric keys: {save_path}")
        messagebox.showinfo("Exported", f"CSV successfully saved to {save_path}")

    except Exception as e:
        logger.exception(f"❌ Failed to export CSV: {e}")
        messagebox.showerror("Export Error", f"{type(e).__name__}: {str(e)}")


def export_last_chart_data(chart_type: str = "bar", format: str = "csv") -> None:
    """Export the latest chart data (bar/pie/heatmap) to CSV or JSON."""
    if not _last_keys or not _last_vals:
        messagebox.showinfo("No Chart", "No chart data available for export.")
        return

    base_filename = f"{_last_title.lower().replace(' ', '_')}_{chart_type}.{format}"
    save_path = EXPORT_DIR / base_filename

    try:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                for k, v in zip(_last_keys, _last_vals):
                    writer.writerow([k, v])
        elif format == "json":
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(dict(zip(_last_keys, _last_vals)), f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"✅ Exported chart ({chart_type}) as {format} to {save_path}")
        messagebox.showinfo("Exported", f"{chart_type.capitalize()} chart exported to {save_path}")

    except Exception as e:
        logger.exception(f"❌ Failed to export chart data: {e}")
        messagebox.showerror("Export Error", f"{type(e).__name__}: {str(e)}")


def export_all_assets() -> None:
    """Export all visualisation components: bar/pie/heatmap charts + overlays."""
    from gui.shared_state import get_shared_state
    from gui.heatmap_renderer import export_heatmap_to_csv, export_heatmap_to_png
    from gui.chart_utils import (
        export_last_chart_data,
        export_overlay_as_json,
        export_html_dashboard,
    )

    shared_state = get_shared_state()
    formats = [fmt for fmt, var in shared_state.export_formats.items() if var.get()]
    include_overlay = shared_state.export_with_overlay.get()

    try:
        export_last_chart_data(formats=formats)
        export_overlay_as_json("overlay_export.json")
        export_html_dashboard("dashboard.html")

        if include_overlay and shared_state.heatmap_frame and shared_state.overlay_tokens:
            export_heatmap_to_csv("exports/heatmap_overlay.csv", shared_state.overlay_tokens)
            export_heatmap_to_png(shared_state.heatmap_frame, "exports/heatmap_overlay.png")

        logger.info("📤 All visual assets exported successfully.")
        messagebox.showinfo("Export Complete", "All charts and overlays exported to /exports/")
    except Exception as e:
        logger.error(f"❌ Failed unified export: {e}")
        messagebox.showerror("Export Error", f"Unified export failed: {e}")
