# File: gui_full_launcher.py

import sys
import subprocess
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

plt.style.use("seaborn-v0_8")

results = {}
chart_canvas = None

# Splash screen with progress bar
splash = tk.Tk()
splash.overrideredirect(True)
splash.geometry("350x120+600+300")
splash.configure(bg="white")
tk.Label(splash, text="Loading Code Analyser GUI...", font=("Segoe UI", 12), bg="white").pack(pady=10)
progress = ttk.Progressbar(splash, mode='indeterminate', length=250)
progress.pack(pady=10)
progress.start(10)

def start_main_gui():
    progress.stop()
    splash.destroy()
    launch_gui()

splash.after(1500, start_main_gui)


def run_file_analysis():
    file_path = filedialog.askopenfilename(filetypes=[("Python files", "*.py")])
    if file_path:
        run_metric_extraction(file_path)


def run_directory_analysis():
    folder = filedialog.askdirectory()
    if not folder:
        return

    py_files = list(Path(folder).rglob("*.py"))
    if not py_files:
        messagebox.showinfo("No Files", "No .py files found in directory.")
        return

    for file in py_files:
        run_metric_extraction(str(file), show_result=False)

    update_tree(results)
    update_footer_summary()


def run_metric_extraction(file_path, show_result=True):
    out_file = "metrics.json"
    try:
        script_path = Path(__file__).resolve().parent / "src" / "metrics" / "main.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--file", file_path,
            "--out", out_file,
            "--format", "json"
        ]

        subprocess.run(cmd, capture_output=True, text=True, check=True)

        if not Path(out_file).exists():
            raise FileNotFoundError("metrics.json not created")

        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        results[file_path] = data
        if show_result:
            update_tree({file_path: data})
            update_footer_summary()
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"{file_path}\n\n{e.stderr}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def update_tree(data: dict):
    for i in tree.get_children():
        tree.delete(i)

    for file, metrics in data.items():
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if filter_var.get().lower() in k.lower() or filter_var.get().lower() in Path(file).name.lower():
                    tree.insert("", "end", values=(Path(file).name, k, v))


def update_footer_summary():
    all_keys = sorted({key for m in results.values() for key in m})
    totals = {k: sum(results[f].get(k, 0) for f in results) for k in all_keys}
    avgs = {k: round(t / len(results), 2) for k, t in totals.items()}

    text = "\nSummary Totals:\n"
    for k in all_keys:
        text += f"{k}: total={totals[k]}, avg={avgs[k]}\n"

    summary_label.config(text=text)


def export_to_csv():
    if not results:
        messagebox.showinfo("No data", "No metrics to export.")
        return

    metrics_set = sorted({key for m in results.values() for key in m})

    with open("metrics.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["File"] + metrics_set)

        for file, metrics in results.items():
            row = [file] + [metrics.get(k, 0) for k in metrics_set]
            writer.writerow(row)

        writer.writerow([])
        writer.writerow(["Summary"])
        totals = [sum(results[f].get(k, 0) for f in results) for k in metrics_set]
        avgs = [round(t / len(results), 2) for t in totals]
        writer.writerow(["Total"] + totals)
        writer.writerow(["Average"] + avgs)

    messagebox.showinfo("Exported", "Metrics with summary exported to metrics.csv")


def draw_chart(keys, vals, title, filename):
    global chart_canvas
    fig, ax = plt.subplots(figsize=(10, 5))

    if chart_type.get() == "bar":
        ax.barh(keys, vals)
        ax.set_xlabel("Value")
        ax.set_ylabel("Metric")
    else:
        ax.pie(vals, labels=keys, autopct='%1.1f%%')

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename)

    for widget in chart_frame.winfo_children():
        widget.destroy()

    chart_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    chart_canvas.draw()
    chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def show_chart():
    selected = tree.selection()
    if not selected:
        messagebox.showinfo("No Selection", "Please select a row.")
        return

    values = tree.item(selected[0], 'values')
    file_name = values[0]

    metrics = {k: v for f, data in results.items() if Path(f).name == file_name for k, v in data.items()}
    keys = [k for k in metrics if include_metric(k)]
    vals = [metrics[k] for k in keys]

    draw_chart(keys, vals, f"Metrics for {file_name}", f"chart_{file_name.replace('.py', '')}.png")


def show_directory_summary_chart():
    if not results:
        messagebox.showinfo("No Data", "No analysis has been run.")
        return

    combined = {}
    for metrics in results.values():
        for k, v in metrics.items():
            if include_metric(k):
                combined[k] = combined.get(k, 0) + v

    keys = list(combined.keys())
    vals = [combined[k] for k in keys]
    draw_chart(keys, vals, "Directory-wide Metric Summary", "summary_chart.png")


def include_metric(metric):
    if metric_scope.get() == "ast":
        return not metric.startswith("number_of_")
    elif metric_scope.get() == "bandit":
        return metric.startswith("number_of_")
    return True


def apply_filter(*args):
    update_tree(results)


def launch_gui():
    global root, include_bandit, chart_type, metric_scope, chart_canvas, filter_var, chart_frame, tree, summary_label

    root = tk.Tk()
    include_bandit = tk.BooleanVar()
    chart_type = tk.StringVar(value="bar")
    metric_scope = tk.StringVar(value="all")
    chart_canvas = None
    filter_var = tk.StringVar()

    root.title("Code Analyser GUI")
    root.geometry("900x850")

    frame = tk.Frame(root)
    frame.pack(pady=10)

    tk.Checkbutton(frame, text="Include Bandit Metrics", variable=include_bandit).grid(row=0, column=0, padx=5)
    tk.Button(frame, text="📂 Choose File", command=run_file_analysis).grid(row=0, column=1, padx=5)
    tk.Button(frame, text="📁 Analyse Folder", command=run_directory_analysis).grid(row=0, column=2, padx=5)
    tk.Button(frame, text="📄 Export CSV", command=export_to_csv).grid(row=0, column=3, padx=5)
    tk.Button(frame, text="📊 File Chart", command=show_chart).grid(row=0, column=4, padx=5)
    tk.Button(frame, text="📊 Dir Chart", command=show_directory_summary_chart).grid(row=0, column=5, padx=5)
    tk.Button(frame, text="Exit", command=root.quit).grid(row=0, column=6, padx=5)

    chart_frame = tk.Frame(root, height=300)
    chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    chart_type_frame = tk.Frame(root)
    chart_type_frame.pack(pady=5)

    tk.Label(chart_type_frame, text="Chart Type:").pack(side=tk.LEFT)
    tk.Radiobutton(chart_type_frame, text="Bar", variable=chart_type, value="bar").pack(side=tk.LEFT)
    tk.Radiobutton(chart_type_frame, text="Pie", variable=chart_type, value="pie").pack(side=tk.LEFT)

    tk.Label(chart_type_frame, text="Metric Scope:").pack(side=tk.LEFT, padx=(20, 5))
    tk.Radiobutton(chart_type_frame, text="AST", variable=metric_scope, value="ast").pack(side=tk.LEFT)
    tk.Radiobutton(chart_type_frame, text="Bandit", variable=metric_scope, value="bandit").pack(side=tk.LEFT)
    tk.Radiobutton(chart_type_frame, text="All", variable=metric_scope, value="all").pack(side=tk.LEFT)

    filter_frame = tk.Frame(root)
    filter_frame.pack(fill=tk.X, padx=10, pady=5)
    filter_var.trace_add("write", apply_filter)
    tk.Label(filter_frame, text="Filter: ").pack(side=tk.LEFT)
    tk.Entry(filter_frame, textvariable=filter_var, width=40).pack(side=tk.LEFT, expand=True, fill=tk.X)

    tree = ttk.Treeview(root, columns=("File", "Metric", "Value"), show="headings")
    tree.heading("File", text="File")
    tree.heading("Metric", text="Metric")
    tree.heading("Value", text="Value")
    tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    summary_label = tk.Label(root, text="", justify=tk.LEFT, anchor="w")
    summary_label.pack(fill=tk.X, padx=10, pady=5)

    root.mainloop()


# Start the splash window loop to trigger launch_gui()
splash.mainloop()
