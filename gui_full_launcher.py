import sys
import subprocess
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
import mplcursors

if hasattr(sys, "_MEIPASS"):
    sys.path.insert(0, str(Path(sys._MEIPASS) / "src"))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

plt.style.use("seaborn-v0_8")

results = {}
chart_canvas = None

splash = tk.Tk()
splash.overrideredirect(True)
splash.geometry("360x120+600+300")
splash.configure(bg="white")
tk.Label(splash, text="Launching Code Analyser GUI...", font=("Segoe UI", 12, "bold"), bg="white").pack(pady=10)
progress = ttk.Progressbar(splash, mode='indeterminate', length=280)
progress.pack(pady=10)
progress.start(10)

def start_main_gui():
    progress.stop()
    splash.destroy()
    launch_gui()

splash.after(1500, start_main_gui)

def run_metric_extraction(file_path, show_result=True):
    out_file = Path("metrics.json")
    try:
        if hasattr(sys, "_MEIPASS"):
            script_path = Path(sys._MEIPASS) / "src" / "metrics" / "main.py"
        else:
            script_path = Path(__file__).resolve().parent / "src" / "metrics" / "main.py"
        python_exec = sys.executable
        if getattr(sys, 'frozen', False) and python_exec.lower().endswith("codeanalysergui.exe"):
            python_exec = "python"
        cmd = [python_exec, str(script_path), "--file", file_path, "--out", str(out_file), "--format", "json"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logging.info(result.stdout)
        if not out_file.exists():
            raise FileNotFoundError("metrics.json not created")
        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        results[file_path] = data
        if show_result:
            update_tree({file_path: data})
            update_footer_summary()
    except subprocess.CalledProcessError as e:
        messagebox.showerror("CLI Error", f"{file_path}\n\nExit Code: {e.returncode}\n\n{e.stderr}")
    except Exception as e:
        messagebox.showerror("Unexpected Error", str(e))

def run_directory_analysis():
    folder = filedialog.askdirectory()
    if not folder:
        return
    py_files = list(Path(folder).rglob("*.py"))
    if not py_files:
        messagebox.showinfo("No Files", "No .py files found.")
        return
    for file in py_files:
        run_metric_extraction(str(file), show_result=False)
    update_tree(results)
    update_footer_summary()

def update_tree(data: dict):
    for i in tree.get_children():
        tree.delete(i)
    for file, top_level in data.items():
        base_name = Path(file).name
        for key, value in top_level.items():
            if key == "metrics" and isinstance(value, dict):
                for metric_key, metric_val in value.items():
                    if filter_var.get().lower() in metric_key.lower() or filter_var.get().lower() in base_name.lower():
                        rounded_value = round(metric_val, 2) if isinstance(metric_val, float) else metric_val
                        tree.insert("", "end", values=(base_name, metric_key, rounded_value))
            else:
                if filter_var.get().lower() in key.lower() or filter_var.get().lower() in base_name.lower():
                    tree.insert("", "end", values=(base_name, key, value))

def update_footer_summary():
    all_keys = sorted({key for m in results.values() for key in m})
    def safe_numeric(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0
    totals = {k: sum(safe_numeric(results[f].get(k, 0)) for f in results) for k in all_keys}
    avgs = {k: round(totals[k] / len(results), 2) for k in all_keys}
    text = "Summary Totals:\n"
    for k in all_keys:
        text += f"{k}: total={totals[k]}, avg={avgs[k]}\n"
    summary_label.config(text=text)

def include_metric(metric):
    scope = metric_scope.get()
    if scope == "ast":
        return not metric.startswith("number_of_") and "docstring" not in metric
    elif scope == "bandit":
        return "security" in metric or "vulnerability" in metric
    elif scope == "flake8":
        return "styling" in metric or "line_length" in metric
    elif scope == "cloc":
        return "line" in metric or "comment" in metric
    elif scope == "lizard":
        return any(key in metric for key in ["complexity", "token", "parameter", "maintainability"])
    elif scope == "pydocstyle":
        return any(key in metric.lower() for key in ["docstring", "pydocstyle", "compliance"])
    return True

def draw_chart(keys, vals, title, filename):
    global chart_canvas
    for widget in chart_frame.winfo_children():
        widget.destroy()

    def prettify(label):
        label = label.replace("metrics.", "").replace("_", " ").strip()
        return label.capitalize()

    pretty_keys = [prettify(k) for k in keys]
    height_per_metric = 0.4
    fig_height = max(4, len(pretty_keys) * height_per_metric)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    if chart_type.get() == "bar":
        bars = ax.barh(pretty_keys, vals)
        ax.set_xlabel("Value")
        ax.set_ylabel("Metric")
        ax.tick_params(axis='y', labelsize=8)
        ax.set_title(title)

        # ✅ Add hover cursor for bar chart
        cursor = mplcursors.cursor(bars, hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(
            f"{pretty_keys[sel.index]}: {vals[sel.index]}"
        ))

    else:
        wedges, _ = ax.pie(vals, labels=None)
        ax.set_title(title)
        tooltip = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                              bbox=dict(boxstyle="round", fc="w"),
                              arrowprops=dict(arrowstyle="->"))
        tooltip.set_visible(False)

        def format_tooltip(event):
            for i, wedge in enumerate(wedges):
                if wedge.contains_point([event.x, event.y]):
                    tooltip.xy = (event.xdata, event.ydata)
                    tooltip.set_text(f"{pretty_keys[i]}: {vals[i]:.1f}%")
                    tooltip.set_visible(True)
                    fig.canvas.draw_idle()
                    return
            tooltip.set_visible(False)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", format_tooltip)

    fig.tight_layout()
    fig.savefig(filename)

    canvas_container = tk.Canvas(chart_frame)
    scrollbar = tk.Scrollbar(chart_frame, orient="vertical", command=canvas_container.yview)
    canvas_container.configure(yscrollcommand=scrollbar.set)
    chart_widget = tk.Frame(canvas_container)
    chart_canvas = FigureCanvasTkAgg(fig, master=chart_widget)
    chart_canvas.draw()
    chart_canvas.get_tk_widget().pack()
    canvas_container.create_window((0, 0), window=chart_widget, anchor="nw")
    canvas_container.update_idletasks()
    canvas_container.config(scrollregion=canvas_container.bbox("all"))
    canvas_container.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")



def show_chart():
    selected = tree.selection()
    if not selected:
        messagebox.showinfo("No Selection", "Please select a row.")
        return
    values = tree.item(selected[0], 'values')
    file_name = values[0]
    metrics = {}
    for f, data in results.items():
        if Path(f).name == file_name:
            metrics = data
            break
    def flatten_metrics(d, prefix=""):
        flat = {}
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (int, float)):
                flat[full_key] = v
            elif isinstance(v, dict):
                flat.update(flatten_metrics(v, full_key))
        return flat
    flat_metrics = flatten_metrics(metrics)
    keys = []
    vals = []
    for k, v in flat_metrics.items():
        if include_metric(k):
            keys.append(k)
            vals.append(round(v, 2))
    if not keys:
        messagebox.showinfo("No Numeric Metrics", f"No numeric metrics available for {file_name}.")
        return
    draw_chart(keys, vals, f"Metrics for {file_name}", f"chart_{file_name.replace('.py', '')}.png")

def show_directory_summary_chart():
    if not results:
        messagebox.showinfo("No Data", "No analysis has been run.")
        return
    def flatten_metrics(d, prefix=""):
        flat = {}
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (int, float)):
                flat[full_key] = v
            elif isinstance(v, dict):
                flat.update(flatten_metrics(v, full_key))
        return flat
    combined = {}
    for data in results.values():
        flat = flatten_metrics(data)
        for k, v in flat.items():
            if include_metric(k):
                combined[k] = combined.get(k, 0) + v
    if not combined:
        messagebox.showinfo("No Numeric Metrics", "No numeric metrics available to summarise.")
        return
    keys = list(combined.keys())
    vals = [round(combined[k], 2) for k in keys]
    draw_chart(keys, vals, "Directory-wide Metric Summary", "summary_chart.png")

def export_to_csv():
    if not results:
        messagebox.showinfo("No Data", "No metrics to export.")
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not save_path:
        return
    metrics_set = sorted({key for m in results.values() for key in m})
    with open(save_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["File"] + metrics_set)
        for file, metrics in results.items():
            row = [file] + [
                round(metrics.get(k, 0), 2) if isinstance(metrics.get(k, 0), float) else metrics.get(k, 0)
                for k in metrics_set
            ]
            writer.writerow(row)
        writer.writerow([])
        writer.writerow(["Summary"])
        totals = [sum(results[f].get(k, 0) for f in results) for k in metrics_set]
        avgs = [round(t / len(results), 2) for t in totals]
        writer.writerow(["Total"] + totals)
        writer.writerow(["Average"] + avgs)
    messagebox.showinfo("Exported", "CSV successfully saved.")

def launch_gui():
    global root, chart_type, metric_scope, chart_canvas, filter_var, chart_frame, tree, summary_label
    root = tk.Tk()
    chart_type = tk.StringVar(value="bar")
    metric_scope = tk.StringVar(value="all")
    chart_canvas = None
    filter_var = tk.StringVar()
    root.title("🧠 Code Analyser GUI")
    root.geometry("1000x900")
    top_frame = tk.Frame(root)
    top_frame.pack(pady=10)
    tk.Button(top_frame, text="📂 File", command=lambda: run_metric_extraction(filedialog.askopenfilename())).grid(row=0, column=0, padx=5)
    tk.Button(top_frame, text="📁 Folder", command=run_directory_analysis).grid(row=0, column=1, padx=5)
    tk.Button(top_frame, text="📊 File Chart", command=show_chart).grid(row=0, column=2, padx=5)
    tk.Button(top_frame, text="📊 Dir Chart", command=show_directory_summary_chart).grid(row=0, column=3, padx=5)
    tk.Button(top_frame, text="📄 Export CSV", command=export_to_csv).grid(row=0, column=4, padx=5)
    tk.Button(top_frame, text="Exit", command=root.destroy).grid(row=0, column=5, padx=5)
    option_frame = tk.Frame(root)
    option_frame.pack(pady=5)
    tk.Label(option_frame, text="Chart Type:").pack(side=tk.LEFT)
    tk.Radiobutton(option_frame, text="Bar", variable=chart_type, value="bar").pack(side=tk.LEFT)
    tk.Radiobutton(option_frame, text="Pie", variable=chart_type, value="pie").pack(side=tk.LEFT)
    tk.Label(option_frame, text="Metric Scope:").pack(side=tk.LEFT, padx=(20, 5))
    for label, value in [("AST", "ast"), ("Bandit", "bandit"), ("Flake8", "flake8"),
                         ("Cloc", "cloc"), ("Lizard", "lizard"), ("Pydocstyle", "pydocstyle"), ("All", "all")]:
        tk.Radiobutton(option_frame, text=label, variable=metric_scope, value=value).pack(side=tk.LEFT)
    filter_frame = tk.Frame(root)
    filter_frame.pack(fill=tk.X, padx=10, pady=5)
    filter_var.trace_add("write", lambda *args: update_tree(results))
    tk.Label(filter_frame, text="Filter: ").pack(side=tk.LEFT)
    tk.Entry(filter_frame, textvariable=filter_var, width=40).pack(side=tk.LEFT, expand=True, fill=tk.X)
    chart_frame = tk.Frame(root)
    chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    tree_frame = tk.Frame(root)
    tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    tree_scroll = tk.Scrollbar(tree_frame)
    tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    tree = ttk.Treeview(tree_frame, columns=("File", "Metric", "Value"), show="headings", yscrollcommand=tree_scroll.set)
    tree.heading("File", text="File")
    tree.heading("Metric", text="Metric")
    tree.heading("Value", text="Value")
    tree.pack(fill=tk.BOTH, expand=True)
    tree_scroll.config(command=tree.yview)
    summary_label = tk.Label(root, text="", justify=tk.LEFT, anchor="w")
    summary_label.pack(fill=tk.X, padx=10, pady=5)
    root.mainloop()

if __name__ == "__main__":
    splash.mainloop()
