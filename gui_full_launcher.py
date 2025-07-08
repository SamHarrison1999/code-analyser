import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import os
import csv
from pathlib import Path

results = {}


def run_file_analysis():
    file_path = filedialog.askopenfilename(filetypes=[("Python files", "*.py")])
    if not file_path:
        return
    run_ast_metrics(file_path)


def run_directory_analysis():
    folder = filedialog.askdirectory()
    if not folder:
        return

    py_files = list(Path(folder).rglob("*.py"))
    if not py_files:
        messagebox.showinfo("No Files", "No .py files found in directory.")
        return

    for file in py_files:
        run_ast_metrics(str(file), show_result=False)

    update_tree(results)


def run_ast_metrics(file_path, show_result=True):
    out_file = "metrics.json"
    try:
        subprocess.run(
            ["ast-metrics", "--file", file_path, "--out", out_file],
            capture_output=True, text=True, check=True
        )
        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            results[file_path] = data
        if show_result:
            update_tree({file_path: data})
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"{file_path}\n\n{e.stderr}")


def update_tree(data: dict):
    for i in tree.get_children():
        tree.delete(i)

    for file, metrics in data.items():
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                tree.insert("", "end", values=(Path(file).name, k, v))


def export_to_csv():
    if not results:
        messagebox.showinfo("No data", "No metrics to export.")
        return

    metrics_set = sorted({key for m in results.values() for key in m})

    with open("metrics.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["File"] + metrics_set)

        # Write per-file rows
        for file, metrics in results.items():
            row = [file] + [metrics.get(k, 0) for k in metrics_set]
            writer.writerow(row)

        # Add summary row
        writer.writerow([])  # Empty line
        writer.writerow(["Summary"])

        totals = [sum(results[f].get(k, 0) for f in results) for k in metrics_set]
        avgs = [round(t / len(results), 2) for t in totals]
        writer.writerow(["Total"] + totals)
        writer.writerow(["Average"] + avgs)

    messagebox.showinfo("Exported", "Metrics with summary exported to metrics.csv")



# GUI setup
root = tk.Tk()
root.title("AST Metrics GUI")
root.geometry("600x500")

frame = tk.Frame(root)
frame.pack(pady=10)

tk.Button(frame, text="📂 Choose File", command=run_file_analysis).grid(row=0, column=0, padx=5)
tk.Button(frame, text="📁 Analyse Folder", command=run_directory_analysis).grid(row=0, column=1, padx=5)
tk.Button(frame, text="💾 Export CSV", command=export_to_csv).grid(row=0, column=2, padx=5)
tk.Button(frame, text="Exit", command=root.quit).grid(row=0, column=3, padx=5)

tree = ttk.Treeview(root, columns=("File", "Metric", "Value"), show="headings")
tree.heading("File", text="File")
tree.heading("Metric", text="Metric")
tree.heading("Value", text="Value")
tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

root.mainloop()
