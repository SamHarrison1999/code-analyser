# File: gui_components.py

# ⚠️ SAST Risk: GUI callback arguments were being evaluated immediately instead of being passed as callables
# ✅ Best Practice: Use lambda to defer function execution

import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import sys
import webbrowser

from gui.shared_state import get_shared_state
from gui.file_ops import run_metric_extraction, run_directory_analysis, export_to_csv
from gui.chart_utils import (
    draw_chart,
    redraw_last_chart,
    filter_metrics_by_scope,
    export_all_assets,
)
from gui.gui_logic import update_tree
from gui.utils import flatten_metrics
from gui.heatmap_renderer import refresh_overlay_for_file
from gui.overlay_loader import (
    load_together_ai_overlay,
    load_rl_overlay_from_json,
    merge_overlay_summaries,
    extract_rl_tokens,
)
from ml.tensorboard_logger import log_metric_bundle_to_tensorboard
from metrics.ast_metrics.gather import gather_ast_metrics_bundle
from ml.config_manager import load_config, save_config


def append_export_log(message: str):
    from datetime import datetime

    log_path = Path("datasets/annotated_fintech/export_log.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path.write_text(f"[{timestamp}] {message}\n", encoding="utf-8", append=True)


def simple_input(title, prompt, default=""):
    import tkinter.simpledialog as sd

    return sd.askstring(title, prompt, initialvalue=default)


def update_footer_summary(tree: ttk.Treeview, flat_metrics: dict) -> None:
    """Update the summary view with flattened metrics and AI overlay signals."""
    shared_state = get_shared_state()
    overlay_summary = (
        shared_state.overlay_summary if hasattr(shared_state, "overlay_summary") else {}
    )
    tree.delete(*tree.get_children())
    metric_groups = {}

    for k, v in flat_metrics.items():
        try:
            metric_groups.setdefault(k, []).append(float(v))
        except Exception:
            continue

    for key, values in metric_groups.items():
        total = round(sum(values), 2)
        avg = round(total / len(values), 2)
        ai_signal = overlay_summary.get(key, {}).get("AI Signal", "").lower()

        if "high" in ai_signal:
            icon = "🔴"
        elif "medium" in ai_signal:
            icon = "🟡"
        elif "low" in ai_signal:
            icon = "🟢"
        else:
            icon = ""

        tree.insert("", "end", values=(key, total, avg, icon))

    if hasattr(shared_state, "overlay_status_label"):
        if overlay_summary:
            shared_state.overlay_status_label.config(
                text="🧠 AI overlay: loaded ✅", fg="green"
            )
        else:
            shared_state.overlay_status_label.config(
                text="🧠 AI overlay: not loaded", fg="darkgrey"
            )


def launch_gui(root: tk.Tk) -> None:
    shared_state = get_shared_state()
    root.title("🧠 Code Analyser GUI")
    root.geometry("1000x900")
    root.bind("<Configure>", on_resize)

    def clean_exit(root: tk.Tk) -> None:
        """Safely shut down the application."""
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception as e:
            logging.warning(f"⚠️ Could not close matplotlib figures: {e}")
        try:
            root.quit()
            root.destroy()
        except Exception as e:
            logging.warning(f"⚠️ Error during GUI shutdown: {e}")
        logging.info("📤 Clean exit triggered")
        sys.exit(0)

    # ✅ Use lambda to defer execution
    root.protocol("WM_DELETE_WINDOW", lambda: clean_exit(root))

    # === Top toolbar ===
    top_frame = tk.Frame(root)
    top_frame.pack(pady=10)

    def create_button(parent, text, command, col, tooltip_text):
        btn = tk.Button(parent, text=text, command=command)
        btn.grid(row=0, column=col, padx=5)
        tooltip = tk.Label(parent, text=tooltip_text, fg="gray", font=("Arial", 7))
        tooltip.grid(row=1, column=col)
        tooltip.grid_remove()
        btn.bind("<Enter>", lambda e: tooltip.grid())
        btn.bind("<Leave>", lambda e: tooltip.grid_remove())

    create_button(
        top_frame,
        "📂 File",
        lambda: prompt_and_extract_file(),
        0,
        "Select a single file to analyse",
    )
    create_button(
        top_frame,
        "📁 Folder",
        lambda: run_directory_analysis(),
        1,
        "Analyse all files in a folder",
    )
    create_button(
        top_frame,
        "📊 File Chart",
        lambda: show_chart(),
        2,
        "Show metrics chart for selected file",
    )
    create_button(
        top_frame,
        "📊 Dir Chart",
        lambda: show_directory_summary_chart(),
        3,
        "Aggregate chart across folder",
    )
    create_button(
        top_frame, "📄 Export CSV", lambda: export_to_csv(), 4, "Export metrics to CSV"
    )
    create_button(
        top_frame,
        "🧠 Annotate File",
        lambda: annotate_selected_file(),
        5,
        "AI annotate the selected file",
    )
    create_button(
        top_frame,
        "📤 Export All",
        lambda: on_export_all(),
        6,
        "Save all charts in selected formats",
    )
    create_button(
        top_frame,
        "📦 Export ZIP",
        lambda: export_zip_bundle(),
        10,
        "Create dashboard zip bundle",
    )
    create_button(
        top_frame,
        "☁️ Upload HF",
        lambda: upload_to_huggingface_gui(),
        11,
        "Upload bundle to HuggingFace",
    )
    create_button(
        top_frame,
        "📨 Email ZIP",
        lambda: email_zip_gui(),
        12,
        "Send ZIP bundle via email",
    )
    create_button(
        top_frame, "📊 Open Dashboard", lambda: open_dashboard(), 13, "Open dashboard"
    )

    def refresh_ai_overlay():
        filepath = shared_state.current_file_path
        if not filepath:
            messagebox.showerror("No File", "No file selected.")
            return

        base = Path(filepath)
        overlay_path = base.with_suffix(".json")
        rl_path = base.with_name(base.stem + ".tb.json")
        alt_rl_path = base.with_name(base.stem + ".rl_log.json")

        together_data = {}
        rl_summary = {}
        rl_tokens = []

        try:
            if overlay_path.exists():
                together_data = load_together_ai_overlay(overlay_path)
                shared_state.overlay_tokens = together_data.get("overlays", [])
        except Exception as e:
            logging.warning(f"⚠️ Failed to load Together.ai overlay: {e}")

        try:
            if rl_path.exists():
                rl_summary = load_rl_overlay_from_json(rl_path)
                rl_tokens = extract_rl_tokens(rl_path)
            elif alt_rl_path.exists():
                rl_summary = load_rl_overlay_from_json(alt_rl_path)
                rl_tokens = extract_rl_tokens(alt_rl_path)
        except Exception as e:
            logging.warning(f"⚠️ Failed to load RL overlay: {e}")

        shared_state.overlay_tokens.extend(rl_tokens)
        summary = merge_overlay_summaries(together_data.get("summary", {}), rl_summary)
        shared_state.overlay_summary = summary

        flat_metrics = flatten_metrics(shared_state.results.get(filepath, {}))
        update_footer_summary(shared_state.summary_tree, flat_metrics)
        refresh_overlay_for_file(filepath, shared_state.overlay_tokens)

    create_button(
        top_frame,
        "🔄 Refresh AI Overlay",
        lambda: refresh_ai_overlay(),
        7,
        "Reload AI overlay from .json",
    )
    create_button(
        top_frame,
        "📈 Export to TB",
        lambda: export_current_to_tensorboard(),
        9,
        "Log AST metrics to TensorBoard",
    )
    tk.Button(top_frame, text="Exit", command=lambda: clean_exit(root)).grid(
        row=0, column=8, padx=5
    )

    # === Export + Overlay Checkboxes ===
    export_format_frame = tk.Frame(root)
    export_format_frame.pack(pady=(0, 10))

    tk.Label(export_format_frame, text="Export Formats:").pack(
        side=tk.LEFT, padx=(0, 5)
    )
    for fmt in ["csv", "json", "png"]:
        cb = tk.Checkbutton(
            export_format_frame,
            text=fmt.upper(),
            variable=shared_state.export_formats[fmt],
        )
        cb.pack(side=tk.LEFT)

    shared_state.export_with_overlay = tk.BooleanVar(value=False)
    overlay_cb = tk.Checkbutton(
        export_format_frame,
        text="Export with overlays",
        variable=shared_state.export_with_overlay,
    )
    overlay_cb.pack(side=tk.LEFT, padx=10)

    overlay_cb_tooltip = tk.Label(
        export_format_frame,
        text="Include AI overlays (heatmap) in exported charts",
        fg="gray",
        font=("Arial", 7),
    )
    overlay_cb_tooltip.pack(side=tk.LEFT)
    overlay_cb_tooltip.pack_forget()
    overlay_cb.bind("<Enter>", lambda e: overlay_cb_tooltip.pack(side=tk.LEFT))
    overlay_cb.bind("<Leave>", lambda e: overlay_cb_tooltip.pack_forget())

    # === Chart type & Metric scope ===
    option_frame = tk.Frame(root)
    option_frame.pack(pady=5)
    tk.Label(option_frame, text="Chart Type:").pack(side=tk.LEFT)
    tk.Radiobutton(
        option_frame, text="Bar", variable=shared_state.chart_type, value="bar"
    ).pack(side=tk.LEFT)

    tk.Label(option_frame, text="Metric Scope:").pack(side=tk.LEFT, padx=(20, 5))

    for label, value in [
        ("AST", "ast"),
        ("Bandit", "bandit"),
        ("Cloc", "cloc"),
        ("Flake8", "flake8"),
        ("Lizard", "lizard"),
        ("Pydocstyle", "pydocstyle"),
        ("Pyflakes", "pyflakes"),
        ("Pylint", "pylint"),
        ("Radon", "radon"),
        ("Vulture", "vulture"),
        ("Sonar", "sonar"),
        ("Together AI", "together_ai"),
        ("RL Agent", "rl_agent"),
        ("AI (Unified)", "ai"),
        ("All", "all"),
    ]:
        tk.Radiobutton(
            option_frame,
            text=label,
            variable=shared_state.metric_scope,
            value=value,
            command=lambda: refresh_chart_on_scope_change(),
        ).pack(side=tk.LEFT)

    # === Filter field ===
    filter_frame = tk.Frame(root)
    filter_frame.pack(fill=tk.X, padx=10, pady=5)
    tk.Label(filter_frame, text="Filter: ").pack(side=tk.LEFT)
    tk.Entry(filter_frame, textvariable=shared_state.filter_var, width=40).pack(
        side=tk.LEFT, expand=True, fill=tk.X
    )

    logging.debug("📌 Calling trace_add from <launch_gui>")
    shared_state.filter_trace_id = shared_state.filter_var.trace_add(
        "write", on_filter_change
    )

    # === Notebook ===
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    # Charts Tab
    chart_tab = tk.Frame(notebook)
    chart_canvas_widget = tk.Canvas(chart_tab)
    chart_scroll = ttk.Scrollbar(
        chart_tab, orient="vertical", command=chart_canvas_widget.yview
    )
    scrollable_chart = tk.Frame(chart_canvas_widget)
    scrollable_chart.bind(
        "<Configure>",
        lambda e: chart_canvas_widget.configure(
            scrollregion=chart_canvas_widget.bbox("all")
        ),
    )
    chart_canvas_widget.create_window((0, 0), window=scrollable_chart, anchor="nw")
    chart_canvas_widget.configure(yscrollcommand=chart_scroll.set)
    chart_canvas_widget.pack(side="left", fill="both", expand=True)
    chart_scroll.pack(side="right", fill="y")

    shared_state.chart_frame = scrollable_chart
    shared_state.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    notebook.add(chart_tab, text="📊 Charts")

    # Metrics Tree Tab
    tree_tab = tk.Frame(notebook)
    tree_scroll = ttk.Scrollbar(tree_tab, orient="vertical")
    tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    shared_state.tree = ttk.Treeview(
        tree_tab,
        columns=("Metric", "Value"),
        show="headings",
        yscrollcommand=tree_scroll.set,
    )
    for col in ("Metric", "Value"):
        shared_state.tree.heading(col, text=col)
        shared_state.tree.column(col, anchor="w")
    shared_state.tree.pack(fill=tk.BOTH, expand=True)
    tree_scroll.config(command=shared_state.tree.yview)
    notebook.add(tree_tab, text="📋 Metrics")

    # Summary Tab
    summary_tab = tk.Frame(notebook)
    summary_scroll = ttk.Scrollbar(summary_tab, orient="vertical")
    summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    shared_state.summary_tree = ttk.Treeview(
        summary_tab,
        columns=("Metric", "Total", "Average", "AI Signal"),
        show="headings",
        yscrollcommand=summary_scroll.set,
    )
    for col in ("Metric", "Total", "Average", "AI Signal"):
        shared_state.summary_tree.heading(col, text=col)
        shared_state.summary_tree.column(col, anchor="center")
    shared_state.summary_tree.column("Metric", anchor="w", width=300)
    shared_state.summary_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    summary_scroll.config(command=shared_state.summary_tree.yview)

    overlay_status = tk.Label(
        summary_tab,
        text="🧠 AI overlay: not loaded",
        font=("Segoe UI", 9, "italic"),
        fg="darkgrey",
    )
    overlay_status.pack(side=tk.BOTTOM, pady=4)
    shared_state.overlay_status_label = overlay_status
    notebook.add(summary_tab, text="📈 Summary")

    # === Footer TensorBoard Launch ===
    footer_frame = tk.Frame(root)
    footer_frame.pack(side=tk.BOTTOM, pady=4)

    from ml.zip_exporter import create_export_zip
    from ml.huggingface_uploader import upload_zip_to_huggingface
    from ml.email_zip_sender import send_zip_email

    def export_zip_bundle():
        shared_state = get_shared_state()
        output_dir = (
            shared_state.output_dir
            if hasattr(shared_state, "output_dir")
            else Path("datasets/annotated_fintech")
        )
        zip_path = create_export_zip(output_dir)
        messagebox.showinfo("ZIP Exported", f"Bundle saved to:\n{zip_path}")
        return zip_path

    def upload_to_huggingface_gui():
        config = load_config()
        zip_path = export_zip_bundle()
        token = simple_input(
            "HuggingFace Token",
            "Enter your token:",
            default=config.get("huggingface_token"),
        )
        repo = simple_input(
            "Repo Name", "Dataset repo:", default=config.get("huggingface_repo")
        )
        try:
            url = upload_zip_to_huggingface(zip_path, repo, token)
            messagebox.showinfo("Uploaded", f"✅ Uploaded to: {url}")
            config.update({"huggingface_token": token, "huggingface_repo": repo})
            save_config(config)
            append_export_log(f"HuggingFace upload: {zip_path.name} → {url}")
        except Exception as e:
            messagebox.showerror("Upload Failed", str(e))

    def email_zip_gui():
        config = load_config()
        zip_path = export_zip_bundle()
        to_email = simple_input("Send To", "Recipient email:")
        smtp_user = simple_input(
            "SMTP User", "Your email:", default=config.get("smtp_user")
        )
        smtp_pass = simple_input(
            "SMTP Pass", "App password:", default=config.get("smtp_pass")
        )
        smtp_host = simple_input(
            "SMTP Host", "SMTP server:", default=config.get("smtp_host")
        )
        smtp_port = simple_input(
            "SMTP Port", "SMTP port:", default=str(config.get("smtp_port"))
        )

        try:
            send_zip_email(
                zip_path,
                to_email,
                smtp_user,
                smtp_host,
                int(smtp_port),
                smtp_user,
                smtp_pass,
            )
            messagebox.showinfo("Sent", f"Email sent to {to_email}")
            config.update(
                {
                    "smtp_user": smtp_user,
                    "smtp_pass": smtp_pass,
                    "smtp_host": smtp_host,
                    "smtp_port": int(smtp_port),
                }
            )
            save_config(config)
            append_export_log(f"📨 Email: {zip_path.name} → {to_email}")
        except Exception as e:
            messagebox.showerror("Email Failed", str(e))

    def launch_tensorboard_gui():
        from main import launch_tensorboard

        launch_tensorboard()

    # Add in GUI under "Export All" button
    create_button(
        top_frame,
        "📈 TensorBoard",
        lambda: launch_tensorboard_gui(),
        8,
        "Launch AI training visualisation",
    )

    root.mainloop()


def on_resize(event: tk.Event) -> None:
    try:
        shared_state = get_shared_state()
    except RuntimeError:
        return
    if event.widget == event.widget.winfo_toplevel():
        redraw_last_chart()


def open_dashboard():
    output_dir = (
        get_shared_state().output_dir
        if hasattr(get_shared_state(), "output_dir")
        else Path("datasets/annotated_fintech")
    )
    dashboard_path = output_dir / "dashboard.html"
    if dashboard_path.exists():
        webbrowser.open(dashboard_path.resolve().as_uri())


def prompt_and_extract_file() -> None:
    logging.debug("📌 prompt_and_extract_file() triggered")
    shared_state = get_shared_state()

    try:
        shared_state.filter_var.trace_remove("write", shared_state.filter_trace_id)
    except Exception as e:
        logging.debug(f"⚠️ Could not remove trace temporarily: {e}")

    path = filedialog.askopenfilename()
    if path:
        shared_state.current_file_path = path
        run_metric_extraction(path)
        update_tree(shared_state.tree, path)
        flat_metrics = flatten_metrics(shared_state.results.get(path, {}))
        update_footer_summary(shared_state.summary_tree, flat_metrics)

    shared_state.filter_trace_id = shared_state.filter_var.trace_add(
        "write", on_filter_change
    )


def annotate_selected_file() -> None:
    """Use Together AI to annotate the selected file."""
    shared_state = get_shared_state()
    filepath = shared_state.current_file_path
    if not filepath or not Path(filepath).is_file():
        messagebox.showerror(
            "No File Selected", "Please select a file before annotating."
        )
        return
    try:
        logging.info(f"🧠 Annotating file with Together AI: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            original_code = f.read()
        annotated_code = annotate_code_with_together_ai(original_code)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(annotated_code)
        run_metric_extraction(filepath)
        update_tree(shared_state.tree, filepath)
        flat_metrics = flatten_metrics(shared_state.results.get(filepath, {}))
        update_footer_summary(shared_state.summary_tree, flat_metrics)
        messagebox.showinfo(
            "Annotation Complete", "File successfully annotated with AI."
        )
    except Exception as e:
        logging.exception(f"❌ Annotation failed: {e}")
        messagebox.showerror(
            "Annotation Error", f"An error occurred during annotation:\n{e}"
        )


def on_filter_change(*args) -> None:
    logging.debug("📌 on_filter_change() triggered")
    shared_state = get_shared_state()
    keys = list(shared_state.results.keys())
    file_path = keys[0] if keys else None
    if file_path:
        update_tree(shared_state.tree, file_path)
        flat_metrics = flatten_metrics(shared_state.results.get(file_path, {}))
        update_footer_summary(shared_state.summary_tree, flat_metrics)


def refresh_chart_on_scope_change() -> None:
    shared_state = get_shared_state()
    if shared_state.current_file_path:
        show_chart()


def show_chart() -> None:
    shared_state = get_shared_state()
    filename = shared_state.current_file_path
    if not filename:
        return
    file_metrics = shared_state.results.get(filename, {})
    filtered = filter_metrics_by_scope(file_metrics)
    if not filtered:
        messagebox.showinfo(
            "No Metrics",
            f"No metrics found for scope: {shared_state.metric_scope.get()}",
        )
        return
    keys = list(filtered.keys())
    vals = [round(float(filtered[k]), 2) for k in keys]
    draw_chart(
        keys,
        vals,
        f"Metrics - Scope: {shared_state.metric_scope.get()}",
        "scope_chart.png",
    )


def show_directory_summary_chart() -> None:
    shared_state = get_shared_state()
    if not shared_state.results:
        messagebox.showinfo("No Data", "No analysis has been run.")
        return
    scope = shared_state.metric_scope.get()
    combined = {}
    for file_data in shared_state.results.values():
        filtered = filter_metrics_by_scope(file_data)
        for k, v in filtered.items():
            try:
                combined[k] = combined.get(k, 0) + float(v)
            except (TypeError, ValueError):
                continue
    if not combined:
        messagebox.showinfo(
            "No Metrics", f"No numeric metrics available for scope: {scope}"
        )
        return
    keys = list(combined.keys())
    vals = [round(combined[k], 2) for k in keys]
    draw_chart(keys, vals, f"Metrics - Scope: {scope}", f"summary_scope_{scope}.png")


def on_export_all():
    path = export_all_assets()
    messagebox.showinfo(
        "Export Complete", f"All charts and overlays exported to: {path}"
    )


def export_current_to_tensorboard():
    shared_state = get_shared_state()
    filepath = shared_state.current_file_path
    if not filepath:
        messagebox.showerror("No File", "No file selected.")
        return

    try:
        bundle = gather_ast_metrics_bundle(filepath)
        log_metric_bundle_to_tensorboard(bundle, run_name="gui_export", step=0)
        messagebox.showinfo("Export Complete", "Logged AST metrics to TensorBoard.")
    except Exception as e:
        logging.exception("❌ TensorBoard export failed")
        messagebox.showerror("Export Error", str(e))
