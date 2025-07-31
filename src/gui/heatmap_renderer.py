import tkinter as tk
from tkinter import messagebox
from typing import List, Dict, Any
from gui.shared_state import get_shared_state
import csv
import logging
from PIL import ImageGrab

# === Colour map for severity ===
SEVERITY_COLOURS = {
    "low": "#B7E4C7",  # green
    "medium": "#FFD966",  # yellow
    "high": "#FF6F61",  # red
}


def draw_token_heatmap(container: tk.Frame, overlays: List[Dict[str, Any]]) -> None:
    """
    Render a simple token-based heatmap in the given container frame.
    Overlays are expected to be a list of dicts with line, token, confidence, severity.
    """
    shared_state = get_shared_state()

    # Clear existing widgets in the container
    for widget in container.winfo_children():
        widget.destroy()

    if not overlays:
        tk.Label(container, text="No overlay data available").pack()
        return

    # Organise tokens by line number
    line_to_tokens = {}
    for item in overlays:
        line = item.get("line")
        token = item.get("token")
        conf = item.get("confidence", 0.0)
        severity = item.get("severity", "unknown").lower()

        # Filter by severity settings
        severity_checkbox = shared_state.overlay_severity_filter.get(
            severity, tk.BooleanVar(value=True)
        )
        min_threshold = shared_state.overlay_conf_threshold.get()

        if not severity_checkbox.get() or conf < min_threshold:
            continue

        line_to_tokens.setdefault(line, []).append((token, conf, severity))

    if not line_to_tokens:
        tk.Label(container, text="No overlays match current filters").pack()
        return

    for line_num in sorted(line_to_tokens):
        line_frame = tk.Frame(container)
        line_frame.pack(anchor="w", pady=1, padx=5)
        tk.Label(line_frame, text=f"{line_num:>3}:", font=("Courier", 10, "bold")).pack(
            side=tk.LEFT
        )

        for token, conf, severity in line_to_tokens[line_num]:
            bg = SEVERITY_COLOURS.get(severity, "#ccc")
            label = tk.Label(
                line_frame,
                text=token,
                bg=bg,
                bd=1,
                relief="solid",
                padx=2,
                font=("Courier", 10),
            )
            label.pack(side=tk.LEFT, padx=1)

            tooltip = (
                f"{token} | Confidence: {conf:.2f} | Severity: {severity.capitalize()}"
            )
            label.bind(
                "<Enter>", lambda e, msg=tooltip: shared_state.status_var.set(msg)
            )
            label.bind("<Leave>", lambda e: shared_state.status_var.set(""))


def export_heatmap_to_csv(filepath: str, overlays: List[Dict[str, Any]]) -> None:
    """
    Save the overlay tokens to a .csv file.
    """
    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["line", "token", "confidence", "severity"]
            )
            writer.writeheader()
            writer.writerows(overlays)
        messagebox.showinfo(
            "Export Successful", f"Overlay heatmap exported to:\n{filepath}"
        )
    except Exception as e:
        logging.error(f"❌ Failed to export CSV: {e}")
        messagebox.showerror("Export Failed", f"Could not export heatmap to CSV:\n{e}")


def export_heatmap_to_png(widget: tk.Widget, filepath: str) -> None:
    """
    Capture the given widget as an image and save it to PNG.
    """
    try:
        widget.update_idletasks()
        x = widget.winfo_rootx()
        y = widget.winfo_rooty()
        w = widget.winfo_width()
        h = widget.winfo_height()
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        img.save(filepath)
        messagebox.showinfo("Export Successful", f"Overlay image saved to:\n{filepath}")
    except Exception as e:
        logging.error(f"❌ Failed to export PNG: {e}")
        messagebox.showerror("Export Failed", f"Could not export image:\n{e}")


def refresh_overlay_for_file(
    file_path: str, overlay_data: List[Dict[str, Any]]
) -> None:
    """
    Called when user switches files or updates filters.
    Applies overlays to heatmap panel.
    """
    shared_state = get_shared_state()
    if not shared_state.heatmap_frame:
        logging.warning("⚠️ heatmap_frame is not initialised.")
        return

    if shared_state.preview_overlay_only.get():
        logging.info("👁 Preview overlay mode active")

    draw_token_heatmap(shared_state.heatmap_frame, overlay_data)
