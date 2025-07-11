import sys
import tkinter as tk
from tkinter import ttk
from gui.gui_components import launch_gui
import logging
logging.basicConfig(level=logging.DEBUG, format="%(message)s")


def start_main_gui(splash: tk.Tk, progress: ttk.Progressbar):
    """Stop splash animation, destroy splash window, and launch main GUI."""
    progress.stop()
    splash.destroy()
    launch_gui()


def show_splash_and_start():
    """Display a simple splash screen while GUI is loading."""
    splash = tk.Tk()
    splash.overrideredirect(True)
    splash.geometry("360x120+600+300")
    splash.configure(bg="white")

    tk.Label(
        splash,
        text="Launching Code Analyser GUI...",
        font=("Segoe UI", 12, "bold"),
        bg="white"
    ).pack(pady=10)

    try:
        style = ttk.Style()
        style.theme_use('default')
        style.configure("TProgressbar", thickness=8)
    except Exception:
        pass  # fallback silently if style config fails

    progress = ttk.Progressbar(splash, mode='indeterminate', length=280)
    progress.pack(pady=10)
    progress.start(10)

    splash.after(1500, lambda: start_main_gui(splash, progress))
    splash.mainloop()


if __name__ == "__main__":
    # ✅ Prevent recursive GUI spawn when subprocess calls `-m metrics.main`
    safe_args = [arg.replace("\\", "/") for arg in sys.argv]
    if (
        "metrics.main" not in sys.argv
        and not any(arg.endswith("metrics/main.py") for arg in safe_args)
    ):
        show_splash_and_start()
