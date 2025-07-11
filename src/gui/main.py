import sys
import os
import logging
import tkinter as tk
from tkinter import ttk
from gui.gui_components import launch_gui

# === Setup debug log location ===
if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.getcwd()

log_path = os.path.join(base_dir, "debug.log")

# Ensure log directory exists
try:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
except Exception as e:
    print(f"⚠️ Could not ensure log directory exists: {e}")

# === Configure root logger ===
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.basicConfig(
    filename=log_path,
    filemode='w',
    encoding='utf-8',
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.debug("✅ Starting Code Analyser GUI")
logging.debug(f"🔍 Debug log path: {log_path}")
logging.debug(f"🧩 sys.argv: {sys.argv}")
logging.debug(f"📦 Frozen (PyInstaller): {getattr(sys, 'frozen', False)}")


def start_main_gui(splash: tk.Tk, progress: ttk.Progressbar) -> None:
    """Stop the splash animation and launch the main GUI."""
    try:
        progress.stop()
    except Exception:
        pass
    splash.destroy()
    launch_gui()


def show_splash_and_start() -> None:
    """Display a splash screen before launching the GUI."""
    splash = tk.Tk()
    splash.overrideredirect(True)

    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    splash_width, splash_height = 360, 120
    x = (screen_width // 2) - (splash_width // 2)
    y = (screen_height // 2) - (splash_height // 2)
    splash.geometry(f"{splash_width}x{splash_height}+{x}+{y}")
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
    except Exception as e:
        logging.debug(f"⚠️ Failed to set progressbar style: {e}")

    progress = ttk.Progressbar(splash, mode='indeterminate', length=280)
    progress.pack(pady=10)
    progress.start(10)

    splash.after(1500, lambda: start_main_gui(splash, progress))
    splash.mainloop()


if __name__ == "__main__":
    # Prevent subprocess metric-only mode from launching GUI
    safe_args = [arg.replace("\\", "/") for arg in sys.argv]
    if (
        "metrics.main" not in sys.argv and
        not any(arg.endswith("metrics/main.py") for arg in safe_args)
    ):
        show_splash_and_start()
