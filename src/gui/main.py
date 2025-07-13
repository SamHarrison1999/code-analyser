import sys
import os
import logging
import tkinter as tk
from tkinter import ttk
from pathlib import Path

from gui.shared_state import setup_shared_gui_state  # ✅ Inject shared state before GUI
from gui.gui_components import launch_gui  # ✅ Accepts root window

# === Configure logging ===
if getattr(sys, 'frozen', False):
    base_dir = Path(sys.executable).parent
else:
    base_dir = Path.cwd()

log_path = base_dir / "debug.log"

try:
    log_path.parent.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"⚠️ Could not create log directory: {e}")

# Suppress matplotlib font loading noise
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# Setup file-based logging
logging.basicConfig(
    filename=str(log_path),
    filemode='w',
    encoding='utf-8',
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.debug("✅ Starting Code Analyser GUI")
logging.debug(f"🔍 Debug log path: {log_path}")
logging.debug(f"🧩 sys.argv: {sys.argv}")
logging.debug(f"📦 Frozen (PyInstaller): {getattr(sys, 'frozen', False)}")

# === App exit logic ===
def clean_exit(root: tk.Tk) -> None:
    """Safely exit the application."""
    try:
        root.quit()
        root.destroy()
    except Exception:
        pass
    logging.info("📤 Exiting application cleanly via clean_exit()")
    sys.exit(0)


def start_main_gui(splash: tk.Tk, progress: ttk.Progressbar) -> None:
    """Stop splash screen and launch the main GUI."""
    try:
        progress.stop()
    except Exception:
        logging.debug("⚠️ Could not stop progress bar")

    try:
        splash.destroy()
    except Exception:
        logging.debug("⚠️ Could not destroy splash window")

    try:
        root = tk.Tk()
        setup_shared_gui_state(root)  # ✅ Initialise shared state
        root.protocol("WM_DELETE_WINDOW", lambda: clean_exit(root))  # 🛑 Handle manual window close
        launch_gui(root)
        root.mainloop()
        clean_exit(root)  # 🧹 Ensure full cleanup on normal close
    except Exception as e:
        logging.exception(f"❌ Failed to launch GUI: {e}")
        print("❌ Failed to launch GUI. See debug.log for details.")
        sys.exit(1)


def show_splash_and_start() -> None:
    """Display splash screen and start the GUI with animation."""
    splash = tk.Tk()
    splash.overrideredirect(True)

    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    splash_width, splash_height = 360, 120
    x = (screen_width - splash_width) // 2
    y = (screen_height - splash_height) // 2
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
        logging.debug(f"⚠️ Failed to style progressbar: {type(e).__name__}: {e}")

    progress = ttk.Progressbar(splash, mode='indeterminate', length=280)
    progress.pack(pady=10)
    progress.start(10)

    splash.after(1500, lambda: start_main_gui(splash, progress))
    splash.mainloop()


if __name__ == "__main__":
    args = [str(arg).replace("\\", "/") for arg in sys.argv]
    if (
        "metrics.main" not in sys.argv and
        not any(arg.endswith("metrics/main.py") for arg in args)
    ):
        show_splash_and_start()
