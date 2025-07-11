import sys
import tkinter as tk
from tkinter import ttk
from gui.gui_components import launch_gui
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG, format="%(message)s")


def start_main_gui(splash: tk.Tk, progress: ttk.Progressbar) -> None:
    """
    Stop splash animation, close splash window, and launch the main GUI.

    Args:
        splash (tk.Tk): Splash screen window.
        progress (ttk.Progressbar): Progress bar widget to stop.
    """
    progress.stop()
    splash.destroy()
    launch_gui()


def show_splash_and_start() -> None:
    """
    Display a splash screen with a progress indicator before launching the main GUI.
    """
    splash = tk.Tk()
    splash.overrideredirect(True)

    # Centre splash on screen (platform-agnostic)
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
        logging.debug(f"⚠️ Failed to set style: {e}")

    progress = ttk.Progressbar(splash, mode='indeterminate', length=280)
    progress.pack(pady=10)
    progress.start(10)

    splash.after(1500, lambda: start_main_gui(splash, progress))
    splash.mainloop()


if __name__ == "__main__":
    # ✅ Prevent recursive GUI spawn when subprocess executes `-m metrics.main`
    safe_args = [arg.replace("\\", "/") for arg in sys.argv]
    if (
            "metrics.main" not in sys.argv and
            not any(arg.endswith("metrics/main.py") for arg in safe_args)
    ):
        show_splash_and_start()

