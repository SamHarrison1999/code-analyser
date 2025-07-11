# gui/main.py

import tkinter as tk
from tkinter import ttk
from gui.gui_components import launch_gui



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

    progress = ttk.Progressbar(splash, mode='indeterminate', length=280)
    progress.pack(pady=10)
    progress.start(10)

    splash.after(1500, lambda: start_main_gui(splash, progress))
    splash.mainloop()


if __name__ == "__main__":
    show_splash_and_start()
