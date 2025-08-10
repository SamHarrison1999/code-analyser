# --- file: src/gui/model_service_gui.py ---
# Annotation: Standard library imports for GUI, files, JSON and OS/env handling.
import json
import os
import sys
import time

# Annotation: Tkinter is used for the desktop GUI (shipped with Python).
import tkinter as tk

# Annotation: ttk provides themed widgets like Treeview.
from tkinter import ttk

# Annotation: filedialog and messagebox support opening/saving files and simple alerts.
from tkinter import filedialog, messagebox

# Annotation: Import the HTTP client that talks to your FastAPI model service.
from src.ml.model_client import AnalyserModelClient

# Annotation: Import the multi-tool rule engine (AST + flake8/pyflakes/pydocstyle/pylint/bandit/SonarLint).
from src.ml.rule_engine import collect_reasons

# Annotation: Heuristic keywords for mapping rule messages to the appropriate label.
_SAST_HINTS = (
    "unsafe",
    "verify=false",
    "exec",
    "eval",
    "unpickl",
    "yaml.load",
    "shell=true",
    "credential",
    "injection",
    "deserial",
    "sql",
    "cwe",
    "bandit",
)
# Annotation: Style/readability hints; bias these to Best Practice.
_BEST_PRACTICE_HINTS = (
    "pep8",
    "docstring",
    "unused",
    "import",
    "naming",
    "format",
    "style",
    "convention",
    "readability",
)
# Annotation: Canonical → pretty label names used in the UI.
_PRETTY = {
    "sast_risk": "⚠️ SAST Risk",
    "ml_signal": "🧠 ML Signal",
    "best_practice": "✅ Best Practice",
}


# Annotation: Decide which canonical label fits a rule message string.
def _label_for_reason(reason: str) -> str:
    low = reason.lower()
    if any(k in low for k in _SAST_HINTS):
        return "sast_risk"
    if any(k in low for k in _BEST_PRACTICE_HINTS):
        return "best_practice"
    return "ml_signal"


# Annotation: Small helper to pretty-print dicts in a stable way inside the GUI.
def _pretty_json(obj) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


# Annotation: Main desktop application window for the Code Analyser model client.
class ModelServiceGUI(tk.Tk):
    # Annotation: Construct the window, widgets and default client.
    def __init__(self):
        super().__init__()
        # Annotation: Window title reflects the tool purpose.
        self.title("Code Analyser — Model Service Client")
        # Annotation: Default geometry gives the panes enough space.
        self.geometry("1100x700")
        # Annotation: Track the service URL and the threshold slider value.
        self.service_var = tk.StringVar(
            value=os.environ.get("MODEL_SERVICE_URL", "http://127.0.0.1:8111")
        )
        self.threshold_var = tk.DoubleVar(value=0.50)
        # Annotation: Status bar variable for brief messages.
        self.status_var = tk.StringVar(value="")
        # Annotation: Client is created lazily once we first use it.
        self.client: AnalyserModelClient | None = None
        # Annotation: Cache of last-used service URL to avoid re-creating the client each action.
        self._last_service_url = None
        # Annotation: Layout the full UI.
        self._build_menu()
        self._build_toolbar()
        self._build_panes()
        self._build_status()
        # Annotation: Initial client creation.
        self._ensure_client()

    # Annotation: File/Help menus including About dialog.
    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open…", command=self.on_open)
        file_menu.add_command(label="Save", command=self.on_save)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=self.on_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)

    # Annotation: The toolbar hosts the service URL entry, threshold slider and action buttons.
    def _build_toolbar(self):
        bar = ttk.Frame(self)
        ttk.Label(bar, text="Service URL:").pack(side="left", padx=(6, 4))
        self.service_entry = ttk.Entry(bar, textvariable=self.service_var, width=40)
        self.service_entry.pack(side="left", padx=(0, 10))
        ttk.Label(bar, text="Threshold:").pack(side="left")
        self.threshold = ttk.Scale(
            bar, from_=0.0, to=1.0, orient="horizontal", variable=self.threshold_var
        )
        self.threshold.pack(side="left", fill="x", expand=True, padx=(6, 10))
        ttk.Button(bar, text="Predict", command=self.on_predict).pack(side="left", padx=3)
        ttk.Button(bar, text="Annotate", command=self.on_annotate).pack(side="left", padx=3)
        ttk.Button(bar, text="Clear", command=self.on_clear).pack(side="left", padx=3)
        ttk.Button(bar, text="Open…", command=self.on_open).pack(side="left", padx=3)
        ttk.Button(bar, text="Save", command=self.on_save).pack(side="left", padx=3)
        ttk.Button(bar, text="Health", command=self.on_health).pack(side="left", padx=3)
        bar.pack(side="top", fill="x", pady=4)

    # Annotation: The main split panes: left = input, right = predictions + raw + annotated preview.
    def _build_panes(self):
        main = ttk.Panedwindow(self, orient="horizontal")
        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=1)
        main.pack(fill="both", expand=True)
        ttk.Label(left, text="Input code / snippet").pack(anchor="w", padx=6)
        self.input_text = tk.Text(left, wrap="none", undo=True, height=10)
        self.input_text.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        right_top = ttk.Frame(right)
        right_mid = ttk.Frame(right)
        right_bot = ttk.Frame(right)
        right_top.pack(fill="both", expand=False, padx=6)
        right_mid.pack(fill="both", expand=True, padx=6, pady=(6, 0))
        right_bot.pack(fill="both", expand=True, padx=6, pady=(6, 6))
        ttk.Label(right_top, text="Prediction results").grid(row=0, column=0, sticky="w")
        ttk.Label(right_top, text="Activated labels:").grid(row=1, column=0, sticky="w")
        self.labels_tree = ttk.Treeview(
            right_top, columns=("label", "score"), show="headings", height=6
        )
        self.labels_tree.heading("label", text="Label")
        self.labels_tree.heading("score", text="Score")
        self.labels_tree.column("label", width=180, anchor="w")
        self.labels_tree.column("score", width=80, anchor="e")
        self.labels_tree.grid(row=2, column=0, sticky="nsew")
        right_top.grid_columnconfigure(0, weight=1)
        ttk.Label(right_mid, text="Raw response:").pack(anchor="w")
        self.raw_response = tk.Text(right_mid, wrap="none", height=10)
        self.raw_response.pack(fill="both", expand=True)
        ttk.Label(right_bot, text="Annotated preview").pack(anchor="w")
        self.annotated_text = tk.Text(right_bot, wrap="none")
        self.annotated_text.pack(fill="both", expand=True)

    # Annotation: Status bar at the bottom to show quick state info.
    def _build_status(self):
        bar = ttk.Frame(self)
        self.status = ttk.Label(bar, textvariable=self.status_var, anchor="w")
        self.status.pack(fill="x")
        bar.pack(side="bottom", fill="x")

    # Annotation: Ensure we have a client bound to the current URL.
    def _ensure_client(self):
        url = self.service_var.get().strip()
        if self._last_service_url != url:
            self.client = AnalyserModelClient(base_url=url, timeout=30.0)
            self._last_service_url = url

    # Annotation: Clear outputs.
    def on_clear(self):
        self.raw_response.delete("1.0", "end")
        self.annotated_text.delete("1.0", "end")
        for row in self.labels_tree.get_children():
            self.labels_tree.delete(row)
        self.status_var.set("Cleared")

    # Annotation: Open a .py or text file and load it into the input pane.
    def on_open(self):
        path = filedialog.askopenfilename(
            filetypes=[("Python", "*.py"), ("Text", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read()
        except Exception as e:
            messagebox.showerror("Open failed", f"Could not read file:\n{e}")
            return
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", data)
        self.status_var.set(f"Loaded {os.path.basename(path)}")

    # Annotation: Save the annotated preview to a file.
    def on_save(self):
        data = self.annotated_text.get("1.0", "end-1c")
        if not data.strip():
            messagebox.showinfo("Nothing to save", "There is no annotated content to save yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".py", filetypes=[("Python", "*.py"), ("Text", "*.txt")]
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
        except Exception as e:
            messagebox.showerror("Save failed", f"Could not write file:\n{e}")
            return
        self.status_var.set(f"Saved {os.path.basename(path)}")

    # Annotation: Simple About box.
    def on_about(self):
        messagebox.showinfo(
            "About",
            "Code Analyser — Model Service Client\nMulti-tool rule integration + model scores.",
        )

    # Annotation: Call the /healthz endpoint and show a toast.
    def on_health(self):
        try:
            self._ensure_client()
            info = self.client.health()
        except Exception as e:
            messagebox.showerror("Service health", f"Failed:\n{e}")
            return
        messagebox.showinfo("Service health", _pretty_json(info))

    # Annotation: Run a prediction over the whole text to show overall label scores and raw JSON.
    def on_predict(self):
        self._ensure_client()
        text = self.input_text.get("1.0", "end-1c")
        thr = float(self.threshold_var.get())
        if not text.strip():
            return
        t0 = time.time()
        try:
            res = self.client.predict([text], threshold=thr)
        except Exception as e:
            messagebox.showerror("Predict failed", f"{e}")
            return
        dt = int((time.time() - t0) * 1000)
        self.raw_response.delete("1.0", "end")
        self.raw_response.insert("1.0", _pretty_json(res))
        for row in self.labels_tree.get_children():
            self.labels_tree.delete(row)
        if res.get("results"):
            item = res["results"][0]
            scores = item.get("scores", {})
            for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
                self.labels_tree.insert("", "end", values=(k, f"{v:.4f}"))
        self.status_var.set(f"Predicted in {dt} ms via {self._last_service_url}")

    # Annotation: Build the full annotated preview; emits ALL rule reasons per relevant line and indents comments to match.
    def on_annotate(self):
        self._ensure_client()
        raw = self.input_text.get("1.0", "end-1c")
        thr = float(self.threshold_var.get())
        if not raw.strip():
            return
        annotated = self._annotate_text_with_rules_and_model(raw, thr)
        self.annotated_text.delete("1.0", "end")
        self.annotated_text.insert("1.0", annotated)
        self.status_var.set(f"Annotated {len(raw.splitlines())} lines via {self._last_service_url}")

    # Annotation: Core routine that blends rule-engine reasons with per-line model scores; inserts comments above lines with matching indentation.
    def _annotate_text_with_rules_and_model(self, text: str, threshold: float) -> str:
        # Annotation: Keep original line endings by capturing them from each line.
        lines = text.splitlines(keepends=True)
        # Annotation: Gather rule-engine reasons once (dict: 1-based line → list of messages).
        reasons_by_line = collect_reasons(text)
        # Annotation: Prepare batch for model scores (only real code lines).
        idx_and_texts = [
            (i, ln) for i, ln in enumerate(lines) if ln.strip() and not ln.lstrip().startswith("#")
        ]
        batch_texts = [t for _, t in idx_and_texts] or [""]

        # Annotation: Call the model service once; if it fails we still show rule comments.
        try:
            res = (
                self.client.predict(batch_texts, threshold=threshold)
                if idx_and_texts
                else {"results": []}
            )
        except Exception:
            res = {"results": [{"scores": {}} for _ in idx_and_texts]}

        results_iter = iter(res["results"]) if idx_and_texts else iter([])
        out_lines = []

        # Annotation: Walk original text, injecting comments just above each relevant line.
        for i, original in enumerate(lines):
            # Annotation: Pass-through for blank/comment lines.
            if not (original.strip() and not original.lstrip().startswith("#")):
                out_lines.append(original)
                continue

            # Annotation: Pull model scores for this line (may be empty).
            item = next(results_iter, {"scores": {}})
            scores = item.get("scores", {})

            # Annotation: Determine indentation (spaces/tabs) and newline style from the target line.
            indent = original[: len(original) - len(original.lstrip())]
            newline = "\r\n" if original.endswith("\r\n") else "\n"

            # Annotation: Fetch and prioritise reasons (SAST first), but DO NOT gate by threshold — tools come first.
            line_no = i + 1
            reasons = reasons_by_line.get(line_no, [])
            reasons_sorted = sorted(
                reasons, key=lambda r: 0 if any(k in r.lower() for k in _SAST_HINTS) else 1
            )

            # Annotation: Emit a comment for each reason with matched indentation; confidence comes from line scores if available.
            if reasons_sorted:
                for reason in reasons_sorted:
                    label_key = _label_for_reason(reason)
                    pretty = _PRETTY[label_key]
                    conf = float(scores.get(label_key, max(scores.values(), default=0.0)))
                    out_lines.append(
                        f"{indent}# {pretty}: {reason} (confidence: {conf:.2f}){newline}"
                    )
            else:
                # Annotation: No tool reasons — optional model-only fallback (respect threshold to avoid noise).
                if scores:
                    best_label = max(scores, key=lambda k: scores[k])
                    conf = float(scores[best_label])
                    if conf >= threshold:
                        pretty = _PRETTY[best_label]

            # Annotation: Finally, append the original source line unchanged.
            out_lines.append(original)

        # Annotation: Return the reconstructed text with inline comments.
        return "".join(out_lines)


# Annotation: Allow module execution via `python -m src.gui.model_service_gui`.
if __name__ == "__main__":
    try:
        import torch

        print(f"✅ sys.path: {sys.path}")
        print(f"✅ torch loaded from: {getattr(torch,'__file__','<unknown>')}")
        print(f"✅ hasattr(torch, '__version__'): {hasattr(torch,'__version__')}")
        print(f"✅ torch.__version__: {getattr(torch,'__version__','<unknown>')}")
    except Exception:
        pass
    app = ModelServiceGUI()
    app.mainloop()
