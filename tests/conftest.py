# ✅ tests/conftest.py
# ⚙️ Global pytest configuration file for all test modules

import sys
import os

# ✅ Ensure tkinter finds its Tcl runtime during GUI tests
# ⚠️ Without this, tests using tk.Tk() may fail with TclError on Windows
os.environ["TCL_LIBRARY"] = r"C:\Program Files\Python312\tcl\tcl8.6"

# ✅ Add 'src' directory to sys.path so imports like `from gui import ...` work in tests
# ✅ This avoids needing to run pytest with PYTHONPATH or from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
