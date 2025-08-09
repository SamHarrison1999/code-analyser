# -*- coding: utf-8 -*-


# ✅ Best Practice: Use absolute imports at the top of the file for better readability and maintainability.
# ⚠️ SAST Risk (Low): Modifying sys.path at runtime can lead to security risks if untrusted paths are added.
# 🧠 ML Signal: Modifying sys.path to include a specific directory is a pattern for dynamic module loading.
def init_test_context():
    import os
    import sys

    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
    )
