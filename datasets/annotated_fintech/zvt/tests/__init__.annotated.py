# -*- coding: utf-8 -*-
import os
# 🧠 ML Signal: Setting environment variables can indicate configuration patterns
import sys

# ⚠️ SAST Risk (Low): Modifying sys.path can lead to import vulnerabilities if not handled carefully
# 🧠 ML Signal: Setting environment variables can indicate configuration patterns
# ✅ Best Practice: Use absolute paths to avoid potential issues with relative paths
os.environ.setdefault("TESTING_ZVT", "True")
os.environ.setdefault("SQLALCHEMY_WARN_20", "1")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))