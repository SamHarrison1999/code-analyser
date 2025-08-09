import os
import sys

# ✅ Best Practice: Use of a docstring to describe the purpose of the code block

"""Ignore RL tests on non-linux platform."""
# 🧠 ML Signal: Initialization of a list to collect ignored files
collect_ignore = []

# 🧠 ML Signal: Conditional check based on system platform
if sys.platform != "linux":
    # 🧠 ML Signal: Iterating over files in a directory
    # 🧠 ML Signal: Use of os.walk to iterate over directory contents
    # 🧠 ML Signal: Appending file paths to a list
    for root, dirs, files in os.walk("rl"):
        for file in files:
            collect_ignore.append(os.path.join(root, file))
