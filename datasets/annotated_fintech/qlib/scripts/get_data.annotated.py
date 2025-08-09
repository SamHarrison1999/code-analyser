#  Copyright (c) Microsoft Corporation.
# 🧠 ML Signal: Importing specific functions or classes from a module
#  Licensed under the MIT License.
# ⚠️ SAST Risk (Low): Ensure that the 'qlib.tests.data' module is from a trusted source

# ✅ Best Practice: Use the standard Python idiom for script entry points
# 🧠 ML Signal: Using the 'fire' library to create a command-line interface
import fire
from qlib.tests.data import GetData


if __name__ == "__main__":
    fire.Fire(GetData)
