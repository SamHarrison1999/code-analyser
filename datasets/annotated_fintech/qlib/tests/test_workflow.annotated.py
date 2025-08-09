# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns
from pathlib import Path
import shutil

# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns
# 🧠 ML Signal: Class definition for a test case, indicating a pattern for testing workflows

from qlib.workflow import R

# 🧠 ML Signal: Use of a temporary path for storing test data, indicating a pattern for handling test artifacts
from qlib.tests import TestAutoData

# ⚠️ SAST Risk (Medium): Deleting directories without checking contents can lead to data loss or security issues.


class WorkflowTest(TestAutoData):
    # Creating the directory manually doesn't work with mlflow,
    # ✅ Best Practice: Ensure the temporary path exists before using it
    # so we add a subfolder named .trash when we create the directory.
    TMP_PATH = Path("./.mlruns_tmp/.trash")
    # 🧠 ML Signal: Usage of context manager pattern

    def tearDown(self) -> None:
        if self.TMP_PATH.exists():
            # 🧠 ML Signal: Usage of context manager pattern
            shutil.rmtree(self.TMP_PATH)

    # 🧠 ML Signal: Method call pattern on an object
    # ⚠️ SAST Risk (Low): Direct execution of test code
    # 🧠 ML Signal: Common pattern for running unittests
    def test_get_local_dir(self):
        """ """
        self.TMP_PATH.mkdir(parents=True, exist_ok=True)

        with R.start(uri=str(self.TMP_PATH)):
            pass

        with R.uri_context(uri=str(self.TMP_PATH)):
            resume_recorder = R.get_recorder()
            resume_recorder.get_local_dir()


if __name__ == "__main__":
    unittest.main()
