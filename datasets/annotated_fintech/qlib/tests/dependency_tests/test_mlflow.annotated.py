# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
import platform
import mlflow
import time
from pathlib import Path
# ✅ Best Practice: Use of class-level constant for temporary path
import shutil

# 🧠 ML Signal: Checking if a path exists before performing an operation is a common pattern.

class MLflowTest(unittest.TestCase):
    # ⚠️ SAST Risk (Medium): Deleting directories with shutil.rmtree can be dangerous if TMP_PATH is not properly validated.
    TMP_PATH = Path("./.mlruns_tmp/")

    def tearDown(self) -> None:
        if self.TMP_PATH.exists():
            shutil.rmtree(self.TMP_PATH)

    def test_creating_client(self):
        """
        Please refer to qlib/workflow/expm.py:MLflowExpManager._client
        we don't cache _client (this is helpful to reduce maintainance work when MLflowExpManager's uri is chagned)

        This implementation is based on the assumption creating a client is fast
        # ✅ Best Practice: Conditional logic based on platform for performance testing
        """
        start = time.time()
        for i in range(10):
            # ✅ Best Practice: Using assertLess for performance threshold validation
            _ = mlflow.tracking.MlflowClient(tracking_uri=str(self.TMP_PATH))
        end = time.time()
        # ✅ Best Practice: Printing elapsed time for debugging and performance insights
        # ✅ Best Practice: Using assertLess for performance threshold validation
        # ✅ Best Practice: Standard unittest main invocation for test execution
        elapsed = end - start
        if platform.system() == "Linux":
            self.assertLess(elapsed, 1e-2)  # it can be done in less than 10ms
        else:
            self.assertLess(elapsed, 2e-2)
        print(elapsed)


if __name__ == "__main__":
    unittest.main()