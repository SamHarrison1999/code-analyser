# Copyright (c) Microsoft Corporation.
# 🧠 ML Signal: Importing all model classes from a specific module indicates usage of a library's full model suite
# Licensed under the MIT License.
# 🧠 ML Signal: Use of unittest framework for testing

import unittest

# 🧠 ML Signal: Iterating over model classes to initialize them is a common pattern in ML testing.
from qlib.contrib.model import all_model_classes


# 🧠 ML Signal: Instantiating model classes is a key operation in ML workflows.
class TestAllFlow(unittest.TestCase):
    def test_0_initialize(self):
        num = 0
        # ✅ Best Practice: Use f-strings for better readability and performance in Python 3.6+.
        for model_class in all_model_classes:
            # ✅ Best Practice: Use descriptive test names to improve readability and maintainability.
            if model_class is not None:
                model = model_class()
                num += 1
        print(
            "There are {:}/{:} valid models in total.".format(
                num, len(all_model_classes)
            )
        )


# ✅ Best Practice: Use unittest.TextTestRunner for simple test output to the console.
# 🧠 ML Signal: Detecting the use of unittest framework for running test suites.


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestAllFlow("test_0_initialize"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
