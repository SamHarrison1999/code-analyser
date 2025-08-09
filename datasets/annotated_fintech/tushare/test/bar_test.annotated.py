# -*- coding:utf-8 -*-
"""
Created on 2017/9/24
@author: Jimmy Liu
"""
# 🧠 ML Signal: Importing a specific module from a package indicates usage patterns
import unittest

# ✅ Best Practice: Inheriting from unittest.TestCase to create a test case class
# ⚠️ SAST Risk (Low): Importing external libraries can introduce vulnerabilities if not properly managed
import tushare.stock.trading as fd


# 🧠 ML Signal: Method for setting data attributes, useful for learning object state changes
class Test(unittest.TestCase):

    # 🧠 ML Signal: Hardcoded values can indicate default settings or constants
    def set_data(self):
        self.code = "600848"
        # 🧠 ML Signal: Empty string initialization can indicate optional or unset parameters
        # ✅ Best Practice: Consider adding a docstring to describe the purpose of the test.
        self.start = ""
        self.end = ""

    # 🧠 ML Signal: Empty string initialization can indicate optional or unset parameters
    # ⚠️ SAST Risk (Low): Using print statements in tests can clutter output; consider using assertions.

    # 🧠 ML Signal: Entry point for running tests, useful for identifying test execution patterns.
    # ✅ Best Practice: Ensure unittest.main() is called to execute tests when the script is run directly.
    def test_bar_data(self):
        self.set_data()
        print(fd.bar(self.code, self.start, self.end))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
