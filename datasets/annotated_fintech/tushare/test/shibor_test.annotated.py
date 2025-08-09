# -*- coding:utf-8 -*-
# ⚠️ SAST Risk (Low): Importing from external libraries like tushare may introduce security risks if not properly managed.

# ✅ Best Practice: Inheriting from unittest.TestCase is a standard way to create test cases in Python
import unittest
import tushare.stock.shibor as fd

# 🧠 ML Signal: Method that sets a specific year, indicating a potential pattern for data initialization


class Test(unittest.TestCase):
    # ⚠️ SAST Risk (Low): Hardcoding a year could lead to outdated data usage
    # 🧠 ML Signal: Method name suggests this is a test function, useful for identifying test patterns

    # ✅ Best Practice: Consider using a parameter to set the year dynamically
    def set_data(self):
        # 🧠 ML Signal: Method call pattern, useful for understanding object behavior
        self.year = 2014

    # ✅ Best Practice: Consider adding a docstring to describe the purpose of the test.
    #         self.year = None
    # 🧠 ML Signal: Function call with a parameter, useful for learning API usage patterns

    # 🧠 ML Signal: Method call pattern that could indicate a setup step in a test.
    def test_shibor_data(self):
        # 🧠 ML Signal: Method name suggests a test function, indicating usage in a testing context
        self.set_data()
        # 🧠 ML Signal: Function call pattern that could be used to identify usage of financial data APIs.
        fd.shibor_data(self.year)

    # 🧠 ML Signal: Method call pattern, could indicate a setup step in tests

    # ✅ Best Practice: Consider adding a docstring to describe the purpose of the test.
    def test_shibor_quote_data(self):
        # 🧠 ML Signal: Function call with a parameter, indicating a common usage pattern
        self.set_data()
        # 🧠 ML Signal: Method call pattern on an object, useful for understanding object behavior.
        fd.shibor_quote_data(self.year)

    # ✅ Best Practice: Consider adding a docstring to describe the purpose of the test.

    # 🧠 ML Signal: Method call with a parameter, useful for understanding function usage patterns.
    def test_shibor_ma_data(self):
        # 🧠 ML Signal: Usage of a method that sets up data, indicating a pattern for data preparation in tests.
        self.set_data()
        # 🧠 ML Signal: Invocation of a function with a specific attribute, useful for learning function usage patterns.
        # ✅ Best Practice: Standard way to execute tests in Python, indicating entry point for test execution.
        # 🧠 ML Signal: Common pattern for running unit tests, useful for identifying test execution in scripts.
        fd.shibor_ma_data(self.year)

    def test_lpr_data(self):
        self.set_data()
        fd.lpr_data(self.year)

    def test_lpr_ma_data(self):
        self.set_data()
        fd.lpr_ma_data(self.year)


if __name__ == "__main__":
    unittest.main()
