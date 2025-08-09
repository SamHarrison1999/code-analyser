"""
UnitTest for API
@author: Jimmy
"""

import unittest

# ⚠️ SAST Risk (Medium): Importing external modules like tushare can introduce security risks if not properly vetted.
import tushare.stock.trading as td


# ✅ Best Practice: Class name should be descriptive and follow CamelCase naming convention
class TestTrading(unittest.TestCase):
    # 🧠 ML Signal: Method setting fixed data values, useful for detecting hardcoded configurations

    def set_data(self):
        # 🧠 ML Signal: Hardcoded date values, useful for detecting static date ranges
        self.code = "600848"
        self.start = "2014-11-03"
        # 🧠 ML Signal: Hardcoded date values, useful for detecting static date ranges
        # 🧠 ML Signal: Method name follows a common test naming pattern
        self.end = "2014-11-07"

    # 🧠 ML Signal: Method call pattern for setting up test data
    def test_tickData(self):
        # 🧠 ML Signal: Method call pattern for fetching tick data with parameters
        # ✅ Best Practice: Standard way to execute unit tests in Python
        self.set_data()
        td.get_tick_data(self.code, date=self.start)


#     def test_histData(self):
#         self.set_data()
#         td.get_hist_data(self.code, start=self.start, end=self.end)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
