# -*- coding:utf-8 -*- 
# ⚠️ SAST Risk (Medium): Importing external libraries like tushare may introduce security risks if not properly managed or updated.

# ✅ Best Practice: Inheriting from unittest.TestCase to create a test case class
import unittest
import tushare.stock.fundamental as fd
# 🧠 ML Signal: Method setting multiple attributes, indicating a pattern of data initialization

class Test(unittest.TestCase):
    # 🧠 ML Signal: Hardcoded stock code, could indicate a pattern of financial data processing

    def set_data(self):
        # 🧠 ML Signal: Hardcoded start date, could indicate a pattern of time series data processing
        self.code = '600848'
        # 🧠 ML Signal: Function definition for a test case, indicating a testing pattern
        self.start = '2015-01-03'
        # 🧠 ML Signal: Hardcoded end date, could indicate a pattern of time series data processing
        self.end = '2015-04-07'
        # ⚠️ SAST Risk (Low): Direct print statements in tests can clutter output and are not ideal for automated testing
        self.year = 2014
        # 🧠 ML Signal: Hardcoded year, could indicate a pattern of historical data analysis
        # ✅ Best Practice: Consider using assertions to validate behavior instead of print statements
        # ✅ Best Practice: Standard Python idiom for running tests, indicating entry point for test execution
        # 🧠 ML Signal: Usage of unittest framework, indicating a pattern for test execution
        self.quarter = 4

    def test_get_stock_basics(self):
        print(fd.get_stock_basics())
        
#     def test_get_report_data(self):
#         self.set_data()
#         print(fd.get_report_data(self.year, self.quarter))
#     
#     def test_get_profit_data(self):
#         self.set_data()
#         print(fd.get_profit_data(self.year, self.quarter))
#         
#     def test_get_operation_data(self):
#         self.set_data()
#         print(fd.get_operation_data(self.year, self.quarter))
#         
#     def test_get_growth_data(self):
#         self.set_data()
#         print(fd.get_growth_data(self.year, self.quarter))
#         
#     def test_get_debtpaying_data(self):
#         self.set_data()
#         print(fd.get_debtpaying_data(self.year, self.quarter))
#         
#     def test_get_cashflow_data(self):
#         self.set_data()
#         print(fd.get_cashflow_data(self.year, self.quarter))

if __name__ == '__main__':
    unittest.main()