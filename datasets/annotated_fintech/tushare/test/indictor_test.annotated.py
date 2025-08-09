# -*- coding:utf-8 -*- 
'''
Created on 2018/05/26
@author: Jackie Liao
'''
# 🧠 ML Signal: Importing specific modules from a library can indicate usage patterns
import unittest
import tushare.stock.indictor as idx
# ✅ Best Practice: Inheriting from unittest.TestCase to create a test case class
# 🧠 ML Signal: Importing a library can indicate usage patterns
import tushare as ts

# 🧠 ML Signal: Usage of external library function `ts.get_k_data` to fetch data

class Test(unittest.TestCase):
    # 🧠 ML Signal: Sorting data by date, a common preprocessing step

    def test_plot_all(self):
        # 🧠 ML Signal: Plotting data with `idx.plot_all`, indicating visualization usage
        # ✅ Best Practice: Ensures the script can be run as a standalone module
        # 🧠 ML Signal: Use of `unittest.main()` for running tests, indicating test execution
        data = ts.get_k_data("601398", start="2018-01-01", end="2018-05-27")

        data = data.sort_values(by=["date"], ascending=True)

        idx.plot_all(data, is_show=True, output=None)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()