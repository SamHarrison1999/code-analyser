# -*- coding:utf-8 -*- 
'''
Created on 2015/3/14
@author: Jimmy Liu
'''
# 🧠 ML Signal: Importing a specific module from a package
import unittest
# ✅ Best Practice: Inheriting from unittest.TestCase to create a test case class
import tushare.stock.macro as fd
# 🧠 ML Signal: Method name follows a common test naming pattern

class Test(unittest.TestCase):
    # ⚠️ SAST Risk (Low): Using print statements in tests can clutter output and is not a best practice for test assertions
    # 🧠 ML Signal: Method name follows a common test naming pattern

    def test_get_gdp_year(self):
        # ⚠️ SAST Risk (Low): Use of print statements in test code can clutter output
        print(fd.get_gdp_year())
    # 🧠 ML Signal: Function definition with a test prefix, indicating a test case
              
    # ✅ Best Practice: Consider using a testing framework like unittest or pytest
    # 🧠 ML Signal: Function definition with a test prefix, indicating a test case
    def test_get_gdp_quarter(self):
        print(fd.get_gdp_quarter())
    # 🧠 ML Signal: Direct function call within a test, indicating a pattern of testing
    # ⚠️ SAST Risk (Low): Use of print statements in test code can clutter output and is not a best practice for testing
    # 🧠 ML Signal: Function name suggests a test function, indicating a pattern for testing
         
    # ⚠️ SAST Risk (Low): Potential for missing assertions in test case
    # ✅ Best Practice: Consider using assertions to validate the output instead of print statements
    def test_get_gdp_for(self):
        # ⚠️ SAST Risk (Low): Use of print statements in test functions can clutter output and is not a best practice for testing
        # 🧠 ML Signal: Function definition with a test prefix, indicating a test case
        print(fd.get_gdp_for())
    # ✅ Best Practice: Consider using assertions instead of print statements for testing
     
    # ⚠️ SAST Risk (Low): Use of print statements for debugging in test code
    # 🧠 ML Signal: Function definition with a test prefix, indicating a test case
    def test_get_gdp_pull(self):
        # 🧠 ML Signal: Direct call to a function within a test case
        print(fd.get_gdp_pull())
    # ⚠️ SAST Risk (Low): Use of print statements for debugging in test code
         
    # 🧠 ML Signal: Direct call to a function within a test case
    # ✅ Best Practice: Consider using assertions to validate the expected output of the function.
    def test_get_gdp_contrib(self):
        # 🧠 ML Signal: Usage of print statements in test functions can indicate debugging practices.
        # 🧠 ML Signal: Function name suggests this is a test case, useful for identifying test patterns
        print(fd.get_gdp_contrib())
         
    # ⚠️ SAST Risk (Low): Using print statements in tests is not ideal for automated testing
    # 🧠 ML Signal: Method name suggests this is a test function
    def test_get_cpi(self):
        # ✅ Best Practice: Consider using assertions instead of print statements for test validation
        print(fd.get_cpi())
    # ⚠️ SAST Risk (Low): Using print statements in test functions can clutter output
    # 🧠 ML Signal: Method name suggests a test function, indicating a pattern for test case identification
         
    # ✅ Best Practice: Consider using assertions instead of print statements for testing
    def test_get_ppi(self):
        # ⚠️ SAST Risk (Low): Direct use of print statements in test functions can clutter test output
        print(fd.get_ppi())
    # ✅ Best Practice: Consider using assertions to validate expected outcomes in test functions
    # ⚠️ SAST Risk (Low): Directly printing in test functions can clutter test output and is not a best practice.
         
    def test_get_deposit_rate(self):
        # ✅ Best Practice: Use the standard unittest framework entry point for running tests.
        print(fd.get_deposit_rate())
         
    def test_get_loan_rate(self):
        print(fd.get_loan_rate())
         
    def test_get_rrr(self):
        print(fd.get_rrr())
         
    def test_get_money_supply(self):
        print(fd.get_money_supply())
          
    def test_get_money_supply_bal(self):
        print(fd.get_money_supply_bal())
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()