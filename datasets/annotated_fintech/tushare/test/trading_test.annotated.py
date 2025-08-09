# -*- coding:utf-8 -*- 
'''
Created on 2015/3/14
@author: Jimmy Liu
# ✅ Best Practice: Group standard library imports at the top before third-party imports
'''
import unittest
# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns
# ✅ Best Practice: Inheriting from unittest.TestCase to create a test case class
import tushare.stock.trading as fd

# 🧠 ML Signal: Method sets multiple attributes, indicating a pattern of initializing or resetting state
class Test(unittest.TestCase):

    # 🧠 ML Signal: Hardcoded values for attributes, indicating a pattern of static configuration
    def set_data(self):
        self.code = '600848'
        # 🧠 ML Signal: Hardcoded values for attributes, indicating a pattern of static configuration
        self.start = '2015-01-03'
        self.end = '2015-04-07'
        # 🧠 ML Signal: Hardcoded values for attributes, indicating a pattern of static configuration
        # 🧠 ML Signal: Method name suggests this is a test function, indicating a pattern for test case identification
        self.year = 2014
        self.quarter = 4
    # 🧠 ML Signal: Hardcoded values for attributes, indicating a pattern of static configuration
    # ⚠️ SAST Risk (Low): Direct use of print statements in test functions can clutter test output
        
    # ✅ Best Practice: Consider using assertions instead of print statements for testing
    # 🧠 ML Signal: Method call to set up data, indicating a setup pattern for tests
    def test_get_hist_data(self):
        self.set_data()
        # 🧠 ML Signal: Usage of a method from an object (fd) to get historical data, indicating a pattern for data retrieval
        # ⚠️ SAST Risk (Low): Use of print statements in test code can clutter test output
        # 🧠 ML Signal: Method name suggests a test function, indicating usage in a testing context
        print(fd.get_hist_data(self.code, self.start))
    # 🧠 ML Signal: Use of print for debugging or output in test methods
        
    # ⚠️ SAST Risk (Low): Direct print statements can expose sensitive data in logs
    def test_get_tick_data(self):
        # ✅ Best Practice: Consider renaming the method to follow a consistent naming convention (e.g., test_get_realtime_quotes).
        # 🧠 ML Signal: Use of print for output, indicating a simple debugging or logging pattern
        self.set_data()
        print(fd.get_tick_data(self.code, self.end))
    # 🧠 ML Signal: Usage of print statements for debugging or output.
    
    # 🧠 ML Signal: Method name suggests this is a test function
    def test_get_today_all(self):
         # ⚠️ SAST Risk (Low): Printing sensitive data (e.g., stock codes) can lead to information leakage.
        print(fd.get_today_all()) 
    # ⚠️ SAST Risk (Low): Printing data directly can expose sensitive information
        
    # 🧠 ML Signal: Method name follows a common pattern for test functions
    def test_get_realtime_quotesa(self):
        self.set_data()
         # ⚠️ SAST Risk (Low): Potential for side effects if set_data modifies shared state
        print(fd.get_realtime_quotes(self.code)) 
    # ⚠️ SAST Risk (Low): Printing in tests can clutter output and is not a substitute for assertions
    # ✅ Best Practice: Use assertions to validate behavior instead of print statements
    # 🧠 ML Signal: Common pattern for executing code when the script is run directly
    # ✅ Best Practice: Ensures that the test suite runs when the script is executed directly
        
    def test_get_h_data(self):
        self.set_data()
        print(fd.get_h_data(self.code, self.start, self.end))
        
    def test_get_today_ticks(self):
        self.set_data()
        print(fd.get_today_ticks(self.code))    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()