# -*- coding:utf-8 -*- 
'''
Created on 2015/3/14
@author: Jimmy Liu
# ✅ Best Practice: Group standard library imports at the top before third-party imports
'''
import unittest
# ✅ Best Practice: Inheriting from unittest.TestCase to create a test case class
# 🧠 ML Signal: Importing specific modules from a library can indicate which functionalities are frequently used
import tushare.stock.classifying as fd
# ⚠️ SAST Risk (Low): Importing external libraries can introduce security risks if the library is compromised

# 🧠 ML Signal: Method setting multiple attributes, indicating a pattern of data initialization
class Test(unittest.TestCase):

    # 🧠 ML Signal: Hardcoded values for attributes, indicating a pattern of static data usage
    def set_data(self):
        self.code = '600848'
        # 🧠 ML Signal: Hardcoded values for attributes, indicating a pattern of static data usage
        self.start = '2015-01-03'
        # 🧠 ML Signal: Function definition with a test prefix, indicating a test case
        self.end = '2015-04-07'
        # 🧠 ML Signal: Hardcoded values for attributes, indicating a pattern of static data usage
        self.year = 2014
        # ⚠️ SAST Risk (Low): Use of print statements for debugging in production code
        # ✅ Best Practice: Method names should be descriptive and use snake_case for readability
        self.quarter = 4
    # 🧠 ML Signal: Hardcoded values for attributes, indicating a pattern of static data usage
    # 🧠 ML Signal: Direct call to a function within a test case
        
    # ✅ Best Practice: Consider using logging instead of print for better control over output
    # 🧠 ML Signal: Function definition with a test prefix, indicating a test case
    def test_get_industry_classified(self):
        print(fd.get_industry_classified())
    # ⚠️ SAST Risk (Low): Use of print statements in test functions can clutter test output
    # 🧠 ML Signal: Method name suggests this is a test function, useful for identifying test patterns
    # 🧠 ML Signal: Tracking the usage of specific function calls can help in understanding common patterns
        
    # ✅ Best Practice: Consider using assertions to validate the output instead of print
    def test_get_concept_classified(self):
        # ⚠️ SAST Risk (Low): Using print statements in test functions can clutter output and is not a best practice for testing
        # 🧠 ML Signal: Method name suggests a test function, indicating a pattern for test case identification
        print(fd.get_concept_classified())
    # ✅ Best Practice: Consider using assertions to validate behavior instead of print statements
    # ✅ Best Practice: Function should have a docstring to describe its purpose
        
    # 🧠 ML Signal: Function definition with a test prefix, indicating a test case
    def test_get_area_classified(self):
        # ⚠️ SAST Risk (Low): Use of print statements in test functions can clutter output; consider using assertions
        print(fd.get_area_classified())
    # ⚠️ SAST Risk (Low): Use of print statements in test functions can clutter test output
    # 🧠 ML Signal: Function definition with a test prefix, indicating a test case or test function
        
    # ✅ Best Practice: Consider using assertions to validate behavior instead of print
    def test_get_gem_classified(self):
        # ⚠️ SAST Risk (Low): Use of print statements for debugging can expose sensitive data in production
        print(fd.get_gem_classified())
    # 🧠 ML Signal: Function definition with a test prefix, indicating a test case
    # ✅ Best Practice: Consider using logging instead of print for better control over output
        
    # ✅ Best Practice: Use of self parameter suggests this is a method in a class
    def test_get_sme_classified(self):
        # 🧠 ML Signal: Use of print statements in test functions can indicate debugging practices.
        print(fd.get_sme_classified())
    # 🧠 ML Signal: Direct function call within a test, indicating a pattern of testing outputs
        
    # ⚠️ SAST Risk (Low): Direct print statements in tests can clutter output and are not ideal for automated testing
    # ✅ Best Practice: Ensure the script is being run directly before executing test cases.
    # 🧠 ML Signal: Use of unittest.main() indicates a pattern for running test cases.
    def test_get_st_classified(self):
        print(fd.get_st_classified())
    
    def test_get_hs300s(self):
        print(fd.get_hs300s())   
        
    def test_get_sz50s(self):
        print(fd.get_sz50s()) 
      
    def test_get_zz500s(self):
        print(fd.get_zz500s())   
        
if __name__ == "__main__":
    unittest.main()
    
#     suite = unittest.TestSuite()  
#     suite.addTest(Test('test_get_gem_classified'))  
#     unittest.TextTestRunner(verbosity=2).run(suite)