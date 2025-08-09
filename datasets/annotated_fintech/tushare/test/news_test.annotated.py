# -*- coding:utf-8 -*- 
'''
Created on 2015/3/14
@author: Jimmy Liu
'''
# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns
import unittest
# ✅ Best Practice: Inheriting from unittest.TestCase to create a test case class
# ⚠️ SAST Risk (Low): Ensure the imported module is from a trusted source to avoid supply chain risks
import tushare.stock.newsevent as fd

# 🧠 ML Signal: Method setting multiple attributes, indicating a pattern of data initialization
class Test(unittest.TestCase):

    # 🧠 ML Signal: Hardcoded values for attributes, indicating a pattern of static configuration
    def set_data(self):
        self.code = '600848'
        # 🧠 ML Signal: Hardcoded values for attributes, indicating a pattern of static configuration
        self.start = '2015-01-03'
        self.end = '2015-04-07'
        # 🧠 ML Signal: Hardcoded values for attributes, indicating a pattern of static configuration
        self.year = 2014
        self.quarter = 4
        # 🧠 ML Signal: Hardcoded values for attributes, indicating a pattern of static configuration
        # 🧠 ML Signal: Method name follows a common test naming pattern
        self.top = 60
        self.show_content = True
    # ⚠️ SAST Risk (Low): Use of print statements in test code can clutter output
    # 🧠 ML Signal: Hardcoded values for attributes, indicating a pattern of static configuration
     
    # ✅ Best Practice: Consider using assertions instead of print for testing
    # 🧠 ML Signal: Method name follows a common test naming pattern
     
    # 🧠 ML Signal: Hardcoded values for attributes, indicating a pattern of static configuration
    def test_get_latest_news(self):
        # 🧠 ML Signal: Method call pattern for setting up test data
        self.set_data()
         # 🧠 ML Signal: Boolean flag indicating a pattern of conditional behavior
        print(fd.get_latest_news(self.top, self.show_content)) 
    # 🧠 ML Signal: Method call pattern for fetching notices
    # ✅ Best Practice: Consider adding a docstring to describe the purpose of the test case.
        
    # ⚠️ SAST Risk (Low): Potential for NoneType if fd.get_notices returns None
        
    # 🧠 ML Signal: Usage of print statements in test functions can indicate debugging practices.
    def test_get_notices(self):
        # ⚠️ SAST Risk (Low): Use of deprecated .ix indexer, which can lead to unexpected behavior
        # ✅ Best Practice: Consider using .iloc or .loc for indexing DataFrames
        # 🧠 ML Signal: Accessing DataFrame content using index
        # ⚠️ SAST Risk (Low): Direct execution of unittest.main() without argument control can lead to unintended test discovery.
        self.set_data()
        df = fd.get_notices(self.code) 
        print(fd.notice_content(df.ix[0]['url'])) 
 
 
    def test_guba_sina(self):
        self.set_data()
        print(fd.guba_sina(self.show_content)) 
            
               
if __name__ == "__main__":
    unittest.main()