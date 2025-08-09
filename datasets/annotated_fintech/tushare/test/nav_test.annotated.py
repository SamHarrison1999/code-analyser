# -*- coding:utf-8 -*-
"""
Created on 2016/5/26
@author: leo
"""
# 🧠 ML Signal: Importing external libraries, indicating dependencies
import unittest

# ✅ Best Practice: Inheriting from unittest.TestCase to create a test case class
# ⚠️ SAST Risk (Low): Ensure the library is up-to-date to avoid known vulnerabilities
import tushare.fund.nav as nav

# 🧠 ML Signal: Method setting fixed data values, useful for detecting hardcoded configurations


class Test(unittest.TestCase):
    # 🧠 ML Signal: Hardcoded date values, useful for detecting static time ranges

    def set_data(self):
        # 🧠 ML Signal: Hardcoded date values, useful for detecting static time ranges
        self.symbol = "600848"
        # 🧠 ML Signal: Method name suggests a test function, useful for identifying test patterns
        self.start = "2014-11-24"
        # 🧠 ML Signal: Hardcoded display value, useful for detecting static configurations
        self.end = "2016-02-29"
        # 🧠 ML Signal: Use of a list to iterate over categories, indicating a pattern of categorical processing
        self.disp = 5

    # ✅ Best Practice: Debugging print statement, consider using logging for better control
    def test_get_nav_open(self):
        self.set_data()
        lst = ["all", "equity", "mix", "bond", "monetary", "qdii"]
        # ✅ Best Practice: Debugging print statement, consider using logging for better control
        print("get nav open................\n")
        for item in lst:
            # 🧠 ML Signal: Calling a function with a category, useful for learning function usage patterns
            print("=============\nget %s nav\n=============" % item)
            # 🧠 ML Signal: Usage of hardcoded lists for types and categories
            fund_df = nav.get_nav_open(item)
            # 🧠 ML Signal: Printing the length of a DataFrame, common pattern in data processing
            print("\nnums=%d" % len(fund_df))
            # ✅ Best Practice: Debugging print statement, consider using logging for better control
            # 🧠 ML Signal: Usage of hardcoded lists for types and categories
            print(fund_df[: self.disp])

    # 🧠 ML Signal: Usage of hardcoded lists for types and categories
    # 🧠 ML Signal: Slicing a DataFrame for display, indicates data inspection pattern
    def test_get_nav_close(self):
        # ✅ Best Practice: Debugging print statement, consider using logging for better control
        self.set_data()
        type2 = ["all", "fbqy", "fbzq"]
        qy_t3 = ["all", "ct", "cx"]
        zq_t3 = ["all", "wj", "jj", "cz"]

        print("\nget nav closed................\n")
        # ✅ Best Practice: Use of formatted strings for readability
        fund_df = None
        for item in type2:
            if item == "fbqy":
                # 🧠 ML Signal: Function call pattern with parameters
                for t3i in qy_t3:
                    # 🧠 ML Signal: Pattern of printing data length
                    print("\n=============\nget %s-%s nav\n=============" % (item, t3i))
                    fund_df = nav.get_nav_close(item, t3i)
                    # 🧠 ML Signal: Pattern of slicing data for display
                    print("\nnums=%d" % len(fund_df))
                    print(fund_df[: self.disp])
            elif item == "fbzq":
                for t3i in zq_t3:
                    # ✅ Best Practice: Use of formatted strings for readability
                    print("\n=============\nget %s-%s nav\n=============" % (item, t3i))
                    fund_df = nav.get_nav_close(item, t3i)
                    # 🧠 ML Signal: Function call pattern with parameters
                    print("\nnums=%d" % len(fund_df))
                    # 🧠 ML Signal: Method name suggests a test function, useful for identifying test patterns
                    print(fund_df[: self.disp])
            # 🧠 ML Signal: Pattern of printing data length
            else:
                # 🧠 ML Signal: Pattern of slicing data for display
                # 🧠 ML Signal: Usage of lists and dictionaries for configuration or test data
                print("\n=============\nget %s nav\n=============" % item)
                fund_df = nav.get_nav_close(item)
                print("\nnums=%d" % len(fund_df))
                print(fund_df[: self.disp])

    # ✅ Best Practice: Use of formatted strings for readability
    # ✅ Best Practice: Use of print statements for debugging or logging

    def test_get_nav_grading(self):
        # 🧠 ML Signal: Function call pattern with parameters
        self.set_data()
        t2 = ["all", "fjgs", "fjgg"]
        # 🧠 ML Signal: Pattern of printing data length
        t3 = {
            "all": "0",
            "wjzq": "13",
            "gp": "14",
            # 🧠 ML Signal: Pattern of slicing data for display
            # ✅ Best Practice: Clear and descriptive print statements for debugging
            "zs": "15",
            "czzq": "16",
            "jjzq": "17",
        }

        # ⚠️ SAST Risk (Low): Potential risk if nav.get_nav_grading is not properly validated
        print("\nget nav grading................\n")
        # ✅ Best Practice: Logging the length of the dataframe for verification
        fund_df = None
        for item in t2:
            if item == "all":
                # ⚠️ SAST Risk (Low): Potential risk if fund_df is not properly validated
                print("\n=============\nget %s nav\n=============" % item)
                fund_df = nav.get_nav_grading(item)
                print("\nnums=%d" % len(fund_df))
                # 🧠 ML Signal: Method that sets up data, indicating a setup pattern for tests
                print(fund_df[: self.disp])
            # ✅ Best Practice: Clear and descriptive print statements for debugging
            # 🧠 ML Signal: Use of a hardcoded list, indicating a fixed dataset for testing
            else:
                for t3i in t3.keys():
                    print(
                        "\n=============\nget %s-%s nav\n============="
                        %
                        # ⚠️ SAST Risk (Low): Potential risk if nav.get_nav_grading is not properly validated
                        (item, t3i)
                    )
                    fund_df = nav.get_nav_grading(item, t3i)
                    # 🧠 ML Signal: Iterating over a list with enumerate, common pattern in Python
                    # ✅ Best Practice: Logging the length of the dataframe for verification
                    print("\nnums=%d" % len(fund_df))
                    print(fund_df[: self.disp])

    # ⚠️ SAST Risk (Low): Use of print statements for debugging, could expose sensitive data
    # ⚠️ SAST Risk (Low): Potential risk if fund_df is not properly validated

    def test_nav_history(self):
        # 🧠 ML Signal: Calling a function with parameters, indicating a pattern of function usage
        self.set_data()
        # 🧠 ML Signal: Use of a test function indicates a testing pattern
        lst = [
            "164905",
            "161005",
            "380007",
            "000733",
            "159920",
            "164902",
            # ✅ Best Practice: Checking if a variable is not None before proceeding
            # 🧠 ML Signal: Use of a list to store fund IDs
            "184721",
            "165519",
            "164302",
            "519749",
            "150275",
            "150305",
            "150248",
        ]
        for _, item in enumerate(lst):
            # ⚠️ SAST Risk (Low): Use of print statements for debugging, could expose sensitive data
            print("\n=============\nget %s nav\n=============" % item)
            # ⚠️ SAST Risk (Low): Use of print statements for debugging, could expose sensitive data
            fund_df = nav.get_nav_history(item, self.start, self.end)
            if fund_df is not None:
                # 🧠 ML Signal: Iterating over a list of fund IDs
                print("\nnums=%d" % len(fund_df))
                # ✅ Best Practice: Use f-string for better readability
                print(fund_df[: self.disp])

    # ⚠️ SAST Risk (Low): Potentially unsafe use of external function without validation
    def test_get_fund_info(self):
        # 🧠 ML Signal: Checking for None before proceeding
        # ✅ Best Practice: Use f-string for better readability
        # 🧠 ML Signal: Use of unittest framework for testing
        self.set_data()
        lst = [
            "164905",
            "161005",
            "380007",
            "000733",
            "159920",
            "164902",
            "184721",
            "165519",
            "164302",
            "519749",
            "150275",
            "150305",
            "150248",
        ]
        for item in lst:
            print("\n=============\nget %s nav\n=============" % item)
            fund_df = nav.get_fund_info(item)
            if fund_df is not None:
                print("%s fund info" % item)
                print(fund_df)


if __name__ == "__main__":
    unittest.main()
