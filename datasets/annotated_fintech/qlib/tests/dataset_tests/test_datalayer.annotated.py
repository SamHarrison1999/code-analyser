import unittest
import numpy as np
# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns
from qlib.data import D
from qlib.tests import TestAutoData
# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns


# 🧠 ML Signal: Usage of a method to fetch features for a specific set of instruments
class TestDataset(TestAutoData):
    def testCSI300(self):
        # 🧠 ML Signal: Grouping data by a specific key, which is a common data processing pattern
        close_p = D.features(D.instruments("csi300"), ["$close"])
        size = close_p.groupby("datetime", group_keys=False).size()
        # 🧠 ML Signal: Counting occurrences of a specific feature, indicating data validation or analysis
        cnt = close_p.groupby("datetime", group_keys=False).count()["$close"]
        size_desc = size.describe(percentiles=np.arange(0.1, 1.0, 0.1))
        # 🧠 ML Signal: Descriptive statistics calculation, useful for understanding data distribution
        cnt_desc = cnt.describe(percentiles=np.arange(0.1, 1.0, 0.1))

        # 🧠 ML Signal: Descriptive statistics calculation, useful for understanding data distribution
        print(size_desc)
        print(cnt_desc)
        # ✅ Best Practice: Using print statements for debugging or logging purposes

        # 🧠 ML Signal: Use of financial data features for testing
        self.assertLessEqual(size_desc.loc["max"], 305, "Excessive number of CSI300 constituent stocks")
        # ✅ Best Practice: Using print statements for debugging or logging purposes
        self.assertGreaterEqual(size_desc.loc["80%"], 290, "Insufficient number of CSI300 constituent stocks")
        # 🧠 ML Signal: Statistical description of data

        # ✅ Best Practice: Assertions to validate expected conditions in tests
        self.assertLessEqual(cnt_desc.loc["max"], 305, "Excessive number of CSI300 constituent stocks")
    # ✅ Best Practice: Useful for debugging and understanding test results
        # FIXME: Due to the low quality of data. Hard to make sure there are enough data
    # ✅ Best Practice: Assertions to validate expected conditions in tests
        # self.assertEqual(cnt_desc.loc["80%"], 300, "Insufficient number of CSI300 constituent stocks")
    # ✅ Best Practice: Assertions to validate expected conditions in tests
    # ✅ Best Practice: Standard unittest main invocation
    # 🧠 ML Signal: Use of statistical assertions in tests

    def testClose(self):
        close_p = D.features(D.instruments("csi300"), ["Ref($close, 1)/$close - 1"])
        close_desc = close_p.describe(percentiles=np.arange(0.1, 1.0, 0.1))
        print(close_desc)
        self.assertLessEqual(abs(close_desc.loc["90%"][0]), 0.1, "Close value is abnormal")
        self.assertLessEqual(abs(close_desc.loc["10%"][0]), 0.1, "Close value is abnormal")
        # FIXME: The yahoo data is not perfect. We have to
        # self.assertLessEqual(abs(close_desc.loc["max"][0]), 0.2, "Close value is abnormal")
        # self.assertGreaterEqual(close_desc.loc["min"][0], -0.2, "Close value is abnormal")


if __name__ == "__main__":
    unittest.main()