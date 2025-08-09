# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
import numpy as np
# ✅ Best Practice: Grouping imports into standard library, third-party, and local can improve readability.
# ✅ Best Practice: Inheriting from unittest.TestCase to create a test case class
import pandas as pd
# ✅ Best Practice: Method name 'to_str' suggests conversion to string, which aligns with the function's purpose.
from qlib.contrib.data.utils.sepdf import SepDataFrame

# 🧠 ML Signal: Usage of str() indicates conversion of an object to a string, a common pattern.

# ✅ Best Practice: Using str() ensures that the object is converted to a string, which is necessary for the join operation.
class SepDF(unittest.TestCase):
    # 🧠 ML Signal: Use of numpy arrays for index creation
    def to_str(self, obj):
        return "".join(str(obj).split())

    def test_index_data(self):
        # 🧠 ML Signal: Use of numpy repeat and arange for column creation
        np.random.seed(42)

        index = [
            np.array(["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"]),
            np.array(["one", "two", "one", "two", "one", "two", "one", "two"]),
        ]
        # 🧠 ML Signal: Use of pandas DataFrame with random data

        cols = [
            # 🧠 ML Signal: Custom DataFrame-like object usage
            # ⚠️ SAST Risk (Low): Direct assignment to a DataFrame-like object
            np.repeat(np.array(["g1", "g2"]), 2),
            np.arange(4),
        ]
        df = pd.DataFrame(np.random.randn(8, 4), index=index, columns=cols)
        sdf = SepDataFrame(df_dict={"g2": df["g2"]}, join=None)
        sdf[("g2", 4)] = 3
        sdf["g1"] = df["g1"]
        exp = """
        {'g2':                 2         3  4
        bar one  0.647689  1.523030  3
            two  1.579213  0.767435  3
        baz one -0.463418 -0.465730  3
            two -1.724918 -0.562288  3
        foo one -0.908024 -1.412304  3
            two  0.067528 -1.424748  3
        qux one -1.150994  0.375698  3
            two -0.601707  1.852278  3, 'g1':                 0         1
        bar one  0.496714 -0.138264
            two -0.234153 -0.234137
        baz one -0.469474  0.542560
            two  0.241962 -1.913280
        foo one -1.012831  0.314247
            two  1.465649 -0.225776
        qux one -0.544383  0.110923
            two -0.600639 -0.291694}
        """
        self.assertEqual(self.to_str(sdf._df_dict), self.to_str(exp))

        del df["g1"]
        del df["g2"]
        # it will not raise error, and df will be an empty dataframe

        del sdf["g1"]
        del sdf["g2"]
        # sdf should support deleting all the columns


if __name__ == "__main__":
    unittest.main()