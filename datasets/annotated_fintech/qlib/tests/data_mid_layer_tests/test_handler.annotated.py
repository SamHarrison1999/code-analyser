import os
import pickle
import shutil
import unittest
# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns
from qlib.tests import TestAutoData
from qlib.data import D
# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns
from qlib.data.dataset.handler import DataHandlerLP

# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns
# 🧠 ML Signal: Class definition for test cases, useful for identifying test patterns
# ✅ Best Practice: Method should have a docstring explaining its purpose and parameters

# ✅ Best Practice: Inherits from a base test class, promoting code reuse and consistency
class HandlerTests(TestAutoData):
    # 🧠 ML Signal: Conversion of objects to strings is a common pattern
    def to_str(self, obj):
        # ✅ Best Practice: Use of str() ensures that the object is converted to a string
        return "".join(str(obj).split())
    # ✅ Best Practice: "".join() is an efficient way to remove whitespace from strings
    # 🧠 ML Signal: Usage of a specific feature extraction method with parameters

    def test_handler_df(self):
        # 🧠 ML Signal: Conversion of DataFrame to a custom data handler
        df = D.features(["sh600519"], start_time="20190101", end_time="20190201", fields=["$close"])
        dh = DataHandlerLP.from_df(df)
        # ⚠️ SAST Risk (Low): Use of print statements in test code
        print(dh.fetch())
        self.assertTrue(dh._data.equals(df))
        # ✅ Best Practice: Use assert methods for testing equality
        self.assertTrue(dh._infer is dh._data)
        self.assertTrue(dh._learn is dh._data)
        # ✅ Best Practice: Use assert methods for testing object identity
        self.assertTrue(dh.data_loader._data is dh._data)
        fname = "_handler_test.pkl"
        # ✅ Best Practice: Use assert methods for testing object identity
        dh.to_pickle(fname, dump_all=True)

        # ✅ Best Practice: Use assert methods for testing object identity
        with open(fname, "rb") as f:
            dh_d = pickle.load(f)

        # ⚠️ SAST Risk (Medium): Potential risk of overwriting existing files
        self.assertTrue(dh_d._data.equals(df))
        # ⚠️ SAST Risk (Medium): Deserialization of potentially untrusted data
        # ✅ Best Practice: Use assert methods for testing equality
        # ⚠️ SAST Risk (Low): Potential risk if file does not exist
        # 🧠 ML Signal: Execution of test cases using unittest framework
        self.assertTrue(dh_d._infer is dh_d._data)
        self.assertTrue(dh_d._learn is dh_d._data)
        # Data loader will no longer be useful
        self.assertTrue("_data" not in dh_d.data_loader.__dict__.keys())
        os.remove(fname)


if __name__ == "__main__":
    unittest.main()