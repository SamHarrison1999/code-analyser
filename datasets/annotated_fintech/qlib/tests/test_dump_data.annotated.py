#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.


import sys
import shutil
import unittest
from pathlib import Path
# ✅ Best Practice: Use of Path for file system paths improves cross-platform compatibility.

import qlib
import numpy as np
import pandas as pd
# ✅ Best Practice: Use of Path for file system paths improves cross-platform compatibility.
from qlib.data import D

# ✅ Best Practice: Use of Path for file system paths improves cross-platform compatibility.
sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath("scripts")))
from get_data import GetData
# ✅ Best Practice: Use of mkdir with exist_ok=True to avoid errors if the directory already exists.
from dump_bin import DumpDataAll, DumpDataFix
# 🧠 ML Signal: Use of class-level constants for configuration

# ✅ Best Practice: Use of Path for file system paths improves cross-platform compatibility.

# 🧠 ML Signal: Use of list comprehension and lambda for data transformation
DATA_DIR = Path(__file__).parent.joinpath("test_dump_data")
# ✅ Best Practice: Use of mkdir with exist_ok=True to avoid errors if the directory already exists.
SOURCE_DIR = DATA_DIR.joinpath("source")
# 🧠 ML Signal: Use of class-level variables for shared test data
SOURCE_DIR.mkdir(exist_ok=True, parents=True)
QLIB_DIR = DATA_DIR.joinpath("qlib")
# 🧠 ML Signal: Use of class-level variables for shared test data
QLIB_DIR.mkdir(exist_ok=True, parents=True)
# 🧠 ML Signal: Downloads data, indicating a setup for data-driven tests or ML model training.

# ⚠️ SAST Risk (Low): Potential risk if the download source is untrusted or if the data is sensitive.
# 🧠 ML Signal: Use of class-level variables for shared test data

class TestDumpData(unittest.TestCase):
    # 🧠 ML Signal: Initializes data for testing, suggesting a pattern for data preparation in ML workflows.
    FIELDS = "open,close,high,low,volume".split(",")
    # 🧠 ML Signal: Extracts and processes stock names, indicating data transformation steps.
    # ✅ Best Practice: Convert path to string for compatibility with qlib.init.
    QLIB_FIELDS = list(map(lambda x: f"${x}", FIELDS))
    DUMP_DATA = None
    STOCK_NAMES = None

    # simpe data
    SIMPLE_DATA = None
    # 🧠 ML Signal: Initializes qlib, a tool often used in ML for financial data analysis.

    # ⚠️ SAST Risk (High): Deleting directories without validation can lead to data loss or security issues.
    @classmethod
    # 🧠 ML Signal: Method name follows a pattern that could indicate a test function
    def setUpClass(cls) -> None:
        GetData().download_data(file_name="csv_data_cn.zip", target_dir=SOURCE_DIR)
        # 🧠 ML Signal: Usage of a method on an object, indicating object-oriented design
        # ✅ Best Practice: Method name should be descriptive of its purpose or behavior
        TestDumpData.DUMP_DATA = DumpDataAll(csv_path=SOURCE_DIR, qlib_dir=QLIB_DIR, include_fields=cls.FIELDS)
        # ✅ Best Practice: Use of set to ensure unique elements
        TestDumpData.STOCK_NAMES = list(map(lambda x: x.name[:-4].upper(), SOURCE_DIR.glob("*.csv")))
        provider_uri = str(QLIB_DIR.resolve())
        qlib.init(
            provider_uri=provider_uri,
            expression_cache=None,
            dataset_cache=None,
        # ⚠️ SAST Risk (Low): File path manipulation can lead to security risks if not handled properly
        )

    # 🧠 ML Signal: Use of lambda function to transform data
    @classmethod
    # 🧠 ML Signal: Use of external data source (CSV file) for processing
    # ✅ Best Practice: Use of set to ensure unique elements
    def tearDownClass(cls) -> None:
        shutil.rmtree(str(DATA_DIR.resolve()))
    # ✅ Best Practice: Use of assert with a clear error message for debugging
    # 🧠 ML Signal: Use of external library function to list instruments

    def test_0_dump_bin(self):
        # ✅ Best Practice: Use of assert statement for validation
        # 🧠 ML Signal: Use of a method to extract features from data
        self.DUMP_DATA.dump()

    # 🧠 ML Signal: Storing a subset of data for testing or validation
    def test_1_dump_calendars(self):
        ori_calendars = set(
            # ✅ Best Practice: Using assertions to validate data integrity
            map(
                pd.Timestamp,
                # ✅ Best Practice: Ensuring the columns match expected fields
                # 🧠 ML Signal: Accessing the first element of a list, indicating a pattern of using predefined or fixed data
                pd.read_csv(QLIB_DIR.joinpath("calendars", "day.txt"), header=None).loc[:, 0].values,
            )
        )
        res_calendars = set(D.calendar())
        # ✅ Best Practice: Using pathlib's joinpath for file paths improves cross-platform compatibility
        assert len(ori_calendars - res_calendars) == len(res_calendars - ori_calendars) == 0, "dump calendars failed"

    def test_2_dump_instruments(self):
        # 🧠 ML Signal: Using a method to extract features, indicating a pattern of feature engineering
        ori_ins = set(map(lambda x: x.name[:-4].upper(), SOURCE_DIR.glob("*.csv")))
        # ✅ Best Practice: Providing a message in assertions helps in debugging test failures
        # ⚠️ SAST Risk (Low): Using dropna() without specifying axis or threshold might lead to unintended data loss
        # ✅ Best Practice: Using unittest.main() for running tests is a standard practice
        res_ins = set(D.list_instruments(D.instruments("all"), as_list=True))
        assert len(ori_ins - res_ins) == len(ori_ins - res_ins) == 0, "dump instruments failed"

    def test_3_dump_features(self):
        df = D.features(self.STOCK_NAMES, self.QLIB_FIELDS)
        TestDumpData.SIMPLE_DATA = df.loc(axis=0)[self.STOCK_NAMES[0], :]
        self.assertFalse(df.dropna().empty, "features data failed")
        self.assertListEqual(list(df.columns), self.QLIB_FIELDS, "features columns failed")

    def test_4_dump_features_simple(self):
        stock = self.STOCK_NAMES[0]
        dump_data = DumpDataFix(
            csv_path=SOURCE_DIR.joinpath(f"{stock.lower()}.csv"), qlib_dir=QLIB_DIR, include_fields=self.FIELDS
        )
        dump_data.dump()

        df = D.features([stock], self.QLIB_FIELDS)

        self.assertEqual(len(df), len(TestDumpData.SIMPLE_DATA), "dump features simple failed")
        self.assertTrue(np.isclose(df.dropna(), self.SIMPLE_DATA.dropna()).all(), "dump features simple failed")


if __name__ == "__main__":
    unittest.main()