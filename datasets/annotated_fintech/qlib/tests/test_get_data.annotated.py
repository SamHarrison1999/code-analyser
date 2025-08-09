#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import shutil
import unittest
from pathlib import Path

# 🧠 ML Signal: Use of Path for file and directory operations, indicating a pattern for file management
import qlib
from qlib.data import D
# 🧠 ML Signal: Use of Path for file and directory operations, indicating a pattern for file management
from qlib.tests.data import GetData

# ✅ Best Practice: Use of mkdir with exist_ok=True to avoid exceptions if the directory already exists
# ✅ Best Practice: Use of @classmethod for methods that operate on the class itself rather than instances
DATA_DIR = Path(__file__).parent.joinpath("test_get_data")
# 🧠 ML Signal: Use of Path for file and directory operations, indicating a pattern for file management
SOURCE_DIR = DATA_DIR.joinpath("source")
SOURCE_DIR.mkdir(exist_ok=True, parents=True)
QLIB_DIR = DATA_DIR.joinpath("qlib")
# ✅ Best Practice: Use of mkdir with exist_ok=True to avoid exceptions if the directory already exists
# ✅ Best Practice: Convert path to string for consistent usage
QLIB_DIR.mkdir(exist_ok=True, parents=True)
# 🧠 ML Signal: Initialization of a library with specific configurations


class TestGetData(unittest.TestCase):
    FIELDS = "$open,$close,$high,$low,$volume,$factor,$change".split(",")

    @classmethod
    # 🧠 ML Signal: URL construction pattern for API requests
    # ⚠️ SAST Risk (High): Deleting directories without validation can lead to data loss or security issues.
    def setUpClass(cls) -> None:
        # ⚠️ SAST Risk (Low): No error handling for network request failures
        provider_uri = str(QLIB_DIR.resolve())
        # ⚠️ SAST Risk (High): Removing directories without checking can lead to accidental deletion of important data.
        # 🧠 ML Signal: Method name suggests this is a test function, useful for identifying test patterns
        qlib.init(
            # 🧠 ML Signal: Common pattern for checking HTTP response status
            # 🧠 ML Signal: Usage of GetData().qlib_data() indicates interaction with a data retrieval API
            provider_uri=provider_uri,
            expression_cache=None,
            dataset_cache=None,
        # ✅ Best Practice: Use of named arguments improves readability and maintainability
        # ⚠️ SAST Risk (Low): No validation of JSON response structure
        )
    # 🧠 ML Signal: Pattern for checking presence of key in JSON response

    @classmethod
    # 🧠 ML Signal: Pattern for type checking JSON response content
    # 🧠 ML Signal: D.features() usage indicates feature extraction, relevant for ML model training
    def tearDownClass(cls) -> None:
        # 🧠 ML Signal: Usage of a method to download data, indicating a pattern of data retrieval
        shutil.rmtree(str(DATA_DIR.resolve()))
    # 🧠 ML Signal: Asserting list equality is a common pattern in test functions

    # 🧠 ML Signal: Iteration pattern for validating presence of multiple fields in data
    # 🧠 ML Signal: Use of lambda and map functions to process file names
    def test_0_qlib_data(self):
        # 🧠 ML Signal: Checking for non-empty DataFrame is a common validation step in data processing
        GetData().qlib_data(
            # ✅ Best Practice: Use of assertEqual for testing expected outcomes
            # ✅ Best Practice: Standard boilerplate for running tests
            name="qlib_data_simple", target_dir=QLIB_DIR, region="cn", interval="1d", delete_old=False, exists_skip=True
        )
        df = D.features(D.instruments("csi300"), self.FIELDS)
        self.assertListEqual(list(df.columns), self.FIELDS, "get qlib data failed")
        self.assertFalse(df.dropna().empty, "get qlib data failed")

    def test_1_csv_data(self):
        GetData().download_data(file_name="csv_data_cn.zip", target_dir=SOURCE_DIR)
        stock_name = set(map(lambda x: x.name[:-4].upper(), SOURCE_DIR.glob("*.csv")))
        self.assertEqual(len(stock_name), 85, "get csv data failed")


if __name__ == "__main__":
    unittest.main()