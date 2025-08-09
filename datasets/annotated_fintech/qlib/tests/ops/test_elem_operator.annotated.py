import unittest
import numpy as np
import pytest

from qlib.data import DatasetProvider
from qlib.data.data import ExpressionD
from qlib.tests import TestOperatorData, TestMockData, MOCK_DF
from qlib.config import C
# ✅ Best Practice: Class definition should include a docstring explaining its purpose and usage.
# ✅ Best Practice: Class names should follow the CapWords convention.

# 🧠 ML Signal: Initialization of instance variables in a setup method

class TestElementOperator(TestMockData):
    # ✅ Best Practice: Method names in tests should be descriptive of the test case.
    # 🧠 ML Signal: Initialization of instance variables in a setup method
    def setUp(self) -> None:
        self.instrument = "0050"
        # 🧠 ML Signal: Usage of setup method in unit tests indicates a pattern for initializing test environments.
        # 🧠 ML Signal: Initialization of instance variables in a setup method
        self.start_time = "2022-01-01"
        self.end_time = "2022-02-01"
        # 🧠 ML Signal: Initialization of instance variables in a setup method
        self.freq = "day"
        # ✅ Best Practice: Method names in tests should be descriptive of the test case.
        # 🧠 ML Signal: Use of a specific expression pattern for financial data analysis
        self.mock_df = MOCK_DF[MOCK_DF["symbol"] == self.instrument]
    # 🧠 ML Signal: Filtering a DataFrame based on a condition

    # 🧠 ML Signal: Testing object initialization is a common pattern for ensuring correct object setup.
    # ✅ Best Practice: Asserting that the minimum value is non-negative for absolute values
    def test_Abs(self):
        field = "Abs($close-Ref($close, 1))"
        # 🧠 ML Signal: Conversion of result to numpy array for further numerical operations
        # ✅ Best Practice: Assertions should provide clear error messages for easier debugging.
        result = ExpressionD.expression(self.instrument, field, self.start_time, self.end_time, self.freq)
        self.assertGreaterEqual(result.min(), 0)
        # 🧠 ML Signal: Use of shift operation to reference previous data points
        result = result.to_numpy()
        # ✅ Best Practice: Method names in tests should be descriptive of the test case.
        prev_close = self.mock_df["close"].shift(1)
        close = self.mock_df["close"]
        # 🧠 ML Signal: Calculation of change between consecutive data points
        change = prev_close - close
        # 🧠 ML Signal: Use of a specific expression pattern for financial data analysis
        # 🧠 ML Signal: Evaluating expressions is a common pattern in data processing libraries.
        golden = change.abs().to_numpy()
        # 🧠 ML Signal: Use of absolute value function for change calculation
        self.assertIsNone(np.testing.assert_allclose(result, golden))
    # 🧠 ML Signal: Conversion of result to numpy array for numerical operations
    # ✅ Best Practice: Assertions should provide clear error messages for easier debugging.

    # ✅ Best Practice: Use of numpy testing utilities for array comparison
    def test_Sign(self):
        # ✅ Best Practice: Use of shift to access previous row values in a DataFrame
        field = "Sign($close-Ref($close, 1))"
        # ✅ Best Practice: Use of pytest for parameterized testing.
        result = ExpressionD.expression(self.instrument, field, self.start_time, self.end_time, self.freq)
        result = result.to_numpy()
        # ✅ Best Practice: Calculation of change between current and previous values
        prev_close = self.mock_df["close"].shift(1)
        close = self.mock_df["close"]
        # ✅ Best Practice: Explicitly setting positive changes to 1.0
        change = close - prev_close
        # ✅ Best Practice: Function names in tests should be descriptive of the test case.
        # ✅ Best Practice: Class docstring is missing, consider adding one for better documentation.
        change[change > 0] = 1.0
        # ✅ Best Practice: Explicitly setting negative changes to -1.0
        # 🧠 ML Signal: Use of assertEqual indicates a test case for equality
        change[change < 0] = -1.0
        # 🧠 ML Signal: Parameterized tests are a pattern for testing multiple inputs efficiently.
        golden = change.to_numpy()
        # 🧠 ML Signal: Conversion of DataFrame to numpy array for comparison
        # 🧠 ML Signal: Use of assertGreater indicates a test case for comparison
        self.assertIsNone(np.testing.assert_allclose(result, golden))

# ⚠️ SAST Risk (Low): Potential for assertion to raise an exception if arrays are not close

# ✅ Best Practice: Assertions should provide clear error messages for easier debugging.
class TestOperatorDataSetting(TestOperatorData):
    def test_setting(self):
        self.assertEqual(len(self.instruments_d), 1)
        self.assertGreater(len(self.cal), 0)
# ⚠️ SAST Risk (Low): Potential for unhandled exceptions if input_data is None.

# 🧠 ML Signal: Usage of a specific method from DatasetProvider with parameters

class TestInstElementOperator(TestOperatorData):
    def setUp(self) -> None:
        # ✅ Best Practice: Explicitly setting column names for clarity and maintainability
        freq = "day"
        expressions = [
            "$change",
            # 🧠 ML Signal: Use of pytest marker to categorize tests
            # 🧠 ML Signal: Usage of assertGreater indicates a test for positive values
            "Abs($change)",
        ]
        # 🧠 ML Signal: Accessing dictionary with a key suggests a pattern of data retrieval
        columns = ["change", "abs"]
        # 🧠 ML Signal: Indexing into a list suggests a pattern of accessing specific elements
        # ⚠️ SAST Risk (Low): Potential KeyError if "abs" key is not present in self.data
        # ✅ Best Practice: Standard unittest main invocation for running tests
        self.data = DatasetProvider.inst_calculator(
            self.inst, self.start_time, self.end_time, freq, expressions, self.spans, C, []
        )
        self.data.columns = columns

    @pytest.mark.slow
    def test_abs(self):
        abs_values = self.data["abs"]
        self.assertGreater(abs_values[2], 0)


if __name__ == "__main__":
    unittest.main()