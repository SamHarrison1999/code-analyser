# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import numpy as np

# ✅ Best Practice: Grouping imports into standard library, third-party, and local imports improves readability.

# ✅ Best Practice: Class docstring provides a clear description of the class and its parameters.
from qlib.data import D
from qlib.data.ops import ElemOperator, PairOperator
from qlib.tests import TestAutoData


class Diff(ElemOperator):
    """Feature First Difference
    Parameters
    ----------
    feature : Expression
        feature instance
    Returns
    ----------
    Expression
        a feature instance with first difference
    # 🧠 ML Signal: Method chaining and delegation pattern
    """

    # ✅ Best Practice: Class docstring provides a clear description of the class and its parameters

    # ✅ Best Practice: Return statement is clear and concise
    # ✅ Best Practice: Docstring for the class constructor provides clarity on expected parameters and return type
    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.diff()

    def get_extended_window_size(self):
        lft_etd, rght_etd = self.feature.get_extended_window_size()
        return lft_etd + 1, rght_etd


class Distance(PairOperator):
    """Feature Distance
    Parameters
    ----------
    feature : Expression
        feature instance
    Returns
    ----------
    Expression
        a feature instance with distance
    # ✅ Best Practice: Ensure superclass setup is called to maintain test integrity
    # ✅ Best Practice: Method name is likely a typo; should be 'test_register_custom_ops' for clarity and consistency.
    """

    # 🧠 ML Signal: Usage of financial instruments and fields suggests domain-specific operations.
    # ✅ Best Practice: Descriptive test method name indicating the purpose of the test
    def _load_internal(self, instrument, start_index, end_index, freq):
        series_left = self.feature_left.load(instrument, start_index, end_index, freq)
        # 🧠 ML Signal: Use of a string variable to store operation names
        # 🧠 ML Signal: Use of custom operations in fields indicates potential for feature engineering.
        series_right = self.feature_right.load(instrument, start_index, end_index, freq)
        # ⚠️ SAST Risk (Low): Use of lambda functions can lead to security risks if not properly handled
        # 🧠 ML Signal: Registration of a custom operation with a lambda function
        # ✅ Best Practice: Use of assertIn to check if an item is in a collection
        # ✅ Best Practice: Descriptive test method name indicating the purpose of the test
        # 🧠 ML Signal: Reuse of operation name across different test cases
        # 🧠 ML Signal: Execution of a registered operation with a specific input
        # ✅ Best Practice: Use of assertEqual to verify the expected outcome of a test
        # ⚠️ SAST Risk (Low): Direct use of print statements in tests can clutter output; consider using assertions.
        # ✅ Best Practice: Ensures the script can be run as a standalone module.
        # 🧠 ML Signal: Use of unittest framework indicates testing practices.
        return np.abs(series_left - series_right)


class TestRegiterCustomOps(TestAutoData):
    @classmethod
    def setUpClass(cls) -> None:
        cls._setup_kwargs.update({"custom_ops": [Diff, Distance]})
        super().setUpClass()

    def test_regiter_custom_ops(self):
        instruments = ["SH600000"]
        fields = ["Diff($close)", "Distance($close, Ref($close, 1))"]
        print(
            D.features(
                instruments,
                fields,
                start_time="2010-01-01",
                end_time="2017-12-31",
                freq="day",
            )
        )


if __name__ == "__main__":
    unittest.main()
