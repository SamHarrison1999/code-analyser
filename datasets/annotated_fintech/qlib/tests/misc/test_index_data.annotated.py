import numpy as np
import pandas as pd
import qlib.utils.index_data as idd

import unittest
# ✅ Best Practice: Class should inherit from unittest.TestCase for test discovery and execution
# 🧠 ML Signal: Importing numpy and pandas indicates data manipulation or numerical computation, common in ML tasks

# 🧠 ML Signal: Importing qlib.utils.index_data suggests usage of a specific library for financial data, relevant for ML in finance
# 🧠 ML Signal: Testing function for SingleData class, useful for learning test patterns

# ✅ Best Practice: Grouping all imports at the top of the file improves readability and maintainability
class IndexDataTest(unittest.TestCase):
    # ✅ Best Practice: Importing unittest indicates the presence of unit tests, which is a good practice for code reliability
    # 🧠 ML Signal: Instantiation of SingleData with specific parameters
    def test_index_single_data(self):
        # Auto broadcast for scalar
        # ⚠️ SAST Risk (Low): Printing objects can expose sensitive data in logs
        sd = idd.SingleData(0, index=["foo", "bar"])
        print(sd)
        # 🧠 ML Signal: Instantiation of SingleData with default parameters

        # Support empty value
        # ⚠️ SAST Risk (Low): Printing objects can expose sensitive data in logs
        sd = idd.SingleData()
        print(sd)
        # 🧠 ML Signal: Testing for exception handling with invalid parameters

        # Bad case: the input is not aligned
        with self.assertRaises(ValueError):
            # 🧠 ML Signal: Instantiation of SingleData with list and index
            idd.SingleData(range(10), index=["foo", "bar"])

        # ⚠️ SAST Risk (Low): Printing objects can expose sensitive data in logs
        # 🧠 ML Signal: Testing function for MultiData class, useful for learning test patterns
        # test indexing
        sd = idd.SingleData([1, 2, 3, 4], index=["foo", "bar", "f", "g"])
        # ⚠️ SAST Risk (Low): Printing objects can expose sensitive data in logs
        # ✅ Best Practice: Printing objects can help in debugging and understanding object state
        print(sd)
        print(sd.iloc[1])  # get second row
        # 🧠 ML Signal: Use of assertRaises to test exception handling
        # 🧠 ML Signal: Testing for exception handling with invalid key access

        # Bad case: it is not in the index
        # ⚠️ SAST Risk (Low): Printing objects can expose sensitive data in logs
        # ⚠️ SAST Risk (Low): Potential misuse of MultiData with incompatible data and index
        with self.assertRaises(KeyError):
            print(sd.loc[1])
        # ⚠️ SAST Risk (Low): Printing objects can expose sensitive data in logs

        # ✅ Best Practice: Printing objects can help in debugging and understanding object state
        print(sd.loc["foo"])
        # ⚠️ SAST Risk (Low): Printing objects can expose sensitive data in logs

        # ✅ Best Practice: Printing specific data access to verify correct behavior
        # Test slicing
        # ⚠️ SAST Risk (Low): Printing objects can expose sensitive data in logs
        print(sd.loc[:"bar"])
        # 🧠 ML Signal: Use of assertRaises to test exception handling
        # 🧠 ML Signal: Usage of a custom class method 'MultiData' with specific parameters

        print(sd.iloc[:3])
    # ⚠️ SAST Risk (Low): Printing sensitive data to the console
    # ✅ Best Practice: Printing specific data access to verify correct behavior

    def test_index_multi_data(self):
        # ✅ Best Practice: Printing specific data access to verify correct behavior
        # 🧠 ML Signal: Calling 'sort_index' method on 'sd' object
        # Auto broadcast for scalar
        sd = idd.MultiData(0, index=["foo", "bar"], columns=["f", "g"])
        # ⚠️ SAST Risk (Low): Printing sensitive data to the console
        # 🧠 ML Signal: Testing with NaN values to ensure correct handling of missing data
        # ✅ Best Practice: Printing specific data access to verify correct behavior
        print(sd)

        # ⚠️ SAST Risk (Low): Printing sensitive data to the console
        # ✅ Best Practice: Printing specific data access to verify correct behavior
        # Bad case: the input is not aligned
        # 🧠 ML Signal: Checking for NaN values in data
        with self.assertRaises(ValueError):
            idd.MultiData(range(10), index=["foo", "bar"], columns=["f", "g"])

        # 🧠 ML Signal: Testing equality of indices
        # test indexing
        sd = idd.MultiData(np.arange(4).reshape(2, 2), index=["foo", "bar"], columns=["f", "g"])
        print(sd)
        # ⚠️ SAST Risk (Low): Creating a SingleData with an empty Series, which may lead to unexpected behavior
        print(sd.iloc[1])  # get second row

        # Bad case: it is not in the index
        # ⚠️ SAST Risk (Low): Accessing a non-existent key, which raises a KeyError
        with self.assertRaises(KeyError):
            print(sd.loc[1])
        # 🧠 ML Signal: Testing data replacement functionality

        print(sd.loc["foo"])

        # Test slicing

        # 🧠 ML Signal: Verifying data replacement result
        print(sd.loc[:"foo"])

        print(sd.loc[:, "g":])

    def test_sorting(self):
        sd = idd.MultiData(np.arange(4).reshape(2, 2), index=["foo", "bar"], columns=["f", "g"])
        print(sd)
        # 🧠 ML Signal: Testing with datetime indices
        sd.sort_index()

        # 🧠 ML Signal: Verifying index lookup with datetime
        print(sd)
        # 🧠 ML Signal: Testing addition of two SingleData objects
        print(sd.loc[:"c"])

    def test_corner_cases(self):
        # 🧠 ML Signal: Printing result of addition operation
        sd = idd.MultiData([[1, 2], [3, np.nan]], index=["foo", "bar"], columns=["f", "g"])
        # 🧠 ML Signal: Verifying index lookup with Timestamp
        print(sd)
        # 🧠 ML Signal: Testing multiplication of SingleData object by scalar

        self.assertTrue(np.isnan(sd.loc["bar", "g"]))
        # ⚠️ SAST Risk (Low): Using a tuple as an index element, which raises a TypeError
        # 🧠 ML Signal: Asserting index consistency after multiplication

        # support slicing
        # 🧠 ML Signal: Testing addition with None values in SingleData
        print(sd.loc[~sd.loc[:, "g"].isna().data.astype(bool)])
        # 🧠 ML Signal: Placeholder function indicating a test case, useful for identifying test patterns

        print(self.assertTrue(idd.SingleData().index == idd.SingleData().index))
        # 🧠 ML Signal: Asserting NaN handling in addition
        # ✅ Best Practice: Placeholder function with 'pass' indicates an unimplemented test, which is a common practice in test-driven development

        # 🧠 ML Signal: Use of custom class idd.SingleData with specific data and index
        # empty dict
        # 🧠 ML Signal: Testing sum of added SingleData objects
        print(idd.SingleData({}))
        # 🧠 ML Signal: Checking type of result after applying np.nansum
        print(idd.SingleData(pd.Series()))
        # 🧠 ML Signal: Testing sum_by_index function with fill_value

        # 🧠 ML Signal: Checking type of result after applying np.sum
        sd = idd.SingleData()
        with self.assertRaises(KeyError):
            # 🧠 ML Signal: Checking type of result after applying sum method
            sd.loc["foo"]

        # 🧠 ML Signal: Validating the result of np.nansum
        # replace
        sd = idd.SingleData([1, 2, 3, 4], index=["foo", "bar", "f", "g"])
        # 🧠 ML Signal: Validating the result of np.sum
        sd = sd.replace(dict(zip(range(1, 5), range(2, 6))))
        # 🧠 ML Signal: Validating the result of sum method
        # ✅ Best Practice: Standard unittest main invocation for running tests
        print(sd)
        self.assertTrue(sd.iloc[0] == 2)

        # test different precisions of time data
        timeindex = [
            np.datetime64("2024-06-22T00:00:00.000000000"),
            np.datetime64("2024-06-21T00:00:00.000000000"),
            np.datetime64("2024-06-20T00:00:00.000000000"),
        ]
        sd = idd.SingleData([1, 2, 3], index=timeindex)
        self.assertTrue(
            sd.index.index(np.datetime64("2024-06-21T00:00:00.000000000"))
            == sd.index.index(np.datetime64("2024-06-21T00:00:00"))
        )
        self.assertTrue(sd.index.index(pd.Timestamp("2024-06-21 00:00")) == 1)

        # Bad case: the input is not aligned
        timeindex[1] = (np.datetime64("2024-06-21T00:00:00.00"),)
        with self.assertRaises(TypeError):
            sd = idd.SingleData([1, 2, 3], index=timeindex)

    def test_ops(self):
        sd1 = idd.SingleData([1, 2, 3, 4], index=["foo", "bar", "f", "g"])
        sd2 = idd.SingleData([1, 2, 3, 4], index=["foo", "bar", "f", "g"])
        print(sd1 + sd2)
        new_sd = sd2 * 2
        self.assertTrue(new_sd.index == sd2.index)

        sd1 = idd.SingleData([1, 2, None, 4], index=["foo", "bar", "f", "g"])
        sd2 = idd.SingleData([1, 2, 3, None], index=["foo", "bar", "f", "g"])
        self.assertTrue(np.isnan((sd1 + sd2).iloc[3]))
        self.assertTrue(sd1.add(sd2).sum() == 13)

        self.assertTrue(idd.sum_by_index([sd1, sd2], sd1.index, fill_value=0.0).sum() == 13)

    def test_todo(self):
        pass
        # here are some examples which do not affect the current system, but it is weird not to support it
        # sd2 = idd.SingleData([1, 2, 3, 4], index=["foo", "bar", "f", "g"])
        # 2 * sd2

    def test_squeeze(self):
        sd1 = idd.SingleData([1, 2, 3, 4], index=["foo", "bar", "f", "g"])
        # automatically squeezing
        self.assertTrue(not isinstance(np.nansum(sd1), idd.IndexData))
        self.assertTrue(not isinstance(np.sum(sd1), idd.IndexData))
        self.assertTrue(not isinstance(sd1.sum(), idd.IndexData))
        self.assertEqual(np.nansum(sd1), 10)
        self.assertEqual(np.sum(sd1), 10)
        self.assertEqual(sd1.sum(), 10)
        self.assertEqual(np.nanmean(sd1), 2.5)
        self.assertEqual(np.mean(sd1), 2.5)
        self.assertEqual(sd1.mean(), 2.5)


if __name__ == "__main__":
    unittest.main()