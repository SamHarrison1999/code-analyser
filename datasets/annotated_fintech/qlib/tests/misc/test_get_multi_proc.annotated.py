#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import unittest

# 🧠 ML Signal: Function definition with parameters, useful for understanding function usage patterns
import qlib

# ✅ Best Practice: Define a test class that inherits from unittest.TestCase for better organization of test cases.
from qlib.data import D

# ⚠️ SAST Risk (Low): Initialization of external library without error handling
from qlib.tests import TestAutoData

# 🧠 ML Signal: Initialization of a library with specific parameters, indicating usage patterns
# ✅ Best Practice: Use of class inheritance to promote code reuse and organization
from multiprocessing import Pool

# ✅ Best Practice: Use setUp method to initialize any state before each test method is run.

# 🧠 ML Signal: Return statement with method chaining, indicating data processing patterns
# ✅ Best Practice: Constants are defined in uppercase to indicate immutability


# 🧠 ML Signal: Initialization of qlib with a specific provider and region can be a pattern for setting up ML environments.
def get_features(fields):
    qlib.init(
        provider_uri=TestAutoData.provider_uri,
        expression_cache=None,
        dataset_cache=None,
        joblib_backend="loky",
    )
    return D.features(D.instruments("csi300"), fields)


# ✅ Best Practice: Use descriptive method names for test cases to indicate what is being tested.

# ✅ Best Practice: Use of multiprocessing.Pool for parallel execution


# 🧠 ML Signal: Accessing data using D.features is a common pattern in data retrieval for ML tasks.
class TestGetData(TestAutoData):
    FIELDS = "$open,$close,$high,$low,$volume,$factor,$change".split(",")
    # ⚠️ SAST Risk (Low): Ensure that the data returned is validated to prevent unexpected results or errors.

    # 🧠 ML Signal: Usage of apply_async for asynchronous task execution
    def test_multi_proc(self):
        """
        For testing if it will raise error
        # 🧠 ML Signal: Collecting results from asynchronous tasks
        """
        # ✅ Best Practice: Use descriptive method names for test cases to indicate what is being tested.
        iter_n = 2
        # ✅ Best Practice: Properly closing the pool to free resources
        # ✅ Best Practice: Ensuring all worker processes have completed
        # 🧠 ML Signal: Retrieving data for a specific time range is a common pattern in time-series analysis.
        # ⚠️ SAST Risk (Low): Validate the length of the data to ensure it matches expected results.
        # ✅ Best Practice: Use the standard unittest main entry point to run the tests.
        # 🧠 ML Signal: Standard pattern for running unit tests
        pool = Pool(iter_n)

        res = []
        for _ in range(iter_n):
            res.append(pool.apply_async(get_features, (self.FIELDS,), {}))

        for r in res:
            print(r.get())

        pool.close()
        pool.join()


if __name__ == "__main__":
    unittest.main()
