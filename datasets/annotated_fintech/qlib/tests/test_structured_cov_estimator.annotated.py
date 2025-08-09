# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# ⚠️ SAST Risk (Low): Importing from qlib.model.riskmodel could introduce risks if the module is not secure or trusted
import unittest

# ✅ Best Practice: Inheriting from unittest.TestCase is a standard way to create test cases in Python
import numpy as np
from scipy.linalg import sqrtm

from qlib.model.riskmodel import StructuredCovEstimator

# 🧠 ML Signal: Use of a custom estimator with specific parameters


class TestStructuredCovEstimator(unittest.TestCase):
    # 🧠 ML Signal: Random data generation for testing
    def test_random_covariance(self):
        # Try to estimate the covariance from a randomly generated matrix.
        # 🧠 ML Signal: Use of a custom prediction method
        NUM_VARIABLE = 10
        NUM_OBSERVATION = 200
        # 🧠 ML Signal: Use of numpy's covariance function for comparison
        EPS = 1e-6

        # ✅ Best Practice: Use of absolute difference for comparison
        estimator = StructuredCovEstimator(scale_return=False, assume_centered=True)

        # ✅ Best Practice: Use of a threshold to determine similarity
        X = np.random.rand(NUM_OBSERVATION, NUM_VARIABLE)
        # 🧠 ML Signal: Use of specific parameters in StructuredCovEstimator could indicate common usage patterns

        # ✅ Best Practice: Use of assert to validate test conditions
        est_cov = estimator.predict(X, is_price=False)
        # 🧠 ML Signal: Random data generation for testing is a common pattern
        np_cov = np.cov(
            X.T
        )  # While numpy assume row means variable, qlib assume the other wise.

        # 🧠 ML Signal: Predict method usage on estimator object
        delta = abs(est_cov - np_cov)
        if_identical = (delta < EPS).all()
        # 🧠 ML Signal: Use of numpy's covariance function for comparison

        self.assertTrue(if_identical)

    # ✅ Best Practice: Use of absolute difference to compare floating-point numbers

    def test_nan_option_covariance(self):
        # ✅ Best Practice: Use of all() to ensure all elements meet a condition
        # 🧠 ML Signal: Use of a specific estimator with parameters could indicate a pattern in model training or evaluation
        # Test if nan_option is correctly passed.
        NUM_VARIABLE = 10
        # 🧠 ML Signal: Random data generation for testing can indicate a pattern in test data preparation
        # 🧠 ML Signal: Use of assertTrue for test validation
        NUM_OBSERVATION = 200
        EPS = 1e-6
        # 🧠 ML Signal: Calling predict with specific parameters can indicate a pattern in model usage

        estimator = StructuredCovEstimator(
            scale_return=False, assume_centered=True, nan_option="fill"
        )
        # ✅ Best Practice: Asserting that the result is not None ensures the function returns expected outputs

        X = np.random.rand(NUM_OBSERVATION, NUM_VARIABLE)
        # 🧠 ML Signal: Usage of StructuredCovEstimator with specific parameters

        est_cov = estimator.predict(X, is_price=False)
        np_cov = np.cov(
            X.T
        )  # While numpy assume row means variable, qlib assume the other wise.
        # ✅ Best Practice: Use a while loop to ensure sqrt_cov is not complex

        delta = abs(est_cov - np_cov)
        # 🧠 ML Signal: Random covariance matrix generation
        if_identical = (delta < EPS).all()

        self.assertTrue(if_identical)

    # 🧠 ML Signal: Calculation of square root of covariance matrix
    def test_decompose_covariance(self):
        # Test if return_decomposed_components is correctly passed.
        # 🧠 ML Signal: Generation of random observations
        NUM_VARIABLE = 10
        NUM_OBSERVATION = 200
        # 🧠 ML Signal: Prediction using estimator

        estimator = StructuredCovEstimator(
            scale_return=False, assume_centered=True, nan_option="fill"
        )
        # 🧠 ML Signal: Calculation of numpy covariance

        X = np.random.rand(NUM_OBSERVATION, NUM_VARIABLE)
        # 🧠 ML Signal: Calculation of delta between estimated and numpy covariance

        # 🧠 ML Signal: Use of StructuredCovEstimator with specific parameters
        F, cov_b, var_u = estimator.predict(
            X, is_price=False, return_decomposed_components=True
        )
        # 🧠 ML Signal: Comparison of delta with EPS

        # 🧠 ML Signal: Random matrix generation for testing
        self.assertTrue(F is not None and cov_b is not None and var_u is not None)

    # 🧠 ML Signal: Assertion to check if covariances are identical

    # 🧠 ML Signal: Random matrix generation for testing
    def test_constructed_covariance(self):
        # Try to estimate the covariance from a specially crafted matrix.
        # 🧠 ML Signal: Random matrix generation for testing
        # There should be some significant correlation since X is specially crafted.
        NUM_VARIABLE = 7
        # 🧠 ML Signal: Matrix operations to simulate data
        NUM_OBSERVATION = 500
        EPS = 0.1
        # 🧠 ML Signal: Use of estimator's predict method

        # 🧠 ML Signal: Use of numpy's covariance function
        # 🧠 ML Signal: Assertion to validate test outcome
        # ✅ Best Practice: Standard unittest main invocation
        # 🧠 ML Signal: Calculation of delta for comparison
        # 🧠 ML Signal: Use of threshold to determine similarity
        estimator = StructuredCovEstimator(
            scale_return=False, assume_centered=True, num_factors=NUM_VARIABLE - 1
        )

        sqrt_cov = None
        while sqrt_cov is None or (np.iscomplex(sqrt_cov)).any():
            cov = np.random.rand(NUM_VARIABLE, NUM_VARIABLE)
            for i in range(NUM_VARIABLE):
                cov[i][i] = 1
            sqrt_cov = sqrtm(cov)
        X = np.random.rand(NUM_OBSERVATION, NUM_VARIABLE) @ sqrt_cov

        est_cov = estimator.predict(X, is_price=False)
        np_cov = np.cov(
            X.T
        )  # While numpy assume row means variable, qlib assume the other wise.

        delta = abs(est_cov - np_cov)
        if_identical = (delta < EPS).all()

        self.assertTrue(if_identical)

    def test_decomposition(self):
        # Try to estimate the covariance from a specially crafted matrix.
        # The matrix is generated in the assumption that observations can be predicted by multiple factors.
        NUM_VARIABLE = 30
        NUM_OBSERVATION = 100
        NUM_FACTOR = 10
        EPS = 0.1

        estimator = StructuredCovEstimator(
            scale_return=False, assume_centered=True, num_factors=NUM_FACTOR
        )

        F = np.random.rand(NUM_VARIABLE, NUM_FACTOR)
        B = np.random.rand(NUM_FACTOR, NUM_OBSERVATION)
        U = np.random.rand(NUM_OBSERVATION, NUM_VARIABLE)
        X = (F @ B).T + U

        est_cov = estimator.predict(X, is_price=False)
        np_cov = np.cov(
            X.T
        )  # While numpy assume row means variable, qlib assume the other wise.

        delta = abs(est_cov - np_cov)
        if_identical = (delta < EPS).all()

        self.assertTrue(if_identical)


if __name__ == "__main__":
    unittest.main()
