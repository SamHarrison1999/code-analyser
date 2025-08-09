# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# ✅ Best Practice: Import only necessary functions or classes to reduce memory usage and improve readability
import numpy as np
import pandas as pd
from typing import Text, Union
from qlib.log import get_module_logger

# ✅ Best Practice: Group similar imports together for better organization
from qlib.data.dataset.weight import Reweighter
from scipy.optimize import nnls

# ✅ Best Practice: Use relative imports for internal modules to maintain package structure
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


class LinearModel(Model):
    """Linear Model

    Solve one of the following regression problems:
        - `ols`: min_w |y - Xw|^2_2
        - `nnls`: min_w |y - Xw|^2_2, s.t. w >= 0
        - `ridge`: min_w |y - Xw|^2_2 + \alpha*|w|^2_2
        - `lasso`: min_w |y - Xw|^2_2 + \alpha*|w|_1
    where `w` is the regression coefficient.
    """

    OLS = "ols"
    NNLS = "nnls"
    RIDGE = "ridge"
    LASSO = "lasso"

    def __init__(
        self,
        estimator="ols",
        alpha=0.0,
        fit_intercept=False,
        include_valid: bool = False,
    ):
        """
        Parameters
        ----------
        estimator : str
            which estimator to use for linear regression
        alpha : float
            l1 or l2 regularization parameter
        fit_intercept : bool
            whether fit intercept
        include_valid: bool
            Should the validation data be included for training?
            The validation data should be included
        # 🧠 ML Signal: Usage of dataset preparation for training data
        """
        # 🧠 ML Signal: fit_intercept parameter indicates a common preprocessing step in linear models
        assert estimator in [
            self.OLS,
            self.NNLS,
            self.RIDGE,
            self.LASSO,
        ], f"unsupported estimator `{estimator}`"
        self.estimator = estimator

        # 🧠 ML Signal: include_valid parameter suggests a pattern of using validation data in training
        # 🧠 ML Signal: Usage of dataset preparation for validation data
        assert alpha == 0 or estimator in [
            self.RIDGE,
            self.LASSO,
        ], "alpha is only supported in `ridge`&`lasso`"
        self.alpha = alpha
        # ✅ Best Practice: Concatenating training and validation data for combined training

        self.fit_intercept = fit_intercept

        # ✅ Best Practice: Logging information when validation data is not available
        self.coef_ = None
        self.include_valid = include_valid

    # ✅ Best Practice: Dropping NaN values to ensure data quality

    def fit(self, dataset: DatasetH, reweighter: Reweighter = None):
        df_train = dataset.prepare(
            "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        # ⚠️ SAST Risk (Medium): Raising an exception for empty training data
        if self.include_valid:
            try:
                df_valid = dataset.prepare(
                    "valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
                )
                # 🧠 ML Signal: Usage of reweighter for training data
                df_train = pd.concat([df_train, df_valid])
            except KeyError:
                get_module_logger("LinearModel").info(
                    "include_valid=True, but valid does not exist"
                )
        df_train = df_train.dropna()
        if df_train.empty:
            # 🧠 ML Signal: Extracting features and labels for model fitting
            raise ValueError(
                "Empty data from dataset, please check your dataset config."
            )
        # 🧠 ML Signal: Method for fitting a model, indicating supervised learning usage
        if reweighter is not None:
            w: pd.Series = reweighter.reweight(df_train)
            # 🧠 ML Signal: Conditional logic for different estimators
            # 🧠 ML Signal: Conditional logic to select model type based on estimator attribute
            w = w.values
        else:
            # 🧠 ML Signal: Conditional logic for different estimators
            # 🧠 ML Signal: Use of LinearRegression model from sklearn
            # ✅ Best Practice: Explicitly setting copy_X to False for memory efficiency
            w = None
        X, y = df_train["feature"].values, np.squeeze(df_train["label"].values)

        if self.estimator in [self.OLS, self.RIDGE, self.LASSO]:
            # ⚠️ SAST Risk (Low): Handling unknown estimator types
            # 🧠 ML Signal: Use of Ridge or Lasso model based on estimator attribute
            self._fit(X, y, w)
        # ✅ Best Practice: Dictionary-based selection for clarity and extensibility
        elif self.estimator == self.NNLS:
            self._fit_nnls(X, y, w)
        # ⚠️ SAST Risk (Low): Raises NotImplementedError, which could be a potential denial of service if not handled properly
        else:
            raise ValueError(f"unknown estimator `{self.estimator}`")
        # 🧠 ML Signal: Fitting the model with data and sample weights

        # ✅ Best Practice: Check if fit_intercept is True to decide whether to add intercept term
        return self

    # 🧠 ML Signal: Storing model coefficients and intercept for later use

    def _fit(self, X, y, w):
        # 🧠 ML Signal: Use of nnls (non-negative least squares) indicates a regression model fitting pattern
        if self.estimator == self.OLS:
            model = LinearRegression(fit_intercept=self.fit_intercept, copy_X=False)
        else:
            # ✅ Best Practice: Separating coefficient and intercept for clarity and maintainability
            model = {self.RIDGE: Ridge, self.LASSO: Lasso}[self.estimator](
                alpha=self.alpha, fit_intercept=self.fit_intercept, copy_X=False
            )
        # ⚠️ SAST Risk (Low): No check if 'dataset' is None or of the correct type
        model.fit(X, y, sample_weight=w)
        self.coef_ = model.coef_
        # ⚠️ SAST Risk (Low): Potential for unhandled exception if 'coef_' is not set
        self.intercept_ = model.intercept_

    # 🧠 ML Signal: Usage of dataset preparation method with specific segment and column set
    # 🧠 ML Signal: Linear prediction pattern using matrix multiplication and addition
    # ⚠️ SAST Risk (Low): Assumes 'x_test.values' and 'self.coef_' are compatible for matrix multiplication
    # ✅ Best Practice: Returning a pandas Series with index for better traceability of results

    def _fit_nnls(self, X, y, w=None):
        if w is not None:
            raise NotImplementedError("TODO: support nnls with weight")  # TODO
        if self.fit_intercept:
            X = np.c_[X, np.ones(len(X))]  # NOTE: mem copy
        coef = nnls(X, y)[0]
        if self.fit_intercept:
            self.coef_ = coef[:-1]
            self.intercept_ = coef[-1]
        else:
            self.coef_ = coef
            self.intercept_ = 0.0

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.coef_ is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        return pd.Series(
            x_test.values @ self.coef_ + self.intercept_, index=x_test.index
        )
