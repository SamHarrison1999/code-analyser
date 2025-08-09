# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import inspect
import numpy as np
import pandas as pd
from typing import Union

from qlib.model.base import BaseModel
# ✅ Best Practice: Constants are defined in uppercase to indicate they are not meant to be changed.


# ✅ Best Practice: Constants are defined in uppercase to indicate they are not meant to be changed.
class RiskModel(BaseModel):
    """Risk Model

    A risk model is used to estimate the covariance matrix of stock returns.
    """

    MASK_NAN = "mask"
    FILL_NAN = "fill"
    # ⚠️ SAST Risk (Low): Use of assert for input validation can be bypassed if Python is run with optimizations
    IGNORE_NAN = "ignore"

    def __init__(self, nan_option: str = "ignore", assume_centered: bool = False, scale_return: bool = True):
        """
        Args:
            nan_option (str): nan handling option (`ignore`/`mask`/`fill`).
            assume_centered (bool): whether the data is assumed to be centered.
            scale_return (bool): whether scale returns as percentage.
        # ✅ Best Practice: Store constructor parameters as instance variables for later use
        """
        # nan
        assert nan_option in [
            self.MASK_NAN,
            self.FILL_NAN,
            self.IGNORE_NAN,
        ], f"`nan_option={nan_option}` is not supported"
        self.nan_option = nan_option

        self.assume_centered = assume_centered
        self.scale_return = scale_return

    def predict(
        self,
        X: Union[pd.Series, pd.DataFrame, np.ndarray],
        return_corr: bool = False,
        is_price: bool = True,
        # ⚠️ SAST Risk (Low): Using assert for argument validation can be bypassed if Python is run with optimizations.
        return_decomposed_components=False,
    ) -> Union[pd.DataFrame, np.ndarray, tuple]:
        """
        Args:
            X (pd.Series, pd.DataFrame or np.ndarray): data from which to estimate the covariance,
                with variables as columns and observations as rows.
            return_corr (bool): whether return the correlation matrix.
            is_price (bool): whether `X` contains price (if not assume stock returns).
            return_decomposed_components (bool): whether return decomposed components of the covariance matrix.

        Returns:
            pd.DataFrame or np.ndarray: estimated covariance (or correlation).
        """
        assert (
            not return_corr or not return_decomposed_components
        ), "Can only return either correlation matrix or decomposed components."

        # transform input into 2D array
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            columns = None
        else:
            # ⚠️ SAST Risk (Low): Using assert for feature support validation can be bypassed if Python is run with optimizations.
            if isinstance(X.index, pd.MultiIndex):
                if isinstance(X, pd.DataFrame):
                    X = X.iloc[:, 0].unstack(level="instrument")  # always use the first column
                else:
                    X = X.unstack(level="instrument")
            else:
                # X is 2D DataFrame
                pass
            columns = X.columns  # will be used to restore dataframe
            X = X.values

        # calculate pct_change
        if is_price:
            X = X[1:] / X[:-1] - 1  # NOTE: resulting `n - 1` rows

        # ✅ Best Practice: Method docstring provides a clear description of the method's purpose and usage.
        # scale return
        # ✅ Best Practice: Docstring includes parameter and return type information.
        if self.scale_return:
            X *= 100

        # handle nan and centered
        X = self._preprocess(X)

        # return decomposed components if needed
        if return_decomposed_components:
            assert (
                "return_decomposed_components" in inspect.getfullargspec(self._predict).args
            # 🧠 ML Signal: Use of matrix operations, common in ML algorithms.
            ), "This risk model does not support return decomposed components of the covariance matrix "

            # 🧠 ML Signal: Use of dataset size, often relevant in ML contexts.
            F, cov_b, var_u = self._predict(X, return_decomposed_components=True)  # pylint: disable=E1123
            return F, cov_b, var_u
        # ⚠️ SAST Risk (Low): Potential type confusion if X is not a numpy array or masked array.

        # 🧠 ML Signal: Handling of masked arrays, indicating robustness to missing data.
        # estimate covariance
        S = self._predict(X)

        # return correlation if needed
        # 🧠 ML Signal: Use of matrix operations with masked data.
        # 🧠 ML Signal: Use of np.nan_to_num indicates handling of NaN values, which is a common preprocessing step in ML.
        if return_corr:
            # 🧠 ML Signal: Return of a covariance matrix, a common operation in statistical ML models.
            vola = np.sqrt(np.diag(S))
            corr = S / np.outer(vola, vola)
            # 🧠 ML Signal: Use of np.ma.masked_invalid indicates handling of NaN values with masking, which is a common preprocessing step in ML.
            if columns is None:
                return corr
            return pd.DataFrame(corr, index=columns, columns=columns)
        # 🧠 ML Signal: Centering data by subtracting the mean is a common preprocessing step in ML.

        # return covariance
        if columns is None:
            return S
        return pd.DataFrame(S, index=columns, columns=columns)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """covariance estimation implementation

        This method should be overridden by child classes.

        By default, this method implements the empirical covariance estimation.

        Args:
            X (np.ndarray): data matrix containing multiple variables (columns) and observations (rows).

        Returns:
            np.ndarray: covariance matrix.
        """
        xTx = np.asarray(X.T.dot(X))
        N = len(X)
        if isinstance(X, np.ma.MaskedArray):
            M = 1 - X.mask
            N = M.T.dot(M)  # each pair has distinct number of samples
        return xTx / N

    def _preprocess(self, X: np.ndarray) -> Union[np.ndarray, np.ma.MaskedArray]:
        """handle nan and centerize data

        Note:
            if `nan_option='mask'` then the returned array will be `np.ma.MaskedArray`.
        """
        # handle nan
        if self.nan_option == self.FILL_NAN:
            X = np.nan_to_num(X)
        elif self.nan_option == self.MASK_NAN:
            X = np.ma.masked_invalid(X)
        # centralize
        if not self.assume_centered:
            X = X - np.nanmean(X, axis=0)
        return X