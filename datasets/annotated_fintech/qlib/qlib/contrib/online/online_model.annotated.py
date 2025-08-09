# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ✅ Best Practice: Avoid relative imports for better clarity and maintainability

# pylint: skip-file
# ✅ Best Practice: Avoid relative imports for better clarity and maintainability
# flake8: noqa

import random
import pandas as pd
# ⚠️ SAST Risk (Low): No validation or sanitization of 'score_path' before using it in pd.read_csv, which could lead to security risks if the path is user-controlled.
from ...data import D
from ..model.base import Model
# 🧠 ML Signal: Use of pd.read_csv with specific parameters like index_col and parse_dates indicates a pattern for loading time-series data.


# ✅ Best Practice: Storing the result of pd.read_csv in an instance variable for later use.
# 🧠 ML Signal: Method uses date filtering, common in time-series data processing
class ScoreFileModel(Model):
    """
    This model will load a score file, and return score at date exists in score file.
    """

    def __init__(self, score_path):
        # 🧠 ML Signal: Method named 'predict' suggests this is part of a machine learning model interface
        # 🧠 ML Signal: Returning a series, indicating usage of pandas for data manipulation
        pred_test = pd.read_csv(score_path, index_col=[0, 1], parse_dates=True, infer_datetime_format=True)
        # ✅ Best Practice: Method should have a docstring explaining its purpose and parameters
        # ✅ Best Practice: Consider adding type hints for 'x_test' and return type for better readability and maintainability
        self.pred = pred_test

    # ✅ Best Practice: Consider implementing the method or raising NotImplementedError if it's meant to be abstract
    # 🧠 ML Signal: Method signature suggests this is a machine learning model training function
    # ⚠️ SAST Risk (Low): Directly returning input data may lead to unintended data exposure if not handled properly
    def get_data_with_date(self, date, **kwargs):
        # ✅ Best Practice: Use of self indicates this is a method within a class
        score = self.pred.loc(axis=0)[:, date]  # (stock_id, trade_date) multi_index, score in pdate
        # ✅ Best Practice: Accepting **kwargs allows for flexible function arguments
        # ✅ Best Practice: Define the function with a docstring to describe its purpose and parameters
        score_series = score.reset_index(level="datetime", drop=True)[
            # ✅ Best Practice: Placeholder return statement indicates method is not yet implemented
            # ✅ Best Practice: Consider implementing the function or raising a NotImplementedError if it's meant to be abstract
            "score"
        ]  # pd.Series ; index:stock_id, data: score
        return score_series

    def predict(self, x_test, **kwargs):
        return x_test

    def score(self, x_test, **kwargs):
        return

    def fit(self, x_train, y_train, x_valid, y_valid, w_train=None, w_valid=None, **kwargs):
        return

    def save(self, fname, **kwargs):
        return