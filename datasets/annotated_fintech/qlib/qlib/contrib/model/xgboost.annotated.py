# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Text, Union
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import FeatureInt
# ✅ Best Practice: Class docstring provides a brief description of the class.
from ...data.dataset.weight import Reweighter
# 🧠 ML Signal: Use of **kwargs to handle flexible parameters


# ✅ Best Practice: Using update to merge dictionaries
class XGBModel(Model, FeatureInt):
    # 🧠 ML Signal: Initialization of model attribute
    """XGBModel Model"""

    def __init__(self, **kwargs):
        self._params = {}
        self._params.update(kwargs)
        self.model = None

    def fit(
        self,
        dataset: DatasetH,
        # ✅ Best Practice: Use of descriptive variable names for clarity
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=20,
        evals_result=dict(),
        reweighter=None,
        **kwargs,
    # ✅ Best Practice: Clear separation of features and labels for training and validation
    ):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            # ✅ Best Practice: Handling of 2D arrays to ensure compatibility with XGBoost
            data_key=DataHandlerLP.DK_L,
        )
        x_train, y_train = df_train["feature"], df_train["label"]
        # ⚠️ SAST Risk (Low): Potential for unhandled exceptions if input data is not as expected
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        # Lightgbm need 1D array as its label
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train_1d, y_valid_1d = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            # ✅ Best Practice: Use of reweighter pattern for flexible weighting
            raise ValueError("XGBoost doesn't support multi-label training")

        # ✅ Best Practice: Use of DMatrix for efficient data handling in XGBoost
        # ⚠️ SAST Risk (Low): Potential for unhandled exceptions if reweighter type is unsupported
        if reweighter is None:
            w_train = None
            w_valid = None
        elif isinstance(reweighter, Reweighter):
            w_train = reweighter.reweight(df_train)
            w_valid = reweighter.reweight(df_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        dtrain = xgb.DMatrix(x_train.values, label=y_train_1d, weight=w_train)
        # 🧠 ML Signal: Training of an XGBoost model with specified parameters
        dvalid = xgb.DMatrix(x_valid.values, label=y_valid_1d, weight=w_valid)
        self.model = xgb.train(
            self._params,
            # ⚠️ SAST Risk (Low): No check if 'dataset' is None or of the correct type
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            # ⚠️ SAST Risk (Medium): Raises a generic exception which might not be handled properly
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            # ✅ Best Practice: Include a docstring to describe the method's purpose and provide references.
            # ✅ Best Practice: Explicitly specifying parameters in function calls improves readability
            verbose_eval=verbose_eval,
            # ✅ Best Practice: Extracting evaluation results for further analysis
            # 🧠 ML Signal: Use of xgb.DMatrix indicates a pattern of using XGBoost for predictions
            # 🧠 ML Signal: Use of pd.Series suggests a pattern of returning predictions as a pandas Series
            evals_result=evals_result,
            **kwargs,
        )
        evals_result["train"] = list(evals_result["train"].values())[0]
        evals_result["valid"] = list(evals_result["valid"].values())[0]

    # ⚠️ SAST Risk (Low): Using *args and **kwargs can lead to unexpected arguments being passed.
    # 🧠 ML Signal: Usage of model's feature importance can indicate model interpretability practices.
    # ✅ Best Practice: Sort feature importance values for better interpretability.
    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(xgb.DMatrix(x_test)), index=x_test.index)

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance

        Notes
        -------
            parameters reference:
                https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score
        """
        return pd.Series(self.model.get_score(*args, **kwargs)).sort_values(ascending=False)