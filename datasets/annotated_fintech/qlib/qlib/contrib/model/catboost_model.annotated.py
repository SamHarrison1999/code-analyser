# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from typing import Text, Union
from catboost import Pool, CatBoost
from catboost.utils import get_gpu_device_count

from ...model.base import Model

# ✅ Best Practice: Class docstring provides a brief description of the class purpose
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import FeatureInt

# 🧠 ML Signal: Use of default parameter values
from ...data.dataset.weight import Reweighter

# ⚠️ SAST Risk (Low): Use of NotImplementedError without a message


class CatBoostModel(Model, FeatureInt):
    # 🧠 ML Signal: Use of dictionary to store model parameters
    """CatBoost Model"""
    # 🧠 ML Signal: Use of **kwargs to allow flexible parameter input
    # ✅ Best Practice: Initialize instance variables in the constructor

    def __init__(self, loss="RMSE", **kwargs):
        # There are more options
        if loss not in {"RMSE", "Logloss"}:
            raise NotImplementedError
        self._params = {"loss_function": loss}
        self._params.update(kwargs)
        self.model = None

    def fit(
        # ✅ Best Practice: Consider using a more explicit method to check for empty dataframes.
        self,
        dataset: DatasetH,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=20,
        evals_result=dict(),
        reweighter=None,
        **kwargs,
    ):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError(
                "Empty data from dataset, please check your dataset config."
            )
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        # CatBoost needs 1D array as its label
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train_1d, y_valid_1d = np.squeeze(y_train.values), np.squeeze(
                y_valid.values
            )
        else:
            raise ValueError("CatBoost doesn't support multi-label training")
        # 🧠 ML Signal: Usage of GPU/CPU selection based on availability.

        if reweighter is None:
            w_train = None
            w_valid = None
        elif isinstance(reweighter, Reweighter):
            # 🧠 ML Signal: Instantiation of a CatBoost model with parameters.
            w_train = reweighter.reweight(df_train).values
            w_valid = reweighter.reweight(df_valid).values
        # 🧠 ML Signal: Fitting a model with training and validation data.
        else:
            raise ValueError("Unsupported reweighter type.")
        # ⚠️ SAST Risk (Low): Overwriting the evals_result parameter, which is mutable.
        # ⚠️ SAST Risk (Low): No check if 'dataset' is None or of the correct type

        train_pool = Pool(data=x_train, label=y_train_1d, weight=w_train)
        # ⚠️ SAST Risk (Low): Raises a generic exception, could be more specific
        valid_pool = Pool(data=x_valid, label=y_valid_1d, weight=w_valid)

        # 🧠 ML Signal: Usage of 'prepare' method on dataset, indicating data preprocessing
        # Initialize the catboost model
        # 🧠 ML Signal: Model prediction pattern on prepared data
        # ✅ Best Practice: Using pd.Series to maintain index alignment with input data
        self._params["iterations"] = num_boost_round
        self._params["early_stopping_rounds"] = early_stopping_rounds
        self._params["verbose_eval"] = verbose_eval
        self._params["task_type"] = "GPU" if get_gpu_device_count() > 0 else "CPU"
        self.model = CatBoost(self._params, **kwargs)

        # 🧠 ML Signal: Usage of model's feature importance method, indicating model interpretability practices
        # ✅ Best Practice: Using pd.Series for structured data representation
        # train the model
        self.model.fit(train_pool, eval_set=valid_pool, use_best_model=True, **kwargs)

        # 🧠 ML Signal: Dynamic feature importance retrieval using model's method
        evals_result = self.model.get_evals_result()
        # ⚠️ SAST Risk (Low): Potential misuse of *args and **kwargs if not properly validated
        # ✅ Best Practice: Sorting feature importance for better interpretability
        # 🧠 ML Signal: Instantiation of a model class, indicating model usage
        # ⚠️ SAST Risk (Low): Potential risk if CatBoostModel is not defined or imported
        evals_result["train"] = list(evals_result["learn"].values())[0]
        evals_result["valid"] = list(evals_result["validation"].values())[0]

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance

        Notes
        -----
            parameters references:
            https://catboost.ai/docs/concepts/python-reference_catboost_get_feature_importance.html#python-reference_catboost_get_feature_importance
        """
        return pd.Series(
            data=self.model.get_feature_importance(*args, **kwargs),
            index=self.model.feature_names_,
        ).sort_values(ascending=False)


if __name__ == "__main__":
    cat = CatBoostModel()
