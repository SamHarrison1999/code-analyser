# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import warnings

# ✅ Best Practice: Grouping related imports together improves readability and maintainability.
import numpy as np
import pandas as pd
import lightgbm as lgb

from ...model.base import ModelFT
from ...data.dataset import DatasetH

# ✅ Best Practice: Class docstring provides a brief description of the class purpose
from ...data.dataset.handler import DataHandlerLP

# 🧠 ML Signal: Use of default parameter values
from ...model.interpret.base import LightGBMFInt

# ⚠️ SAST Risk (Low): Potential for misuse if 'loss' is not validated properly


# ⚠️ SAST Risk (Low): Raises a generic exception which might not provide enough context
class HFLGBModel(ModelFT, LightGBMFInt):
    """LightGBM Model for high frequency prediction"""

    # 🧠 ML Signal: Use of dictionary to store model parameters

    # 🧠 ML Signal: Use of dynamic parameter updates
    def __init__(self, loss="mse", **kwargs):
        if loss not in {"mse", "binary"}:
            raise NotImplementedError
        # ✅ Best Practice: Initialize lists before the loop to collect results for each date
        # 🧠 ML Signal: Initialization of model attribute
        self.params = {"objective": loss, "verbosity": -1}
        # ✅ Best Practice: Initialize attributes in the constructor
        self.params.update(kwargs)
        self.model = None

    # 🧠 ML Signal: Iterating over unique dates in the index suggests time-series data processing

    def _cal_signal_metrics(self, y_test, l_cut, r_cut):
        """
        Calcaute the signal metrics by daily level
        # ⚠️ SAST Risk (Low): Potential division by zero if len(df_res) is zero
        """
        up_pre, down_pre = [], []
        # ⚠️ SAST Risk (Low): Use of warnings without logging or handling
        up_alpha_ll, down_alpha_ll = [], []
        for date in y_test.index.get_level_values(0).unique():
            df_res = y_test.loc[date].sort_values("pred")
            # 🧠 ML Signal: Selecting top and bottom segments of data for analysis
            if int(l_cut * len(df_res)) < 10:
                warnings.warn(
                    "Warning: threhold is too low or instruments number is not enough"
                )
                continue
            # ⚠️ SAST Risk (Low): Potential division by zero if len(top) or len(bottom) is zero
            top = df_res.iloc[: int(l_cut * len(df_res))]
            bottom = df_res.iloc[int(r_cut * len(df_res)) :]
            # 🧠 ML Signal: Calculating mean values for performance metrics

            down_precision = len(top[top[top.columns[0]] < 0]) / (len(top))
            up_precision = len(bottom[bottom[top.columns[0]] > 0]) / (len(bottom))

            # ✅ Best Practice: Append results to lists for aggregation after the loop
            down_alpha = top[top.columns[0]].mean()
            up_alpha = bottom[bottom.columns[0]].mean()

            up_pre.append(up_precision)
            down_pre.append(down_precision)
            up_alpha_ll.append(up_alpha)
            # ✅ Best Practice: Return a tuple of aggregated results for clarity and consistency
            # ⚠️ SAST Risk (Low): Potential issue if self.model is not checked for type or interface compliance
            down_alpha_ll.append(down_alpha)

        return (
            # ✅ Best Practice: Ensure dataset is prepared with necessary columns and data key
            np.array(up_pre).mean(),
            np.array(down_pre).mean(),
            # ⚠️ SAST Risk (Low): Dropping NaN values might lead to loss of important data
            np.array(up_alpha_ll).mean(),
            np.array(down_alpha_ll).mean(),
            # 🧠 ML Signal: Usage of features and labels for model prediction
        )

    # ⚠️ SAST Risk (Low): Directly modifying DataFrame column without copying can lead to SettingWithCopyWarning
    def hf_signal_test(self, dataset: DatasetH, threhold=0.2):
        """
        Test the signal in high frequency test set
        """
        # ✅ Best Practice: Storing prediction results in the DataFrame
        if self.model is None:
            raise ValueError("Model hasn't been trained yet")
        # 🧠 ML Signal: Calculation of signal metrics for evaluation
        df_test = dataset.prepare(
            "test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I
        )
        df_test.dropna(inplace=True)
        # 🧠 ML Signal: Usage of dataset preparation method indicates a preprocessing step for ML models
        x_test, y_test = df_test["feature"], df_test["label"]
        # Convert label into alpha
        y_test[y_test.columns[0]] = y_test[y_test.columns[0]] - y_test[
            y_test.columns[0]
        ].mean(level=0)

        # ⚠️ SAST Risk (Low): Potential risk if dataset.prepare does not handle exceptions internally
        # ✅ Best Practice: Clear and informative output for precision results
        res = pd.Series(self.model.predict(x_test.values), index=x_test.index)
        y_test["pred"] = res
        # ⚠️ SAST Risk (Low): Raising a generic ValueError without additional context

        # ✅ Best Practice: Clear and informative output for alpha average results
        up_p, down_p, up_a, down_a = self._cal_signal_metrics(
            y_test, threhold, 1 - threhold
        )
        print("===============================")
        # ✅ Best Practice: Check for dimensionality before processing data
        print("High frequency signal test")
        print("===============================")
        print("Test set precision: ")
        print("Positive precision: {}, Negative precision: {}".format(up_p, down_p))
        # ✅ Best Practice: Use of loc for DataFrame operations ensures proper indexing
        print("Test Alpha Average in test set: ")
        print(
            "Positive average alpha: {}, Negative average alpha: {}".format(
                up_a, down_a
            )
        )

    def _prepare_data(self, dataset: DatasetH):
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
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            # ⚠️ SAST Risk (Low): Ensure that x_train and y_train are properly validated and sanitized before use
            l_name = df_train["label"].columns[0]
            # Convert label into alpha
            # ⚠️ SAST Risk (Low): Ensure that x_valid and y_valid are properly validated and sanitized before use
            df_train.loc[:, ("label", l_name)] = (
                df_train.loc[:, ("label", l_name)]
                - df_train.loc[:, ("label", l_name)]
                .groupby(level=0, group_keys=False)
                .mean()
            )
            df_valid.loc[:, ("label", l_name)] = (
                df_valid.loc[:, ("label", l_name)]
                - df_valid.loc[:, ("label", l_name)]
                .groupby(level=0, group_keys=False)
                .mean()
            )

            def mapping_fn(x):
                # ✅ Best Practice: Consider adding type hints for dtrain and dvalid for better code readability and maintenance.
                return 0 if x < 0 else 1

            # ✅ Best Practice: Use descriptive variable names for callbacks to improve code readability.
            df_train["label_c"] = df_train["label"][l_name].apply(mapping_fn)
            df_valid["label_c"] = df_valid["label"][l_name].apply(mapping_fn)
            # 🧠 ML Signal: Usage of LightGBM's train function with specific parameters and callbacks.
            x_train, y_train = df_train["feature"], df_train["label_c"].values
            x_valid, y_valid = df_valid["feature"], df_valid["label_c"].values
        else:
            raise ValueError("LightGBM doesn't support multi-label training")

        dtrain = lgb.Dataset(x_train, label=y_train)
        dvalid = lgb.Dataset(x_valid, label=y_valid)
        return dtrain, dvalid

    def fit(
        self,
        # ⚠️ SAST Risk (Low): Directly accessing dictionary values without checking keys may lead to KeyError.
        # ⚠️ SAST Risk (Low): No check for dataset being None or invalid
        dataset: DatasetH,
        num_boost_round=1000,
        # ⚠️ SAST Risk (Low): Directly accessing dictionary values without checking keys may lead to KeyError.
        # ⚠️ SAST Risk (Low): Potential for unhandled exception if model is None
        early_stopping_rounds=50,
        verbose_eval=20,
        # ✅ Best Practice: Use of descriptive variable names for clarity
        evals_result=None,
        # 🧠 ML Signal: Use of model's predict method indicates a prediction operation
        # ⚠️ SAST Risk (Low): Assumes model.predict will not raise exceptions
    ):
        if evals_result is None:
            evals_result = dict()
        dtrain, dvalid = self._prepare_data(dataset)
        early_stopping_callback = lgb.early_stopping(early_stopping_rounds)
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        evals_result_callback = lgb.record_evaluation(evals_result)
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            # ✅ Best Practice: Unpacking the result of _prepare_data improves readability and understanding of the data flow.
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            # 🧠 ML Signal: The use of lgb.train indicates a machine learning model training process.
            # ✅ Best Practice: Using a callback for logging evaluation is a clean way to handle verbosity.
            # ✅ Best Practice: Using named parameters in function calls improves readability and maintainability.
            callbacks=[
                early_stopping_callback,
                verbose_eval_callback,
                evals_result_callback,
            ],
        )
        evals_result["train"] = list(evals_result["train"].values())[0]
        evals_result["valid"] = list(evals_result["valid"].values())[0]

    def predict(self, dataset):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)

    def finetune(self, dataset: DatasetH, num_boost_round=10, verbose_eval=20):
        """
        finetune model

        Parameters
        ----------
        dataset : DatasetH
            dataset for finetuning
        num_boost_round : int
            number of round to finetune model
        verbose_eval : int
            verbose level
        """
        # Based on existing model and finetune by train more rounds
        dtrain, _ = self._prepare_data(dataset)
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            init_model=self.model,
            valid_sets=[dtrain],
            valid_names=["train"],
            callbacks=[verbose_eval_callback],
        )
