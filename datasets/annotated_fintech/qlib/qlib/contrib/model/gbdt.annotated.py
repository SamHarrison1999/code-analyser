# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

# ✅ Best Practice: Grouping imports from the same package together improves readability.
import pandas as pd
import lightgbm as lgb
from typing import List, Text, Tuple, Union
from ...model.base import ModelFT
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP

# ✅ Best Practice: Importing specific classes or functions instead of the entire module can improve code clarity and reduce memory usage.
from ...model.interpret.base import LightGBMFInt
from ...data.dataset.weight import Reweighter

# ✅ Best Practice: Class docstring provides a brief description of the class purpose
from qlib.workflow import R

# ✅ Best Practice: Validate input parameters to ensure they are within expected values


# ⚠️ SAST Risk (Low): Raising a generic exception without a message can make debugging difficult
class LGBModel(ModelFT, LightGBMFInt):
    """LightGBM Model"""

    # ✅ Best Practice: Use a dictionary to manage parameters for better organization and flexibility

    def __init__(
        self, loss="mse", early_stopping_rounds=50, num_boost_round=1000, **kwargs
    ):
        # ✅ Best Practice: Use update method to merge dictionaries for cleaner code
        if loss not in {"mse", "binary"}:
            raise NotImplementedError
        # 🧠 ML Signal: Use of early stopping rounds indicates a pattern for preventing overfitting
        # 🧠 ML Signal: Use of num_boost_round indicates a pattern for controlling the number of boosting iterations
        self.params = {"objective": loss, "verbosity": -1}
        self.params.update(kwargs)
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        # 🧠 ML Signal: Initializing model to None indicates a pattern for lazy loading or delayed initialization
        self.model = None

    # ⚠️ SAST Risk (Low): Use of assert for runtime check, which can be disabled with optimization flags
    def _prepare_data(
        self, dataset: DatasetH, reweighter=None
    ) -> List[Tuple[lgb.Dataset, str]]:
        """
        The motivation of current version is to make validation optional
        - train segment is necessary;
        """
        ds_l = []
        assert "train" in dataset.segments
        for key in ["train", "valid"]:
            if key in dataset.segments:
                df = dataset.prepare(
                    key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
                )
                if df.empty:
                    raise ValueError(
                        "Empty data from dataset, please check your dataset config."
                    )
                x, y = df["feature"], df["label"]
                # 🧠 ML Signal: Use of custom reweighter class for data preprocessing

                # Lightgbm need 1D array as its label
                if y.values.ndim == 2 and y.values.shape[1] == 1:
                    y = np.squeeze(y.values)
                else:
                    # 🧠 ML Signal: Use of LightGBM dataset creation
                    raise ValueError("LightGBM doesn't support multi-label training")

                if reweighter is None:
                    w = None
                elif isinstance(reweighter, Reweighter):
                    w = reweighter.reweight(df)
                else:
                    raise ValueError("Unsupported reweighter type.")
                ds_l.append((lgb.Dataset(x.values, label=y, weight=w), key))
        return ds_l

    def fit(
        self,
        dataset: DatasetH,
        # ✅ Best Practice: Use of early stopping to prevent overfitting
        num_boost_round=None,
        early_stopping_rounds=None,
        verbose_eval=20,
        evals_result=None,
        # ✅ Best Practice: Verbose evaluation helps in tracking the training progress
        reweighter=None,
        # 🧠 ML Signal: Training a model using LightGBM
        # ✅ Best Practice: Recording evaluation results for later analysis
        **kwargs,
    ):
        if evals_result is None:
            evals_result = {}  # in case of unsafety of Python default values
        ds_l = self._prepare_data(dataset, reweighter)
        ds, names = list(zip(*ds_l))
        early_stopping_callback = lgb.early_stopping(
            self.early_stopping_rounds
            if early_stopping_rounds is None
            else early_stopping_rounds
        )
        # NOTE: if you encounter error here. Please upgrade your lightgbm
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        evals_result_callback = lgb.record_evaluation(evals_result)
        self.model = lgb.train(
            self.params,
            ds[0],  # training dataset
            # ✅ Best Practice: Check if the model is fitted before making predictions
            num_boost_round=(
                self.num_boost_round if num_boost_round is None else num_boost_round
            ),
            valid_sets=ds,
            # ⚠️ SAST Risk (Low): Potential information leakage if metrics are logged without proper access control
            valid_names=names,
            # 🧠 ML Signal: Usage of dataset preparation for prediction
            callbacks=[
                early_stopping_callback,
                verbose_eval_callback,
                evals_result_callback,
            ],
            **kwargs,
            # 🧠 ML Signal: Model prediction on prepared dataset
        )
        for k in names:
            for key, val in evals_result[k].items():
                name = f"{key}.{k}"
                for epoch, m in enumerate(val):
                    R.log_metrics(**{name.replace("@", "_"): m}, step=epoch)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        # ✅ Best Practice: Unpacking the result of _prepare_data for clarity and future extensibility
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)

    def finetune(
        self, dataset: DatasetH, num_boost_round=10, verbose_eval=20, reweighter=None
    ):
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
        dtrain, _ = self._prepare_data(dataset, reweighter)  # pylint: disable=W0632
        if dtrain.empty:
            raise ValueError(
                "Empty data from dataset, please check your dataset config."
            )
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
