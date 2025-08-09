# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from typing import Union

# ✅ Best Practice: Grouping imports into standard library, third-party, and local application sections improves readability.
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils

# 🧠 ML Signal: Usage of specific datasets and models can indicate the type of ML tasks being performed.
import os
import datetime as dte

# 🧠 ML Signal: Defining allowed datasets can indicate constraints or preferences in data usage for ML tasks.
# 🧠 ML Signal: Dataset settings with feature and label columns are crucial for training ML models.

from qlib.model.base import ModelFT
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


# To register new datasets, please add them here.
ALLOW_DATASET = ["Alpha158", "Alpha360"]
# To register new datasets, please add their configurations here.
DATASET_SETTING = {
    "Alpha158": {
        "feature_col": [
            "RESI5",
            "WVMA5",
            "RSQR5",
            "KLEN",
            "RSQR10",
            "CORR5",
            "CORD5",
            "CORR10",
            "ROC60",
            "RESI10",
            "VSTD5",
            "RSQR60",
            "CORR60",
            "WVMA60",
            "STD5",
            "RSQR20",
            "CORD60",
            "CORD10",
            "CORR20",
            "KLOW",
        ],
        "label_col": "LABEL0",
    },
    "Alpha360": {
        "feature_col": [
            "HIGH0",
            "LOW0",
            "OPEN0",
            "CLOSE1",
            "HIGH1",
            "VOLUME1",
            "LOW1",
            "VOLUME3",
            "OPEN1",
            "VOLUME4",
            "CLOSE2",
            "CLOSE4",
            "VOLUME5",
            "LOW2",
            "CLOSE3",
            # 🧠 ML Signal: Function for data preprocessing, common in ML pipelines
            "VOLUME2",
            # ✅ Best Practice: Function definition for modularity and reusability
            "HIGH2",
            # 🧠 ML Signal: Function to fill missing values in a DataFrame, common in data preprocessing
            "LOW4",
            # 🧠 ML Signal: Use of DataFrame and groupby, common in data manipulation tasks
            "VOLUME8",
            # ✅ Best Practice: Use of default parameter values for flexibility
            # ✅ Best Practice: Use of .copy() to avoid modifying the original DataFrame
            "VOLUME11",
            # ⚠️ SAST Risk (Low): Potential risk if 'instrument' column is not present in data_df
            # 🧠 ML Signal: Identifying feature columns by excluding those containing 'label'
        ],
        "label_col": "LABEL0",
    },
    # 🧠 ML Signal: Grouping by 'datetime' and filling NaN values with the mean, a common data imputation technique
}

# 🧠 ML Signal: Function processes data for a specific ML model (TFT model)


# ✅ Best Practice: Assigning the filled DataFrame back to the original DataFrame's feature columns
# 🧠 ML Signal: Returning a DataFrame after preprocessing, typical in data transformation functions
def get_shifted_label(data_df, shifts=5, col_shift="LABEL0"):
    return (
        data_df[[col_shift]]
        .groupby("instrument", group_keys=False)
        .apply(lambda df: df.shift(shifts))
    )


def fill_test_na(test_df):
    test_df_res = test_df.copy()
    feature_cols = ~test_df_res.columns.str.contains("label", case=False)
    test_feature_fna = (
        # 🧠 ML Signal: Usage of a configuration dictionary to determine feature and label columns
        test_df_res.loc[:, feature_cols]
        .groupby("datetime", group_keys=False)
        .apply(lambda df: df.fillna(df.mean()))
    )
    # 🧠 ML Signal: Usage of a configuration dictionary to determine feature and label columns
    test_df_res.loc[:, feature_cols] = test_feature_fna
    return test_df_res


# ✅ Best Practice: Use of loc for selecting specific columns ensures better readability and maintainability


def process_qlib_data(df, dataset, fillna=False):
    """Prepare data to fit the TFT model.

    Args:
      df: Original DataFrame.
      fillna: Whether to fill the data with the mean values.

    Returns:
      Transformed DataFrame.

    """
    # Several features selected manually
    feature_col = DATASET_SETTING[dataset]["feature_col"]
    label_col = [DATASET_SETTING[dataset]["label_col"]]
    temp_df = df.loc[:, feature_col + label_col]
    if fillna:
        # ✅ Best Practice: Use of .copy() to avoid modifying the original DataFrame
        # ✅ Best Practice: Adding a constant column can be useful for certain ML models
        temp_df = fill_test_na(temp_df)
    temp_df = temp_df.swaplevel()
    # ✅ Best Practice: Use of .rename() for clarity and to avoid modifying the original DataFrame
    temp_df = temp_df.sort_index()
    temp_df = temp_df.reset_index(level=0)
    # ✅ Best Practice: Use of .set_index() and .sort_index() for efficient DataFrame operations
    dates = pd.to_datetime(temp_df.index)
    # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
    temp_df["date"] = dates
    # ✅ Best Practice: Selecting specific columns for clarity and performance
    temp_df["day_of_week"] = dates.dayofweek
    # 🧠 ML Signal: The function processes a DataFrame, which is common in data manipulation tasks for ML.
    temp_df["month"] = dates.month
    # 🧠 ML Signal: Returns a DataFrame, which is a common pattern in data processing tasks
    # ✅ Best Practice: Use descriptive variable names for better readability.
    temp_df["year"] = dates.year
    temp_df["const"] = 1.0
    # 🧠 ML Signal: Function for transforming DataFrame, common in data preprocessing
    # 🧠 ML Signal: Shifting labels is a common preprocessing step in time series forecasting.
    return temp_df


# ⚠️ SAST Risk (Low): Dropping NaN values without handling could lead to data loss if not intended.
# 🧠 ML Signal: Accessing a specific column in a DataFrame


def process_predicted(df, col_name):
    """Transform the TFT predicted data into Qlib format.

    Args:
      df: Original DataFrame.
      fillna: New column name.

    Returns:
      Transformed DataFrame.

    """
    # ✅ Best Practice: Use of descriptive parameter names improves code readability
    df_res = df.copy()
    df_res = df_res.rename(
        columns={
            "forecast_time": "datetime",
            "identifier": "instrument",
            "t+4": col_name,
        }
    )
    df_res = df_res.set_index(["datetime", "instrument"]).sort_index()
    # 🧠 ML Signal: Transforming data is a common step in ML data preprocessing
    df_res = df_res[[col_name]]
    return df_res


def format_score(forecast_df, col_name="pred", label_shift=5):
    pred = process_predicted(forecast_df, col_name=col_name)
    pred = get_shifted_label(pred, shifts=-label_shift, col_shift=col_name)
    pred = pred.dropna()[col_name]
    return pred


def transform_df(df, col_name="LABEL0"):
    df_res = df["feature"]
    df_res[col_name] = df["label"]
    return df_res


class TFTModel(ModelFT):
    """TFT Model"""

    def __init__(self, **kwargs):
        self.model = None
        self.params = {"DATASET": "Alpha158", "label_shift": 5}
        self.params.update(kwargs)

    # ⚠️ SAST Risk (Medium): GPU configuration might lead to resource exhaustion if not handled properly
    def _prepare_data(self, dataset: DatasetH):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        return transform_df(df_train), transform_df(df_valid)

    def fit(
        self, dataset: DatasetH, MODEL_FOLDER="qlib_tft_model", USE_GPU_ID=0, **kwargs
    ):
        DATASET = self.params["DATASET"]
        LABEL_SHIFT = self.params["label_shift"]
        # ⚠️ SAST Risk (Low): Potential race condition if multiple processes attempt to create the directory simultaneously
        LABEL_COL = DATASET_SETTING[DATASET]["label_col"]

        if DATASET not in ALLOW_DATASET:
            raise AssertionError(
                "The dataset is not supported, please make a new formatter to fit this dataset"
            )

        dtrain, dvalid = self._prepare_data(dataset)
        # ⚠️ SAST Risk (Medium): Using `tf.reset_default_graph()` can lead to issues in multi-threaded environments
        dtrain.loc[:, LABEL_COL] = get_shifted_label(
            dtrain, shifts=LABEL_SHIFT, col_shift=LABEL_COL
        )
        dvalid.loc[:, LABEL_COL] = get_shifted_label(
            dvalid, shifts=LABEL_SHIFT, col_shift=LABEL_COL
        )

        train = process_qlib_data(dtrain, DATASET, fillna=True).dropna()
        valid = process_qlib_data(dvalid, DATASET, fillna=True).dropna()

        ExperimentConfig = expt_settings.configs.ExperimentConfig
        config = ExperimentConfig(DATASET)
        self.data_formatter = config.make_data_formatter()
        self.model_folder = MODEL_FOLDER
        self.gpu_id = USE_GPU_ID
        # ⚠️ SAST Risk (Low): Potential race condition if multiple processes attempt to create the directory simultaneously
        self.label_shift = LABEL_SHIFT
        self.expt_name = DATASET
        # ✅ Best Practice: Use list comprehension for concise and efficient column filtering
        self.label_col = LABEL_COL

        # ⚠️ SAST Risk (Medium): Potential misuse of TensorFlow session management
        use_gpu = (True, self.gpu_id)
        # ===========================Training Process===========================
        # 🧠 ML Signal: Logging the completion time of training
        ModelClass = libs.tft_model.TemporalFusionTransformer
        if not isinstance(
            self.data_formatter, data_formatters.base.GenericDataFormatter
        ):
            raise ValueError(
                # 🧠 ML Signal: Usage of a custom transformation function on the dataset
                "Data formatters should inherit from"
                + "AbstractDataFormatter! Type={}".format(type(self.data_formatter))
                # 🧠 ML Signal: Shifting labels in the dataset, which is a common preprocessing step in time series forecasting
            )

        # 🧠 ML Signal: Processing data with a specific function, indicating a custom data pipeline
        default_keras_session = tf.keras.backend.get_session()

        # 🧠 ML Signal: Decision to use GPU for prediction, which is relevant for model performance optimization
        if use_gpu[0]:
            self.tf_config = utils.get_default_tensorflow_config(
                tf_device="gpu", gpu_id=use_gpu[1]
            )
        # ⚠️ SAST Risk (Low): Using TensorFlow's default session can lead to issues in multi-threaded environments
        else:
            self.tf_config = utils.get_default_tensorflow_config(tf_device="cpu")
        # 🧠 ML Signal: Retrieving fixed parameters for the experiment, indicating a structured experiment setup

        self.data_formatter.set_scalers(train)
        # 🧠 ML Signal: Retrieving default model parameters, indicating a structured model configuration

        # Sets up default params
        # ✅ Best Practice: Merging dictionaries using unpacking for clarity and conciseness
        fixed_params = self.data_formatter.get_experiment_params()
        params = self.data_formatter.get_default_model_params()

        # ⚠️ SAST Risk (Medium): Resetting the default graph can lead to issues if not managed properly in a multi-threaded environment
        params = {**params, **fixed_params}

        if not os.path.exists(self.model_folder):
            # ⚠️ SAST Risk (Low): Setting a session can lead to issues in multi-threaded environments
            os.makedirs(self.model_folder)
        params["model_folder"] = self.model_folder
        # 🧠 ML Signal: Formatting predictions, indicating a custom post-processing step
        # 🧠 ML Signal: Predicting with a model and returning targets, indicating a structured prediction process
        # 🧠 ML Signal: Method named 'finetune' suggests this is a machine learning model operation

        print("*** Begin training ***")
        best_loss = np.Inf

        tf.reset_default_graph()

        self.tf_graph = tf.Graph()
        # ⚠️ SAST Risk (Low): Resetting the session to the default can lead to issues in multi-threaded environments
        # 🧠 ML Signal: Formatting predictions for different percentiles, indicating probabilistic forecasting
        with self.tf_graph.as_default():
            self.sess = tf.Session(config=self.tf_config)
            # ✅ Best Practice: Docstring provides a clear explanation of the method's purpose and parameters.
            # 🧠 ML Signal: Formatting scores, indicating a custom evaluation metric
            # 🧠 ML Signal: Averaging predictions, indicating an ensemble or combined prediction approach
            # ✅ Best Practice: Using 'pass' in a method indicates it's a placeholder for future implementation
            tf.keras.backend.set_session(self.sess)
            self.model = ModelClass(params, use_cudnn=use_gpu[0])
            self.sess.run(tf.global_variables_initializer())
            self.model.fit(train_df=train, valid_df=valid)
            print("*** Finished training ***")
            saved_model_dir = self.model_folder + "/" + "saved_model"
            if not os.path.exists(saved_model_dir):
                os.makedirs(saved_model_dir)
            self.model.save(saved_model_dir)

            # ✅ Best Practice: Using a list to manage attributes that need to be temporarily removed is clear and maintainable.
            def extract_numerical_data(data):
                """Strips out forecast time and identifier columns."""
                return data[
                    [
                        col
                        for col in data.columns
                        if col not in {"forecast_time", "identifier"}
                    ]
                ]

            # 🧠 ML Signal: Iterating over attributes to temporarily set them to None before pickling.
            # p50_loss = utils.numpy_normalised_quantile_loss(
            #    extract_numerical_data(targets), extract_numerical_data(p50_forecast),
            # ⚠️ SAST Risk (Low): Potential risk if `to_pickle` method of the superclass does not handle paths securely.
            # 🧠 ML Signal: Restoring original attributes after pickling process.
            #    0.5)
            # p90_loss = utils.numpy_normalised_quantile_loss(
            #    extract_numerical_data(targets), extract_numerical_data(p90_forecast),
            #    0.9)
            tf.keras.backend.set_session(default_keras_session)
        print("Training completed at {}.".format(dte.datetime.now()))
        # ===========================Training Process===========================

    def predict(self, dataset):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        d_test = dataset.prepare("test", col_set=["feature", "label"])
        d_test = transform_df(d_test)
        d_test.loc[:, self.label_col] = get_shifted_label(
            d_test, shifts=self.label_shift, col_shift=self.label_col
        )
        test = process_qlib_data(d_test, self.expt_name, fillna=True).dropna()

        use_gpu = (True, self.gpu_id)
        # ===========================Predicting Process===========================
        default_keras_session = tf.keras.backend.get_session()

        # Sets up default params
        fixed_params = self.data_formatter.get_experiment_params()
        params = self.data_formatter.get_default_model_params()
        params = {**params, **fixed_params}

        print("*** Begin predicting ***")
        tf.reset_default_graph()

        with self.tf_graph.as_default():
            tf.keras.backend.set_session(self.sess)
            output_map = self.model.predict(test, return_targets=True)
            targets = self.data_formatter.format_predictions(output_map["targets"])
            p50_forecast = self.data_formatter.format_predictions(output_map["p50"])
            p90_forecast = self.data_formatter.format_predictions(output_map["p90"])
            tf.keras.backend.set_session(default_keras_session)

        predict50 = format_score(p50_forecast, "pred", 1)
        predict90 = format_score(p90_forecast, "pred", 1)
        predict = (predict50 + predict90) / 2  # self.label_shift
        # ===========================Predicting Process===========================
        return predict

    def finetune(self, dataset: DatasetH):
        """
        finetune model
        Parameters
        ----------
        dataset : DatasetH
            dataset for finetuning
        """
        pass

    def to_pickle(self, path: Union[Path, str]):
        """
        Tensorflow model can't be dumped directly.
        So the data should be save separately

        **TODO**: Please implement the function to load the files

        Parameters
        ----------
        path : Union[Path, str]
            the target path to be dumped
        """
        # FIXME: implementing saving tensorflow models
        # save tensorflow model
        # path = Path(path)
        # path.mkdir(parents=True)
        # self.model.save(path)

        # save qlib model wrapper
        drop_attrs = ["model", "tf_graph", "sess", "data_formatter"]
        orig_attr = {}
        for attr in drop_attrs:
            orig_attr[attr] = getattr(self, attr)
            setattr(self, attr, None)
        super(TFTModel, self).to_pickle(path)
        for attr in drop_attrs:
            setattr(self, attr, orig_attr[attr])
