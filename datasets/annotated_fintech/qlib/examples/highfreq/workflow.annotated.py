#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import fire

import qlib
import pickle
from qlib.constant import REG_CN
from qlib.config import HIGH_FREQ_CONFIG

# ✅ Best Practice: Group related imports together and separate them with a blank line for better readability.
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP

# 🧠 ML Signal: Use of specific custom operations for data processing
from qlib.data.ops import Operators
from qlib.data.data import Cal

# 🧠 ML Signal: Use of a specific market for data handling
from qlib.tests.data import GetData

# 🧠 ML Signal: Specific start time for data processing
from highfreq_ops import (
    get_calendar_day,
    DayLast,
    FFillNan,
    BFillNan,
    Date,
    Select,
    IsNull,
    Cut,
)

# 🧠 ML Signal: Specific end time for data processing
# 🧠 ML Signal: Specific train end time for data segmentation


class HighfreqWorkflow:
    SPEC_CONF = {
        "custom_ops": [DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut],
        "expression_cache": None,
    }

    MARKET = "all"

    start_time = "2020-09-15 00:00:00"
    end_time = "2021-01-18 16:00:00"
    # ✅ Best Practice: Use of a configuration dictionary for data handler settings
    train_end_time = "2020-11-30 16:00:00"
    test_start_time = "2020-12-01 00:00:00"

    DATA_HANDLER_CONFIG0 = {
        "start_time": start_time,
        # 🧠 ML Signal: Use of a specific data processing class and module
        # ✅ Best Practice: Use of a configuration dictionary for data handler settings
        # ✅ Best Practice: Use of a task dictionary to organize dataset configurations
        "end_time": end_time,
        "fit_start_time": start_time,
        "fit_end_time": train_end_time,
        "instruments": MARKET,
        "infer_processors": [
            {"class": "HighFreqNorm", "module_path": "highfreq_processor"}
        ],
    }
    DATA_HANDLER_CONFIG1 = {
        "start_time": start_time,
        "end_time": end_time,
        "instruments": MARKET,
    }

    task = {
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "HighFreqHandler",
                    "module_path": "highfreq_handler",
                    "kwargs": DATA_HANDLER_CONFIG0,
                },
                "segments": {
                    "train": (start_time, train_end_time),
                    "test": (
                        test_start_time,
                        end_time,
                    ),
                },
            },
        },
        "dataset_backtest": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "HighFreqBacktestHandler",
                    "module_path": "highfreq_handler",
                    # 🧠 ML Signal: Use of a specific dataset class and module for backtesting
                    # 🧠 ML Signal: Use of a specific backtest handler class and module
                    # 🧠 ML Signal: Specific data segmentation for training and testing
                    "kwargs": DATA_HANDLER_CONFIG1,
                },
                # 🧠 ML Signal: Method for initializing a library with specific configurations
                "segments": {
                    "train": (start_time, train_end_time),
                    # ⚠️ SAST Risk (Low): Potential risk if HIGH_FREQ_CONFIG or self.SPEC_CONF contains sensitive data
                    "test": (
                        # 🧠 ML Signal: Specific data segmentation for training and testing in backtest
                        test_start_time,
                        # 🧠 ML Signal: Usage pattern for fetching data with specific parameters
                        end_time,
                    ),
                    # 🧠 ML Signal: Initialization pattern for a library with dynamic configuration
                    # ✅ Best Practice: Use of a docstring to describe the method's purpose
                },
            },
            # 🧠 ML Signal: Method call with a specific frequency parameter
        },
    }
    # ✅ Best Practice: Consider adding type hints for the method parameters and return type for better readability and maintainability.
    # 🧠 ML Signal: Function call with a specific frequency parameter

    def _init_qlib(self):
        """initialize qlib"""
        # 🧠 ML Signal: Usage of a configuration-based initialization pattern for datasets.
        # use cn_data_1min data
        QLIB_INIT_CONFIG = {**HIGH_FREQ_CONFIG, **self.SPEC_CONF}
        # 🧠 ML Signal: Common pattern of splitting data into training and testing sets.
        provider_uri = QLIB_INIT_CONFIG.get("provider_uri")
        GetData().qlib_data(
            target_dir=provider_uri, interval="1min", region=REG_CN, exists_skip=True
        )
        # ⚠️ SAST Risk (Low): Printing data directly can lead to exposure of sensitive information.
        qlib.init(**QLIB_INIT_CONFIG)

    # 🧠 ML Signal: Usage of a configuration-based initialization pattern for datasets.
    def _prepare_calender_cache(self):
        """preload the calendar for cache"""
        # 🧠 ML Signal: Common pattern of splitting data into training and testing sets.
        # ✅ Best Practice: Initialize necessary components before use

        # This code used the copy-on-write feature of Linux to avoid calculating the calendar multiple times in the subprocess
        # ⚠️ SAST Risk (Low): Printing data directly can lead to exposure of sensitive information.
        # ✅ Best Practice: Prepare cache to optimize performance
        # This code may accelerate, but may be not useful on Windows and Mac Os
        Cal.calendar(freq="1min")
        # ✅ Best Practice: Explicitly return a value, even if it's None, for better readability.
        # 🧠 ML Signal: Usage of configuration-based initialization
        get_calendar_day(freq="1min")

    # 🧠 ML Signal: Usage of configuration-based initialization
    def get_data(self):
        """use dataset to get highreq data"""
        # ⚠️ SAST Risk (Low): Potential data leakage if sensitive data is stored in pickle
        self._init_qlib()
        self._prepare_calender_cache()
        # ⚠️ SAST Risk (Low): Potential data leakage if sensitive data is stored in pickle

        dataset = init_instance_by_config(self.task["dataset"])
        # ⚠️ SAST Risk (Low): Unvalidated deserialization of data from pickle
        xtrain, xtest = dataset.prepare(["train", "test"])
        print(xtrain, xtest)

        dataset_backtest = init_instance_by_config(self.task["dataset_backtest"])
        backtest_train, backtest_test = dataset_backtest.prepare(["train", "test"])
        print(backtest_train, backtest_test)

        return

    def dump_and_load_dataset(self):
        """dump and load dataset state on disk"""
        self._init_qlib()
        # ✅ Best Practice: Prepare cache again after loading data
        # 🧠 ML Signal: Configuration of dataset with specific time range
        self._prepare_calender_cache()
        dataset = init_instance_by_config(self.task["dataset"])
        dataset_backtest = init_instance_by_config(self.task["dataset_backtest"])

        ##=============dump dataset=============
        # 🧠 ML Signal: Setup of data with specific handler type
        dataset.to_pickle(path="dataset.pkl")
        dataset_backtest.to_pickle(path="dataset_backtest.pkl")

        del dataset, dataset_backtest
        ##=============reload dataset=============
        with open("dataset.pkl", "rb") as file_dataset:
            dataset = pickle.load(file_dataset)

        with open("dataset_backtest.pkl", "rb") as file_dataset_backtest:
            dataset_backtest = pickle.load(file_dataset_backtest)

        self._prepare_calender_cache()
        # 🧠 ML Signal: Configuration of dataset_backtest with specific time range
        ##=============reinit dataset=============
        dataset.config(
            handler_kwargs={
                "start_time": "2021-01-19 00:00:00",
                "end_time": "2021-01-25 16:00:00",
            },
            # 🧠 ML Signal: Setup of data without specific handler type
            # 🧠 ML Signal: Preparation of test data for model evaluation
            # ✅ Best Practice: Use logging instead of print for better control over output
            # ✅ Best Practice: Use of __name__ guard to ensure code is only executed when the script is run directly
            # 🧠 ML Signal: Command-line interface usage for workflow execution
            segments={
                "test": (
                    "2021-01-19 00:00:00",
                    "2021-01-25 16:00:00",
                ),
            },
        )
        dataset.setup_data(
            handler_kwargs={
                "init_type": DataHandlerLP.IT_LS,
            },
        )
        dataset_backtest.config(
            handler_kwargs={
                "start_time": "2021-01-19 00:00:00",
                "end_time": "2021-01-25 16:00:00",
            },
            segments={
                "test": (
                    "2021-01-19 00:00:00",
                    "2021-01-25 16:00:00",
                ),
            },
        )
        dataset_backtest.setup_data(handler_kwargs={})

        ##=============get data=============
        xtest = dataset.prepare("test")
        backtest_test = dataset_backtest.prepare("test")

        print(xtest, backtest_test)
        return


if __name__ == "__main__":
    fire.Fire(HighfreqWorkflow)
