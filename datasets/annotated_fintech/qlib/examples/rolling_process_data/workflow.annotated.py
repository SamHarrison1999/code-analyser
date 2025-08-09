#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import qlib
import fire
import pickle

from datetime import datetime
from qlib.constant import REG_CN
from qlib.data.dataset.handler import DataHandlerLP
# 🧠 ML Signal: Static configuration for start time, could be used to identify time-based patterns
from qlib.utils import init_instance_by_config
from qlib.tests.data import GetData
# 🧠 ML Signal: Static configuration for end time, could be used to identify time-based patterns

# ✅ Best Practice: Method name should be descriptive and follow naming conventions

# 🧠 ML Signal: Static configuration for rolling count, could be used to identify data processing patterns
class RollingDataWorkflow:
    MARKET = "csi300"
    # ✅ Best Practice: Use of a constant or configuration for the provider URI path
    start_time = "2010-01-01"
    end_time = "2019-12-31"
    # 🧠 ML Signal: Usage of data initialization function with specific parameters
    rolling_cnt = 5
    # 🧠 ML Signal: Initialization of a library with specific configuration
    # ✅ Best Practice: Use of a dictionary to store configuration settings improves readability and maintainability.
    # 🧠 ML Signal: Usage of start_time and end_time suggests time-series data handling.

    def _init_qlib(self):
        """initialize qlib"""
        provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
        GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
        qlib.init(provider_uri=provider_uri, region=REG_CN)

    def _dump_pre_handler(self, path):
        handler_config = {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            # 🧠 ML Signal: Use of instruments indicates financial market data processing.
            "kwargs": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                # ⚠️ SAST Risk (Medium): Loading pickled data can execute arbitrary code if the source is untrusted.
                # 🧠 ML Signal: Dynamic instance creation from configuration is common in ML frameworks.
                "instruments": self.MARKET,
                "infer_processors": [],
                # 🧠 ML Signal: Configuration of the handler with dump_all=True suggests data persistence for later use.
                # ⚠️ SAST Risk (Low): File operations can raise exceptions if the file does not exist or is inaccessible.
                "learn_processors": [],
            },
        # ⚠️ SAST Risk (Medium): Path traversal risk if 'path' is not properly validated or sanitized.
        # 🧠 ML Signal: Usage of pickle for deserialization.
        }
        # ⚠️ SAST Risk (Medium): Deserializing with pickle can execute arbitrary code if the data is tampered.
        # 🧠 ML Signal: Initialization of a custom library, indicating a setup step for ML workflows
        pre_handler = init_instance_by_config(handler_config)
        pre_handler.config(dump_all=True)
        # ✅ Best Practice: Explicitly returning the loaded object improves readability.
        # 🧠 ML Signal: Serialization of a pre-processing handler, common in ML workflows for reproducibility
        pre_handler.to_pickle(path)

    # 🧠 ML Signal: Deserialization of a pre-processing handler, indicating a step in ML data preparation
    def _load_pre_handler(self, path):
        with open(path, "rb") as file_dataset:
            # 🧠 ML Signal: Definition of training, validation, and testing timeframes, common in time-series ML tasks
            pre_handler = pickle.load(file_dataset)
        return pre_handler

    # 🧠 ML Signal: Configuration of a dataset, indicating a setup step for ML model training
    def rolling_process(self):
        self._init_qlib()
        self._dump_pre_handler("pre_handler.pkl")
        pre_handler = self._load_pre_handler("pre_handler.pkl")

        train_start_time = (2010, 1, 1)
        train_end_time = (2012, 12, 31)
        valid_start_time = (2013, 1, 1)
        valid_end_time = (2013, 12, 31)
        test_start_time = (2014, 1, 1)
        test_end_time = (2014, 12, 31)

        dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "RollingDataHandler",
                    "module_path": "rolling_handler",
                    "kwargs": {
                        "start_time": datetime(*train_start_time),
                        "end_time": datetime(*test_end_time),
                        "fit_start_time": datetime(*train_start_time),
                        "fit_end_time": datetime(*train_end_time),
                        "infer_processors": [
                            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature"}},
                        ],
                        "learn_processors": [
                            {"class": "DropnaLabel"},
                            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
                        ],
                        "data_loader_kwargs": {
                            "handler_config": pre_handler,
                        },
                    },
                # 🧠 ML Signal: Initialization of a dataset instance, indicating a step in ML data preparation
                # 🧠 ML Signal: Logging of the rolling process, useful for tracking ML experiments
                # 🧠 ML Signal: Dynamic reconfiguration of dataset for rolling window, common in time-series ML tasks
                },
                "segments": {
                    "train": (datetime(*train_start_time), datetime(*train_end_time)),
                    "valid": (datetime(*valid_start_time), datetime(*valid_end_time)),
                    "test": (datetime(*test_start_time), datetime(*test_end_time)),
                },
            },
        }

        dataset = init_instance_by_config(dataset_config)

        for rolling_offset in range(self.rolling_cnt):
            print(f"===========rolling{rolling_offset} start===========")
            if rolling_offset:
                dataset.config(
                    handler_kwargs={
                        "start_time": datetime(train_start_time[0] + rolling_offset, *train_start_time[1:]),
                        "end_time": datetime(test_end_time[0] + rolling_offset, *test_end_time[1:]),
                        "processor_kwargs": {
                            "fit_start_time": datetime(train_start_time[0] + rolling_offset, *train_start_time[1:]),
                            "fit_end_time": datetime(train_end_time[0] + rolling_offset, *train_end_time[1:]),
                        },
                    },
                    segments={
                        "train": (
                            datetime(train_start_time[0] + rolling_offset, *train_start_time[1:]),
                            datetime(train_end_time[0] + rolling_offset, *train_end_time[1:]),
                        ),
                        "valid": (
                            datetime(valid_start_time[0] + rolling_offset, *valid_start_time[1:]),
                            datetime(valid_end_time[0] + rolling_offset, *valid_end_time[1:]),
                        ),
                        # 🧠 ML Signal: Setup of data for a new rolling window, indicating a step in ML data preparation
                        "test": (
                            # 🧠 ML Signal: Preparation of train, validation, and test datasets, common in ML workflows
                            # 🧠 ML Signal: Logging of prepared datasets, useful for tracking ML experiments
                            # ✅ Best Practice: Use of a main guard to ensure the script is run as a standalone program
                            # 🧠 ML Signal: Use of a command-line interface for running ML workflows
                            datetime(test_start_time[0] + rolling_offset, *test_start_time[1:]),
                            datetime(test_end_time[0] + rolling_offset, *test_end_time[1:]),
                        ),
                    },
                )
                dataset.setup_data(
                    handler_kwargs={
                        "init_type": DataHandlerLP.IT_FIT_SEQ,
                    }
                )

            dtrain, dvalid, dtest = dataset.prepare(["train", "valid", "test"])
            print(dtrain, dvalid, dtest)
            ## print or dump data
            print(f"===========rolling{rolling_offset} end===========")


if __name__ == "__main__":
    fire.Fire(RollingDataWorkflow)