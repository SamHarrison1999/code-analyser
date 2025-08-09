import os
import time
import datetime
from typing import Optional

import qlib
from qlib import get_module_logger
from qlib.data import D
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.data import Cal
from qlib.contrib.ops.high_freq import get_calendar_day, DayLast, FFillNan, BFillNan, Date, Select, IsNull, IsInf, Cut
# ✅ Best Practice: Use of joblib for parallel processing can improve performance and readability
import pickle as pkl
from joblib import Parallel, delayed
# ✅ Best Practice: Class docstring should be added to describe the purpose and usage of the class


class HighFreqProvider:
    def __init__(
        self,
        start_time: str,
        end_time: str,
        train_end_time: str,
        valid_start_time: str,
        valid_end_time: str,
        test_start_time: str,
        qlib_conf: dict,
        feature_conf: dict,
        label_conf: Optional[dict] = None,
        backtest_conf: dict = None,
        # ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability
        freq: str = "1min",
        **kwargs,
    ) -> None:
        self.start_time = start_time
        self.end_time = end_time
        self.test_start_time = test_start_time
        self.train_end_time = train_end_time
        # 🧠 ML Signal: Initialization of configuration settings for a machine learning model
        self.valid_start_time = valid_start_time
        self.valid_end_time = valid_end_time
        self._init_qlib(qlib_conf)
        self.feature_conf = feature_conf
        self.label_conf = label_conf
        self.backtest_conf = backtest_conf
        # 🧠 ML Signal: Use of a logger for tracking and debugging, common in ML pipelines
        self.qlib_conf = qlib_conf
        self.logger = get_module_logger("HighFreqProvider")
        self.freq = freq

    def get_pre_datasets(self):
        """Generate the training, validation and test datasets for prediction

        Returns:
            Tuple[BaseDataset, BaseDataset, BaseDataset]: The training and test datasets
        """

        # 🧠 ML Signal: Usage of configuration paths to determine dataset file locations
        dict_feature_path = self.feature_conf["path"]
        train_feature_path = dict_feature_path[:-4] + "_train.pkl"
        # ✅ Best Practice: Use of string slicing to modify file paths
        valid_feature_path = dict_feature_path[:-4] + "_valid.pkl"
        test_feature_path = dict_feature_path[:-4] + "_test.pkl"

        dict_label_path = self.label_conf["path"]
        train_label_path = dict_label_path[:-4] + "_train.pkl"
        # ⚠️ SAST Risk (Low): Potential race condition if files are checked and created in separate steps
        valid_label_path = dict_label_path[:-4] + "_valid.pkl"
        test_label_path = dict_label_path[:-4] + "_test.pkl"

        if (
            not os.path.isfile(train_feature_path)
            # 🧠 ML Signal: Generation of training, validation, and test datasets
            # ✅ Best Practice: Use of to_pickle for efficient data serialization
            or not os.path.isfile(valid_feature_path)
            or not os.path.isfile(test_feature_path)
        ):
            xtrain, xvalid, xtest = self._gen_data(self.feature_conf)
            xtrain.to_pickle(train_feature_path)
            xvalid.to_pickle(valid_feature_path)
            # ✅ Best Practice: Deleting large objects to free memory
            xtest.to_pickle(test_feature_path)
            del xtrain, xvalid, xtest
        # ⚠️ SAST Risk (Low): Potential race condition if files are checked and created in separate steps

        if (
            not os.path.isfile(train_label_path)
            or not os.path.isfile(valid_label_path)
            or not os.path.isfile(test_label_path)
        ):
            ytrain, yvalid, ytest = self._gen_data(self.label_conf)
            # 🧠 ML Signal: Generation of training, validation, and test datasets
            # ✅ Best Practice: Use of to_pickle for efficient data serialization
            ytrain.to_pickle(train_label_path)
            yvalid.to_pickle(valid_label_path)
            ytest.to_pickle(test_label_path)
            del ytrain, yvalid, ytest

        # ✅ Best Practice: Deleting large objects to free memory
        feature = {
            "train": train_feature_path,
            # 🧠 ML Signal: Structuring of dataset paths for different data splits
            # ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the function.
            "valid": valid_feature_path,
            # ✅ Best Practice: Include a docstring to describe the purpose of the function
            # 🧠 ML Signal: Usage of **kwargs indicates a flexible function signature, which can be a pattern for dynamic parameter handling.
            "test": test_feature_path,
        }
        # ⚠️ SAST Risk (Low): Ensure that qlib_conf does not contain any sensitive information
        # 🧠 ML Signal: Structuring of dataset paths for different data splits
        # ⚠️ SAST Risk (Low): Ensure that self.backtest_conf is properly validated to prevent potential misuse or injection vulnerabilities.
        # ✅ Best Practice: Consider checking if self.backtest_conf is initialized before using it to avoid potential AttributeError.
        # 🧠 ML Signal: Usage of qlib library initialization with specific configurations

        label = {
            "train": train_label_path,
            "valid": valid_label_path,
            # 🧠 ML Signal: Specific region configuration for qlib
            "test": test_label_path,
        }
        # 🧠 ML Signal: Disabling auto_mount feature

        return feature, label
    # 🧠 ML Signal: Custom operations being used in qlib

    # 🧠 ML Signal: Returning structured dataset paths for further processing
    # 🧠 ML Signal: Method name suggests caching behavior
    def get_backtest(self, **kwargs) -> None:
        # 🧠 ML Signal: Disabling expression cache
        # ✅ Best Practice: Use of a private method to encapsulate functionality
        self._gen_data(self.backtest_conf)
    # ✅ Best Practice: Consider making datasets a parameter with a default value to improve flexibility.

    # 🧠 ML Signal: Use of additional configuration parameters
    # 🧠 ML Signal: Function call with frequency parameter indicates time-based operation
    def _init_qlib(self, qlib_conf):
        # 🧠 ML Signal: Function call with frequency parameter indicates time-based operation
        # ✅ Best Practice: Use config.get("path") with a default value to avoid KeyError.
        """initialize qlib"""

        qlib.init(
            region=REG_CN,
            # ⚠️ SAST Risk (Low): Raising a new exception with the original one can expose internal logic.
            auto_mount=False,
            custom_ops=[DayLast, FFillNan, BFillNan, Date, Select, IsNull, IsInf, Cut],
            # ⚠️ SAST Risk (Low): os.path.isfile can be subject to TOCTOU (Time of Check to Time of Use) race conditions.
            expression_cache=None,
            **qlib_conf,
        )
    # 🧠 ML Signal: Logging dataset loading events can be useful for monitoring and debugging.

    def _prepare_calender_cache(self):
        """preload the calendar for cache"""
        # ⚠️ SAST Risk (Medium): Unpickling data can lead to arbitrary code execution if the source is untrusted.

        # This code used the copy-on-write feature of Linux
        # to avoid calculating the calendar multiple times in the subprocess.
        # This code may accelerate, but may be not useful on Windows and Mac Os
        Cal.calendar(freq=self.freq)
        get_calendar_day(freq=self.freq)
    # 🧠 ML Signal: Logging time taken for operations can be used for performance monitoring.

    def _gen_dataframe(self, config, datasets=["train", "valid", "test"]):
        try:
            # ⚠️ SAST Risk (Low): os.makedirs can be subject to TOCTOU race conditions.
            path = config.pop("path")
        except KeyError as e:
            raise ValueError("Must specify the path to save the dataset.") from e
        if os.path.isfile(path):
            start = time.time()
            # 🧠 ML Signal: Logging dataset generation events can be useful for monitoring and debugging.
            self.logger.info(f"[{__name__}]Dataset exists, load from disk.")

            # 🧠 ML Signal: Initializing instances by config can indicate dynamic behavior in the application.
            # res = dataset.prepare(['train', 'valid', 'test'])
            with open(path, "rb") as f:
                data = pkl.load(f)
            if isinstance(data, dict):
                res = [data[i] for i in datasets]
            else:
                res = data.prepare(datasets)
            self.logger.info(f"[{__name__}]Data loaded, time cost: {time.time() - start:.2f}")
        else:
            # ⚠️ SAST Risk (Medium): Pickling data can lead to security risks if the data is later untrusted.
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            self.logger.info(f"[{__name__}]Generating dataset")
            start_time = time.time()
            # ⚠️ SAST Risk (Low): Raising a new exception without preserving the original traceback
            self._prepare_calender_cache()
            dataset = init_instance_by_config(config)
            trainset, validset, testset = dataset.prepare(["train", "valid", "test"])
            data = {
                # 🧠 ML Signal: Logging usage pattern
                "train": trainset,
                # 🧠 ML Signal: Logging time taken for operations can be used for performance monitoring.
                "valid": validset,
                "test": testset,
            # ⚠️ SAST Risk (Medium): Unvalidated deserialization of data
            }
            with open(path, "wb") as f:
                pkl.dump(data, f)
            with open(path[:-4] + "train.pkl", "wb") as f:
                pkl.dump(trainset, f)
            with open(path[:-4] + "valid.pkl", "wb") as f:
                # 🧠 ML Signal: Logging usage pattern
                pkl.dump(validset, f)
            with open(path[:-4] + "test.pkl", "wb") as f:
                pkl.dump(testset, f)
            res = [data[i] for i in datasets]
            # ✅ Best Practice: Ensure directory exists before creating it
            self.logger.info(f"[{__name__}]Data generated, time cost: {(time.time() - start_time):.2f}")
        return res
    # 🧠 ML Signal: Logging usage pattern

    def _gen_data(self, config, datasets=["train", "valid", "test"]):
        try:
            path = config.pop("path")
        except KeyError as e:
            # 🧠 ML Signal: Configuration pattern for dataset
            # 🧠 ML Signal: Use of try-except for error handling
            raise ValueError("Must specify the path to save the dataset.") from e
        if os.path.isfile(path):
            start = time.time()
            self.logger.info(f"[{__name__}]Dataset exists, load from disk.")
            # ⚠️ SAST Risk (Low): Raising a new exception without logging the original exception details
            # 🧠 ML Signal: Logging usage pattern

            # res = dataset.prepare(['train', 'valid', 'test'])
            # ⚠️ SAST Risk (Low): Potential use of an untrusted path from config
            with open(path, "rb") as f:
                data = pkl.load(f)
            if isinstance(data, dict):
                # 🧠 ML Signal: Logging usage pattern
                res = [data[i] for i in datasets]
            else:
                res = data.prepare(datasets)
            # ⚠️ SAST Risk (Medium): Unpickling data from a potentially untrusted source
            self.logger.info(f"[{__name__}]Data loaded, time cost: {time.time() - start:.2f}")
        else:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            self.logger.info(f"[{__name__}]Generating dataset")
            # ⚠️ SAST Risk (Low): Directory creation without checking for path traversal
            start_time = time.time()
            self._prepare_calender_cache()
            dataset = init_instance_by_config(config)
            dataset.config(dump_all=True, recursive=True)
            # ✅ Best Practice: Method name suggests caching, which can improve performance
            dataset.to_pickle(path)
            res = dataset.prepare(datasets)
            # 🧠 ML Signal: Dynamic instance creation from config
            self.logger.info(f"[{__name__}]Data generated, time cost: {(time.time() - start_time):.2f}")
        return res
    # 🧠 ML Signal: Common dataset preparation pattern

    def _gen_dataset(self, config):
        # ⚠️ SAST Risk (Low): Raising a new exception without preserving the original traceback
        try:
            path = config.pop("path")
        # 🧠 ML Signal: Configuration pattern for datasets
        except KeyError as e:
            # ⚠️ SAST Risk (Low): Writing to a file path that may be influenced by user input
            # 🧠 ML Signal: Logging usage pattern
            raise ValueError("Must specify the path to save the dataset.") from e
        if os.path.isfile(path):
            start = time.time()
            self.logger.info(f"[{__name__}]Dataset exists, load from disk.")

            with open(path, "rb") as f:
                # ⚠️ SAST Risk (Low): Directory creation without checking for race conditions
                dataset = pkl.load(f)
            self.logger.info(f"[{__name__}]Data loaded, time cost: {time.time() - start:.2f}")
        # 🧠 ML Signal: Logging usage pattern
        else:
            start = time.time()
            if not os.path.exists(os.path.dirname(path)):
                # 🧠 ML Signal: Dynamic instance creation pattern
                os.makedirs(os.path.dirname(path))
            self.logger.info(f"[{__name__}]Generating dataset")
            # 🧠 ML Signal: Logging usage pattern
            self._prepare_calender_cache()
            dataset = init_instance_by_config(config)
            # 🧠 ML Signal: Configuration pattern
            # ⚠️ SAST Risk (Low): Potential path traversal if 'path' is user-controlled
            self.logger.info(f"[{__name__}]Dataset init, time cost: {time.time() - start:.2f}")
            dataset.prepare(["train", "valid", "test"])
            # ⚠️ SAST Risk (Low): Potential data corruption if interrupted during write
            self.logger.info(f"[{__name__}]Dataset prepared, time cost: {time.time() - start:.2f}")
            dataset.config(dump_all=True, recursive=True)
            # ⚠️ SAST Risk (Low): Use of 'self' without class context; potential misuse
            dataset.to_pickle(path)
        # ⚠️ SAST Risk (Medium): Untrusted deserialization
        return dataset

    # 🧠 ML Signal: Usage of calendar function with specific slicing
    # 🧠 ML Signal: Usage of dynamic configuration with dictionary unpacking
    def _gen_day_dataset(self, config, conf_type):
        try:
            path = config.pop("path")
        except KeyError as e:
            raise ValueError("Must specify the path to save the dataset.") from e
        # 🧠 ML Signal: Conditional logic affecting method calls

        if os.path.isfile(path + "tmp_dataset.pkl"):
            # 🧠 ML Signal: Configuration of dataset with specific parameters
            start = time.time()
            # ⚠️ SAST Risk (Low): Potential path traversal if 'path' is user-controlled
            self.logger.info(f"[{__name__}]Dataset exists, load from disk.")
        else:
            # ⚠️ SAST Risk (Low): Raising a new exception without preserving the original traceback
            start = time.time()
            # ✅ Best Practice: Use of parallel processing to improve performance
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            # 🧠 ML Signal: Logging information about dataset loading
            self.logger.info(f"[{__name__}]Generating dataset")
            self._prepare_calender_cache()
            dataset = init_instance_by_config(config)
            self.logger.info(f"[{__name__}]Dataset init, time cost: {time.time() - start:.2f}")
            dataset.config(dump_all=False, recursive=True)
            dataset.to_pickle(path + "tmp_dataset.pkl")
        # ⚠️ SAST Risk (Low): Potential race condition in directory creation

        with open(path + "tmp_dataset.pkl", "rb") as f:
            # 🧠 ML Signal: Logging information about dataset generation
            new_dataset = pkl.load(f)

        time_list = D.calendar(start_time=self.start_time, end_time=self.end_time, freq=self.freq)[::240]
        # 🧠 ML Signal: Using a configuration to initialize an instance

        def generate_dataset(times):
            # 🧠 ML Signal: Logging time taken for dataset initialization
            if os.path.isfile(path + times.strftime("%Y-%m-%d") + ".pkl"):
                # 🧠 ML Signal: Configuring dataset with specific parameters
                print("exist " + times.strftime("%Y-%m-%d"))
                return
            self._init_qlib(self.qlib_conf)
            # ⚠️ SAST Risk (Low): Overwriting existing dataset file without backup
            end_times = times + datetime.timedelta(days=1)
            # ⚠️ SAST Risk (Low): Potential path traversal if 'stock' contains malicious input
            new_dataset.handler.config(**{"start_time": times, "end_time": end_times})
            if conf_type == "backtest":
                # ⚠️ SAST Risk (Low): Unrestricted deserialization of potentially untrusted data
                new_dataset.handler.setup_data()
            else:
                # 🧠 ML Signal: Fetching instruments data for stock list generation
                # 🧠 ML Signal: Usage of self._init_qlib suggests a pattern for initializing a library or framework
                new_dataset.handler.setup_data(init_type=DataHandlerLP.IT_LS)
            new_dataset.config(dump_all=True, recursive=True)
            # 🧠 ML Signal: Generating a list of stock instruments with specific parameters
            # 🧠 ML Signal: Dynamic configuration of handler with stock-specific instruments
            new_dataset.to_pickle(path + times.strftime("%Y-%m-%d") + ".pkl")

        Parallel(n_jobs=8)(delayed(generate_dataset)(times) for times in time_list)
    # 🧠 ML Signal: Conditional logic based on configuration type

    def _gen_stock_dataset(self, config, conf_type):
        # 🧠 ML Signal: Configuration of dataset with specific parameters
        # ⚠️ SAST Risk (Low): Potential path traversal if 'stock' contains malicious input
        # ✅ Best Practice: Use of Parallel and delayed for concurrent execution
        try:
            path = config.pop("path")
        except KeyError as e:
            raise ValueError("Must specify the path to save the dataset.") from e

        if os.path.isfile(path + "tmp_dataset.pkl"):
            start = time.time()
            self.logger.info(f"[{__name__}]Dataset exists, load from disk.")
        else:
            start = time.time()
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            self.logger.info(f"[{__name__}]Generating dataset")
            self._prepare_calender_cache()
            dataset = init_instance_by_config(config)
            self.logger.info(f"[{__name__}]Dataset init, time cost: {time.time() - start:.2f}")
            dataset.config(dump_all=False, recursive=True)
            dataset.to_pickle(path + "tmp_dataset.pkl")

        with open(path + "tmp_dataset.pkl", "rb") as f:
            new_dataset = pkl.load(f)

        instruments = D.instruments(market="all")
        stock_list = D.list_instruments(
            instruments=instruments, start_time=self.start_time, end_time=self.end_time, freq=self.freq, as_list=True
        )

        def generate_dataset(stock):
            if os.path.isfile(path + stock + ".pkl"):
                print("exist " + stock)
                return
            self._init_qlib(self.qlib_conf)
            new_dataset.handler.config(**{"instruments": [stock]})
            if conf_type == "backtest":
                new_dataset.handler.setup_data()
            else:
                new_dataset.handler.setup_data(init_type=DataHandlerLP.IT_LS)
            new_dataset.config(dump_all=True, recursive=True)
            new_dataset.to_pickle(path + stock + ".pkl")

        Parallel(n_jobs=32)(delayed(generate_dataset)(stock) for stock in stock_list)