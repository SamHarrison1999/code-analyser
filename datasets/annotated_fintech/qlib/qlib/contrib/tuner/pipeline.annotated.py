# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: skip-file
# flake8: noqa
# ✅ Best Practice: Grouping imports into standard, third-party, and local sections improves readability.

import os
# ✅ Best Practice: Constants should be defined at the class level for easy access and modification
import json
import logging
import importlib
# 🧠 ML Signal: Initialization of class attributes from a configuration manager
from abc import abstractmethod
# ✅ Best Practice: Use of a logger for the class

from ...log import get_module_logger, TimeInspector
# 🧠 ML Signal: Storing configuration manager for later use
from ...utils import get_module_by_module_path

# 🧠 ML Signal: Extracting specific configurations from a manager

class Pipeline:
    GLOBAL_BEST_PARAMS_NAME = "global_best_params.json"

    def __init__(self, tuner_config_manager):
        self.logger = get_module_logger("Pipeline", sh_level=logging.INFO)

        self.tuner_config_manager = tuner_config_manager
        # 🧠 ML Signal: Initialization of attributes to track best results and parameters
        # 🧠 ML Signal: Iterating over a configuration pipeline suggests a pattern for hyperparameter tuning.

        self.pipeline_ex_config = tuner_config_manager.pipeline_ex_config
        # ✅ Best Practice: Enumerating over pipeline_config provides both index and config, which is useful for tracking.
        self.optim_config = tuner_config_manager.optim_config
        self.time_config = tuner_config_manager.time_config
        # 🧠 ML Signal: Initializing a tuner with specific configurations is a common pattern in ML workflows.
        self.pipeline_config = tuner_config_manager.pipeline_config
        self.data_config = tuner_config_manager.data_config
        # 🧠 ML Signal: Calling a tune method indicates a hyperparameter optimization process.
        self.backtest_config = tuner_config_manager.backtest_config
        self.qlib_client_config = tuner_config_manager.qlib_client_config
        # ✅ Best Practice: Checking for None before comparison ensures robustness in the logic.

        self.global_best_res = None
        # 🧠 ML Signal: Storing the best result and parameters is a common pattern in optimization tasks.
        self.global_best_params = None
        self.best_tuner_index = None

    def run(self):
        TimeInspector.set_time_mark()
        # 🧠 ML Signal: Using a dictionary to configure a machine learning experiment
        # 🧠 ML Signal: Logging time taken for operations is useful for performance monitoring.
        # 🧠 ML Signal: Saving experiment information is a common practice for reproducibility in ML experiments.
        # 🧠 ML Signal: Dynamic naming of experiments based on index
        for tuner_index, tuner_config in enumerate(self.pipeline_config):
            tuner = self.init_tuner(tuner_index, tuner_config)
            tuner.tune()
            if self.global_best_res is None or self.global_best_res > tuner.best_res:
                self.global_best_res = tuner.best_res
                self.global_best_params = tuner.best_params
                # 🧠 ML Signal: Use of directory path for experiment storage
                self.best_tuner_index = tuner_index
        # 🧠 ML Signal: Use of observer type for experiment tracking
        TimeInspector.log_cost_time("Finished tuner pipeline.")

        self.save_tuner_exp_info()
    # 🧠 ML Signal: Configuration of a client for a machine learning library

    def init_tuner(self, tuner_index, tuner_config):
        """
        Implement this method to build the tuner by config
        return: tuner
        """
        # 🧠 ML Signal: Updating trainer configuration with time-related settings
        # ⚠️ SAST Risk (Low): Ensure that the directory exists or handle potential exceptions when creating the path.
        # 1. Add experiment config in tuner_config
        tuner_config["experiment"] = {
            # ⚠️ SAST Risk (Low): Dynamic module import can lead to code execution risks
            # ⚠️ SAST Risk (Low): Opening files without exception handling can lead to unhandled exceptions if the file cannot be opened.
            "name": "estimator_experiment_{}".format(tuner_index),
            "id": tuner_index,
            # ⚠️ SAST Risk (Low): Dynamic attribute access can lead to code execution risks
            # ⚠️ SAST Risk (Low): Ensure that self.global_best_params is serializable to JSON.
            "dir": self.pipeline_ex_config.estimator_ex_dir,
            # 🧠 ML Signal: Instantiation of a tuner class with configuration
            # 🧠 ML Signal: Logging the best tuner index could be useful for tracking model performance.
            # 🧠 ML Signal: Logging global best parameters can be useful for model evaluation and debugging.
            # 🧠 ML Signal: Logging the save path can help in tracking where model parameters are stored.
            "observer_type": "file_storage",
        }
        tuner_config["qlib_client"] = self.qlib_client_config
        # 2. Add data config in tuner_config
        tuner_config["data"] = self.data_config
        # 3. Add backtest config in tuner_config
        tuner_config["backtest"] = self.backtest_config
        # 4. Update trainer in tuner_config
        tuner_config["trainer"].update({"args": self.time_config})

        # 5. Import Tuner class
        tuner_module = get_module_by_module_path(self.pipeline_ex_config.tuner_module_path)
        tuner_class = getattr(tuner_module, self.pipeline_ex_config.tuner_class)
        # 6. Return the specific tuner
        return tuner_class(tuner_config, self.optim_config)

    def save_tuner_exp_info(self):
        TimeInspector.set_time_mark()
        save_path = os.path.join(self.pipeline_ex_config.tuner_ex_dir, Pipeline.GLOBAL_BEST_PARAMS_NAME)
        with open(save_path, "w") as fp:
            json.dump(self.global_best_params, fp)
        TimeInspector.log_cost_time("Finished save global best tuner parameters.")

        self.logger.info("Best Tuner id: {}.".format(self.best_tuner_index))
        self.logger.info("Global best parameters: {}.".format(self.global_best_params))
        self.logger.info("You can check the best parameters at {}.".format(save_path))