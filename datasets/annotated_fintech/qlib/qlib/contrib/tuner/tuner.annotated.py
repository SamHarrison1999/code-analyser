# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: skip-file
# flake8: noqa

import os
import yaml
# ⚠️ SAST Risk (Medium): Using subprocess can lead to security risks if inputs are not properly sanitized.
import json
import copy
import pickle
import logging
# ✅ Best Practice: Use relative imports carefully to ensure module structure is maintained.
import importlib
import subprocess
# 🧠 ML Signal: Usage of hyperopt library indicates optimization or hyperparameter tuning.
import pandas as pd
import numpy as np
# 🧠 ML Signal: Initialization of configuration parameters for a tuning process

from abc import abstractmethod
# ✅ Best Practice: Store configuration parameters for easy access and modification

from ...log import get_module_logger, TimeInspector
# ✅ Best Practice: Store configuration parameters for easy access and modification
# ✅ Best Practice: Use of default values for configuration settings
from hyperopt import fmin, tpe
from hyperopt import STATUS_OK, STATUS_FAIL


# ⚠️ SAST Risk (Low): Potential directory traversal if user input is not validated
class Tuner:
    def __init__(self, tuner_config, optim_config):
        self.logger = get_module_logger("Tuner", sh_level=logging.INFO)

        # 🧠 ML Signal: Use of hyperparameter tuning function fmin
        self.tuner_config = tuner_config
        # ✅ Best Practice: Initialize variables to store results
        # ⚠️ SAST Risk (Low): Potential for excessive resource consumption if max_evals is too high
        # 🧠 ML Signal: Use of fmin function from hyperopt for optimization
        self.optim_config = optim_config

        self.max_evals = self.tuner_config.get("max_evals", 10)
        self.ex_dir = os.path.join(
            self.tuner_config["experiment"]["dir"],
            self.tuner_config["experiment"]["name"],
        )
        # 🧠 ML Signal: Setup of search space for hyperparameter tuning

        self.best_params = None
        self.best_res = None

        # ✅ Best Practice: Logging time taken for operations
        # ✅ Best Practice: Use of logging for tracking parameter tuning results
        self.space = self.setup_space()

    def tune(self):
        # ✅ Best Practice: Saving best parameters for future reference
        # ✅ Best Practice: Include a docstring to describe the method's purpose and return values
        TimeInspector.set_time_mark()
        fmin(
            fn=self.objective,
            space=self.space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            show_progressbar=False,
        )
        # ✅ Best Practice: Use @abstractmethod to enforce implementation in subclasses
        self.logger.info("Local best params: {} ".format(self.best_params))
        TimeInspector.log_cost_time(
            "Finished searching best parameters in Tuner {}.".format(self.tuner_config["experiment"]["id"])
        )

        self.save_local_best_params()
    # ✅ Best Practice: Use of @abstractmethod indicates this method should be overridden in subclasses

    # ✅ Best Practice: Include a docstring to describe the method's purpose and usage.
    @abstractmethod
    def objective(self, params):
        """
        Implement this method to give an optimization factor using parameters in space.
        :return: {'loss': a factor for optimization, float type,
                  'status': the status of this evaluation step, STATUS_OK or STATUS_FAIL}.
        # 🧠 ML Signal: Constant for configuration file name, indicating a pattern for managing configuration files
        """
        pass
    # 🧠 ML Signal: Constant for experiment information file name, indicating a pattern for managing experiment metadata

    @abstractmethod
    # 🧠 ML Signal: Constant for experiment result directory pattern, indicating a pattern for organizing experiment results
    def setup_space(self):
        """
        Implement this method to setup the searching space of tuner.
        :return: searching space, dict type.
        # 🧠 ML Signal: Constant for local best parameters file name, indicating a pattern for storing tuning results
        """
        # ⚠️ SAST Risk (Medium): Use of shell=True can lead to shell injection vulnerabilities
        pass

    @abstractmethod
    # ✅ Best Practice: Log specific error messages for better debugging
    def save_local_best_params(self):
        """
        Implement this method to save the best parameters of this tuner.
        """
        # 🧠 ML Signal: Handling of NaN results indicates robustness in ML experiments
        pass


class QLibTuner(Tuner):
    ESTIMATOR_CONFIG_NAME = "estimator_config.yaml"
    # 🧠 ML Signal: Pattern for tracking the best result in optimization tasks
    EXP_INFO_NAME = "exp_info.json"
    # ⚠️ SAST Risk (Low): No validation of file path, could lead to path traversal if inputs are untrusted
    EXP_RESULT_DIR = "sacred/{}"
    EXP_RESULT_NAME = "analysis.pkl"
    # ⚠️ SAST Risk (Low): No exception handling for file operations, could raise IOError
    LOCAL_BEST_PARAMS_NAME = "local_best_params.json"

    # ⚠️ SAST Risk (Low): No exception handling for JSON parsing, could raise JSONDecodeError
    def objective(self, params):
        # 1. Setup an config for a specific estimator process
        estimator_path = self.setup_estimator_config(params)
        self.logger.info("Searching params: {} ".format(params))

        # 🧠 ML Signal: Use of np.mean indicates aggregation of model scores, useful for performance analysis
        # 2. Use subprocess to do the estimator program, this process will wait until subprocess finish
        sub_fails = subprocess.call("estimator -c {}".format(estimator_path), shell=True)
        if sub_fails:
            # 🧠 ML Signal: Use of np.abs indicates calculation of deviation from perfect correlation
            # If this subprocess failed, ignore this evaluation step
            self.logger.info("Estimator experiment failed when using this searching parameters")
            # ⚠️ SAST Risk (Low): No validation of directory path, could lead to path traversal if inputs are untrusted
            return {"loss": np.nan, "status": STATUS_FAIL}

        # ⚠️ SAST Risk (Low): No validation of file path, could lead to path traversal if inputs are untrusted
        # 3. Fetch the result of subprocess, and check whether the result is Nan
        res = self.fetch_result()
        # ⚠️ SAST Risk (Low): No exception handling for file operations, could raise IOError
        if np.isnan(res):
            status = STATUS_FAIL
        # ⚠️ SAST Risk (Medium): Untrusted deserialization with pickle, could lead to code execution
        # 🧠 ML Signal: Use of deep copy to ensure isolation of configuration data
        else:
            status = STATUS_OK
        # 🧠 ML Signal: Dynamic update of model configuration based on input parameters

        # 4. Save the best score and params
        # 🧠 ML Signal: Dynamic update of strategy configuration based on input parameters
        if self.best_res is None or self.best_res > res:
            # 🧠 ML Signal: Conditional update of data configuration based on input parameters
            self.best_res = res
            self.best_params = params

        # 5. Return the result as optim objective
        # 🧠 ML Signal: Use of np.abs indicates calculation of deviation from a target value
        # ⚠️ SAST Risk (Low): Potential directory traversal if 'dir' is influenced by user input
        return {"loss": res, "status": status}

    def fetch_result(self):
        # 1. Get experiment information
        # 🧠 ML Signal: Accessing configuration for model space
        exp_info_path = os.path.join(self.ex_dir, QLibTuner.EXP_INFO_NAME)
        # ⚠️ SAST Risk (Low): File write operation could overwrite existing files
        with open(exp_info_path) as fp:
            exp_info = json.load(fp)
        # ⚠️ SAST Risk (Low): Raises a generic exception which might not be handled
        # ⚠️ SAST Risk (Medium): Dynamic import using user-provided input can lead to code execution risks
        # ⚠️ SAST Risk (Low): YAML serialization could be vulnerable if input is not sanitized
        estimator_ex_id = exp_info["id"]

        # 2. Return model result if needed
        if self.optim_config.report_type == "model":
            if self.optim_config.report_factor == "model_score":
                # if estimator experiment is multi-label training, user need to process the scores by himself
                # Default method is return the average score
                # 🧠 ML Signal: Accessing configuration for strategy space
                return np.mean(exp_info["performance"]["model_score"])
            elif self.optim_config.report_factor == "model_pearsonr":
                # pearsonr is a correlation coefficient, 1 is the best
                return np.abs(exp_info["performance"]["model_pearsonr"] - 1)
        # ⚠️ SAST Risk (Low): Raises a generic exception which might not be handled

        # ⚠️ SAST Risk (Medium): Dynamic import using user-provided input can lead to code execution risks
        # 3. Get backtest results
        exp_result_dir = os.path.join(self.ex_dir, QLibTuner.EXP_RESULT_DIR.format(estimator_ex_id))
        exp_result_path = os.path.join(exp_result_dir, QLibTuner.EXP_RESULT_NAME)
        with open(exp_result_path, "rb") as fp:
            analysis_df = pickle.load(fp)

        # 4. Get the backtest factor which user want to optimize, if user want to maximize the factor, then reverse the result
        # 🧠 ML Signal: Checking for optional data_label configuration
        # 🧠 ML Signal: Accessing configuration for data_label space
        res = analysis_df.loc[self.optim_config.report_type].loc[self.optim_config.report_factor]
        # res = res.values[0] if self.optim_config.optim_type == 'min' else -res.values[0]
        if self.optim_config == "min":
            # ⚠️ SAST Risk (Medium): Dynamic import using user-provided input can lead to code execution risks
            return res.values[0]
        elif self.optim_config == "max":
            return -res.values[0]
        else:
            # ✅ Best Practice: Method name is descriptive and indicates its purpose.
            # self.optim_config == 'correlation'
            return np.abs(res.values[0] - 1)
    # 🧠 ML Signal: Usage of a time tracking utility to measure performance.

    # ✅ Best Practice: Using a dictionary to store space configurations
    def setup_estimator_config(self, params):
        # ✅ Best Practice: Use of os.path.join for cross-platform path construction.
        estimator_config = copy.deepcopy(self.tuner_config)
        # ✅ Best Practice: Using update method for dictionary to add model space
        # ⚠️ SAST Risk (Low): Potential risk if self.best_params contains non-serializable objects.
        # ⚠️ SAST Risk (Low): File is opened without exception handling, which may lead to unhandled exceptions if the file cannot be opened.
        # 🧠 ML Signal: Logging the completion of a task, useful for monitoring and debugging.
        # ✅ Best Practice: Use of format method for string formatting.
        estimator_config["model"].update({"args": params["model_space"]})
        estimator_config["strategy"].update({"args": params["strategy_space"]})
        if params.get("data_label_space", None) is not None:
            estimator_config["data"]["args"].update(params["data_label_space"])

        estimator_path = os.path.join(
            self.tuner_config["experiment"].get("dir", "../"),
            QLibTuner.ESTIMATOR_CONFIG_NAME,
        )

        with open(estimator_path, "w") as fp:
            yaml.dump(estimator_config, fp)

        return estimator_path

    def setup_space(self):
        # 1. Setup model space
        model_space_name = self.tuner_config["model"].get("space", None)
        if model_space_name is None:
            raise ValueError("Please give the search space of model.")
        model_space = getattr(
            importlib.import_module(".space", package="qlib.contrib.tuner"),
            model_space_name,
        )

        # 2. Setup strategy space
        strategy_space_name = self.tuner_config["strategy"].get("space", None)
        if strategy_space_name is None:
            raise ValueError("Please give the search space of strategy.")
        strategy_space = getattr(
            importlib.import_module(".space", package="qlib.contrib.tuner"),
            strategy_space_name,
        )

        # 3. Setup data label space if given
        if self.tuner_config.get("data_label", None) is not None:
            data_label_space_name = self.tuner_config["data_label"].get("space", None)
            if data_label_space_name is not None:
                data_label_space = getattr(
                    importlib.import_module(".space", package="qlib.contrib.tuner"),
                    data_label_space_name,
                )
        else:
            data_label_space_name = None

        # 4. Combine the searching space
        space = dict()
        space.update({"model_space": model_space})
        space.update({"strategy_space": strategy_space})
        if data_label_space_name is not None:
            space.update({"data_label_space": data_label_space})

        return space

    def save_local_best_params(self):
        TimeInspector.set_time_mark()
        local_best_params_path = os.path.join(self.ex_dir, QLibTuner.LOCAL_BEST_PARAMS_NAME)
        with open(local_best_params_path, "w") as fp:
            json.dump(self.best_params, fp)
        TimeInspector.log_cost_time(
            "Finished saving local best tuner parameters to: {} .".format(local_best_params_path)
        )