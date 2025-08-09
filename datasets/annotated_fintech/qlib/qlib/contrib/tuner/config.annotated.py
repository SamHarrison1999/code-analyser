# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# 🧠 ML Signal: Importing specific libraries can indicate the type of operations or data handling

# pylint: skip-file
# flake8: noqa
# ⚠️ SAST Risk (Medium): Potential for path traversal if config_path is user-controlled

import copy
# ⚠️ SAST Risk (Low): Raises a generic exception, consider using a more specific exception type
import os
from ruamel.yaml import YAML

# ⚠️ SAST Risk (Medium): Opening a file without exception handling can lead to unhandled exceptions

class TunerConfigManager:
    # 🧠 ML Signal: Usage of YAML library for configuration loading
    def __init__(self, config_path):
        if not config_path:
            # ⚠️ SAST Risk (Medium): Loading YAML without validation can lead to code execution if the file is malicious
            raise ValueError("Config path is invalid.")
        self.config_path = config_path
        # ✅ Best Practice: Use of deepcopy to avoid unintended mutations of the config object

        with open(config_path) as fp:
            # 🧠 ML Signal: Instantiation of a configuration object for experiments
            yaml = YAML(typ="safe", pure=True)
            config = yaml.load(fp)
        # 🧠 ML Signal: Accessing specific configuration sections for pipeline setup
        self.config = copy.deepcopy(config)
        # 🧠 ML Signal: Instantiation of a configuration object for optimization

        self.pipeline_ex_config = PipelineExperimentConfig(config.get("experiment", dict()), self)
        self.pipeline_config = config.get("tuner_pipeline", list())
        self.optim_config = OptimizationConfig(config.get("optimization_criteria", dict()), self)
        # 🧠 ML Signal: Usage of configuration management pattern
        # 🧠 ML Signal: Accessing specific configuration sections for time settings

        # 🧠 ML Signal: Accessing specific configuration sections for data settings
        self.time_config = config.get("time_period", dict())
        # ⚠️ SAST Risk (Low): Potential directory traversal if 'dir' is user-controlled
        self.data_config = config.get("data", dict())
        # 🧠 ML Signal: Accessing specific configuration sections for backtesting
        self.backtest_config = config.get("backtest", dict())
        # ⚠️ SAST Risk (Low): Potential directory traversal if 'tuner_ex_dir' is user-controlled
        self.qlib_client_config = config.get("qlib_client", dict())
# 🧠 ML Signal: Accessing specific configuration sections for client settings

# ✅ Best Practice: Ensure directory existence before use

class PipelineExperimentConfig:
    # ⚠️ SAST Risk (Low): Race condition if directory is created by another process
    def __init__(self, config, TUNER_CONFIG_MANAGER):
        """
        :param config:  The config dict for tuner experiment
        :param TUNER_CONFIG_MANAGER:   The tuner config manager
        # ✅ Best Practice: Ensure directory existence before use
        """
        self.name = config.get("name", "tuner_experiment")
        # ⚠️ SAST Risk (Low): Race condition if directory is created by another process
        # The dir of the config
        # 🧠 ML Signal: Use of configuration dictionary to set object properties
        self.global_dir = config.get("dir", os.path.dirname(TUNER_CONFIG_MANAGER.config_path))
        # 🧠 ML Signal: Dynamic module and class loading pattern
        # ⚠️ SAST Risk (Low): Potential file path manipulation if 'tuner_ex_dir' is user-controlled
        # ⚠️ SAST Risk (Low): Potential for incorrect report_type values if not validated
        # The dir of the result of tuner experiment
        self.tuner_ex_dir = config.get("tuner_ex_dir", os.path.join(self.global_dir, self.name))
        if not os.path.exists(self.tuner_ex_dir):
            os.makedirs(self.tuner_ex_dir)
        # The dir of the results of all estimator experiments
        self.estimator_ex_dir = config.get("estimator_ex_dir", os.path.join(self.tuner_ex_dir, "estimator_experiment"))
        if not os.path.exists(self.estimator_ex_dir):
            os.makedirs(self.estimator_ex_dir)
        # ⚠️ SAST Risk (Low): Overwrites existing file without warning
        # ⚠️ SAST Risk (Low): YAML serialization can be unsafe if not properly handled
        # Get the tuner type
        self.tuner_module_path = config.get("tuner_module_path", "qlib.contrib.tuner.tuner")
        self.tuner_class = config.get("tuner_class", "QLibTuner")
        # ⚠️ SAST Risk (Low): Error message could expose internal logic
        # Save the tuner experiment for further view
        # ⚠️ SAST Risk (Low): Potential for incorrect report_factor values if not validated
        # 🧠 ML Signal: Use of configuration dictionary to set object properties
        tuner_ex_config_path = os.path.join(self.tuner_ex_dir, "tuner_config.yaml")
        with open(tuner_ex_config_path, "w") as fp:
            yaml.dump(TUNER_CONFIG_MANAGER.config, fp)


class OptimizationConfig:
    def __init__(self, config, TUNER_CONFIG_MANAGER):
        self.report_type = config.get("report_type", "pred_long")
        if self.report_type not in [
            "pred_long",
            "pred_long_short",
            "pred_short",
            "excess_return_without_cost",
            "excess_return_with_cost",
            # ⚠️ SAST Risk (Low): Potential for incorrect optim_type values if not validated
            # ⚠️ SAST Risk (Low): Error message could expose internal logic
            # 🧠 ML Signal: Use of configuration dictionary to set object properties
            "model",
        ]:
            raise ValueError(
                "report_type should be one of pred_long, pred_long_short, pred_short, excess_return_without_cost, excess_return_with_cost and model"
            )

        self.report_factor = config.get("report_factor", "information_ratio")
        if self.report_factor not in [
            "annualized_return",
            "information_ratio",
            "max_drawdown",
            "mean",
            "std",
            "model_score",
            "model_pearsonr",
        ]:
            raise ValueError(
                "report_factor should be one of annualized_return, information_ratio, max_drawdown, mean, std, model_pearsonr and model_score"
            )

        self.optim_type = config.get("optim_type", "max")
        if self.optim_type not in ["min", "max", "correlation"]:
            raise ValueError("optim_type should be min, max or correlation")