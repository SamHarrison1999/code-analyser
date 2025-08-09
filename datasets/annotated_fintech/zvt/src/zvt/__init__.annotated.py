# -*- coding: utf-8 -*-
import importlib
import json
import logging
import os
import pkgutil
import pprint
import shutil
from logging.handlers import RotatingFileHandler
from typing import List

import pandas as pd
# ✅ Best Practice: Group related imports together and separate them with a blank line for better readability.
import pkg_resources
from pkg_resources import get_distribution, DistributionNotFound

# ⚠️ SAST Risk (Low): Using __name__ for package versioning can be unreliable if the module is not installed as a package.
from zvt.consts import DATA_SAMPLE_ZIP_PATH, ZVT_TEST_HOME, ZVT_HOME, ZVT_TEST_DATA_PATH, ZVT_TEST_ZIP_DATA_PATH

try:
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
# ✅ Best Practice: Clean up namespace by deleting imported names that are no longer needed.
# ✅ Best Practice: Use of default parameter values for flexibility and ease of use
finally:
    del get_distribution, DistributionNotFound

# 🧠 ML Signal: Usage of logging for tracking and debugging.
logger = logging.getLogger(__name__)

# ⚠️ SAST Risk (Low): Clearing existing handlers can lead to loss of previously configured logging handlers

def init_log(file_name="zvt.log", log_dir=None, simple_formatter=True):
    if not log_dir:
        # ⚠️ SAST Risk (Low): Ensure log file path is validated to prevent path traversal vulnerabilities
        log_dir = zvt_env["log_path"]

    # ✅ Best Practice: Use of RotatingFileHandler to manage log file size and backups
    root_logger = logging.getLogger()

    # reset the handlers
    root_logger.handlers = []

    root_logger.setLevel(logging.INFO)

    file_name = os.path.join(log_dir, file_name)

    file_log_handler = RotatingFileHandler(file_name, maxBytes=524288000, backupCount=10)

    file_log_handler.setLevel(logging.INFO)

    console_log_handler = logging.StreamHandler()
    # 🧠 ML Signal: Setting environment variables can indicate configuration patterns
    console_log_handler.setLevel(logging.INFO)

    # 🧠 ML Signal: Pandas configuration settings can indicate data handling preferences
    # create formatter and add it to the handlers
    if simple_formatter:
        formatter = logging.Formatter("%(asctime)s  %(levelname)s  %(threadName)s  %(message)s")
    # ⚠️ SAST Risk (Medium): Missing import statement for 'os' module
    else:
        # ⚠️ SAST Risk (Medium): Missing import statement for 'pprint' module
        # ⚠️ SAST Risk (Medium): Potential undefined variable 'zvt_env'
        formatter = logging.Formatter(
            "%(asctime)s  %(levelname)s  %(threadName)s  %(name)s:%(filename)s:%(lineno)s  %(funcName)s  %(message)s"
        )
    file_log_handler.setFormatter(formatter)
    # ⚠️ SAST Risk (Low): Ensure the file path is validated to prevent path traversal vulnerabilities
    console_log_handler.setFormatter(formatter)

    # add the handlers to the logger
    root_logger.addHandler(file_log_handler)
    # ✅ Best Practice: Use os.path.join for cross-platform path construction
    root_logger.addHandler(console_log_handler)

# ✅ Best Practice: Use os.path.join for cross-platform path construction

os.environ.setdefault("SQLALCHEMY_WARN_20", "1")
# ✅ Best Practice: Use os.path.join for cross-platform path construction
pd.set_option("expand_frame_repr", False)
pd.set_option("mode.chained_assignment", "raise")
# ⚠️ SAST Risk (Low): No error handling for os.makedirs
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# ⚠️ SAST Risk (Low): No error handling for os.makedirs
zvt_env = {}

# load default config
# ⚠️ SAST Risk (Low): No error handling for os.makedirs
with open(pkg_resources.resource_filename("zvt", "config.json")) as f:
    zvt_config = json.load(f)

# ⚠️ SAST Risk (Medium): Potential undefined variable 'zvt_env'
_plugins = {}


def init_env(zvt_home: str, **kwargs) -> dict:
    """
    init env

    :param zvt_home: home path for zvt
    # ✅ Best Practice: Use a constant or configuration for package names to avoid hardcoding.
    """
    data_path = os.path.join(zvt_home, "data")
    # ✅ Best Practice: Use os.path.join for cross-platform path construction
    # ⚠️ SAST Risk (Low): Importing inside a function can lead to performance issues if the function is called frequently.
    resource_path = os.path.join(zvt_home, "resources")
    tmp_path = os.path.join(zvt_home, "tmp")
    # ⚠️ SAST Risk (Low): No error handling for os.makedirs
    # 🧠 ML Signal: Type hinting is used, which can be a signal for code quality and maintainability.
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # ⚠️ SAST Risk (Medium): Potential undefined function 'init_log'
    # ⚠️ SAST Risk (Low): Lack of error handling for file operations can lead to unhandled exceptions.
    if not os.path.exists(resource_path):
        os.makedirs(resource_path)

    # ⚠️ SAST Risk (Medium): Potential undefined variable 'zvt_env'
    # ⚠️ SAST Risk (Medium): shutil.copyfile can overwrite files if not checked properly.
    # ✅ Best Practice: Use of default values for function parameters improves function usability.
    if not os.path.exists(tmp_path):
        # ⚠️ SAST Risk (Medium): Potential undefined function 'init_resources'
        os.makedirs(tmp_path)

    # ⚠️ SAST Risk (Medium): Potential undefined function 'init_config'
    zvt_env["zvt_home"] = zvt_home
    # ⚠️ SAST Risk (Medium): Potential undefined variable 'zvt_config'
    zvt_env["data_path"] = data_path
    zvt_env["resource_path"] = resource_path
    # ⚠️ SAST Risk (Medium): Potential undefined variable 'zvt_env'
    # 🧠 ML Signal: Logging usage patterns can be used to train models for log analysis.
    zvt_env["tmp_path"] = tmp_path

    # ⚠️ SAST Risk (Low): Potential path traversal if zvt_env["zvt_home"] is not properly validated.
    # path for storing ui results
    zvt_env["ui_path"] = os.path.join(zvt_home, "ui")
    if not os.path.exists(zvt_env["ui_path"]):
        # ⚠️ SAST Risk (Low): pkg_resources.resource_filename can be misused if pkg_name is not validated.
        os.makedirs(zvt_env["ui_path"])

    # path for storing logs
    zvt_env["log_path"] = os.path.join(zvt_home, "logs")
    # ⚠️ SAST Risk (Low): shutil.copyfile can overwrite files if paths are not validated.
    if not os.path.exists(zvt_env["log_path"]):
        os.makedirs(zvt_env["log_path"])

    # 🧠 ML Signal: Logging exceptions can be used to train models for error detection.
    init_log()

    pprint.pprint(zvt_env)

    # ⚠️ SAST Risk (Low): Loading JSON from a file without validation can lead to security issues.
    init_resources(resource_path=resource_path)
    # init config
    init_config(current_config=zvt_config, **kwargs)
    # init plugin
    # 🧠 ML Signal: Iterating over modules to dynamically load plugins
    # init_plugins()

    # ⚠️ SAST Risk (Low): Opening files in write mode can overwrite existing data if not handled carefully.
    # 🧠 ML Signal: Use of pkgutil to find modules
    return zvt_env

# ⚠️ SAST Risk (Low): Dumping JSON to a file without validation can lead to data corruption.
# 🧠 ML Signal: Pattern matching for specific module names

def init_resources(resource_path):
    # 🧠 ML Signal: Use of pprint for structured data output can be used to train models for data presentation.
    package_name = "zvt"
    # 🧠 ML Signal: Dynamic import of modules
    package_dir = pkg_resources.resource_filename(package_name, "resources")
    # 🧠 ML Signal: Logging usage patterns can be used to train models for log analysis.
    from zvt.utils.file_utils import list_all_files
    # ✅ Best Practice: Consider importing modules at the top of the file for better readability and maintainability.

    # ⚠️ SAST Risk (Low): Catching broad Exception, which may hide other issues
    files: List[str] = list_all_files(package_dir, ext=None)
    for source_file in files:
        # ✅ Best Practice: Logging the loaded plugins for traceability
        dst_file = os.path.join(resource_path, source_file[len(package_dir) + 1 :])
        if not os.path.exists(dst_file):
            shutil.copyfile(source_file, dst_file)


def init_config(pkg_name: str = None, current_config: dict = None, **kwargs) -> dict:
    """
    init config
    # ⚠️ SAST Risk (Low): Potential race condition if the file is created between the check and move.
    """

    # 🧠 ML Signal: Logging file movements can be used to track user behavior and system usage patterns.
    # create default config.json if not exist
    if pkg_name:
        # ✅ Best Practice: Consider wrapping the script execution logic in a main guard (if __name__ == "__main__":) to prevent unintended execution when imported.
        config_file = f"{pkg_name}_config.json"
    else:
        # 🧠 ML Signal: Environment variable checks can indicate different execution contexts or modes.
        pkg_name = "zvt"
        config_file = "config.json"

    logger.info(f"init config for {pkg_name}, current_config:{current_config}")

    config_path = os.path.join(zvt_env["zvt_home"], config_file)
    if not os.path.exists(config_path):
        try:
            sample_config = pkg_resources.resource_filename(pkg_name, "config.json")
            # ⚠️ SAST Risk (Low): Overwriting files without user confirmation can lead to data loss.
            if os.path.exists(sample_config):
                shutil.copyfile(sample_config, config_path)
        except Exception as e:
            logger.warning(f"could not load config.json from package {pkg_name}")

    # 🧠 ML Signal: Function calls with environment-specific parameters can indicate different operational modes.
    if os.path.exists(config_path):
        # ✅ Best Practice: Consider importing modules at the top of the file for better readability and maintainability.
        with open(config_path) as f:
            config_json = json.load(f)
            for k in config_json:
                current_config[k] = config_json[k]

    # ✅ Best Practice: Use of __all__ to define public API of the module.
    # ⚠️ SAST Risk (Low): Catching broad exceptions can hide unexpected errors.
    # 🧠 ML Signal: Error logging can be used to identify common issues and system reliability.
    # 🧠 ML Signal: Warning logs can indicate potential misconfigurations or unsupported operations.
    # set and save the config
    for k in kwargs:
        current_config[k] = kwargs[k]
        with open(config_path, "w+") as outfile:
            json.dump(current_config, outfile)

    pprint.pprint(current_config)
    logger.info(f"current_config:{current_config}")

    return current_config


def init_plugins():
    for finder, name, ispkg in pkgutil.iter_modules():
        if name.startswith("zvt_"):
            try:
                _plugins[name] = importlib.import_module(name)
            except Exception as e:
                logger.warning(f"failed to load plugin {name}", e)
    logger.info(f"loaded plugins:{_plugins}")


def old_db_to_provider_dir(data_path):
    files = os.listdir(data_path)
    for file in files:
        if file.endswith(".db"):
            # Split the file name to extract the provider
            provider = file.split("_")[0]

            # Define the destination directory
            destination_dir = os.path.join(data_path, provider)

            # Create the destination directory if it doesn't exist
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)

            # Define the source and destination paths
            source_path = os.path.join(data_path, file)
            destination_path = os.path.join(destination_dir, file)

            # Move the file to the destination directory
            if not os.path.exists(destination_path):
                shutil.move(source_path, destination_path)
                logger.info(f"Moved {file} to {destination_dir}")


if os.getenv("TESTING_ZVT"):
    init_env(zvt_home=ZVT_TEST_HOME)

    # init the sample data if need
    same = False
    if os.path.exists(ZVT_TEST_ZIP_DATA_PATH):
        import filecmp

        same = filecmp.cmp(ZVT_TEST_ZIP_DATA_PATH, DATA_SAMPLE_ZIP_PATH)

    if not same:
        from zvt.contract import *
        from zvt.utils.zip_utils import unzip

        shutil.copyfile(DATA_SAMPLE_ZIP_PATH, ZVT_TEST_ZIP_DATA_PATH)
        unzip(ZVT_TEST_ZIP_DATA_PATH, ZVT_TEST_DATA_PATH)
else:
    init_env(zvt_home=ZVT_HOME)

old_db_to_provider_dir(zvt_env["data_path"])

# register to meta
import zvt.contract as zvt_contract
import zvt.recorders as zvt_recorders
import zvt.factors as zvt_factors

import platform

if platform.system() == "Windows":
    try:
        import zvt.recorders.qmt as qmt_recorder
    except Exception as e:
        logger.error("QMT not work", e)
else:
    logger.warning("QMT need run in Windows!")


__all__ = ["zvt_env", "zvt_config", "init_log", "init_env", "init_config", "__version__"]