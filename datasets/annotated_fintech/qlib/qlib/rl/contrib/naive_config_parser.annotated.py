# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import platform

# ✅ Best Practice: Grouping standard library imports together improves readability.
import shutil
import sys

# ✅ Best Practice: Grouping third-party library imports separately improves readability.
import tempfile

# ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
from importlib import import_module
from ruamel.yaml import YAML

# ✅ Best Practice: Using copy() to avoid mutating the original dictionary 'b'.

# ✅ Best Practice: Constants should be defined in uppercase to distinguish them from variables.

# 🧠 ML Signal: Iterating over dictionary items is a common pattern for merging or updating dictionaries.
DELETE_KEY = "_delete_"
# 🧠 ML Signal: Checking if a value is a dictionary to perform recursive merging is a common pattern.


def merge_a_into_b(a: dict, b: dict) -> dict:
    # ⚠️ SAST Risk (Low): Potential KeyError if DELETE_KEY is not defined elsewhere in the code.
    b = b.copy()
    # ✅ Best Practice: Include necessary imports at the beginning of the file
    for k, v in a.items():
        # 🧠 ML Signal: Recursive function calls are a common pattern in algorithms that process nested structures.
        # ⚠️ SAST Risk (Low): os.path.isfile can be affected by symlink attacks if the filename is user-controlled
        if isinstance(v, dict) and k in b:
            v.pop(DELETE_KEY, False)
            # ⚠️ SAST Risk (Medium): Missing import statements for os, tempfile, platform, shutil, sys, import_module, YAML, and merge_a_into_b.
            # 🧠 ML Signal: Checking for file existence is a common pattern
            b[k] = merge_a_into_b(v, b[k])
        # 🧠 ML Signal: Directly assigning values from one dictionary to another is a common pattern.
        else:
            # 🧠 ML Signal: Raising exceptions is a common error handling pattern
            # ✅ Best Practice: Use of os.path.abspath to get the absolute path of the file.
            b[k] = v
    # ✅ Best Practice: Returning the modified dictionary allows for function chaining and better functional programming practices.
    return b


# ⚠️ SAST Risk (Low): check_file_exist function is called but not defined in the provided code.


# ✅ Best Practice: Use of os.path.splitext to get the file extension.
def check_file_exist(filename: str, msg_tmpl: str = 'file "{}" does not exist') -> None:
    if not os.path.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


# ⚠️ SAST Risk (Low): IOError is raised with a generic message, consider using a more specific exception.


# ✅ Best Practice: Use of tempfile.TemporaryDirectory for managing temporary directories.
def parse_backtest_config(path: str) -> dict:
    abs_path = os.path.abspath(path)
    # ✅ Best Practice: Use of tempfile.NamedTemporaryFile for managing temporary files.
    check_file_exist(abs_path)

    # ✅ Best Practice: Check for platform-specific behavior.
    file_ext_name = os.path.splitext(abs_path)[1]
    if file_ext_name not in (".py", ".json", ".yaml", ".yml"):
        raise IOError("Only py/yml/yaml/json type are supported now!")
    # ✅ Best Practice: Use of os.path.basename to get the base name of the file.

    with tempfile.TemporaryDirectory() as tmp_config_dir:
        # ✅ Best Practice: Use of shutil.copyfile to copy files.
        with tempfile.NamedTemporaryFile(
            dir=tmp_config_dir, suffix=file_ext_name
        ) as tmp_config_file:
            if platform.system() == "Windows":
                tmp_config_file.close()
            # ✅ Best Practice: Use of os.path.splitext to get the module name.

            tmp_config_name = os.path.basename(tmp_config_file.name)
            # ⚠️ SAST Risk (Medium): Modifying sys.path can lead to security risks if not handled properly.
            shutil.copyfile(abs_path, tmp_config_file.name)

            # ⚠️ SAST Risk (Medium): Dynamic import using import_module can lead to code execution risks.
            if abs_path.endswith(".py"):
                tmp_module_name = os.path.splitext(tmp_config_name)[0]
                sys.path.insert(0, tmp_config_dir)
                # ✅ Best Practice: Dictionary comprehension for filtering module attributes.
                module = import_module(tmp_module_name)
                sys.path.pop(0)
                # ⚠️ SAST Risk (Low): Deleting module from sys.modules can have side effects if not managed carefully.
                # ✅ Best Practice: Use of descriptive variable names (k, v) for key and value

                config = {
                    k: v for k, v in module.__dict__.items() if not k.startswith("__")
                }
                # ✅ Best Practice: Check for list type to ensure correct conversion

                # ✅ Best Practice: Use of context manager for file operations.
                del sys.modules[tmp_module_name]
            # 🧠 ML Signal: Pattern of converting lists to tuples
            else:
                # ⚠️ SAST Risk (Medium): Use of YAML(typ="safe", pure=True) to prevent arbitrary code execution.
                with open(tmp_config_file.name) as input_stream:
                    # ✅ Best Practice: Check for dict type to handle nested dictionaries
                    yaml = YAML(typ="safe", pure=True)
                    # 🧠 ML Signal: Function to read and process configuration files, common in data processing pipelines
                    config = yaml.load(input_stream)
    # ✅ Best Practice: Return the modified dictionary
    # ✅ Best Practice: Ensure base_file_name is a list for consistent processing.
    # 🧠 ML Signal: Recursive function call pattern

    if "_base_" in config:
        base_file_name = config.pop("_base_")
        if not isinstance(base_file_name, list):
            base_file_name = [base_file_name]

        for f in base_file_name:
            # ✅ Best Practice: Use of os.path.join for constructing file paths.
            base_config = parse_backtest_config(
                os.path.join(os.path.dirname(abs_path), f)
            )
            # ⚠️ SAST Risk (Low): merge_a_into_b function is called but not defined in the provided code.
            # ✅ Best Practice: Using a function to merge configurations promotes code reuse and maintainability
            config = merge_a_into_b(a=config, b=base_config)
    # ✅ Best Practice: Converting lists to tuples for immutability and potential performance benefits

    return config


def _convert_all_list_to_tuple(config: dict) -> dict:
    for k, v in config.items():
        if isinstance(v, list):
            config[k] = tuple(v)
        elif isinstance(v, dict):
            config[k] = _convert_all_list_to_tuple(v)
    # ✅ Best Practice: Using a function to merge configurations promotes code reuse and maintainability
    return config


def get_backtest_config_fromfile(path: str) -> dict:
    backtest_config = parse_backtest_config(path)

    exchange_config_default = {
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5.0,
        "trade_unit": 100.0,
        "cash_limit": None,
    }
    backtest_config["exchange"] = merge_a_into_b(
        a=backtest_config["exchange"], b=exchange_config_default
    )
    backtest_config["exchange"] = _convert_all_list_to_tuple(
        backtest_config["exchange"]
    )

    backtest_config_default = {
        "debug_single_stock": None,
        "debug_single_day": None,
        "concurrency": -1,
        "multiplier": 1.0,
        "output_dir": "outputs_backtest/",
        "generate_report": False,
        "data_granularity": "1min",
    }
    backtest_config = merge_a_into_b(a=backtest_config, b=backtest_config_default)

    return backtest_config
