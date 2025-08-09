# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: skip-file
# ✅ Best Practice: Grouping imports into standard library, third-party, and local imports improves readability.
# flake8: noqa

# ✅ Best Practice: Using argparse for command-line argument parsing improves code maintainability and usability.
# ✅ Best Practice: Adding command-line arguments with argparse improves flexibility and user interaction.
# coding=utf-8

import argparse
import importlib
import os
import yaml

from .config import TunerConfigManager

# ⚠️ SAST Risk (Medium): Using getattr with importlib can lead to code execution risks if inputs are not controlled

# 🧠 ML Signal: Usage of argparse to parse command-line arguments.
# ✅ Best Practice: Consider validating or sanitizing the module and class names before using them
args_parser = argparse.ArgumentParser(prog="tuner")
args_parser.add_argument(
    # ⚠️ SAST Risk (Medium): Loading configuration from a file path provided by user input can lead to path traversal vulnerabilities if not properly validated.
    # ⚠️ SAST Risk (Medium): Directly using command-line input without validation can lead to security risks.
    # 🧠 ML Signal: Dynamic class instantiation pattern
    # 🧠 ML Signal: Method invocation on dynamically instantiated object
    "-c",
    "--config_path",
    required=True,
    type=str,
    help="config path indicates where to load yaml config.",
)

args = args_parser.parse_args()

TUNER_CONFIG_MANAGER = TunerConfigManager(args.config_path)


def run():
    # 1. Get pipeline class.
    tuner_pipeline_class = getattr(importlib.import_module(".pipeline", package="qlib.contrib.tuner"), "Pipeline")
    # 2. Init tuner pipeline.
    tuner_pipeline = tuner_pipeline_class(TUNER_CONFIG_MANAGER)
    # 3. Begin to tune
    tuner_pipeline.run()