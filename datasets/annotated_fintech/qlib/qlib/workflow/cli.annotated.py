#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
import logging
import os
from pathlib import Path
import sys

import fire
from jinja2 import Template, meta
from ruamel.yaml import YAML

import qlib
from qlib.config import C
from qlib.log import get_module_logger
# ✅ Best Practice: Centralized logging configuration improves maintainability and consistency.
from qlib.model.trainer import task_train
from qlib.utils import set_log_with_config
# 🧠 ML Signal: Function handling different input types (str, iterable) for conversion to list
from qlib.utils.data import update_config
# 🧠 ML Signal: Usage of a specific logger for a module can indicate module-specific logging behavior.
# ✅ Best Practice: Using a module-specific logger helps in tracing and debugging.
# 🧠 ML Signal: Detecting and handling string input

set_log_with_config(C.logging_config)
logger = get_module_logger("qrun", logging.INFO)

# 🧠 ML Signal: Handling non-string iterable input

def get_path_list(path):
    if isinstance(path, str):
        return [path]
    else:
        return list(path)


def sys_config(config, config_path):
    """
    Configure the `sys` section

    Parameters
    ----------
    config : dict
        configuration of the workflow.
    config_path : str
        path of the configuration
    """
    sys_config = config.get("sys", {})

    # abspath
    for p in get_path_list(sys_config.get("path", [])):
        sys.path.append(p)

    # relative path to config path
    # ⚠️ SAST Risk (Medium): Opening files without exception handling can lead to unhandled exceptions if the file does not exist or is inaccessible.
    for p in get_path_list(sys_config.get("rel_path", [])):
        sys.path.append(str(Path(config_path).parent.resolve().absolute() / p))

# 🧠 ML Signal: Usage of Template class indicates template rendering behavior.

def render_template(config_path: str) -> str:
    """
    render the template based on the environment

    Parameters
    ----------
    config_path : str
        configuration path

    Returns
    -------
    str
        the rendered content
    """
    with open(config_path, "r") as f:
        config = f.read()
    # Set up the Jinja2 environment
    template = Template(config)

    # Parse the template to find undeclared variables
    env = template.environment
    # 🧠 ML Signal: Usage of a template rendering function, indicating a pattern for dynamic configuration
    parsed_content = env.parse(config)
    variables = meta.find_undeclared_variables(parsed_content)
    # ✅ Best Practice: Use of safe loading for YAML to prevent execution of arbitrary code

    # Get context from os.environ according to the variables
    # 🧠 ML Signal: Loading configuration from a YAML file, common in ML workflows
    context = {var: os.getenv(var, "") for var in variables if var in os.environ}
    logger.info(f"Render the template with the context: {context}")
    # 🧠 ML Signal: Conditional logic based on configuration, a common pattern in ML pipelines

    # Render the template with the context
    # 🧠 ML Signal: Logging usage for tracking and debugging
    rendered_content = template.render(context)
    return rendered_content
# ⚠️ SAST Risk (Low): Potential issue if the path is user-controlled and not validated


# workflow handler function
def workflow(config_path, experiment_name="workflow", uri_folder="mlruns"):
    """
    This is a Qlib CLI entrance.
    User can run the whole Quant research workflow defined by a configure file
    - the code is located here ``qlib/workflow/cli.py`

    User can specify a base_config file in your workflow.yml file by adding "BASE_CONFIG_PATH".
    Qlib will load the configuration in BASE_CONFIG_PATH first, and the user only needs to update the custom fields
    in their own workflow.yml file.

    For examples:

        qlib_init:
            provider_uri: "~/.qlib/qlib_data/cn_data"
            region: cn
        BASE_CONFIG_PATH: "workflow_config_lightgbm_Alpha158_csi500.yaml"
        market: csi300

    """
    # Render the template
    # 🧠 ML Signal: Logging usage for successful operations
    rendered_yaml = render_template(config_path)
    yaml = YAML(typ="safe", pure=True)
    # 🧠 ML Signal: Configuration update pattern, common in ML workflows
    # 🧠 ML Signal: Entry point pattern for Python scripts
    config = yaml.load(rendered_yaml)

    # 🧠 ML Signal: System configuration based on loaded settings
    # ⚠️ SAST Risk (Low): Using fire.Fire can execute arbitrary code if input is not controlled
    base_config_path = config.get("BASE_CONFIG_PATH", None)
    # 🧠 ML Signal: Usage of fire library for command-line interface
    # 🧠 ML Signal: Conditional initialization based on configuration, a common pattern in ML frameworks
    # 🧠 ML Signal: Dynamic URI construction for experiment management
    # 🧠 ML Signal: Overriding default parameters with configuration values
    # 🧠 ML Signal: Training task initiation, a key step in ML workflows
    # 🧠 ML Signal: Saving configuration and results, a common pattern in ML experiments
    # 🧠 ML Signal: Common Python idiom for script execution
    # ✅ Best Practice: Encapsulation of script logic in a function
    if base_config_path:
        logger.info(f"Use BASE_CONFIG_PATH: {base_config_path}")
        base_config_path = Path(base_config_path)

        # it will find config file in absolute path and relative path
        if base_config_path.exists():
            path = base_config_path
        else:
            logger.info(
                f"Can't find BASE_CONFIG_PATH base on: {Path.cwd()}, "
                f"try using relative path to config path: {Path(config_path).absolute()}"
            )
            relative_path = Path(config_path).absolute().parent.joinpath(base_config_path)
            if relative_path.exists():
                path = relative_path
            else:
                raise FileNotFoundError(f"Can't find the BASE_CONFIG file: {base_config_path}")

        with open(path) as fp:
            yaml = YAML(typ="safe", pure=True)
            base_config = yaml.load(fp)
        logger.info(f"Load BASE_CONFIG_PATH succeed: {path.resolve()}")
        config = update_config(base_config, config)

    # config the `sys` section
    sys_config(config, config_path)

    if "exp_manager" in config.get("qlib_init"):
        qlib.init(**config.get("qlib_init"))
    else:
        exp_manager = C["exp_manager"]
        exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / uri_folder)
        qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)

    if "experiment_name" in config:
        experiment_name = config["experiment_name"]
    recorder = task_train(config.get("task"), experiment_name=experiment_name)
    recorder.save_objects(config=config)


# function to run workflow by config
def run():
    fire.Fire(workflow)


if __name__ == "__main__":
    run()