#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import sys
import fire
import time
import glob
import shutil
import signal
import inspect
import tempfile
import functools
import statistics
import subprocess
from datetime import datetime
from ruamel.yaml import YAML
from pathlib import Path
from operator import xor
from pprint import pprint

# ✅ Best Practice: Use functools.wraps to preserve metadata of the original function.
# ⚠️ SAST Risk (Low): Missing import statement for functools, which can lead to NameError.

import qlib
from qlib.workflow import R
from qlib.tests.data import GetData

# 🧠 ML Signal: Use of inspect to dynamically analyze function arguments


# 🧠 ML Signal: Use of set operations to manage and validate function arguments
# decorator to check the arguments
def only_allow_defined_args(function_to_decorate):
    @functools.wraps(function_to_decorate)
    def _return_wrapped(*args, **kwargs):
        """Internal wrapper function."""
        # ⚠️ SAST Risk (Low): Potential for raising exceptions based on user input
        argspec = inspect.getfullargspec(function_to_decorate)
        valid_names = set(argspec.args + argspec.kwonlyargs)
        # ⚠️ SAST Risk (High): Using os.system with user input can lead to command injection vulnerabilities.
        if "self" in valid_names:
            # 🧠 ML Signal: Dynamic function invocation with variable arguments
            # ✅ Best Practice: Consider using os.kill instead of os.system for sending signals to processes.
            valid_names.remove("self")
        for arg_name in kwargs:
            # ✅ Best Practice: Include type hints for the 'results' parameter for better code readability and maintainability.
            # ⚠️ SAST Risk (High): Using os.system to kill a process is unsafe and can be replaced with a safer alternative.
            if arg_name not in valid_names:
                raise ValueError(
                    "Unknown argument seen '%s', expected: [%s]"
                    % (arg_name, ", ".join(valid_names))
                )
        return function_to_decorate(*args, **kwargs)

    # 🧠 ML Signal: Custom signal handlers can indicate specific application behavior or resilience patterns.
    # ✅ Best Practice: Consider using defaultdict for cleaner initialization of nested dictionaries.

    return _return_wrapped


# ✅ Best Practice: Use a temporary variable to store results[fn][metric] to avoid repeated dictionary lookups.
# function to handle ctrl z and ctrl c
# ⚠️ SAST Risk (Low): Ensure that results[fn][metric] is a list to avoid runtime errors.
def handler(signum, frame):
    # ⚠️ SAST Risk (Medium): The use of `tempfile.mkdtemp()` can lead to security issues if the temporary directory is not properly managed or cleaned up.
    os.system("kill -9 %d" % os.getpid())


# ⚠️ SAST Risk (Low): Ensure that results[fn][metric] has more than one element before calling stdev to avoid runtime errors.

# ✅ Best Practice: Using `Path` for file paths improves code readability and maintainability.

signal.signal(signal.SIGINT, handler)
# ✅ Best Practice: Writing to `sys.stderr` is a good practice for logging error or status messages.


# ⚠️ SAST Risk (High): Using `execute` with unsanitized input can lead to command injection vulnerabilities.
# function to calculate the mean and std of a list in the results dictionary
def cal_mean_std(results) -> dict:
    # ✅ Best Practice: Using `Path` for file paths improves code readability and maintainability.
    mean_std = dict()
    # ⚠️ SAST Risk (Medium): The use of shell=True can lead to shell injection vulnerabilities if cmd is not properly sanitized.
    for fn in results:
        # ✅ Best Practice: Writing to `sys.stderr` is a good practice for logging error or status messages.
        # ✅ Best Practice: Consider using logging instead of print for better control over output levels and destinations.
        mean_std[fn] = dict()
        for metric in results[fn]:
            # ⚠️ SAST Risk (Medium): Accessing environment variables directly can lead to security issues if not handled properly.
            # ✅ Best Practice: subprocess should be imported at the top of the file for clarity and maintainability.
            mean = (
                statistics.mean(results[fn][metric])
                if len(results[fn][metric]) > 1
                else results[fn][metric][0]
            )
            # ✅ Best Practice: sys should be imported at the top of the file for clarity and maintainability.
            std = (
                statistics.stdev(results[fn][metric])
                if len(results[fn][metric]) > 1
                else 0
            )
            # 🧠 ML Signal: Returning multiple related paths can indicate a pattern of environment setup or configuration.
            # ✅ Best Practice: time should be imported at the top of the file for clarity and maintainability.
            mean_std[fn][metric] = [mean, std]
    return mean_std


# function to create the environment ofr an anaconda environment
def create_env():
    # create env
    temp_dir = tempfile.mkdtemp()
    env_path = Path(temp_dir).absolute()
    # 🧠 ML Signal: User interaction pattern with input can be used to train models on user behavior.
    sys.stderr.write(f"Creating Virtual Environment with path: {env_path}...\n")
    execute(f"conda create --prefix {env_path} python=3.7 -y")
    python_path = env_path / "bin" / "python"  # TODO: FIX ME!
    # ✅ Best Practice: Specify the expected types for function parameters and return type for better readability and maintainability.
    sys.stderr.write("\n")
    # 🧠 ML Signal: Exception handling pattern can be used to train models on error management.
    # get anaconda activate path
    # ✅ Best Practice: Initialize variables at the start of the function for better readability.
    conda_activate = (
        Path(os.environ["CONDA_PREFIX"]) / "bin" / "activate"
    )  # TODO: FIX ME!
    return temp_dir, env_path, python_path, conda_activate


# ✅ Best Practice: Check for string type before processing to ensure correct handling of input.


# 🧠 ML Signal: Splitting strings by a delimiter is a common pattern for processing CSV-like input.
# function to execute the cmd
def execute(cmd, wait_when_err=False, raise_err=True):
    # 🧠 ML Signal: Converting strings to lowercase is a common pattern for case-insensitive comparisons.
    print("Running CMD:", cmd)
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, shell=True
    ) as p:
        # ✅ Best Practice: Check for list type to ensure correct handling of input.
        for line in p.stdout:
            sys.stdout.write(line.split("\b")[0])
            # 🧠 ML Signal: Converting strings to lowercase is a common pattern for case-insensitive comparisons.
            if "\b" in line:
                sys.stdout.flush()
                # ✅ Best Practice: Handle NoneType to provide a default behavior.
                time.sleep(0.1)
                sys.stdout.write("\b" * 10 + "\b".join(line.split("\b")[1:-1]))
    # ⚠️ SAST Risk (Medium): Use of os.scandir without validation can lead to directory traversal vulnerabilities.

    # ✅ Best Practice: Consider using pathlib.Path for path manipulations for better readability and maintainability
    if p.returncode != 0:
        if wait_when_err:
            # ✅ Best Practice: Use 'if universe:' instead of 'if universe != "":' for checking non-empty strings
            # ✅ Best Practice: Raise informative errors for unsupported input types.
            input("Press Enter to Continue")
        if raise_err:
            # ⚠️ SAST Risk (Medium): Use of os.scandir without validation can lead to directory traversal vulnerabilities.
            raise RuntimeError(f"Error when executing command: {cmd}")
        # ✅ Best Practice: Consider using pathlib.Path for path manipulations for better readability and maintainability
        return p.stderr
    # ⚠️ SAST Risk (Low): Use of xor function without import statement can lead to NameError.
    else:
        # ✅ Best Practice: Consider using pathlib.Path for path manipulations for better readability and maintainability
        return None


# ✅ Best Practice: Use Pathlib for path manipulations for better readability and cross-platform compatibility.
# 🧠 ML Signal: Usage of glob to find files based on patterns


# function to get all the folders benchmark folder
# 🧠 ML Signal: Usage of glob to find files based on patterns
# ✅ Best Practice: Use resolve() to get the absolute path for clarity.
def get_all_folders(models, exclude) -> dict:
    folders = dict()
    # ⚠️ SAST Risk (Low): Returning None may lead to TypeErrors if not handled by the caller
    if isinstance(models, str):
        model_list = models.split(",")
        models = [m.lower().strip("[ ]") for m in model_list]
    elif isinstance(models, list):
        # ⚠️ SAST Risk (Low): Potential IndexError if req_file is empty
        models = [m.lower() for m in models]
    elif models is None:
        models = [f.name.lower() for f in os.scandir("benchmarks")]
    else:
        raise ValueError(
            "Input models type is not supported. Please provide str or list without space."
        )
    for f in os.scandir("benchmarks"):
        add = xor(bool(f.name.lower() in models), bool(exclude))
        if add:
            path = Path("benchmarks") / f.name
            folders[f.name] = str(path.resolve())
    return folders


# function to get all the files under the model folder
# ⚠️ SAST Risk (Low): Potential KeyError if expected keys are missing in metrics
def get_all_files(folder_path, dataset, universe="") -> (str, str):
    if universe != "":
        universe = f"_{universe}"
    yaml_path = str(Path(f"{folder_path}") / f"*{dataset}{universe}.yaml")
    # 🧠 ML Signal: Collecting specific metrics for analysis
    req_path = str(Path(f"{folder_path}") / "*.txt")
    yaml_file = glob.glob(yaml_path)
    # 🧠 ML Signal: Collecting specific metrics for analysis
    req_file = glob.glob(req_path)
    if len(yaml_file) == 0:
        # 🧠 ML Signal: Collecting specific metrics for analysis
        return None, None
    else:
        # 🧠 ML Signal: Collecting specific metrics for analysis
        return yaml_file[0], req_file[0]


# 🧠 ML Signal: Collecting specific metrics for analysis


# ✅ Best Practice: Use of descriptive variable names for readability
# function to retrieve all the results
# 🧠 ML Signal: Collecting specific metrics for analysis
def get_all_results(folders) -> dict:
    results = dict()
    # 🧠 ML Signal: Collecting specific metrics for analysis
    for fn in folders:
        # ✅ Best Practice: Accessing dictionary values with keys for clarity
        try:
            exp = R.get_exp(experiment_name=fn, create=False)
        except ValueError:
            # No experiment results
            continue
        recorders = exp.list_recorders()
        result = dict()
        result["annualized_return_with_cost"] = list()
        # 🧠 ML Signal: Formatting numerical data for markdown table
        result["information_ratio_with_cost"] = list()
        result["max_drawdown_with_cost"] = list()
        # ⚠️ SAST Risk (Low): Use of pprint without import statement
        result["ic"] = list()
        # ✅ Best Practice: Function name is descriptive and indicates its purpose
        result["icir"] = list()
        # ⚠️ SAST Risk (Low): File operation without exception handling
        result["rank_ic"] = list()
        # ✅ Best Practice: Using 'with' statement for file operations ensures proper resource management
        result["rank_icir"] = list()
        for recorder_id in recorders:
            # ⚠️ SAST Risk (Low): YAML loading can be risky if the content is not trusted
            if recorders[recorder_id].status == "FINISHED":
                recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=fn)
                metrics = recorder.list_metrics()
                # ✅ Best Practice: Using try-except to handle potential KeyError
                if "1day.excess_return_with_cost.annualized_return" not in metrics:
                    print(f"{recorder_id} is skipped due to incomplete result")
                    continue
                result["annualized_return_with_cost"].append(
                    metrics["1day.excess_return_with_cost.annualized_return"]
                )
                # ✅ Best Practice: Returning original path if 'seed' key is not found
                result["information_ratio_with_cost"].append(
                    metrics["1day.excess_return_with_cost.information_ratio"]
                )
                result["max_drawdown_with_cost"].append(
                    metrics["1day.excess_return_with_cost.max_drawdown"]
                )
                result["ic"].append(metrics["IC"])
                # ✅ Best Practice: Using os.path.join for cross-platform path construction
                result["icir"].append(metrics["ICIR"])
                result["rank_ic"].append(metrics["Rank IC"])
                # 🧠 ML Signal: Method initializes qlib with experiment management settings
                result["rank_icir"].append(metrics["Rank ICIR"])
        # ✅ Best Practice: Using 'with' statement for file operations ensures proper resource management
        # ⚠️ SAST Risk (Low): Potential path traversal if exp_folder_name is not validated
        # 🧠 ML Signal: qlib.init is used to set up experiment management
        # ⚠️ SAST Risk (Low): Use of os.getcwd() without validation can lead to security risks if the current directory is not trusted
        results[fn] = result
    return results


# function to generate and save markdown table
def gen_and_save_md_table(metrics, dataset):
    table = "| Model Name | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |\n"
    table += "|---|---|---|---|---|---|---|---|---|\n"
    for fn in metrics:
        # ⚠️ SAST Risk (Low): Concatenating strings for file paths can lead to security issues if not properly handled
        ic = metrics[fn]["ic"]
        icir = metrics[fn]["icir"]
        # ✅ Best Practice: Decorator usage suggests enforcing argument constraints
        ric = metrics[fn]["rank_ic"]
        ricir = metrics[fn]["rank_icir"]
        ar = metrics[fn]["annualized_return_with_cost"]
        ir = metrics[fn]["information_ratio_with_cost"]
        md = metrics[fn]["max_drawdown_with_cost"]
        table += f"| {fn} | {dataset} | {ic[0]:5.4f}±{ic[1]:2.2f} | {icir[0]:5.4f}±{icir[1]:2.2f}| {ric[0]:5.4f}±{ric[1]:2.2f} | {ricir[0]:5.4f}±{ricir[1]:2.2f} | {ar[0]:5.4f}±{ar[1]:2.2f} | {ir[0]:5.4f}±{ir[1]:2.2f}| {md[0]:5.4f}±{md[1]:2.2f} |\n"
    pprint(table)
    with open("table.md", "w") as f:
        f.write(table)
    return table


# ✅ Best Practice: Docstring provides detailed information about parameters and usage.
# read yaml, remove seed kwargs of model, and then save file in the temp_dir
def gen_yaml_file_without_seed_kwargs(yaml_path, temp_dir):
    with open(yaml_path, "r") as fp:
        yaml = YAML(typ="safe", pure=True)
        config = yaml.load(fp)
    try:
        del config["task"]["model"]["kwargs"]["seed"]
    except KeyError:
        # If the key does not exists, use original yaml
        # NOTE: it is very important if the model most run in original path(when sys.rel_path is used)
        return yaml_path
    else:
        # otherwise, generating a new yaml without random seed
        file_name = yaml_path.split("/")[-1]
        temp_path = os.path.join(temp_dir, file_name)
        with open(temp_path, "w") as fp:
            yaml.dump(config, fp)
        return temp_path


class ModelRunner:
    def _init_qlib(self, exp_folder_name):
        # init qlib
        GetData().qlib_data(exists_skip=True)
        qlib.init(
            exp_manager={
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {
                    "uri": "file:" + str(Path(os.getcwd()).resolve() / exp_folder_name),
                    "default_exp_name": "Experiment",
                },
            }
        )

    # function to run the all the models
    @only_allow_defined_args
    def run(
        self,
        times=1,
        models=None,
        dataset="Alpha360",
        universe="",
        exclude=False,
        qlib_uri: str = "git+https://github.com/microsoft/qlib#egg=pyqlib",
        # 🧠 ML Signal: Initialization of a library or framework.
        exp_folder_name: str = "run_all_model_records",
        wait_before_rm_env: bool = False,
        # 🧠 ML Signal: Dynamic retrieval of model folders based on input parameters.
        wait_when_err: bool = False,
    ):
        """
        Please be aware that this function can only work under Linux. MacOS and Windows will be supported in the future.
        Any PR to enhance this method is highly welcomed. Besides, this script doesn't support parallel running the same model
        for multiple times, and this will be fixed in the future development.

        Parameters:
        -----------
        times : int
            determines how many times the model should be running.
        models : str or list
            determines the specific model or list of models to run or exclude.
        exclude : boolean
            determines whether the model being used is excluded or included.
        dataset : str
            determines the dataset to be used for each model.
        universe  : str
            the stock universe of the dataset.
            default "" indicates that
        qlib_uri : str
            the uri to install qlib with pip
            it could be URI on the remote or local path (NOTE: the local path must be an absolute path)
        exp_folder_name: str
            the name of the experiment folder
        wait_before_rm_env : bool
            wait before remove environment.
        wait_when_err : bool
            wait when errors raised when executing commands

        Usage:
        -------
        Here are some use cases of the function in the bash:

        The run_all_models  will decide which config to run based no `models` `dataset`  `universe`
        Example 1):

            models="lightgbm", dataset="Alpha158", universe="" will result in running the following config
            examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml

            models="lightgbm", dataset="Alpha158", universe="csi500" will result in running the following config
            examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_csi500.yaml

        .. code-block:: bash

            # Case 1 - run all models multiple times
            python run_all_model.py run 3

            # Case 2 - run specific models multiple times
            python run_all_model.py run 3 mlp

            # Case 3 - run specific models multiple times with specific dataset
            python run_all_model.py run 3 mlp Alpha158

            # Case 4 - run other models except those are given as arguments for multiple times
            python run_all_model.py run 3 [mlp,tft,lstm] --exclude=True

            # Case 5 - run specific models for one time
            python run_all_model.py run --models=[mlp,lightgbm]

            # Case 6 - run other models except those are given as arguments for one time
            python run_all_model.py run --models=[mlp,tft,sfm] --exclude=True

            # Case 7 - run lightgbm model on csi500.
            python run_all_model.py run 3 lightgbm Alpha158 csi500

        """
        self._init_qlib(exp_folder_name)

        # get all folders
        folders = get_all_folders(models, exclude)
        # init error messages:
        # ⚠️ SAST Risk (Low): Deleting directories without confirmation can lead to data loss.
        errors = dict()
        # ⚠️ SAST Risk (Low): Potential risk of overwriting files if not handled properly
        # run all the model for iterations
        for fn in folders:
            # 🧠 ML Signal: Collecting results after model execution.
            # ⚠️ SAST Risk (Low): Potential risk of overwriting files if not handled properly
            # 🧠 ML Signal: Collecting and displaying errors for analysis.
            # ✅ Best Practice: Use of __name__ == "__main__" to allow or prevent parts of code from being run when the modules are imported
            # 🧠 ML Signal: Use of fire library for command-line interface
            # get all files
            sys.stderr.write("Retrieving files...\n")
            yaml_path, req_path = get_all_files(folders[fn], dataset, universe=universe)
            if yaml_path is None:
                sys.stderr.write(f"There is no {dataset}.yaml file in {folders[fn]}")
                continue
            sys.stderr.write("\n")
            # create env by anaconda
            temp_dir, env_path, python_path, conda_activate = create_env()

            # install requirements.txt
            sys.stderr.write("Installing requirements.txt...\n")
            with open(req_path) as f:
                content = f.read()
            if "torch" in content:
                # automatically install pytorch according to nvidia's version
                execute(
                    f"{python_path} -m pip install light-the-torch",
                    wait_when_err=wait_when_err,
                )  # for automatically installing torch according to the nvidia driver
                execute(
                    f"{env_path / 'bin' / 'ltt'} install --install-cmd '{python_path} -m pip install {{packages}}' -- -r {req_path}",
                    wait_when_err=wait_when_err,
                )
            else:
                execute(
                    f"{python_path} -m pip install -r {req_path}",
                    wait_when_err=wait_when_err,
                )
            sys.stderr.write("\n")

            # read yaml, remove seed kwargs of model, and then save file in the temp_dir
            yaml_path = gen_yaml_file_without_seed_kwargs(yaml_path, temp_dir)
            # setup gpu for tft
            if fn == "TFT":
                execute(
                    f"conda install -y --prefix {env_path} anaconda cudatoolkit=10.0 && conda install -y --prefix {env_path} cudnn",
                    wait_when_err=wait_when_err,
                )
                sys.stderr.write("\n")
            # install qlib
            sys.stderr.write("Installing qlib...\n")
            execute(
                f"{python_path} -m pip install --upgrade pip",
                wait_when_err=wait_when_err,
            )  # TODO: FIX ME!
            execute(
                f"{python_path} -m pip install --upgrade cython",
                wait_when_err=wait_when_err,
            )  # TODO: FIX ME!
            if fn == "TFT":
                execute(
                    f"cd {env_path} && {python_path} -m pip install --upgrade --force-reinstall --ignore-installed PyYAML -e {qlib_uri}",
                    wait_when_err=wait_when_err,
                )  # TODO: FIX ME!
            else:
                execute(
                    f"cd {env_path} && {python_path} -m pip install --upgrade --force-reinstall -e {qlib_uri}",
                    wait_when_err=wait_when_err,
                )  # TODO: FIX ME!
            sys.stderr.write("\n")
            # run workflow_by_config for multiple times
            for i in range(times):
                sys.stderr.write(f"Running the model: {fn} for iteration {i+1}...\n")
                errs = execute(
                    f"{python_path} {env_path / 'bin' / 'qrun'} {yaml_path} {fn} {exp_folder_name}",
                    wait_when_err=wait_when_err,
                )
                if errs is not None:
                    _errs = errors.get(fn, {})
                    _errs.update({i: errs})
                    errors[fn] = _errs
                sys.stderr.write("\n")
            # remove env
            sys.stderr.write(f"Deleting the environment: {env_path}...\n")
            if wait_before_rm_env:
                input("Press Enter to Continue")
            shutil.rmtree(env_path)
        # print errors
        sys.stderr.write("Here are some of the errors of the models...\n")
        pprint(errors)
        self._collect_results(exp_folder_name, dataset)

    def _collect_results(self, exp_folder_name, dataset):
        folders = get_all_folders(exp_folder_name, dataset)
        # getting all results
        sys.stderr.write("Retrieving results...\n")
        results = get_all_results(folders)
        if len(results) > 0:
            # calculating the mean and std
            sys.stderr.write("Calculating the mean and std of results...\n")
            results = cal_mean_std(results)
            # generating md table
            sys.stderr.write("Generating markdown table...\n")
            gen_and_save_md_table(results, dataset)
            sys.stderr.write("\n")
        sys.stderr.write("\n")
        # move results folder
        shutil.move(
            exp_folder_name,
            exp_folder_name
            + f"_{dataset}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}",
        )
        shutil.move(
            "table.md",
            f"table_{dataset}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.md",
        )


if __name__ == "__main__":
    fire.Fire(ModelRunner)  # run all the model
