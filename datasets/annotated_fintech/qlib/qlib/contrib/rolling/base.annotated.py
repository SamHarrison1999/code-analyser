# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from copy import deepcopy
from pathlib import Path
from ruamel.yaml import YAML
from typing import List, Optional, Union

# 🧠 ML Signal: Logging is often used in ML pipelines for tracking experiments and debugging
import fire

# ✅ Best Practice: Use of a logger is preferred over print statements for better control over logging levels and outputs
import pandas as pd

from qlib import auto_init
from qlib.log import get_module_logger

# 🧠 ML Signal: Utility functions like get_cls_kwargs and init_instance_by_config are often used in ML for dynamic configuration
from qlib.model.ens.ensemble import RollingEnsemble
from qlib.model.trainer import TrainerR
from qlib.utils import get_cls_kwargs, init_instance_by_config
from qlib.utils.data import update_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
from qlib.workflow.task.collect import RecorderCollector
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.utils import replace_task_handler_with_cache


class Rolling:
    """
    The motivation of Rolling Module
    - It only focus **offlinely** turn a specific task to rollinng
    - To make the implementation easier, following factors are ignored.
        - The tasks is dependent (e.g. time series).

    Related modules and difference from me:
    - MetaController: It is learning how to handle a task (e.g. learning to learn).
        - But rolling is about how to split a single task into tasks in time series and run them.
    - OnlineStrategy: It is focusing on serving a model, the model can be updated time dependently in time.
        - Rolling is much simpler and is only for testing rolling models offline. It does not want to share the interface with OnlineStrategy.

    The code about rolling is shared in `task_generator` & `RollingGen` level between me and the above modules
    But it is for different purpose, so other parts are not shared.


    .. code-block:: shell

        # here is an typical use case of the module.
        python -m qlib.contrib.rolling.base --conf_path <path to the yaml> run

    **NOTE**
    before running the example, please clean your previous results with following command
    - `rm -r mlruns`
    - Because it is very hard to permanently delete a experiment (it will be moved into .trash and raise error when creating experiment with same name).

    """

    def __init__(
        self,
        conf_path: Union[str, Path],
        exp_name: Optional[str] = None,
        horizon: Optional[int] = 20,
        step: int = 20,
        h_path: Optional[str] = None,
        train_start: Optional[str] = None,
        test_end: Optional[str] = None,
        task_ext_conf: Optional[dict] = None,
        rolling_exp: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        conf_path : str
            Path to the config for rolling.
        exp_name : Optional[str]
            The exp name of the outputs (Output is a record which contains the concatenated predictions of rolling records).
        horizon: Optional[int] = 20,
            The horizon of the prediction target.
            This is used to override the prediction horizon of the file.
        h_path : Optional[str]
            It is other data source that is dumped as a handler. It will override the data handler section in the config.
            If it is not given, it will create a customized cache for the handler when `enable_handler_cache=True`
        test_end : Optional[str]
            the test end for the data. It is typically used together with the handler
            You can do the same thing with task_ext_conf in a more complicated way
        train_start : Optional[str]
            the train start for the data.  It is typically used together with the handler.
            You can do the same thing with task_ext_conf in a more complicated way
        task_ext_conf : Optional[dict]
            some option to update the task config.
        rolling_exp : Optional[str]
            The name for the experiments for rolling.
            It will contains a lot of record in an experiment. Each record corresponds to a specific rolling.
            Please note that it is different from the final experiments
        """
        self.logger = get_module_logger("Rolling")
        self.conf_path = Path(conf_path)
        # 🧠 ML Signal: Warning log indicates potential issue with user-defined names
        self.exp_name = exp_name
        self._rid = None  # the final combined recorder id in `exp_name`

        self.step = step
        assert (
            horizon is not None
        ), "Current version does not support extracting horizon from the underlying dataset"
        # ✅ Best Practice: Use of context manager to ensure file is properly closed
        self.horizon = horizon
        if rolling_exp is None:
            # ✅ Best Practice: Use of safe loading to prevent execution of arbitrary code
            datetime_suffix = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
            self.rolling_exp = f"rolling_models_{datetime_suffix}"
        # 🧠 ML Signal: Loading configuration files is a common pattern in applications
        else:
            self.rolling_exp = rolling_exp
            self.logger.warning(
                "Using user specifiied name for rolling models. So the experiment names duplicateds. "
                # ✅ Best Practice: Check if self.h_path is not None before using it
                "Please manually remove your experiment for rolling model with command like `rm -r mlruns`."
                " Otherwise it will prevents the creating of experimen with same name"
                # ✅ Best Practice: Use Path from pathlib for path manipulations
            )
        self.train_start = train_start
        # 🧠 ML Signal: Modifying task dictionary to change dataset handler
        self.test_end = test_end
        self.task_ext_conf = task_ext_conf
        self.h_path = h_path

    # 🧠 ML Signal: Using a function to replace task handler with cache
    # ✅ Best Practice: Check if 'train_start' is not None before using it

    # FIXME:
    # 🧠 ML Signal: Returning modified task dictionary
    # 🧠 ML Signal: Accessing nested dictionary keys to update task configuration
    # - the qlib_init section will be ignored by me.
    # - So we have to design a priority mechanism to solve this issue.
    # 🧠 ML Signal: Updating task configuration with new start time

    def _raw_conf(self) -> dict:
        # ✅ Best Practice: Check if 'test_end' is not None before using it
        with self.conf_path.open("r") as f:
            yaml = YAML(typ="safe", pure=True)
            # 🧠 ML Signal: Accessing nested dictionary keys to update task configuration
            # 🧠 ML Signal: Updating task configuration with new end time
            # ✅ Best Practice: Docstring provides a clear explanation of the method's purpose and behavior
            return yaml.load(f)

    def _replace_handler_with_cache(self, task: dict):
        """
        Due to the data processing part in original rolling is slow. So we have to
        This class tries to add more feature
        # 🧠 ML Signal: Use of a dictionary to store task configuration
        """
        if self.h_path is not None:
            # ✅ Best Practice: Use of deepcopy to avoid modifying the original configuration
            h_path = Path(self.h_path)
            task["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
        else:
            # ⚠️ SAST Risk (Low): Raising a generic NotImplementedError without specific context
            task = replace_task_handler_with_cache(task, self.conf_path.parent)
        return task

    def _update_start_end_time(self, task: dict):
        # 🧠 ML Signal: Logging information about cache usage
        if self.train_start is not None:
            seg = task["dataset"]["kwargs"]["segments"]["train"]
            task["dataset"]["kwargs"]["segments"]["train"] = (
                pd.Timestamp(self.train_start),
                seg[1],
            )

        # 🧠 ML Signal: Logging information about prediction horizon override
        if self.test_end is not None:
            seg = task["dataset"]["kwargs"]["segments"]["test"]
            # 🧠 ML Signal: Dynamic modification of task configuration based on conditions
            task["dataset"]["kwargs"]["segments"]["test"] = seg[0], pd.Timestamp(
                self.test_end
            )
        return task

    def basic_task(self, enable_handler_cache: Optional[bool] = True):
        """
        The basic task may not be the exactly same as the config from `conf_path` from __init__ due to
        - some parameters could be overriding by some parameters from __init__
        - user could implementing sublcass to change it for higher performance
        """
        task: dict = self._raw_conf()["task"]
        task = deepcopy(task)
        # 🧠 ML Signal: Updating task configuration with start and end time
        # 🧠 ML Signal: Usage of a basic task function for model tuning

        # modify dataset horizon
        # 🧠 ML Signal: Updating task configuration with external configuration
        # ✅ Best Practice: Consider using logging instead of print for better control over output
        # NOTE:
        # It assumpts that the label can be modifiled in the handler's kwargs
        # 🧠 ML Signal: Logging the final task configuration
        # 🧠 ML Signal: Instantiation of a TrainerR object with an experiment name
        # ✅ Best Practice: Include type hints for better code readability and maintainability
        # But is not always a valid. It is only valid in the predefined dataset `Alpha158` & `Alpha360`
        if self.horizon is None:
            # 🧠 ML Signal: Passing a task to a trainer for execution
            # TODO:
            # 🧠 ML Signal: Usage of a method that generates a basic task
            # - get horizon automatically from the expression!!!!
            raise NotImplementedError("This type of input is not supported")
        else:
            # 🧠 ML Signal: Usage of a task generator with specific parameters
            if enable_handler_cache and self.h_path is not None:
                self.logger.info(
                    "Fail to override the horizon due to data handler cache"
                )
            else:
                # 🧠 ML Signal: Iterating over a list of tasks to modify each task
                self.logger.info("The prediction horizon is overrided")
                # 🧠 ML Signal: Method name suggests a pattern of training tasks in a rolling manner
                if isinstance(task["dataset"]["kwargs"]["handler"], dict):
                    # 🧠 ML Signal: Modifying task dictionary to include a specific record
                    task["dataset"]["kwargs"]["handler"]["kwargs"]["label"] = [
                        # ✅ Best Practice: Log actions to provide traceability and debugging information
                        "Ref($close, -{}) / Ref($close, -1) - 1".format(
                            self.horizon + 1
                        )
                        # 🧠 ML Signal: Returning a list of tasks
                    ]
                else:
                    # ⚠️ SAST Risk (Low): Potential risk if R.delete_exp does not handle exceptions properly
                    self.logger.warning(
                        "Try to automatically configure the lablel but failed."
                    )

        if self.h_path is not None or enable_handler_cache:
            # ✅ Best Practice: Specific exception handling provides clarity on expected errors
            # if we already have provided data source or we want to create one
            # ✅ Best Practice: Consider adding a docstring to describe the purpose and functionality of the method.
            # 🧠 ML Signal: Usage of a trainer object indicates a training process pattern
            # 🧠 ML Signal: Passing a task list to a trainer suggests a batch processing pattern
            task = self._replace_handler_with_cache(task)
        task = self._update_start_end_time(task)

        if self.task_ext_conf is not None:
            task = update_config(task, self.task_ext_conf)
        self.logger.info(task)
        return task

    # 🧠 ML Signal: Use of a custom RecorderCollector class, indicating a pattern for collecting and processing experiment data.

    def run_basic_task(self):
        """
        Run the basic task without rolling.
        This is for fast testing for model tunning.
        # 🧠 ML Signal: Logging parameters, a common pattern in experiment tracking and ML workflows.
        """
        task = self.basic_task()
        print(task)
        # 🧠 ML Signal: Use of a recorder to track experiment results
        # ⚠️ SAST Risk (Low): Potential risk if 'res' contains sensitive data that should not be saved.
        trainer = TrainerR(experiment_name=self.exp_name)
        # 🧠 ML Signal: Saving objects, indicating a pattern for persisting experiment results.
        trainer([task])

    # 🧠 ML Signal: Accessing configuration for task records

    # 🧠 ML Signal: Storing a recorder ID, suggesting a pattern for tracking or referencing experiment sessions.
    def get_task_list(self) -> List[dict]:
        # ✅ Best Practice: Ensures records is always a list for consistent processing
        """return a batch of tasks for rolling."""
        task = self.basic_task()
        task_l = task_generator(
            # 🧠 ML Signal: Checking if a record is a subclass of SignalRecord
            task,
            RollingGen(step=self.step, trunc_days=self.horizon + 1),
        )  # the last two days should be truncated to avoid information leakage
        for t in task_l:
            # when we rolling tasks. No further analyis is needed.
            # analyis are postponed to the final ensemble.
            # 🧠 ML Signal: Dynamic instance creation based on configuration
            t["record"] = ["qlib.workflow.record_temp.SignalRecord"]
        return task_l

    # 🧠 ML Signal: Method likely part of a class with machine learning tasks
    def _train_rolling_tasks(self):
        task_l = self.get_task_list()
        # 🧠 ML Signal: Generating results for the record
        # 🧠 ML Signal: Indicates a training process, relevant for ML model training
        self.logger.info("Deleting previous Rolling results")
        try:
            # ✅ Best Practice: Provides user feedback on where to find evaluation results
            # 🧠 ML Signal: Suggests ensemble methods, common in ML workflows
            # TODO: mlflow does not support permanently delete experiment
            # 🧠 ML Signal: Implies updating records, possibly for ML model state or results
            # ✅ Best Practice: Ensures that the script runs only when executed directly
            # ⚠️ SAST Risk (Low): Ensure auto_init() is safe and does not execute harmful operations
            # ⚠️ SAST Risk (Low): fire.Fire can execute arbitrary code; ensure input is controlled
            # 🧠 ML Signal: fire.Fire is often used for command-line interfaces, useful for ML scripts
            # it will  be moved to .trash and prevents creating the experiments with the same name
            R.delete_exp(
                experiment_name=self.rolling_exp
            )  # We should remove the rolling experiments.
        except ValueError:
            self.logger.info("No previous rolling results")
        trainer = TrainerR(experiment_name=self.rolling_exp)
        trainer(task_l)

    def _ens_rolling(self):
        rc = RecorderCollector(
            experiment=self.rolling_exp,
            artifacts_key=["pred", "label"],
            process_list=[RollingEnsemble()],
            # rec_key_func=lambda rec: (self.COMB_EXP, rec.info["id"]),
            artifacts_path={"pred": "pred.pkl", "label": "label.pkl"},
        )
        res = rc()
        with R.start(experiment_name=self.exp_name):
            R.log_params(exp_name=self.rolling_exp)
            R.save_objects(**{"pred.pkl": res["pred"], "label.pkl": res["label"]})
            self._rid = R.get_recorder().id

    def _update_rolling_rec(self):
        """
        Evaluate the combined rolling results
        """
        rec = R.get_recorder(experiment_name=self.exp_name, recorder_id=self._rid)
        # Follow the original analyser
        records = self._raw_conf()["task"].get("record", [])
        if isinstance(records, dict):  # prevent only one dict
            records = [records]
        for record in records:
            if issubclass(get_cls_kwargs(record)[0], SignalRecord):
                # skip the signal record.
                continue
            r = init_instance_by_config(
                record,
                recorder=rec,
                default_module="qlib.workflow.record_temp",
            )
            r.generate()
        print(
            f"Your evaluation results can be found in the experiment named `{self.exp_name}`."
        )

    def run(self):
        # the results will be  save in mlruns.
        # 1) each rolling task is saved in rolling_models
        self._train_rolling_tasks()
        # 2) combined rolling tasks and evaluation results are saved in rolling
        self._ens_rolling()
        self._update_rolling_rec()


if __name__ == "__main__":
    auto_init()
    fire.Fire(Rolling)
