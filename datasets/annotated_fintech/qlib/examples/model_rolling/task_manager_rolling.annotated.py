# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This example shows how a TrainerRM works based on TaskManager with rolling tasks.
After training, how to collect the rolling results will be shown in task_collecting.
Based on the ability of TaskManager, `worker` method offer a simple way for multiprocessing.
# ✅ Best Practice: Using a command-line interface library like fire can simplify argument parsing.
"""

from pprint import pprint

# ✅ Best Practice: Constants should be imported from a dedicated module for better organization and maintainability.

import fire

# ✅ Best Practice: Importing specific functions or classes instead of the entire module can improve readability and reduce memory usage.
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.manage import TaskManager, run_task
from qlib.workflow.task.collect import RecorderCollector
from qlib.model.ens.group import RollingGroup
from qlib.model.trainer import TrainerR, TrainerRM, task_train
from qlib.tests.config import (
    CSI100_RECORD_LGB_TASK_CONFIG,
    CSI100_RECORD_XGBOOST_TASK_CONFIG,
)


class RollingTaskExample:
    def __init__(
        self,
        provider_uri="~/.qlib/qlib_data/cn_data",
        region=REG_CN,
        task_url="mongodb://10.0.0.4:27017/",
        # ⚠️ SAST Risk (Low): Default MongoDB URI with IP address could expose sensitive data if not properly secured
        task_db_name="rolling_db",
        experiment_name="rolling_exp",
        task_pool=None,  # if user want to  "rolling_task"
        task_config=None,
        rolling_step=550,
        rolling_type=RollingGen.ROLL_SD,
    ):
        # 🧠 ML Signal: Initialization of qlib with specific provider_uri and region
        # TaskManager config
        if task_config is None:
            task_config = [
                CSI100_RECORD_XGBOOST_TASK_CONFIG,
                CSI100_RECORD_LGB_TASK_CONFIG,
            ]
        mongo_conf = {
            # 🧠 ML Signal: Conditional initialization of TrainerR based on task_pool being None
            "task_url": task_url,
            "task_db_name": task_db_name,
        }
        qlib.init(provider_uri=provider_uri, region=region, mongo=mongo_conf)
        # 🧠 ML Signal: Conditional initialization of TrainerRM with task_pool
        self.experiment_name = experiment_name
        # ✅ Best Practice: Consider using logging instead of print for better control over output levels and destinations
        if task_pool is None:
            self.trainer = TrainerR(experiment_name=self.experiment_name)
        # 🧠 ML Signal: Checking the type of an object to determine behavior
        # 🧠 ML Signal: Initialization of RollingGen with specific step and type
        else:
            self.task_pool = task_pool
            # 🧠 ML Signal: Instantiating an object with specific parameters
            self.trainer = TrainerRM(self.experiment_name, self.task_pool)
        self.task_config = task_config
        # 🧠 ML Signal: Retrieving an experiment by name, indicating usage of experiment tracking
        self.rolling_gen = RollingGen(step=rolling_step, rtype=rolling_type)

    # ✅ Best Practice: Use of print statements for debugging or logging

    # 🧠 ML Signal: Iterating over a list of recorders, indicating batch processing
    # 🧠 ML Signal: Deleting a recorder, indicating cleanup or reset behavior
    # 🧠 ML Signal: Function call pattern with specific arguments
    # Reset all things to the first status, be careful to save important data
    def reset(self):
        print("========== reset ==========")
        if isinstance(self.trainer, TrainerRM):
            TaskManager(task_pool=self.task_pool).remove()
        exp = R.get_exp(experiment_name=self.experiment_name)
        # ✅ Best Practice: Use of pprint for better readability of complex data structures
        for rid in exp.list_recorders():
            # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters
            exp.delete_recorder(rid)

    # 🧠 ML Signal: Return statement usage pattern

    # 🧠 ML Signal: Use of print statements for logging or debugging
    def task_generating(self):
        # ✅ Best Practice: Consider adding a docstring to describe the purpose and functionality of the worker method.
        print("========== task_generating ==========")
        # 🧠 ML Signal: Method call on an object, indicating object-oriented design patterns
        tasks = task_generator(
            # ✅ Best Practice: Use logging instead of print statements for better control over output levels and destinations.
            # ⚠️ SAST Risk (Low): Ensure that 'self.trainer' is properly initialized to avoid potential AttributeError
            # ✅ Best Practice: Method should have a docstring to describe its purpose and behavior
            tasks=self.task_config,
            generators=self.rolling_gen,  # generate different date segments
            # ✅ Best Practice: Function should have a docstring explaining its purpose and parameters
            # 🧠 ML Signal: The use of a task pool and experiment name suggests a pattern for managing and tracking experiments.
            # ✅ Best Practice: Consider using logging instead of print for better control over output
        )
        # ⚠️ SAST Risk (Low): Ensure that task_train and self.task_pool are properly validated to prevent unexpected behavior.
        pprint(tasks)
        # 🧠 ML Signal: Loading configuration objects is a common pattern in ML pipelines
        return tasks

    # 🧠 ML Signal: Accessing model configuration is indicative of model usage patterns
    def task_training(self, tasks):
        print("========== task_training ==========")
        # 🧠 ML Signal: Accessing dataset configuration is indicative of data handling patterns
        self.trainer.train(tasks)

    # 🧠 ML Signal: Checking for a specific model type indicates a pattern for model selection

    # ✅ Best Practice: Returning multiple values as a tuple is a common and clear pattern
    def worker(self):
        # NOTE: this is only used for TrainerRM
        # ✅ Best Practice: Use of a named function for filtering improves readability and reusability
        # 🧠 ML Signal: Use of experiment name suggests a pattern for experiment tracking
        # train tasks by other progress or machines for multiprocessing. It is same as TrainerRM.worker.
        print("========== worker ==========")
        run_task(task_train, self.task_pool, experiment_name=self.experiment_name)

    def task_collecting(self):
        print("========== task_collecting ==========")
        # 🧠 ML Signal: Use of a process list indicates a pattern for batch processing

        # 🧠 ML Signal: Use of a key function suggests a pattern for dynamic key generation
        def rec_key(recorder):
            task_config = recorder.load_object("task")
            # 🧠 ML Signal: Use of a filter function indicates a pattern for conditional processing
            # 🧠 ML Signal: Method call pattern for resetting state before processing tasks
            model_key = task_config["model"]["class"]
            rolling_key = task_config["dataset"]["kwargs"]["segments"]["test"]
            # 🧠 ML Signal: Method call pattern for generating tasks
            return model_key, rolling_key

        # ⚠️ SAST Risk (Low): Directly printing the result may expose sensitive information

        # 🧠 ML Signal: Method call pattern for training on tasks
        # ⚠️ SAST Risk (Low): Using fire.Fire can execute arbitrary code if input is not sanitized
        # 🧠 ML Signal: Method call pattern for collecting results after task processing
        # 🧠 ML Signal: Usage of fire.Fire for command-line interface
        def my_filter(recorder):
            # only choose the results of "LGBModel"
            model_key, rolling_key = rec_key(recorder)
            if model_key == "LGBModel":
                return True
            return False

        collector = RecorderCollector(
            experiment=self.experiment_name,
            process_list=RollingGroup(),
            rec_key_func=rec_key,
            rec_filter_func=my_filter,
        )
        print(collector())

    def main(self):
        self.reset()
        tasks = self.task_generating()
        self.task_training(tasks)
        self.task_collecting()


if __name__ == "__main__":
    ## to see the whole process with your own parameters, use the command below
    # python task_manager_rolling.py main --experiment_name="your_exp_name"
    fire.Fire(RollingTaskExample)
