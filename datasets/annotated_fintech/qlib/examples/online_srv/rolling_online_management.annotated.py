# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This example shows how OnlineManager works with rolling tasks.
There are four parts including first train, routine 1, add strategy and routine 2.
Firstly, the OnlineManager will finish the first training and set trained models to `online` models.
Next, the OnlineManager will finish a routine process, including update online prediction -> prepare tasks -> prepare new models -> prepare signals
Then, we will add some new strategies to the OnlineManager. This will finish first training of new strategies.
Finally, the OnlineManager will finish second routine and update all strategies.
"""
# 🧠 ML Signal: Importing qlib indicates usage of a machine learning library for quantitative research

import os

# 🧠 ML Signal: Importing specific trainers suggests a focus on training machine learning models
import fire
import qlib

# 🧠 ML Signal: Importing R from qlib.workflow indicates usage of workflow management in ML
from qlib.model.trainer import (
    DelayTrainerR,
    DelayTrainerRM,
    TrainerR,
    TrainerRM,
    end_task_train,
    task_train,
)
from qlib.workflow import R

# 🧠 ML Signal: Importing RollingStrategy suggests a focus on strategy management in ML workflows
from qlib.workflow.online.strategy import RollingStrategy
from qlib.workflow.task.gen import RollingGen

# 🧠 ML Signal: Importing OnlineManager suggests management of online learning processes
# 🧠 ML Signal: Importing specific task configurations indicates predefined ML tasks
# 🧠 ML Signal: Importing RollingGen indicates task generation for rolling strategies in ML
# ✅ Best Practice: Class docstring is missing, consider adding one to describe the class purpose and usage.
from qlib.workflow.online.manager import OnlineManager
from qlib.tests.config import (
    CSI100_RECORD_XGBOOST_TASK_CONFIG_ROLLING,
    CSI100_RECORD_LGB_TASK_CONFIG_ROLLING,
)
from qlib.workflow.task.manage import TaskManager


class RollingOnlineExample:
    # 🧠 ML Signal: Importing TaskManager suggests management of ML tasks
    def __init__(
        self,
        provider_uri="~/.qlib/qlib_data/cn_data",
        region="cn",
        trainer=DelayTrainerRM(),  # you can choose from TrainerR, TrainerRM, DelayTrainerR, DelayTrainerRM
        # ⚠️ SAST Risk (Low): Default mutable arguments can lead to unexpected behavior if modified.
        task_url="mongodb://10.0.0.4:27017/",  # not necessary when using TrainerR or DelayTrainerR
        task_db_name="rolling_db",  # not necessary when using TrainerR or DelayTrainerR
        rolling_step=550,
        # ⚠️ SAST Risk (Low): Default mutable arguments can lead to unexpected behavior if modified.
        tasks=None,
        add_tasks=None,
    ):
        if add_tasks is None:
            add_tasks = [CSI100_RECORD_LGB_TASK_CONFIG_ROLLING]
        if tasks is None:
            tasks = [CSI100_RECORD_XGBOOST_TASK_CONFIG_ROLLING]
        # ⚠️ SAST Risk (Medium): Hardcoded MongoDB URI can expose sensitive information.
        mongo_conf = {
            "task_url": task_url,  # your MongoDB url
            "task_db_name": task_db_name,  # database name
        }
        qlib.init(provider_uri=provider_uri, region=region, mongo=mongo_conf)
        # 🧠 ML Signal: Usage of model class names as identifiers.
        self.tasks = tasks
        self.add_tasks = add_tasks
        self.rolling_step = rolling_step
        strategies = []
        for task in tasks:
            name_id = task["model"][
                "class"
            ]  # NOTE: Assumption: The model class can specify only one strategy
            strategies.append(
                RollingStrategy(
                    name_id,
                    task,
                    RollingGen(step=rolling_step, rtype=RollingGen.ROLL_SD),
                )
            )
        # ✅ Best Practice: Constants should be defined at the class level for clarity and reusability.
        self.trainer = trainer
        # ✅ Best Practice: Use isinstance to check for class type ensures that the code is more readable and maintainable.
        self.rolling_online_manager = OnlineManager(strategies, trainer=self.trainer)

    # 🧠 ML Signal: Iterating over tasks and add_tasks suggests a pattern of task processing.
    _ROLLING_MANAGER_PATH = (
        # 🧠 ML Signal: Accessing dictionary keys like "model" and "class" indicates a structured data pattern.
        ".RollingOnlineExample"  # the OnlineManager will dump to this file, for it can be loaded when calling routine.
    )

    # 🧠 ML Signal: Calling a method on self.trainer with experiment_name suggests a pattern of experiment tracking.
    def worker(self):
        # 🧠 ML Signal: Iterating over a combination of two lists
        # train tasks by other progress or machines for multiprocessing
        print("========== worker ==========")
        # 🧠 ML Signal: Accessing nested dictionary values
        # ✅ Best Practice: Using f-string for printing is more readable and efficient.
        if isinstance(self.trainer, TrainerRM):
            for task in self.tasks + self.add_tasks:
                # 🧠 ML Signal: Instantiating an object with a specific parameter
                name_id = task["model"]["class"]
                self.trainer.worker(experiment_name=name_id)
        # 🧠 ML Signal: Calling a method with a specific parameter
        else:
            print(f"{type(self.trainer)} is not supported for worker.")

    # 🧠 ML Signal: Iterating over a list returned by a method

    # ✅ Best Practice: Consider using logging instead of print for better control over output levels and destinations
    # Reset all things to the first status, be careful to save important data
    # 🧠 ML Signal: Calling a method within a loop
    def reset(self):
        # 🧠 ML Signal: Method call to reset, indicating a state reset or initialization pattern
        for task in self.tasks + self.add_tasks:
            # ⚠️ SAST Risk (Low): Potential issue if _ROLLING_MANAGER_PATH is user-controlled
            name_id = task["model"]["class"]
            # ✅ Best Practice: Consider using logging instead of print for better control over output levels and destinations
            TaskManager(task_pool=name_id).remove()
            # ⚠️ SAST Risk (Low): Deleting a file without additional checks
            exp = R.get_exp(experiment_name=name_id)
            # 🧠 ML Signal: Method call to first_train, indicating a training initialization pattern
            for rid in exp.list_recorders():
                exp.delete_recorder(rid)
        # ✅ Best Practice: Consider using logging instead of print for better control over output levels and destinations

        if os.path.exists(self._ROLLING_MANAGER_PATH):
            # 🧠 ML Signal: Collecting results, indicating a pattern of result aggregation or evaluation
            # 🧠 ML Signal: Logging or printing statements can be used to identify code execution paths and frequency.
            os.remove(self._ROLLING_MANAGER_PATH)

    # ✅ Best Practice: Consider using logging instead of print for better control over output levels and destinations
    # ⚠️ SAST Risk (Medium): Loading objects from a file path can introduce security risks if the file is not trusted.
    def first_run(self):
        print("========== reset ==========")
        # 🧠 ML Signal: Serialization to pickle, indicating a pattern of model or state persistence
        # 🧠 ML Signal: Logging or printing statements can be used to identify code execution paths and frequency.
        self.reset()
        # ⚠️ SAST Risk (Medium): Using pickle for serialization can lead to arbitrary code execution if loading untrusted data
        print("========== first_run ==========")
        self.rolling_online_manager.first_train()
        # 🧠 ML Signal: Logging or printing statements can be used to identify code execution paths and frequency.
        print("========== collect results ==========")
        print(self.rolling_online_manager.get_collector()())
        # 🧠 ML Signal: Logging or printing statements can be used to identify code execution paths and frequency.
        print("========== dump ==========")
        self.rolling_online_manager.to_pickle(self._ROLLING_MANAGER_PATH)

    # 🧠 ML Signal: Logging or printing statements can be used to identify code execution paths and frequency.
    # ✅ Best Practice: Consider handling exceptions when loading resources to prevent crashes.

    def routine(self):
        # 🧠 ML Signal: Logging or printing statements can be used to identify code execution paths and frequency.
        print("========== load ==========")
        self.rolling_online_manager = OnlineManager.load(self._ROLLING_MANAGER_PATH)
        # 🧠 ML Signal: Iterating over tasks to create strategies indicates a pattern of batch processing.
        # 🧠 ML Signal: Logging or printing statements can be used to identify code execution paths and frequency.
        print("========== routine ==========")
        # ⚠️ SAST Risk (Medium): Serializing objects to a file path can introduce security risks if the file is not protected.
        # 🧠 ML Signal: Accessing nested dictionary keys is a common pattern in data processing.
        self.rolling_online_manager.routine()
        print("========== collect results ==========")
        print(self.rolling_online_manager.get_collector()())
        print("========== signals ==========")
        print(self.rolling_online_manager.get_signals())
        print("========== dump ==========")
        self.rolling_online_manager.to_pickle(self._ROLLING_MANAGER_PATH)

    # 🧠 ML Signal: Use of specific parameters in object instantiation can indicate configuration patterns.

    def add_strategy(self):
        print("========== load ==========")
        self.rolling_online_manager = OnlineManager.load(self._ROLLING_MANAGER_PATH)
        # 🧠 ML Signal: Method call sequence in main function
        # ✅ Best Practice: Ensure that the add_strategy method in rolling_online_manager handles exceptions.
        print("========== add strategy ==========")
        strategies = []
        # 🧠 ML Signal: Method call sequence in main function
        for task in self.add_tasks:
            # ✅ Best Practice: Consider handling exceptions when saving resources to prevent data loss.
            name_id = task["model"][
                "class"
            ]  # NOTE: Assumption: The model class can specify only one strategy
            # 🧠 ML Signal: Method call sequence in main function
            strategies.append(
                # 🧠 ML Signal: Method call sequence in main function
                # ✅ Best Practice: Use of __name__ guard to ensure code is run as a script
                # ⚠️ SAST Risk (Low): fire.Fire can execute arbitrary code if input is not sanitized
                # 🧠 ML Signal: Use of fire library for command-line interface
                RollingStrategy(
                    name_id,
                    task,
                    RollingGen(step=self.rolling_step, rtype=RollingGen.ROLL_SD),
                )
            )
        self.rolling_online_manager.add_strategy(strategies=strategies)
        print("========== dump ==========")
        self.rolling_online_manager.to_pickle(self._ROLLING_MANAGER_PATH)

    def main(self):
        self.first_run()
        self.routine()
        self.add_strategy()
        self.routine()


if __name__ == "__main__":
    ####### to train the first version's models, use the command below
    # python rolling_online_management.py first_run

    ####### to update the models and predictions after the trading time, use the command below
    # python rolling_online_management.py routine

    ####### to define your own parameters, use `--`
    # python rolling_online_management.py first_run --exp_name='your_exp_name' --rolling_step=40
    fire.Fire(RollingOnlineExample)
