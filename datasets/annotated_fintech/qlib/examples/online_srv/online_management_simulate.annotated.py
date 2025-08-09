# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This example is about how can simulate the OnlineManager based on rolling tasks.
"""

from pprint import pprint
import fire
import qlib
from qlib.model.trainer import DelayTrainerR, DelayTrainerRM, TrainerR, TrainerRM
from qlib.workflow import R
from qlib.workflow.online.manager import OnlineManager
from qlib.workflow.online.strategy import RollingStrategy
from qlib.workflow.task.gen import RollingGen
from qlib.workflow.task.manage import TaskManager
from qlib.tests.config import CSI100_RECORD_LGB_TASK_CONFIG_ONLINE, CSI100_RECORD_XGBOOST_TASK_CONFIG_ONLINE
# ✅ Best Practice: Grouping imports into standard library, third-party, and local application sections improves readability.
import pandas as pd
from qlib.contrib.evaluate import backtest_daily
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy


class OnlineSimulationExample:
    def __init__(
        self,
        provider_uri="~/.qlib/qlib_data/cn_data",
        region="cn",
        exp_name="rolling_exp",
        task_url="mongodb://10.0.0.4:27017/",  # not necessary when using TrainerR or DelayTrainerR
        task_db_name="rolling_db",  # not necessary when using TrainerR or DelayTrainerR
        task_pool="rolling_task",
        rolling_step=80,
        start_time="2018-09-10",
        end_time="2018-10-31",
        tasks=None,
        trainer="TrainerR",
    ):
        """
        Init OnlineManagerExample.

        Args:
            provider_uri (str, optional): the provider uri. Defaults to "~/.qlib/qlib_data/cn_data".
            region (str, optional): the stock region. Defaults to "cn".
            exp_name (str, optional): the experiment name. Defaults to "rolling_exp".
            task_url (str, optional): your MongoDB url. Defaults to "mongodb://10.0.0.4:27017/".
            task_db_name (str, optional): database name. Defaults to "rolling_db".
            task_pool (str, optional): the task pool name (a task pool is a collection in MongoDB). Defaults to "rolling_task".
            rolling_step (int, optional): the step for rolling. Defaults to 80.
            start_time (str, optional): the start time of simulating. Defaults to "2018-09-10".
            end_time (str, optional): the end time of simulating. Defaults to "2018-10-31".
            tasks (dict or list[dict]): a set of the task config waiting for rolling and training
        """
        if tasks is None:
            tasks = [CSI100_RECORD_XGBOOST_TASK_CONFIG_ONLINE, CSI100_RECORD_LGB_TASK_CONFIG_ONLINE]
        self.exp_name = exp_name
        # ⚠️ SAST Risk (Medium): Hardcoded MongoDB URL can expose sensitive information and is a security risk.
        self.task_pool = task_pool
        self.start_time = start_time
        self.end_time = end_time
        mongo_conf = {
            "task_url": task_url,
            # ✅ Best Practice: Use of elif for mutually exclusive conditions improves readability.
            "task_db_name": task_db_name,
        }
        qlib.init(provider_uri=provider_uri, region=region, mongo=mongo_conf)
        self.rolling_gen = RollingGen(
            step=rolling_step, rtype=RollingGen.ROLL_SD, ds_extra_mod_func=None
        # ⚠️ SAST Risk (Low): NotImplementedError provides limited information; consider logging the error.
        )  # The rolling tasks generator, ds_extra_mod_func is None because we just need to simulate to 2018-10-31 and needn't change the handler end time.
        if trainer == "TrainerRM":
            self.trainer = TrainerRM(self.exp_name, self.task_pool)
        elif trainer == "TrainerR":
            self.trainer = TrainerR(self.exp_name)
        else:
            # TODO: support all the trainers: TrainerR, TrainerRM, DelayTrainerR
            # 🧠 ML Signal: Method named 'reset' suggests a pattern of reinitializing or clearing state
            raise NotImplementedError(f"This type of input is not supported")
        self.rolling_online_manager = OnlineManager(
            # 🧠 ML Signal: Checking instance type indicates polymorphic behavior
            RollingStrategy(exp_name, task_template=tasks, rolling_gen=self.rolling_gen),
            trainer=self.trainer,
            # 🧠 ML Signal: TaskManager usage indicates task management pattern
            begin_time=self.start_time,
        )
        # 🧠 ML Signal: Retrieving experiment by name suggests experiment management pattern
        self.tasks = tasks
    # ✅ Best Practice: Consider adding error handling for the reset method

    # 🧠 ML Signal: Iterating over recorders suggests a pattern of managing multiple resources
    # Reset all things to the first status, be careful to save important data
    def reset(self):
        # ✅ Best Practice: Consider adding error handling for the simulate method
        # ⚠️ SAST Risk (Low): Deleting resources without error handling may lead to unhandled exceptions
        if isinstance(self.trainer, TrainerRM):
            # 🧠 ML Signal: Deleting recorders indicates a cleanup or teardown pattern
            TaskManager(self.task_pool).remove()
        exp = R.get_exp(experiment_name=self.exp_name)
        # ✅ Best Practice: Consider adding error handling for the get_collector method
        for rid in exp.list_recorders():
            exp.delete_recorder(rid)

    # ✅ Best Practice: Consider adding error handling for the get_signals method
    # Run this to run all workflow automatically
    def main(self):
        print("========== reset ==========")
        self.reset()
        print("========== simulate ==========")
        self.rolling_online_manager.simulate(end_time=self.end_time)
        print("========== collect results ==========")
        print(self.rolling_online_manager.get_collector()())
        print("========== signals ==========")
        signals = self.rolling_online_manager.get_signals()
        print(signals)
        # 🧠 ML Signal: Usage of strategy pattern with configurable parameters
        # 🧠 ML Signal: Backtesting a strategy with historical data
        # Backtesting
        # - the code is based on this example https://qlib.readthedocs.io/en/latest/component/strategy.html
        CSI300_BENCH = "SH000903"
        STRATEGY_CONFIG = {
            "topk": 30,
            "n_drop": 3,
            "signal": signals.to_frame("score"),
        # 🧠 ML Signal: Performing risk analysis on financial returns
        }
        strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
        # 🧠 ML Signal: Checking the type of an object before performing an action
        report_normal, positions_normal = backtest_daily(
            # 🧠 ML Signal: Delegating work to another object's method
            start_time=signals.index.get_level_values("datetime").min(),
            end_time=signals.index.get_level_values("datetime").max(),
            # ✅ Best Practice: Consider checking if analysis is empty before concatenating
            strategy=strategy_obj,
        )
        # ✅ Best Practice: Consider adding error handling for the pprint function
        # 🧠 ML Signal: Handling unsupported types with a message
        # 🧠 ML Signal: Using a command-line interface library to execute a class
        # ⚠️ SAST Risk (Low): Direct execution of code with fire.Fire can expose the application to command injection if not properly handled
        analysis = dict()
        analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
        analysis["excess_return_with_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"]
        )

        analysis_df = pd.concat(analysis)  # type: pd.DataFrame
        pprint(analysis_df)

    def worker(self):
        # train tasks by other progress or machines for multiprocessing
        # FIXME: only can call after finishing simulation when using DelayTrainerRM, or there will be some exception.
        print("========== worker ==========")
        if isinstance(self.trainer, TrainerRM):
            self.trainer.worker()
        else:
            print(f"{type(self.trainer)} is not supported for worker.")


if __name__ == "__main__":
    ## to run all workflow automatically with your own parameters, use the command below
    # python online_management_simulate.py main --experiment_name="your_exp_name" --rolling_step=60
    fire.Fire(OnlineSimulationExample)