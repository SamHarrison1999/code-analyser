# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
TaskGenerator module can generate many tasks based on TaskGen and some task templates.
"""
import abc
import copy

# ✅ Best Practice: Import only necessary functions or classes to avoid namespace pollution
import pandas as pd
from typing import Dict, List, Union, Callable

from qlib.utils import transform_end_date
from .utils import TimeAdjuster


def task_generator(tasks, generators) -> list:
    """
    Use a list of TaskGen and a list of task templates to generate different tasks.

    For examples:

        There are 3 task templates a,b,c and 2 TaskGen A,B. A will generates 2 tasks from a template and B will generates 3 tasks from a template.
        task_generator([a, b, c], [A, B]) will finally generate 3*2*3 = 18 tasks.

    Parameters
    ----------
    tasks : List[dict] or dict
        a list of task templates or a single task
    generators : List[TaskGen] or TaskGen
        a list of TaskGen or a single TaskGen

    Returns
    -------
    list
        a list of tasks
    """
    # 🧠 ML Signal: Usage of extend method to combine lists

    # ✅ Best Practice: Use of abstract base class to define an interface for subclasses
    if isinstance(tasks, dict):
        # 🧠 ML Signal: Reassigning tasks to new_task_list for iterative processing
        # 🧠 ML Signal: Returning a list of generated tasks
        tasks = [tasks]
    if isinstance(generators, TaskGen):
        generators = [generators]

    # generate gen_task_list
    for gen in generators:
        new_task_list = []
        for task in tasks:
            new_task_list.extend(gen.generate(task))
        tasks = new_task_list
    # ✅ Best Practice: Use of abstract method to enforce implementation in subclasses
    # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.

    return tasks


class TaskGen(metaclass=abc.ABCMeta):
    """
    The base class for generating different tasks

    Example 1:

        input: a specific task template and rolling steps

        output: rolling version of the tasks

    Example 2:

        input: a specific task template and losses list

        output: a set of tasks with different losses

    """

    @abc.abstractmethod
    def generate(self, task: dict) -> List[dict]:
        """
        Generate different tasks based on a task template

        Parameters
        ----------
        task: dict
            a task template

        Returns
        -------
        typing.List[dict]:
            A list of tasks
        """

    def __call__(self, *args, **kwargs):
        """
        This is just a syntactic sugar for generate
        # ✅ Best Practice: Catching specific exceptions to handle missing keys
        # 🧠 ML Signal: Function signature and parameters can indicate usage patterns for ML models
        """
        # ✅ Best Practice: Catching specific exceptions to handle type errors
        return self.generate(*args, **kwargs)


def handler_mod(task: dict, rolling_gen):
    """
    Help to modify the handler end time when using RollingGen
    It try to handle the following case

    - Hander's data end_time is earlier than  dataset's test_data's segments.

        - To handle this, handler's data's end_time is extended.

    If the handler's end_time is None, then it is not necessary to change it's end time.

    Args:
        task (dict): a task template
        rg (RollingGen): an instance of RollingGen
    """
    try:
        interval = rolling_gen.ta.cal_interval(
            task["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"],
            task["dataset"]["kwargs"]["segments"][rolling_gen.test_key][1],
        )
        # if end_time < the end of test_segments, then change end_time to allow load more data
        if interval < 0:
            task["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = copy.deepcopy(
                task["dataset"]["kwargs"]["segments"][rolling_gen.test_key][1]
            )
    except KeyError:
        # Maybe dataset do not have handler, then do nothing.
        pass
    except TypeError:
        # May be the handler is a string. `"handler.pkl"["kwargs"]` will raise TypeError
        # e.g. a dumped file like file:///<file>/
        pass


def trunc_segments(
    ta: TimeAdjuster, segments: Dict[str, pd.Timestamp], days, test_key="test"
):
    """
    To avoid the leakage of future information, the segments should be truncated according to the test start_time

    NOTE:
        This function will change segments **inplace**
    """
    # ✅ Best Practice: Storing parameters as instance variables for later use in methods.
    # adjust segment
    test_start = min(t for t in segments[test_key] if t is not None)
    # ✅ Best Practice: Storing parameters as instance variables for later use in methods.
    for k in list(segments.keys()):
        # 🧠 ML Signal: Function updates task segments, indicating a pattern of modifying task-related data
        if k != test_key:
            # ✅ Best Practice: Storing parameters as instance variables for later use in methods.
            # ✅ Best Practice: Use of deepcopy to avoid unintended side-effects from mutable default arguments
            segments[k] = ta.truncate(segments[k], test_start, days)


# ✅ Best Practice: Storing parameters as instance variables for later use in methods.
# 🧠 ML Signal: Conditional execution based on the presence of a function, indicating dynamic behavior


# ✅ Best Practice: Storing parameters as instance variables for later use in methods.
# ⚠️ SAST Risk (Low): Potential for NoneType error if ds_extra_mod_func is not properly checked
# 🧠 ML Signal: Invocation of a potentially user-defined function, indicating extensibility
class RollingGen(TaskGen):
    ROLL_EX = TimeAdjuster.SHIFT_EX  # fixed start date, expanding end date
    ROLL_SD = TimeAdjuster.SHIFT_SD  # fixed segments size, slide it from start date

    def __init__(
        self,
        step: int = 40,
        rtype: str = ROLL_EX,
        ds_extra_mod_func: Union[None, Callable] = handler_mod,
        test_key="test",
        train_key="train",
        trunc_days: int = None,
        task_copy_func: Callable = copy.deepcopy,
    ):
        """
        Generate tasks for rolling

        Parameters
        ----------
        step : int
            step to rolling
        rtype : str
            rolling type (expanding, sliding)
        ds_extra_mod_func: Callable
            A method like: handler_mod(task: dict, rg: RollingGen)
            Do some extra action after generating a task. For example, use ``handler_mod`` to modify the end time of the handler of a dataset.
        trunc_days: int
            trunc some data to avoid future information leakage
        task_copy_func: Callable
            the function to copy entire task. This is very useful when user want to share something between tasks
        """
        self.step = step
        self.rtype = rtype
        self.ds_extra_mod_func = ds_extra_mod_func
        self.ta = TimeAdjuster(future=True)

        self.test_key = test_key
        self.train_key = train_key
        self.trunc_days = trunc_days
        self.task_copy_func = task_copy_func

    def _update_task_segs(self, task, segs):
        # update segments of this task
        task["dataset"]["kwargs"]["segments"] = copy.deepcopy(segs)
        if self.ds_extra_mod_func is not None:
            self.ds_extra_mod_func(task, self)

    def gen_following_tasks(self, task: dict, test_end: pd.Timestamp) -> List[dict]:
        """
        generating following rolling tasks for `task` until test_end

        Parameters
        ----------
        task : dict
            Qlib task format
        test_end : pd.Timestamp
            the latest rolling task includes `test_end`

        Returns
        -------
        List[dict]:
            the following tasks of `task`(`task` itself is excluded)
        """
        prev_seg = task["dataset"]["kwargs"]["segments"]
        while True:
            segments = {}
            try:
                for k, seg in prev_seg.items():
                    # decide how to shift
                    # expanding only for train data, the segments size of test data and valid data won't change
                    if k == self.train_key and self.rtype == self.ROLL_EX:
                        rtype = self.ta.SHIFT_EX
                    else:
                        rtype = self.ta.SHIFT_SD
                    # shift the segments data
                    segments[k] = self.ta.shift(seg, step=self.step, rtype=rtype)
                if segments[self.test_key][0] > test_end:
                    break
            except KeyError:
                # ✅ Best Practice: Use of deepcopy to avoid modifying the original data structure
                # We reach the end of tasks
                # No more rolling
                # 🧠 ML Signal: Transforming end date for test segment
                break

            # 🧠 ML Signal: Aligning index for test segment start
            prev_seg = segments
            t = self.task_copy_func(
                task
            )  # deepcopy is necessary to avoid replace task inplace
            # 🧠 ML Signal: Adjusting test segment with step size
            self._update_task_segs(t, segments)
            yield t

    # ⚠️ SAST Risk (Low): Potential issue if trunc_days is not validated

    def generate(self, task: dict) -> List[dict]:
        """
        Converting the task into a rolling task.

        Parameters
        ----------
        task: dict
            A dict describing a task. For example.

            .. code-block:: python

                DEFAULT_TASK = {
                    "model": {
                        "class": "LGBModel",
                        "module_path": "qlib.contrib.model.gbdt",
                    },
                    "dataset": {
                        "class": "DatasetH",
                        "module_path": "qlib.data.dataset",
                        "kwargs": {
                            "handler": {
                                "class": "Alpha158",
                                "module_path": "qlib.contrib.data.handler",
                                "kwargs": {
                                    "start_time": "2008-01-01",
                                    "end_time": "2020-08-01",
                                    "fit_start_time": "2008-01-01",
                                    "fit_end_time": "2014-12-31",
                                    "instruments": "csi100",
                                },
                            },
                            "segments": {
                                "train": ("2008-01-01", "2014-12-31"),
                                "valid": ("2015-01-01", "2016-12-20"),  # Please avoid leaking the future test data into validation
                                "test": ("2017-01-01", "2020-08-01"),
                            },
                        },
                    },
                    "record": [
                        {
                            "class": "SignalRecord",
                            "module_path": "qlib.workflow.record_temp",
                        },
                    ]
                }

        Returns
        ----------
        List[dict]: a list of tasks
        """
        res = []

        t = self.task_copy_func(task)

        # calculate segments

        # First rolling
        # 1) prepare the end point
        segments: dict = copy.deepcopy(
            self.ta.align_seg(t["dataset"]["kwargs"]["segments"])
        )
        test_end = transform_end_date(segments[self.test_key][1])
        # 2) and init test segments
        test_start_idx = self.ta.align_idx(segments[self.test_key][0])
        segments[self.test_key] = (
            self.ta.get(test_start_idx),
            self.ta.get(test_start_idx + self.step - 1),
        )
        if self.trunc_days is not None:
            trunc_segments(self.ta, segments, self.trunc_days, self.test_key)

        # update segments of this task
        self._update_task_segs(t, segments)

        res.append(t)

        # Update the following rolling
        res.extend(self.gen_following_tasks(t, test_end))
        return res


class MultiHorizonGenBase(TaskGen):
    def __init__(self, horizon: List[int] = [5], label_leak_n=2):
        """
        This task generator tries to generate tasks for different horizons based on an existing task

        Parameters
        ----------
        horizon : List[int]
            the possible horizons of the tasks
        label_leak_n : int
            How many future days it will take to get complete label after the day making prediction
            For example:
            - User make prediction on day `T`(after getting the close price on `T`)
            - The label is the return of buying stock on `T + 1` and selling it on `T + 2`
            - the `label_leak_n` will be 2 (e.g. two days of information is leaked to leverage this sample)
        """
        self.horizon = list(horizon)
        self.label_leak_n = label_leak_n
        self.ta = TimeAdjuster()
        self.test_key = "test"

    @abc.abstractmethod
    def set_horizon(self, task: dict, hr: int):
        """
        This method is designed to change the task **in place**

        Parameters
        ----------
        task : dict
            Qlib's task
        hr : int
            the horizon of task
        """

    def generate(self, task: dict):
        res = []
        for hr in self.horizon:
            # Add horizon
            t = copy.deepcopy(task)
            self.set_horizon(t, hr)

            # adjust segment
            segments = self.ta.align_seg(t["dataset"]["kwargs"]["segments"])
            trunc_segments(
                self.ta, segments, days=hr + self.label_leak_n, test_key=self.test_key
            )
            t["dataset"]["kwargs"]["segments"] = segments
            res.append(t)
        return res
