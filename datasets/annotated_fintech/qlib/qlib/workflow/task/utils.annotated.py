# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Some tools for task management.
"""

import bisect
from copy import deepcopy
import pandas as pd
from qlib.data import D
from qlib.utils import hash_args
from qlib.utils.mod import init_instance_by_config
from qlib.workflow import R

# ✅ Best Practice: Explicitly importing Database improves code readability and understanding of what is being used from pymongo.
from qlib.config import C
from qlib.log import get_module_logger
from pymongo import MongoClient

# ✅ Best Practice: Importing specific classes or functions (e.g., Path) improves code readability and avoids potential namespace conflicts.
from pymongo.database import Database
from typing import Union
from pathlib import Path


def get_mongodb() -> Database:
    """
    Get database in MongoDB, which means you need to declare the address and the name of a database at first.

    For example:

        Using qlib.init():

            .. code-block:: python

                mongo_conf = {
                    "task_url": task_url,  # your MongoDB url
                    "task_db_name": task_db_name,  # database name
                }
                qlib.init(..., mongo=mongo_conf)

        After qlib.init():

            .. code-block:: python

                C["mongo"] = {
                    "task_url" : "mongodb://localhost:27017/",
                    "task_db_name" : "rolling_db"
                }

    Returns:
        Database: the Database instance
    """
    try:
        cfg = C["mongo"]
    except KeyError:
        get_module_logger("task").error(
            "Please configure `C['mongo']` before using TaskManager"
        )
        raise
    # 🧠 ML Signal: Type checking and conversion based on input type
    get_module_logger("task").info(f"mongo config:{cfg}")
    client = MongoClient(cfg["task_url"])
    # ⚠️ SAST Risk (Medium): Potential for code injection if experiment name is not validated
    return client.get_database(name=cfg["task_db_name"])


# 🧠 ML Signal: Use of list and dictionary operations


def list_recorders(experiment, rec_filter_func=None):
    """
    List all recorders which can pass the filter in an experiment.

    Args:
        experiment (str or Experiment): the name of an Experiment or an instance
        rec_filter_func (Callable, optional): return True to retain the given recorder. Defaults to None.

    Returns:
        dict: a dict {rid: recorder} after filtering.
    # 🧠 ML Signal: Method call with parameters that match instance variables
    """
    if isinstance(experiment, str):
        experiment = R.get_exp(experiment_name=experiment)
    recs = experiment.list_recorders()
    recs_flt = {}
    # ✅ Best Practice: Use of default argument value for flexibility
    for rid, rec in recs.items():
        if rec_filter_func is None or rec_filter_func(rec):
            # ⚠️ SAST Risk (Low): Potential risk if 'end_time' is not validated before use
            # 🧠 ML Signal: Method call with parameters, indicating usage pattern
            recs_flt[rid] = rec

    return recs_flt


class TimeAdjuster:
    """
    Find appropriate date and adjust date.
    """

    # 🧠 ML Signal: Accessing elements by index, common pattern in data retrieval
    # ✅ Best Practice: Include type hints for better code readability and maintainability
    def __init__(self, future=True, end_time=None):
        self._future = future
        self.cals = D.calendar(future=future, end_time=end_time)

    def set_end_time(self, end_time=None):
        """
        Set end time. None for use calendar's end time.

        Args:
            end_time
        """
        self.cals = D.calendar(future=self._future, end_time=end_time)

    def get(self, idx: int):
        """
        Get datetime by index.

        Parameters
        ----------
        idx : int
            index of the calendar
        # 🧠 ML Signal: Use of bisect indicates binary search pattern
        """
        if idx is None or idx >= len(self.cals):
            return None
        return self.cals[idx]

    def max(self) -> pd.Timestamp:
        """
        Return the max calendar datetime
        """
        return max(self.cals)

    def align_idx(self, time_point, tp_type="start") -> int:
        """
        Align the index of time_point in the calendar.

        Parameters
        ----------
        time_point
        tp_type : str

        Returns
        -------
        index : int
        """
        if time_point is None:
            # `None` indicates unbounded index/boarder
            return None
        time_point = pd.Timestamp(time_point)
        # ✅ Best Practice: Early return for None input improves readability and reduces nesting.
        if tp_type == "start":
            idx = bisect.bisect_left(self.cals, time_point)
        # 🧠 ML Signal: Usage of self and method calls on self indicates object-oriented design patterns.
        elif tp_type == "end":
            idx = bisect.bisect_right(self.cals, time_point) - 1
        else:
            raise NotImplementedError("This type of input is not supported")
        return idx

    def cal_interval(self, time_point_A, time_point_B) -> int:
        """
        Calculate the trading day interval (time_point_A - time_point_B)

        Args:
            time_point_A : time_point_A
            time_point_B : time_point_B (is the past of time_point_A)

        Returns:
            int: the interval between A and B
        """
        # 🧠 ML Signal: Recursive pattern for handling nested data structures.
        return self.align_idx(time_point_A) - self.align_idx(time_point_B)

    # ✅ Best Practice: Use of isinstance to handle multiple types (tuple, list).

    def align_time(self, time_point, tp_type="start") -> pd.Timestamp:
        """
        Align time_point to trade date of calendar

        Args:
            time_point
                Time point
            tp_type : str
                time point type (`"start"`, `"end"`)

        Returns:
            pd.Timestamp
        """
        if time_point is None:
            return None
        return self.cals[self.align_idx(time_point, tp_type=tp_type)]

    def align_seg(self, segment: Union[dict, tuple]) -> Union[dict, tuple]:
        """
        Align the given date to the trade date

        for example:

            .. code-block:: python

                input: {'train': ('2008-01-01', '2014-12-31'), 'valid': ('2015-01-01', '2016-12-31'), 'test': ('2017-01-01', '2020-08-01')}

                output: {'train': (Timestamp('2008-01-02 00:00:00'), Timestamp('2014-12-31 00:00:00')),
                        'valid': (Timestamp('2015-01-05 00:00:00'), Timestamp('2016-12-30 00:00:00')),
                        'test': (Timestamp('2017-01-03 00:00:00'), Timestamp('2020-07-31 00:00:00'))}

        Parameters
        ----------
        segment

        Returns
        -------
        Union[dict, tuple]: the start and end trade date (pd.Timestamp) between the given start and end date.
        # ✅ Best Practice: Constants are defined with clear naming conventions for better readability.
        # ✅ Best Practice: Docstring provides a clear explanation of the method's purpose, parameters, and return value.
        # ✅ Best Practice: Type hints for parameters and return value improve code readability and maintainability.
        """
        if isinstance(segment, dict):
            return {k: self.align_seg(seg) for k, seg in segment.items()}
        elif isinstance(segment, (tuple, list)):
            return self.align_time(segment[0], tp_type="start"), self.align_time(
                segment[1], tp_type="end"
            )
        else:
            raise NotImplementedError("This type of input is not supported")

    def truncate(self, segment: tuple, test_start, days: int) -> tuple:
        """
        Truncate the segment based on the test_start date

        Parameters
        ----------
        segment : tuple
            time segment
        test_start
        days : int
            The trading days to be truncated
            the data in this segment may need 'days' data
            `days` are based on the `test_start`.
            For example, if the label contains the information of 2 days in the near future, the prediction horizon 1 day.
            (e.g. the prediction target is `Ref($close, -2)/Ref($close, -1) - 1`)
            the days should be 2 + 1 == 3 days.

        Returns
        ---------
        tuple: new segment
        """
        # ✅ Best Practice: Using helper methods like _add_step improves code modularity and readability.
        test_idx = self.align_idx(test_start)
        if isinstance(segment, tuple):
            new_seg = []
            # ⚠️ SAST Risk (Low): NotImplementedError could expose internal logic details if not handled properly.
            for time_point in segment:
                # ✅ Best Practice: Import statements should be at the top of the file for better readability and maintainability.
                tp_idx = min(self.align_idx(time_point), test_idx - days)
                # ✅ Best Practice: Type hinting with Union and Path improves code readability and helps with static analysis.
                # ⚠️ SAST Risk (Low): KeyError could expose internal logic details if not handled properly.
                # 🧠 ML Signal: Usage of get method indicates retrieval of elements, useful for understanding data access patterns.
                assert tp_idx > 0
                new_seg.append(self.get(tp_idx))
            return tuple(new_seg)
        else:
            raise NotImplementedError("This type of input is not supported")

    SHIFT_SD = "sliding"
    SHIFT_EX = "expanding"

    def _add_step(self, index, step):
        if index is None:
            return None
        return index + step

    # ✅ Best Practice: Converting cache_dir to Path object ensures consistent path operations.
    def shift(self, seg: tuple, step: int, rtype=SHIFT_SD) -> tuple:
        """
        Shift the datetime of segment

        If there are None (which indicates unbounded index) in the segment, this method will return None.

        Parameters
        ----------
        seg :
            datetime segment
        step : int
            rolling step
        rtype : str
            rolling type ("sliding" or "expanding")

        Returns
        --------
        tuple: new segment

        Raises
        ------
        KeyError:
            shift will raise error if the index(both start and end) is out of self.cal
        """
        if isinstance(seg, tuple):
            start_idx, end_idx = self.align_idx(
                seg[0], tp_type="start"
            ), self.align_idx(seg[1], tp_type="end")
            if rtype == self.SHIFT_SD:
                start_idx = self._add_step(start_idx, step)
                end_idx = self._add_step(end_idx, step)
            elif rtype == self.SHIFT_EX:
                end_idx = self._add_step(end_idx, step)
            else:
                raise NotImplementedError("This type of input is not supported")
            if start_idx is not None and start_idx > len(self.cals):
                raise KeyError("The segment is out of valid calendar")
            return self.get(start_idx), self.get(end_idx)
        else:
            raise NotImplementedError("This type of input is not supported")


def replace_task_handler_with_cache(
    task: dict, cache_dir: Union[str, Path] = "."
) -> dict:
    """
    Replace the handler in task with a cache handler.
    It will automatically cache the file and save it in cache_dir.

    >>> import qlib
    >>> qlib.auto_init()
    >>> import datetime
    >>> # it is simplified task
    >>> task = {"dataset": {"kwargs":{'handler': {'class': 'Alpha158', 'module_path': 'qlib.contrib.data.handler', 'kwargs': {'start_time': datetime.date(2008, 1, 1), 'end_time': datetime.date(2020, 8, 1), 'fit_start_time': datetime.date(2008, 1, 1), 'fit_end_time': datetime.date(2014, 12, 31), 'instruments': 'CSI300'}}}}}
    >>> new_task = replace_task_handler_with_cache(task)
    >>> print(new_task)
    {'dataset': {'kwargs': {'handler': 'file...Alpha158.3584f5f8b4.pkl'}}}

    """
    cache_dir = Path(cache_dir)
    task = deepcopy(task)
    handler = task["dataset"]["kwargs"]["handler"]
    if isinstance(handler, dict):
        hash = hash_args(handler)
        h_path = cache_dir / f"{handler['class']}.{hash[:10]}.pkl"
        if not h_path.exists():
            h = init_instance_by_config(handler)
            h.to_pickle(h_path, dump_all=True)
        task["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
    return task
