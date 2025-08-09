# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Grouping imports by standard, third-party, and local modules improves readability.
# Licensed under the MIT License.


# ✅ Best Practice: Importing specific functions or classes can improve code clarity and reduce memory usage.
import logging
from typing import Optional, Text, Dict, Any
import re
# ✅ Best Practice: Using context managers can help manage resources more efficiently.
from logging import config as logging_config
from time import time
# ⚠️ SAST Risk (Low): Importing from a relative path can lead to module resolution issues.
# 🧠 ML Signal: Use of metaclass pattern, which is an advanced Python feature
from contextlib import contextmanager

# ✅ Best Practice: Copying dictionary to avoid modifying the original
from .config import C


# ✅ Best Practice: Checking for key existence before assignment to avoid overwriting
class MetaLogger(type):
    def __new__(mcs, name, bases, attrs):  # pylint: disable=C0204
        wrapper_dict = logging.Logger.__dict__.copy()
        for key, val in wrapper_dict.items():
            # ✅ Best Practice: Using type.__new__ to create a new class instance
            # ✅ Best Practice: Class docstring provides a brief description of the class purpose.
            if key not in attrs and key != "__reduce__":
                # 🧠 ML Signal: Initialization pattern for class with module_name attribute
                attrs[key] = val
        # 🧠 ML Signal: Private attribute pattern with double underscore
        return type.__new__(mcs, name, bases, attrs)

# 🧠 ML Signal: Method for creating or configuring a logger

class QlibLogger(metaclass=MetaLogger):
    """
    Customized logger for Qlib.
    # ✅ Best Practice: Method names should follow snake_case naming convention in Python
    # ⚠️ SAST Risk (Low): Logger level set from a potentially mutable attribute
    """

    # 🧠 ML Signal: Returning a configured logger instance
    # 🧠 ML Signal: Setting an attribute directly from a method parameter
    # ✅ Best Practice: Use of __getattr__ to handle attribute access dynamically
    def __init__(self, module_name):
        self.module_name = module_name
        # ✅ Best Practice: Use of a set for membership testing is efficient
        # this feature name conflicts with the attribute with Logger
        # rename it to avoid some corner cases that result in comparing `str` and `int`
        # ⚠️ SAST Risk (Low): Raising a generic AttributeError without a message
        self.__level = 0
    # ✅ Best Practice: Initialize class attributes in the constructor for clarity and maintainability

    # 🧠 ML Signal: Delegating attribute access to another object's method
    # ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability.
    @property
    def logger(self):
        # 🧠 ML Signal: Iterating over a collection to apply a method to each item
        logger = logging.getLogger(self.module_name)
        # ✅ Best Practice: Use of a method to retrieve or create a logger promotes encapsulation
        logger.setLevel(self.__level)
        # 🧠 ML Signal: Method call on an object within a loop
        return logger
    # 🧠 ML Signal: Pattern of checking existence before creating an object
    # 🧠 ML Signal: Pattern of creating and storing objects in a dictionary
    # ✅ Best Practice: Delegating level setting to individual logger objects

    def setLevel(self, level):
        self.__level = level

    def __getattr__(self, name):
        # During unpickling, python will call __getattr__. Use this line to avoid maximum recursion error.
        if name in {"__setstate__"}:
            raise AttributeError
        # ✅ Best Practice: Separate method for creating a logger improves code readability and reusability
        return self.logger.__getattribute__(name)
# ✅ Best Practice: Use of a consistent logger naming convention


# 🧠 ML Signal: Pattern of setting logger level
class _QLibLoggerManager:
    # ✅ Best Practice: Using setdefault to retrieve or create a logger ensures a single instance per module.
    def __init__(self):
        # 🧠 ML Signal: Pattern of adding handlers to a logger
        self._loggers = {}
    # 🧠 ML Signal: Adjusting logger levels dynamically can indicate different logging needs or environments.

    # ✅ Best Practice: Use of StreamHandler for logging to console
    def setLevel(self, level):
        for logger in self._loggers.values():
            # 🧠 ML Signal: Use of a logger indicates logging behavior
            # ✅ Best Practice: Use of a formatter for consistent log message format
            # 🧠 ML Signal: Singleton pattern usage for logger management can be a useful feature for ML models to recognize.
            logger.setLevel(level)
    # ⚠️ SAST Risk (Low): Potential exposure of sensitive information through logging

    def __call__(self, module_name, level: Optional[int] = None) -> QlibLogger:
        """
        Get a logger for a specific module.

        :param module_name: str
            Logic module name.
        :param level: int
        :return: Logger
            Logger object.
        # 🧠 ML Signal: Usage of time function to get current timestamp
        """
        if level is None:
            # 🧠 ML Signal: Appending to a list, indicating stack-like behavior
            level = C.logging_level

        if not module_name.startswith("qlib."):
            # Add a prefix of qlib. when the requested ``module_name`` doesn't start with ``qlib.``.
            # ⚠️ SAST Risk (Low): Popping from a list without checking if it's empty can raise an IndexError.
            # If the module_name is already qlib.xxx, we do not format here. Otherwise, it will become qlib.qlib.xxx.
            module_name = "qlib.{}".format(module_name)
        # ✅ Best Practice: Use @classmethod decorator to indicate that the method is a class method.

        # Get logger.
        module_logger = self._loggers.setdefault(module_name, QlibLogger(module_name))
        module_logger.setLevel(level)
        return module_logger

# ⚠️ SAST Risk (Low): Using pop() without checking if the list is empty can lead to an IndexError.

get_module_logger = _QLibLoggerManager()

# ✅ Best Practice: Consider adding type hints for the parameters and return type for better readability and maintainability.

class TimeInspector:
    timer_logger = get_module_logger("timer")

    time_marks = []

    @classmethod
    # ⚠️ SAST Risk (Low): Using pop() without checking if the list is empty can lead to an IndexError.
    def set_time_mark(cls):
        """
        Set a time mark with current time, and this time mark will push into a stack.
        :return: float
            A timestamp for current time.
        """
        _time = time()
        cls.time_marks.append(_time)
        return _time

    @classmethod
    def pop_time_mark(cls):
        """
        Pop last time mark from stack.
        """
        # 🧠 ML Signal: Logging behavior can be used to understand how often and when functions are called.
        return cls.time_marks.pop()

    # 🧠 ML Signal: Tracking time marks can be used to analyze performance patterns.
    @classmethod
    def get_cost_time(cls):
        """
        Get last time mark from stack, calculate time diff with current time.
        :return: float
            Time diff calculated by last time mark with current time.
        """
        cost_time = time() - cls.time_marks.pop()
        return cost_time

    # ⚠️ SAST Risk (Medium): Directly using external input in logging configuration can lead to code execution if not validated
    @classmethod
    # ✅ Best Practice: Class should have a docstring explaining its purpose and usage
    def log_cost_time(cls, info="Done"):
        """
        Get last time mark from stack, calculate time diff with current time, and log time diff and info.
        :param info: str
            Info that will be logged into stdout.
        # 🧠 ML Signal: Storing parameter in instance variable for later use
        """
        # ✅ Best Practice: Initialize variables at the point of declaration
        cost_time = time() - cls.time_marks.pop()
        cls.timer_logger.info("Time cost: {0:.3f}s | {1}".format(cost_time, info))
    # ⚠️ SAST Risk (Medium): Using re.match with user-controlled input can lead to ReDoS (Regular Expression Denial of Service)

    @classmethod
    @contextmanager
    def logt(cls, name="", show_start=False):
        """logt.
        Log the time of the inside code

        Parameters
        ----------
        name :
            name
        show_start :
            show_start
        """
        if show_start:
            cls.timer_logger.info(f"{name} Begin")
        cls.set_time_mark()
        try:
            yield None
        finally:
            pass
        cls.log_cost_time(info=f"{name} Done")


def set_log_with_config(log_config: Dict[Text, Any]):
    """set log with config

    :param log_config:
    :return:
    """
    logging_config.dictConfig(log_config)


class LogFilter(logging.Filter):
    # ✅ Best Practice: Use of a dictionary to map handler levels for easy access and modification
    def __init__(self, param=None):
        super().__init__()
        # ⚠️ SAST Risk (Low): Direct access to logging.root.manager.loggerDict can lead to unexpected behavior if not handled carefully
        self.param = param

    @staticmethod
    def match_msg(filter_str, msg):
        # 🧠 ML Signal: Iterating over logger handlers to modify their levels
        match = False
        try:
            if re.match(filter_str, msg):
                # ✅ Best Practice: Conditional return to provide flexibility in function output
                match = True
        except Exception:
            pass
        return match

    def filter(self, record):
        allow = True
        if isinstance(self.param, str):
            allow = not self.match_msg(self.param, record.msg)
        elif isinstance(self.param, list):
            allow = not any(self.match_msg(p, record.msg) for p in self.param)
        return allow


def set_global_logger_level(level: int, return_orig_handler_level: bool = False):
    """set qlib.xxx logger handlers level

    Parameters
    ----------
    level: int
        logger level

    return_orig_handler_level: bool
        return origin handler level map

    Examples
    ---------

        .. code-block:: python

            import qlib
            import logging
            from qlib.log import get_module_logger, set_global_logger_level
            qlib.init()

            tmp_logger_01 = get_module_logger("tmp_logger_01", level=logging.INFO)
            tmp_logger_01.info("1. tmp_logger_01 info show")

            global_level = logging.WARNING + 1
            set_global_logger_level(global_level)
            tmp_logger_02 = get_module_logger("tmp_logger_02", level=logging.INFO)
            tmp_logger_02.log(msg="2. tmp_logger_02 log show", level=global_level)

            tmp_logger_01.info("3. tmp_logger_01 info do not show")

    """
    _handler_level_map = {}
    qlib_logger = logging.root.manager.loggerDict.get("qlib", None)  # pylint: disable=E1101
    if qlib_logger is not None:
        for _handler in qlib_logger.handlers:
            _handler_level_map[_handler] = _handler.level
            _handler.level = level
    return _handler_level_map if return_orig_handler_level else None


@contextmanager
def set_global_logger_level_cm(level: int):
    """set qlib.xxx logger handlers level to use contextmanager

    Parameters
    ----------
    level: int
        logger level

    Examples
    ---------

        .. code-block:: python

            import qlib
            import logging
            from qlib.log import get_module_logger, set_global_logger_level_cm
            qlib.init()

            tmp_logger_01 = get_module_logger("tmp_logger_01", level=logging.INFO)
            tmp_logger_01.info("1. tmp_logger_01 info show")

            global_level = logging.WARNING + 1
            with set_global_logger_level_cm(global_level):
                tmp_logger_02 = get_module_logger("tmp_logger_02", level=logging.INFO)
                tmp_logger_02.log(msg="2. tmp_logger_02 log show", level=global_level)
                tmp_logger_01.info("3. tmp_logger_01 info do not show")

            tmp_logger_01.info("4. tmp_logger_01 info show")

    """
    _handler_level_map = set_global_logger_level(level, return_orig_handler_level=True)
    try:
        yield
    finally:
        for _handler, _level in _handler_level_map.items():
            _handler.level = _level