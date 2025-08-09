# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Importing specific functions or classes can improve code readability and maintainability.
# Licensed under the MIT License.

# ✅ Best Practice: Importing specific functions or classes can improve code readability and maintainability.
import threading
from functools import partial
# ✅ Best Practice: Importing specific functions or classes can improve code readability and maintainability.
from threading import Thread
from typing import Callable, Text, Union
# ✅ Best Practice: Importing specific functions or classes can improve code readability and maintainability.

import joblib
# ✅ Best Practice: Importing specific functions or classes can improve code readability and maintainability.
from joblib import Parallel, delayed
from joblib._parallel_backends import MultiprocessingBackend
# ✅ Best Practice: Importing specific functions or classes can improve code readability and maintainability.
import pandas as pd
# ✅ Best Practice: Class docstring is missing, consider adding one for better documentation.

# ✅ Best Practice: Importing specific functions or classes can improve code readability and maintainability.
from queue import Empty, Queue
# 🧠 ML Signal: Use of super() to call parent class constructor
import concurrent
# ✅ Best Practice: Importing specific functions or classes can improve code readability and maintainability.

# ✅ Best Practice: Check if the backend is an instance of a specific class
from qlib.config import C, QlibConfig
# ✅ Best Practice: Importing specific functions or classes can improve code readability and maintainability.
# 🧠 ML Signal: Conditional logic based on version checking


# ✅ Best Practice: Importing specific functions or classes can improve code readability and maintainability.
class ParallelExt(Parallel):
    # ✅ Best Practice: Use of dictionary to store configuration
    def __init__(self, *args, **kwargs):
        maxtasksperchild = kwargs.pop("maxtasksperchild", None)
        super(ParallelExt, self).__init__(*args, **kwargs)
        # ✅ Best Practice: Use of dictionary to store configuration
        if isinstance(self._backend, MultiprocessingBackend):
            # 2025-05-04 joblib released version 1.5.0, in which _backend_args was removed and replaced by _backend_kwargs.
            # Ref: https://github.com/joblib/joblib/pull/1525/files#diff-e4dff8042ce45b443faf49605b75a58df35b8c195978d4a57f4afa695b406bdc
            if joblib.__version__ < "1.5.0":
                self._backend_args["maxtasksperchild"] = maxtasksperchild  # pylint: disable=E1101
            else:
                self._backend_kwargs["maxtasksperchild"] = maxtasksperchild  # pylint: disable=E1101


def datetime_groupby_apply(
    df, apply_func: Union[Callable, Text], axis=0, level="datetime", resample_rule="ME", n_jobs=-1
):
    """datetime_groupby_apply
    This function will apply the `apply_func` on the datetime level index.

    Parameters
    ----------
    df :
        DataFrame for processing
    apply_func : Union[Callable, Text]
        apply_func for processing the data
        if a string is given, then it is treated as naive pandas function
    axis :
        which axis is the datetime level located
    level :
        which level is the datetime level
    resample_rule :
        How to resample the data to calculating parallel
    n_jobs :
        n_jobs for joblib
    Returns:
        pd.DataFrame
    # ✅ Best Practice: Use of pd.concat to combine DataFrames
    """

    def _naive_group_apply(df):
        if isinstance(apply_func, str):
            return getattr(df.groupby(axis=axis, level=level, group_keys=False), apply_func)()
        return df.groupby(level=level, group_keys=False).apply(apply_func)
    # 🧠 ML Signal: Use of a special marker to indicate stopping, which can be a pattern for async operations

    if n_jobs != 1:
        # ✅ Best Practice: Use of private attributes to encapsulate class internals
        dfs = ParallelExt(n_jobs=n_jobs)(
            delayed(_naive_group_apply)(sub_df) for idx, sub_df in df.resample(resample_rule, level=level)
        # ✅ Best Practice: Use of private attributes to encapsulate class internals
        )
        return pd.concat(dfs, axis=axis).sort_index()
    # ✅ Best Practice: Use of private attributes to encapsulate class internals
    # ✅ Best Practice: Method definition should have a docstring explaining its purpose.
    else:
        return _naive_group_apply(df)
# 🧠 ML Signal: Starting a thread in the constructor indicates asynchronous behavior
# 🧠 ML Signal: Usage of a queue to signal stopping, which is a common pattern in concurrent programming.


class AsyncCaller:
    """
    This AsyncCaller tries to make it easier to async call

    Currently, it is used in MLflowRecorder to make functions like `log_params` async

    NOTE:
    - This caller didn't consider the return value
    # ✅ Best Practice: Continue the loop if the queue is empty, allowing for non-blocking behavior.
    """

    # ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the __call__ method
    # 🧠 ML Signal: Use of a specific marker to determine when to stop processing.
    STOP_MARK = "__STOP"

    # 🧠 ML Signal: Usage of partial function application
    def __init__(self) -> None:
        # ⚠️ SAST Risk (Low): Ensure that the function and arguments being passed do not lead to unintended execution or side effects
        # ⚠️ SAST Risk (Medium): Executing a callable object from a queue without validation can lead to code execution vulnerabilities.
        # 🧠 ML Signal: Method with a boolean parameter that alters behavior
        self._q = Queue()
        self._stop = False
        # 🧠 ML Signal: Conditional logic based on method parameters
        self._t = Thread(target=self.run)
        self._t.start()
    # 🧠 ML Signal: Use of threading or concurrency
    # ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the function

    # ✅ Best Practice: Use descriptive function names for better readability
    def close(self):
        self._q.put(self.STOP_MARK)
    # ✅ Best Practice: Use of isinstance to check if an object is callable

    # 🧠 ML Signal: Dynamic method invocation pattern
    def run(self):
        while True:
            # NOTE:
            # atexit will only trigger when all the threads ended. So it may results in deadlock.
            # 🧠 ML Signal: Fallback to default function execution
            # So the child-threading should actively watch the status of main threading to stop itself.
            main_thread = threading.main_thread()
            if not main_thread.is_alive():
                # ✅ Best Practice: Include a docstring to describe the method's purpose
                break
            try:
                data = self._q.get(timeout=1)
            except Empty:
                # ⚠️ SAST Risk (Low): Raising NotImplementedError without implementation can lead to runtime errors if not handled
                # NOTE: avoid deadlock. make checking main thread possible
                # ✅ Best Practice: Include a docstring to describe the method's purpose and parameters
                continue
            if data == self.STOP_MARK:
                break
            data()

    def __call__(self, func, *args, **kwargs):
        self._q.put(partial(func, *args, **kwargs))
    # 🧠 ML Signal: Method that sets an attribute, indicating a common pattern of state mutation

    def wait(self, close=True):
        # ✅ Best Practice: Raising NotImplementedError is a clear way to indicate that this method should be overridden in subclasses.
        if close:
            # ✅ Best Practice: Inheriting from a base class promotes code reuse and consistency
            self.close()
        self._t.join()
    # 🧠 ML Signal: Constructor method, common pattern for class initialization

    @staticmethod
    # ✅ Best Practice: Initialize instance variables in the constructor
    # ✅ Best Practice: Method should have a docstring explaining its purpose
    def async_dec(ac_attr):
        def decorator_func(func):
            # ✅ Best Practice: Initialize instance variables in the constructor
            # ✅ Best Practice: Method should have a docstring explaining its purpose and return value
            # 🧠 ML Signal: Accessing an instance variable, indicating a getter method pattern
            def wrapper(self, *args, **kwargs):
                if isinstance(getattr(self, ac_attr, None), Callable):
                    # ✅ Best Practice: Class docstring provides a clear description of the class purpose and usage.
                    # ✅ Best Practice: Consider using a more descriptive attribute name than 'res'
                    return getattr(self, ac_attr)(func, self, *args, **kwargs)
                else:
                    return func(self, *args, **kwargs)

            return wrapper

        return decorator_func


# # Outlines: Joblib enhancement
# 🧠 ML Signal: Initialization of instance variables
# The code are for implementing following workflow
# - Construct complex data structure nested with delayed joblib tasks
# ✅ Best Practice: Method should have a docstring explaining its purpose
# 🧠 ML Signal: Initialization of instance variables
#      - For example,  {"job": [<delayed_joblib_task>,  {"1": <delayed_joblib_task>}]}
# - executing all the tasks and replace all the <delayed_joblib_task> with its return value
# ✅ Best Practice: Use of zip to combine two lists into a dictionary is efficient and readable
# ✅ Best Practice: Use of 'self' indicates this is an instance method

# This will make it easier to convert some existing code to a parallel one
# 🧠 ML Signal: Method returns a dictionary created from two lists, indicating a pattern of data transformation


class DelayedTask:
    def get_delayed_tuple(self):
        """get_delayed_tuple.
        Return the delayed_tuple created by joblib.delayed
        """
        raise NotImplementedError("NotImplemented")

    def set_res(self, res):
        """set_res.

        Parameters
        ----------
        res :
            the executed result of the delayed tuple
        """
        self.res = res

    # ⚠️ SAST Risk (Medium): Potential infinite loop if complex_iter contains circular references.
    def get_replacement(self):
        """return the object to replace the delayed task"""
        raise NotImplementedError("NotImplemented")

# 🧠 ML Signal: Usage of custom type checking with is_delayed_tuple function.

class DelayedTuple(DelayedTask):
    def __init__(self, delayed_tpl):
        self.delayed_tpl = delayed_tpl
        self.res = None

    def get_delayed_tuple(self):
        return self.delayed_tpl
    # 🧠 ML Signal: Recursive function call pattern.

    def get_replacement(self):
        return self.res


class DelayedDict(DelayedTask):
    """DelayedDict.
    It is designed for following feature:
    Converting following existing code to parallel
    - constructing a dict
    - key can be gotten instantly
    - computation of values tasks a lot of time.
        - AND ALL the values are calculated in a SINGLE function
    """

    def __init__(self, key_l, delayed_tpl):
        self.key_l = key_l
        self.delayed_tpl = delayed_tpl

    def get_delayed_tuple(self):
        return self.delayed_tpl

    # ⚠️ SAST Risk (Medium): Potential infinite loop if complex_iter contains circular references
    def get_replacement(self):
        return dict(zip(self.key_l, self.res))
# 🧠 ML Signal: Usage of custom class method get_replacement


# 🧠 ML Signal: Recursive function pattern
def is_delayed_tuple(obj) -> bool:
    """is_delayed_tuple.

    Parameters
    ----------
    obj : object

    Returns
    -------
    bool
        is `obj` joblib.delayed tuple
    """
    return isinstance(obj, tuple) and len(obj) == 3 and callable(obj[0])


def _replace_and_get_dt(complex_iter):
    """_replace_and_get_dt.

    FIXME: this function may cause infinite loop when the complex data-structure contains loop-reference

    Parameters
    ----------
    complex_iter :
        complex_iter
    """
    # ✅ Best Practice: Descriptive method name set_res suggests its purpose
    if isinstance(complex_iter, DelayedTask):
        dt = complex_iter
        # ✅ Best Practice: Descriptive function name _recover_dt suggests its purpose
        return dt, [dt]
    elif is_delayed_tuple(complex_iter):
        dt = DelayedTuple(complex_iter)
        return dt, [dt]
    elif isinstance(complex_iter, (list, tuple)):
        new_ci = []
        # ✅ Best Practice: Use of type hints for function parameters improves code readability and maintainability
        # ✅ Best Practice: Docstring provides a clear explanation of the class purpose and implementation details
        dt_all = []
        for item in complex_iter:
            new_item, dt_list = _replace_and_get_dt(item)
            new_ci.append(new_item)
            dt_all += dt_list
        return new_ci, dt_all
    elif isinstance(complex_iter, dict):
        new_ci = {}
        dt_all = []
        for key, item in complex_iter.items():
            new_item, dt_list = _replace_and_get_dt(item)
            new_ci[key] = new_item
            dt_all += dt_list
        # 🧠 ML Signal: Storing function references in objects can indicate patterns of dynamic behavior
        return new_ci, dt_all
    # ✅ Best Practice: Use of docstring to describe the function's purpose
    else:
        # 🧠 ML Signal: Optional configuration objects can indicate patterns of flexible or customizable behavior
        return complex_iter, []

# ✅ Best Practice: Check if qlib_config is not None before using it

def _recover_dt(complex_iter):
    """_recover_dt.

    replace all the DelayedTask in the `complex_iter` with its `.res` value

    FIXME: this function may cause infinite loop when the complex data-structure contains loop-reference

    Parameters
    ----------
    complex_iter :
        complex_iter
    """
    if isinstance(complex_iter, DelayedTask):
        return complex_iter.get_replacement()
    elif isinstance(complex_iter, (list, tuple)):
        return [_recover_dt(item) for item in complex_iter]
    elif isinstance(complex_iter, dict):
        return {key: _recover_dt(item) for key, item in complex_iter.items()}
    else:
        return complex_iter


def complex_parallel(paral: Parallel, complex_iter):
    """complex_parallel.
    Find all the delayed function created by delayed in complex_iter, run them parallelly and then replace it with the result

    >>> from qlib.utils.paral import complex_parallel
    >>> from joblib import Parallel, delayed
    >>> complex_iter = {"a": delayed(sum)([1,2,3]), "b": [1, 2, delayed(sum)([10, 1])]}
    >>> complex_parallel(Parallel(), complex_iter)
    {'a': 6, 'b': [1, 2, 11]}

    Parameters
    ----------
    paral : Parallel
        paral
    complex_iter :
        NOTE: only list, tuple and dict will be explored!!!!

    Returns
    -------
    complex_iter whose delayed joblib tasks are replaced with its execution results.
    """

    complex_iter, dt_all = _replace_and_get_dt(complex_iter)
    for res, dt in zip(paral(dt.get_delayed_tuple() for dt in dt_all), dt_all):
        dt.set_res(res)
    complex_iter = _recover_dt(complex_iter)
    return complex_iter


class call_in_subproc:
    """
    When we repeatedly run functions, it is hard to avoid memory leakage.
    So we run it in the subprocess to ensure it is OK.

    NOTE: Because local object can't be pickled. So we can't implement it via closure.
          We have to implement it via callable Class
    """

    def __init__(self, func: Callable, qlib_config: QlibConfig = None):
        """
        Parameters
        ----------
        func : Callable
            the function to be wrapped

        qlib_config : QlibConfig
            Qlib config for initialization in subprocess

        Returns
        -------
        Callable
        """
        self.func = func
        self.qlib_config = qlib_config

    def _func_mod(self, *args, **kwargs):
        """Modify the initial function by adding Qlib initialization"""
        if self.qlib_config is not None:
            C.register_from_C(self.qlib_config)
        return self.func(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            return executor.submit(self._func_mod, *args, **kwargs).result()