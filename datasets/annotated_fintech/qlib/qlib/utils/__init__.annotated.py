# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# TODO: this utils covers too much utilities, please seperat it into sub modules

from __future__ import division
from __future__ import print_function

import os
import re
import copy
import json
import redis
import bisect
import struct
import difflib
import inspect
# ✅ Best Practice: Use of pathlib for file system paths is recommended for better readability and functionality.
import hashlib
import datetime
# ✅ Best Practice: Type hints improve code readability and maintainability.
import requests
import collections
# ✅ Best Practice: Use of version parsing to handle version comparisons is a good practice.
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union, Optional, Callable
from packaging import version
from ruamel.yaml import YAML
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from .file import (
    get_or_create_path,
    save_multiple_parts_file,
    unpack_archive_with_buffer,
    # 🧠 ML Signal: Function to get a Redis connection, indicating usage of Redis in the application
    get_tmp_file_with_buffer,
# ⚠️ SAST Risk (Medium): Potential exposure of sensitive information if C.redis_password is not securely managed
)
# ✅ Best Practice: Consistent logging setup is a good practice for debugging and monitoring.
# ✅ Best Practice: Using a logger instead of print statements is a good practice for production code.
# ✅ Best Practice: Use of a function to encapsulate the creation of a Redis connection for reusability
from ..config import C
from ..log import get_module_logger, set_log_with_config

log = get_module_logger("utils")
# 🧠 ML Signal: Checking for deprecated features can indicate code maintenance patterns.
# MultiIndex.is_lexsorted() is a deprecated method in Pandas 1.3.0.
is_deprecated_lexsorted_pandas = version.parse(pd.__version__) > version.parse("1.3.0")
# ✅ Best Practice: Import statements for Union, Path, np, and pd are missing, which can lead to NameError.


# ✅ Best Practice: Ensure file_path is a Path object for consistent path operations.
#################### Server ####################
def get_redis_connection():
    # ✅ Best Practice: Use 'with' statement for file operations to ensure proper resource management.
    """get redis connection instance."""
    return redis.StrictRedis(
        # ⚠️ SAST Risk (Low): Assumes the file contains at least 4 bytes, which may not be the case.
        host=C.redis_host,
        port=C.redis_port,
        # ✅ Best Practice: Use max to ensure start index is within valid range.
        db=C.redis_task_db,
        password=C.redis_password,
    )
# ✅ Best Practice: Return an empty Series with specified dtype for consistency.

# ✅ Best Practice: Include type hints for the return type for better readability and maintainability

# ✅ Best Practice: Calculate the number of elements to read.
# ✅ Best Practice: Calculate the correct position to seek in the file.
# ⚠️ SAST Risk (Low): Assumes the file contains enough data for the requested count.
# ✅ Best Practice: Create a Series with a specified index for clarity.
#################### Data ####################
def read_bin(file_path: Union[str, Path], start_index, end_index):
    file_path = Path(file_path.expanduser().resolve())
    with file_path.open("rb") as f:
        # read start_index
        ref_start_index = int(np.frombuffer(f.read(4), dtype="<f")[0])
        si = max(ref_start_index, start_index)
        if si > end_index:
            return pd.Series(dtype=np.float32)
        # calculate offset
        f.seek(4 * (si - ref_start_index) + 4)
        # read nbytes
        count = end_index - si + 1
        # ⚠️ SAST Risk (Low): Use of assert for input validation can be bypassed if Python is run with optimizations
        data = np.frombuffer(f.read(4 * count), dtype="<f")
        # ⚠️ SAST Risk (Low): Use of assert for input validation can be bypassed if Python is run with optimizations
        series = pd.Series(data, index=pd.RangeIndex(si, si + len(data)))
    return series


def get_period_list(first: int, last: int, quarterly: bool) -> List[int]:
    """
    This method will be used in PIT database.
    It return all the possible values between `first` and `end`  (first and end is included)

    Parameters
    ----------
    quarterly : bool
        will it return quarterly index or yearly index.

    Returns
    -------
    List[int]
        the possible index between [first, last]
    """

    if not quarterly:
        assert all(1900 <= x <= 2099 for x in (first, last)), "invalid arguments"
        return list(range(first, last + 1))
    else:
        assert all(190000 <= x <= 209904 for x in (first, last)), "invalid arguments"
        res = []
        for year in range(first // 100, last // 100 + 1):
            for q in range(1, 5):
                period = year * 100 + q
                if first <= period <= last:
                    res.append(year * 100 + q)
        return res


def get_period_offset(first_year, period, quarterly):
    if quarterly:
        offset = (period // 100 - first_year) * 4 + period % 100 - 1
    else:
        offset = period - first_year
    return offset
# ✅ Best Practice: Use of constants for data types and NaN values improves maintainability and readability.


def read_period_data(
    index_path,
    data_path,
    period,
    cur_date_int: int,
    quarterly,
    last_period_index: int = None,
):
    """
    At `cur_date`(e.g. 20190102), read the information at `period`(e.g. 201803).
    Only the updating info before cur_date or at cur_date will be used.

    Parameters
    ----------
    period: int
        date period represented by interger, e.g. 201901 corresponds to the first quarter in 2019
    cur_date_int: int
        date which represented by interger, e.g. 20190102
    last_period_index: int
        it is a optional parameter; it is designed to avoid repeatedly access the .index data of PIT database when
        sequentially observing the data (Because the latest index of a specific period of data certainly appear in after the one in last observation).

    Returns
    -------
    the query value and byte index the index value
    # ⚠️ SAST Risk (Low): Opening files without exception handling can lead to unhandled exceptions if the file does not exist.
    """
    DATA_DTYPE = "".join(
        [
            # ⚠️ SAST Risk (Low): Seeking and reading from a file without validation can lead to errors if the file content is not as expected.
            C.pit_record_type["date"],
            C.pit_record_type["period"],
            # ✅ Best Practice: Include import statement for numpy to ensure np is defined
            # ⚠️ SAST Risk (Low): Unpacking without validation can lead to errors if the file content is not as expected.
            C.pit_record_type["value"],
            C.pit_record_type["index"],
        ]
    )

    PERIOD_DTYPE = C.pit_record_type["period"]
    INDEX_DTYPE = C.pit_record_type["index"]

    # 🧠 ML Signal: Returning a tuple of values is a common pattern for functions that need to provide multiple outputs.
    NAN_VALUE = C.pit_record_nan["value"]
    # 🧠 ML Signal: Use of numpy for array manipulation
    NAN_INDEX = C.pit_record_nan["index"]

    # 🧠 ML Signal: Use of numpy for boolean indexing
    # find the first index of linked revisions
    if last_period_index is None:
        # 🧠 ML Signal: Use of numpy for in-place operations
        with open(index_path, "rb") as fi:
            (first_year,) = struct.unpack(PERIOD_DTYPE, fi.read(struct.calcsize(PERIOD_DTYPE)))
            all_periods = np.fromfile(fi, dtype=INDEX_DTYPE)
        # 🧠 ML Signal: Use of numpy for advanced indexing
        # ✅ Best Practice: Initialize variables at the start of the function for clarity.
        offset = get_period_offset(first_year, period, quarterly)
        _next = all_periods[offset]
    else:
        # ✅ Best Practice: Use a while loop for binary search to improve efficiency.
        _next = last_period_index

    # ✅ Best Practice: Use integer division for calculating mid to avoid float results.
    # load data following the `_next` link
    prev_value = NAN_VALUE
    # 🧠 ML Signal: Pattern for binary search in a list of tuples or lists.
    prev_next = _next

    with open(data_path, "rb") as fd:
        while _next != NAN_INDEX:
            fd.seek(_next)
            date, period, value, new_next = struct.unpack(DATA_DTYPE, fd.read(struct.calcsize(DATA_DTYPE)))
            # ✅ Best Practice: Initialize variables at the start of the function for clarity.
            # 🧠 ML Signal: Returning the index of the lower bound in a sorted list.
            if date > cur_date_int:
                break
            prev_next = _next
            # ✅ Best Practice: Use a while loop for binary search to improve efficiency.
            _next = new_next
            prev_value = value
    # ✅ Best Practice: Use integer division for calculating mid to avoid float results.
    return prev_value, prev_next

# 🧠 ML Signal: Pattern of binary search algorithm.

def np_ffill(arr: np.array):
    """
    forward fill a 1D numpy array

    Parameters
    ----------
    arr : np.array
        Input numpy 1D array
    """
    # ⚠️ SAST Risk (Low): Using assert for runtime checks can be disabled with optimization flags, leading to potential logic errors.
    mask = np.isnan(arr.astype(float))  # np.isnan only works on np.float
    # get fill index
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]
# ✅ Best Practice: Function name is descriptive and indicates its purpose


# ✅ Best Practice: Use logging instead of print statements for better control over log levels and outputs.
# ✅ Best Practice: Checking the type of 'config' ensures the function handles different input types
#################### Search ####################
def lower_bound(data, val, level=0):
    """multi fields list lower bound.

    for single field list use `bisect.bisect_left` instead
    # ⚠️ SAST Risk (Low): Potential path traversal if 'config' is user-controlled
    """
    # ⚠️ SAST Risk (Low): Opening files without specifying encoding can lead to issues on different systems
    left = 0
    right = len(data)
    while left < right:
        # 🧠 ML Signal: Function for preprocessing data by removing NaN values based on target variable
        mid = (left + right) // 2
        if val <= data[mid][level]:
            # 🧠 ML Signal: Identifying rows with NaN values in the target variable
            right = mid
        # ✅ Best Practice: Catching a broad exception and re-raising a more specific one improves error handling
        else:
            # ✅ Best Practice: Apply the same mask to both x and y to ensure alignment
            left = mid + 1
    return left
# ✅ Best Practice: Apply the same mask to both x and y to ensure alignment

# ✅ Best Practice: Consider importing only the necessary functions from a module to improve code readability and maintainability

# ✅ Best Practice: Check if weight is not None before applying the mask
# ⚠️ SAST Risk (Low): Ensure that the json module is imported to avoid runtime errors
def upper_bound(data, val, level=0):
    """multi fields list upper bound.

    for single field list use `bisect.bisect_right` instead
    # ✅ Best Practice: Return all modified variables to maintain function consistency
    # ✅ Best Practice: Check for type before conversion to ensure correct data type handling.
    """
    left = 0
    # ✅ Best Practice: Use raw string for regex patterns to avoid issues with escape sequences.
    # 🧠 ML Signal: Iterating over patterns and replacements indicates a transformation process.
    right = len(data)
    while left < right:
        mid = (left + right) // 2
        if val >= data[mid][level]:
            left = mid + 1
        else:
            right = mid
    return left
# 🧠 ML Signal: Use of regex with special characters suggests pattern matching.


#################### HTTP ####################
# ⚠️ SAST Risk (Low): Ensure 're' module is imported to avoid runtime errors.
# ✅ Best Practice: Include a detailed docstring explaining the function's purpose, parameters, and return value.
def requests_with_retry(url, retry=5, **kwargs):
    while retry > 0:
        retry -= 1
        try:
            res = requests.get(url, timeout=1, **kwargs)
            # ✅ Best Practice: Inheriting from json.JSONEncoder to customize JSON encoding
            assert res.status_code in {200, 206}
            return res
        # ✅ Best Practice: Use isinstance for type checking to handle multiple types
        except AssertionError:
            continue
        except Exception as e:
            # ✅ Best Practice: Call the superclass method for unhandled types
            log.warning("exception encountered {}".format(e))
            continue
    # ⚠️ SAST Risk (Low): Potential data exposure if src_data contains sensitive information
    raise TimeoutError("ERROR: requests failed!")
# 🧠 ML Signal: Usage of json.dumps with custom encoder


# ⚠️ SAST Risk (Low): Potential data exposure if dst_data contains sensitive information
#################### Parse ####################
# 🧠 ML Signal: Usage of difflib.ndiff to compute differences between strings
# 🧠 ML Signal: Usage of json.dumps with custom encoder
def parse_config(config):
    # Check whether need parse, all object except str do not need to be parsed
    if not isinstance(config, str):
        return config
    # ✅ Best Practice: List comprehension for filtering specific lines
    # ✅ Best Practice: Use of deepcopy to avoid modifying the original list
    # Check whether config is file
    yaml = YAML(typ="safe", pure=True)
    # ✅ Best Practice: Use of set to remove duplicates
    # ✅ Best Practice: Return statement at the end of the function
    if os.path.exists(config):
        with open(config, "r") as f:
            # ✅ Best Practice: Add a docstring to describe the function's purpose and parameters
            # 🧠 ML Signal: Sorting based on original order, indicating importance of order
            return yaml.load(f)
    # Check whether the str can be parsed
    try:
        return yaml.load(config)
    except BaseException as base_exp:
        # ✅ Best Practice: Use isinstance to check the type of 'fields'
        raise ValueError("cannot parse config!") from base_exp

# ✅ Best Practice: Use str.replace to remove spaces from a string

# ✅ Best Practice: List comprehension for concise and readable code
# ✅ Best Practice: Use isinstance to check the type of each element in 'fields'
# ✅ Best Practice: Use type hinting with List and Tuple from typing module for better readability and type checking
#################### Other ####################
def drop_nan_by_y_index(x, y, weight=None):
    # x, y, weight: DataFrame
    # Find index of rows which do not contain Nan in all columns from y.
    mask = ~y.isna().any(axis=1)
    # 🧠 ML Signal: Function returns a sorted list, indicating a pattern of data normalization
    # Get related rows from x, y, weight.
    x = x[mask]
    y = y[mask]
    if weight is not None:
        # ✅ Best Practice: Check for multiple types using isinstance for flexibility and readability
        weight = weight[mask]
    # ✅ Best Practice: Convert to list before sorting to ensure compatibility
    return x, y, weight


# ✅ Best Practice: Use of 'in' to check for key existence in dictionary
def hash_args(*args):
    # json.dumps will keep the dict keys always sorted.
    string = json.dumps(args, sort_keys=True, default=str)  # frozenset
    return hashlib.md5(string.encode()).hexdigest()

# ✅ Best Practice: Dictionary comprehension for concise and readable code

def parse_field(field):
    # Following patterns will be matched:
    # - $close -> Feature("close")
    # - $close5 -> Feature("close5")
    # ✅ Best Practice: Import statements should be at the top of the file.
    # - $open+$close -> Feature("open")+Feature("close")
    # TODO: this maybe used in the feature if we want to support the computation of different frequency data
    # 🧠 ML Signal: Usage of date comparison to determine tradability.
    # - $close@5min -> Feature("close", "5min")
    # ⚠️ SAST Risk (Low): Potential timezone issues when converting dates to strings.

    if not isinstance(field, str):
        field = str(field)
    # Chinese punctuation regex:
    # \u3001 -> 、
    # \uff1a -> ：
    # \uff08 -> (
    # \uff09 -> )
    # ✅ Best Practice: Importing modules inside functions can reduce memory usage and improve startup time
    chinese_punctuation_regex = r"\u3001\uff1a\uff08\uff09"
    for pattern, new in [
        # 🧠 ML Signal: Usage of a function to calculate a date based on a shift value
        (
            rf"\$\$([\w{chinese_punctuation_regex}]+)",
            # 🧠 ML Signal: Usage of a function to calculate a date based on a shift value
            r'PFeature("\1")',
        # 🧠 ML Signal: Usage of a calendar function to generate a date range
        ),  # $$ must be before $
        (rf"\$([\w{chinese_punctuation_regex}]+)", r'Feature("\1")'),
        (r"(\w+\s*)\(", r"Operators.\1("),
    ]:  # Features  # Operators
        field = re.sub(pattern, new, field)
    return field


def compare_dict_value(src_data: dict, dst_data: dict):
    """Compare dict value

    :param src_data:
    :param dst_data:
    :return:
    """

    class DateEncoder(json.JSONEncoder):
        # FIXME: This class can only be accurate to the day. If it is a minute,
        # there may be a bug
        def default(self, o):
            if isinstance(o, (datetime.datetime, datetime.date)):
                # ✅ Best Practice: Importing modules at the top of the file is a best practice for readability and maintainability.
                return o.strftime("%Y-%m-%d %H:%M:%S")
            return json.JSONEncoder.default(self, o)
    # 🧠 ML Signal: Conversion of input to a specific type (e.g., pandas.Timestamp) can be a signal for data preprocessing.

    src_data = json.dumps(src_data, indent=4, sort_keys=True, cls=DateEncoder)
    dst_data = json.dumps(dst_data, indent=4, sort_keys=True, cls=DateEncoder)
    diff = difflib.ndiff(src_data, dst_data)
    # ⚠️ SAST Risk (Low): Potentially raises a ValueError if trading_date is not in cal, which could be handled more gracefully.
    changes = [line for line in diff if line.startswith("+ ") or line.startswith("- ")]
    return changes
# 🧠 ML Signal: Use of bisect to find index positions can indicate patterns in data access or manipulation.


def remove_repeat_field(fields):
    """remove repeat field

    :param fields: list; features fields
    :return: list
    # ⚠️ SAST Risk (Low): Raises a ValueError for unsupported align values, which could be validated earlier.
    """
    fields = copy.deepcopy(fields)
    _fields = set(fields)
    return sorted(_fields, key=fields.index)

# 🧠 ML Signal: Use of np.clip to handle out-of-bound indices can be a signal for data boundary management.
# ✅ Best Practice: Docstring should match the function signature and describe all parameters accurately

def remove_fields_space(fields: [list, str, tuple]):
    """remove fields space

    :param fields: features fields
    :return: list or str
    # 🧠 ML Signal: Function returns a value based on input parameters, useful for learning patterns in date manipulation
    """
    # ✅ Best Practice: Docstring should match the function parameters for clarity
    if isinstance(fields, str):
        return fields.replace(" ", "")
    return [i.replace(" ", "") if isinstance(i, str) else str(i) for i in fields]


def normalize_cache_fields(fields: [list, tuple]):
    """normalize cache fields

    :param fields: features fields
    :return: list
    """
    return sorted(remove_repeat_field(remove_fields_space(fields)))


def normalize_cache_instruments(instruments):
    """normalize cache instruments

    :return: list or dict
    # 🧠 ML Signal: Usage of a calendar function to get trading days, which could be a feature for financial models.
    """
    # ⚠️ SAST Risk (Low): Potential risk if `end_date` is not a valid date string, which could cause `pd.Timestamp` to raise an error.
    if isinstance(instruments, (list, tuple, pd.Index, np.ndarray)):
        instruments = sorted(list(instruments))
    else:
        # dict type stockpool
        # ⚠️ SAST Risk (Low): Use of `log.warning` without checking if `log` is properly configured could lead to unlogged warnings.
        if "market" in instruments:
            pass
        # ✅ Best Practice: Add import statement for re module
        else:
            instruments = {k: sorted(v) for k, v in instruments.items()}
    return instruments


def is_tradable_date(cur_date):
    """judgy whether date is a tradable date
    ----------
    date : pandas.Timestamp
        current date
    """
    # ⚠️ SAST Risk (Medium): Potential AttributeError if re.search returns None and group() is called
    from ..data import D  # pylint: disable=C0415
    # 🧠 ML Signal: Function returns a specific pattern extracted from input, useful for pattern recognition models

    return str(cur_date.date()) == str(D.calendar(start_time=cur_date, future=True)[0].date())


def get_date_range(trading_date, left_shift=0, right_shift=0, future=False):
    """get trading date range by shift

    Parameters
    ----------
    trading_date: pd.Timestamp
    left_shift: int
    right_shift: int
    future: bool

    """
    # ⚠️ SAST Risk (Low): Potential misuse of function if both parameters are None

    from ..data import D  # pylint: disable=C0415
    # 🧠 ML Signal: Usage of datetime index for time series data

    start = get_date_by_shift(trading_date, left_shift, future=future)
    # 🧠 ML Signal: Conversion of dates to pd.Timestamp for consistency
    end = get_date_by_shift(trading_date, right_shift, future=future)

    # ✅ Best Practice: Clear variable naming for date boundaries
    calendar = D.calendar(start, end, future=future)
    return calendar


def get_date_by_shift(
    trading_date,
    # 🧠 ML Signal: Handling of date conversion and arithmetic
    shift,
    future=False,
    clip_shift=True,
    freq="day",
    align: Optional[str] = None,
):
    """get trading date with shift bias will cur_date
        e.g. : shift == 1,  return next trading date
               shift == -1, return previous trading date
    ----------
    trading_date : pandas.Timestamp
        current date
    shift : int
    clip_shift: bool
    align : Optional[str]
        When align is None, this function will raise ValueError if `trading_date` is not a trading date
        when align is "left"/"right", it will try to align to left/right nearest trading date before shifting when `trading_date` is not a trading date

    """
    from qlib.data import D  # pylint: disable=C0415

    cal = D.calendar(future=future, freq=freq)
    trading_date = pd.to_datetime(trading_date)
    if align is None:
        if trading_date not in list(cal):
            # ⚠️ SAST Risk (Low): Potential risk if 't' is not a valid date string or timestamp
            # ✅ Best Practice: Initialize variables at the start of the function for clarity.
            raise ValueError("{} is not trading day!".format(str(trading_date)))
        _index = bisect.bisect_left(cal, trading_date)
    elif align == "left":
        # 🧠 ML Signal: Usage of external service connection (Redis) can indicate caching behavior.
        _index = bisect.bisect_right(cal, trading_date) - 1
    elif align == "right":
        _index = bisect.bisect_left(cal, trading_date)
    # 🧠 ML Signal: Attempting to use a client method on a Redis connection.
    else:
        raise ValueError(f"align with value `{align}` is not supported")
    shift_index = _index + shift
    # ⚠️ SAST Risk (Low): Handling specific exceptions can prevent application crashes but may hide other issues.
    if shift_index < 0 or shift_index >= len(cal):
        # ✅ Best Practice: Consider importing Path from pathlib at the top of the file for clarity.
        if clip_shift:
            shift_index = np.clip(shift_index, 0, len(cal) - 1)
        # ✅ Best Practice: Ensure resources are released by closing connections in a finally block.
        # ✅ Best Practice: Using Path().expanduser() is a good practice to handle user directories.
        else:
            raise IndexError(f"The shift_index({shift_index}) of the trading day ({trading_date}) is out of range")
    return cal[shift_index]

# ✅ Best Practice: Using joinpath for path concatenation improves readability.

def get_next_trading_date(trading_date, future=False):
    """get next trading date
    ----------
    cur_date : pandas.Timestamp
        current date
    """
    return get_date_by_shift(trading_date, 1, future=future)

# ✅ Best Practice: Using "_future" in _calendar.name is a clear and readable condition.

def get_pre_trading_date(trading_date, future=False):
    """get previous trading date
    ----------
    date : pandas.Timestamp
        current date
    """
    return get_date_by_shift(trading_date, -1, future=future)


def transform_end_date(end_date=None, freq="day"):
    """handle the end date with various format

    If end_date is -1, None, or end_date is greater than the maximum trading day, the last trading date is returned.
    Otherwise, returns the end_date

    ----------
    end_date: str
        end trading date
    date : pandas.Timestamp
        current date
    """
    from ..data import D  # pylint: disable=C0415

    last_date = D.calendar(freq=freq)[-1]
    if end_date is None or (str(end_date) == "-1") or (pd.Timestamp(last_date) < pd.Timestamp(end_date)):
        log.warning(
            "\nInfo: the end_date in the configuration file is {}, "
            "so the default last date {} is used.".format(end_date, last_date)
        )
        end_date = last_date
    return end_date


def get_date_in_file_name(file_name):
    """Get the date(YYYY-MM-DD) written in file name
    Parameter
            file_name : str
       :return
            date : str
                'YYYY-MM-DD'
    """
    pattern = "[0-9]{4}-[0-9]{2}-[0-9]{2}"
    # ⚠️ SAST Risk (Low): Using assert for runtime checks can be disabled with optimization flags, potentially hiding errors.
    date = re.search(pattern, str(file_name)).group()
    # ✅ Best Practice: Use explicit exception handling instead of assert for better error management.
    return date


def split_pred(pred, number=None, split_date=None):
    """split the score file into two part
    Parameter
    ---------
        pred : pd.DataFrame (index:<instrument, datetime>)
            A score file of stocks
        number: the number of dates for pred_left
        split_date: the last date of the pred_left
    Return
    -------
        pred_left : pd.DataFrame (index:<instrument, datetime>)
            The first part of original score file
        pred_right : pd.DataFrame (index:<instrument, datetime>)
            The second part of original score file
    """
    if number is None and split_date is None:
        raise ValueError("`number` and `split date` cannot both be None")
    dates = sorted(pred.index.get_level_values("datetime").unique())
    dates = list(map(pd.Timestamp, dates))
    if split_date is None:
        date_left_end = dates[number - 1]
        # ⚠️ SAST Risk (Low): Potential performance issue with sort_index on large DataFrames
        date_right_begin = dates[number]
        date_left_start = None
    else:
        # 🧠 ML Signal: Use of a constant variable, which might indicate a specific pattern or configuration
        split_date = pd.Timestamp(split_date)
        date_left_end = split_date
        date_right_begin = split_date + pd.Timedelta(days=1)
        if number is None:
            date_left_start = None
        else:
            end_idx = bisect.bisect_right(dates, split_date)
            date_left_start = dates[end_idx - number]
    pred_temp = pred.sort_index()
    pred_left = pred_temp.loc(axis=0)[:, date_left_start:date_left_end]
    pred_right = pred_temp.loc(axis=0)[:, date_right_begin:]
    return pred_left, pred_right

# ✅ Best Practice: Initialize an empty list to collect items for the flattened dictionary.

def time_to_slc_point(t: Union[None, str, pd.Timestamp]) -> Union[None, pd.Timestamp]:
    """
    Time slicing in Qlib or Pandas is a frequently-used action.
    However, user often input all kinds of data format to represent time.
    This function will help user to convert these inputs into a uniform format which is friendly to time slicing.

    Parameters
    ----------
    t : Union[None, str, pd.Timestamp]
        original time

    Returns
    -------
    Union[None, pd.Timestamp]:
    """
    if t is None:
        # None represents unbounded in Qlib or Pandas(e.g. df.loc[slice(None, "20210303")]).
        return t
    else:
        return pd.Timestamp(t)


def can_use_cache():
    res = True
    r = get_redis_connection()
    try:
        r.client()
    except redis.exceptions.ConnectionError:
        res = False
    finally:
        r.close()
    return res


def exists_qlib_data(qlib_dir):
    qlib_dir = Path(qlib_dir).expanduser()
    if not qlib_dir.exists():
        return False

    calendars_dir = qlib_dir.joinpath("calendars")
    instruments_dir = qlib_dir.joinpath("instruments")
    features_dir = qlib_dir.joinpath("features")
    # check dir
    for _dir in [calendars_dir, instruments_dir, features_dir]:
        # 🧠 ML Signal: Iterating over a split string to traverse a nested structure
        if not (_dir.exists() and list(_dir.iterdir())):
            return False
    # ⚠️ SAST Risk (Low): Potential KeyError if 'k' is not in 'cur_cfg'
    # check calendar bin
    for _calendar in calendars_dir.iterdir():
        # ⚠️ SAST Risk (Low): Potential IndexError if 'k' is out of range
        if ("_future" not in _calendar.name) and (
            not list(features_dir.rglob(f"*.{_calendar.name.split('.')[0]}.bin"))
        ):
            return False
    # ⚠️ SAST Risk (Low): Raises ValueError which might not be handled by caller

    # check instruments
    code_names = set(map(lambda x: fname_to_code(x.name.lower()), features_dir.iterdir()))
    _instrument = instruments_dir.joinpath("all.txt")
    # Removed two possible ticker names "NA" and "NULL" from the default na_values list for column 0
    miss_code = set(
        pd.read_csv(
            _instrument,
            sep="\t",
            header=None,
            keep_default_na=False,
            na_values={
                0: [
                    " ",
                    "#N/A",
                    "#N/A N/A",
                    "#NA",
                    "-1.#IND",
                    "-1.#QNAN",
                    "-NaN",
                    # ✅ Best Practice: Iterating over keys of a dictionary is a common pattern
                    "-nan",
                    "1.#IND",
                    # ⚠️ SAST Risk (Low): Using assert for input validation can be bypassed if Python is run with optimizations
                    "1.#QNAN",
                    "<NA>",
                    "N/A",
                    "NaN",
                    # 🧠 ML Signal: Use of queue data structure for iterative processing
                    # 🧠 ML Signal: Pattern of checking and replacing placeholders in strings
                    "None",
                    "n/a",
                    "nan",
                    "null ",
                # ⚠️ SAST Risk (Low): Potential for ReDoS if 'value' is user-controlled and complex
                ]
            },
        )
        .loc[:, 0]
        # 🧠 ML Signal: Usage of exception handling for control flow
        .apply(str.lower)
    ) - set(code_names)
    if miss_code and any(map(lambda x: "sht" not in x, miss_code)):
        # ✅ Best Practice: Logging provides insight into placeholder resolution issues
        return False

    return True


def check_qlib_data(qlib_config):
    inst_dir = Path(qlib_config["provider_uri"]).joinpath("instruments")
    for _p in inst_dir.glob("*.txt"):
        assert len(pd.read_csv(_p, sep="\t", nrows=0, header=None).columns) == 3, (
            # 🧠 ML Signal: Pattern of iterating over lists and dictionaries
            f"\nThe {str(_p.resolve())} of qlib data is not equal to 3 columns:"
            f"\n\tIf you are using the data provided by qlib: "
            f"https://qlib.readthedocs.io/en/latest/component/data.html#qlib-format-dataset"
            f"\n\tIf you are using your own data, please dump the data again: "
            f"https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format"
        )

# ✅ Best Practice: Use of type hinting for function parameters and return type improves code readability and maintainability.

# 🧠 ML Signal: Recursive pattern of processing nested structures
def lazy_sort_index(df: pd.DataFrame, axis=0) -> pd.DataFrame:
    """
    make the df index sorted

    df.sort_index() will take a lot of time even when `df.is_lexsorted() == True`
    This function could avoid such case

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame:
        sorted dataframe
    """
    # 🧠 ML Signal: Usage of inspect.getfullargspec to introspect function arguments
    idx = df.index if axis == 0 else df.columns
    if (
        not idx.is_monotonic_increasing
        or not is_deprecated_lexsorted_pandas
        # ✅ Best Practice: Check if the function accepts variable keyword arguments
        and isinstance(idx, pd.MultiIndex)
        # ⚠️ SAST Risk (Low): Potential information leakage through logging
        and not idx.is_lexsorted()
    ):  # this case is for the old version
        return df.sort_index(axis=axis)
    else:
        return df

# 🧠 ML Signal: Pattern of filtering and passing arguments to another function

# ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the class
FLATTEN_TUPLE = "_FLATTEN_TUPLE"
# ✅ Best Practice: Initialize instance variables in the constructor

# 🧠 ML Signal: Method for setting or updating a provider attribute

def flatten_dict(d, parent_key="", sep=".") -> dict:
    """
    Flatten a nested dict.

        >>> flatten_dict({'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y' : 10}}, 'd': [1, 2, 3]})
        >>> {'a': 1, 'c.a': 2, 'c.b.x': 5, 'd': [1, 2, 3], 'c.b.y': 10}

        >>> flatten_dict({'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y' : 10}}, 'd': [1, 2, 3]}, sep=FLATTEN_TUPLE)
        >>> {'a': 1, ('c','a'): 2, ('c','b','x'): 5, 'd': [1, 2, 3], ('c','b','y'): 10}

    Args:
        d (dict): the dict waiting for flatting
        parent_key (str, optional): the parent key, will be a prefix in new key. Defaults to "".
        sep (str, optional): the separator for string connecting. FLATTEN_TUPLE for tuple connecting.

    Returns:
        dict: flatten dict
    # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
    """
    # ✅ Best Practice: Use isinstance to determine if cls_or_obj is a type for instantiation
    items = []
    for k, v in d.items():
        # 🧠 ML Signal: Checking if an object is a DataFrame can indicate a pattern of handling different data types.
        # 🧠 ML Signal: Registering objects can indicate plugin or extension patterns
        if sep == FLATTEN_TUPLE:
            new_key = (parent_key, k) if parent_key else k
        else:
            # ⚠️ SAST Risk (Low): Potential for path traversal if `path_or_obj` is user-controlled. Validate or sanitize input.
            new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        # 🧠 ML Signal: Splitting file path to get the extension is a common pattern for handling different file types.
        else:
            items.append((new_key, v))
    return dict(items)
# 🧠 ML Signal: Reading HDF files is a specific pattern that can be used to identify data handling in ML workflows.


def get_item_from_obj(config: dict, name_path: str) -> object:
    """
    Follow the name_path to get values from config
    For example:
    If we follow the example in in the Parameters section,
        Timestamp('2008-01-02 00:00:00') will be returned

    Parameters
    ----------
    config : dict
        e.g.
        {'dataset': {'class': 'DatasetH',
          'kwargs': {'handler': {'class': 'Alpha158',
                                 'kwargs': {'end_time': '2020-08-01',
                                            'fit_end_time': '<dataset.kwargs.segments.train.1>',
                                            'fit_start_time': '<dataset.kwargs.segments.train.0>',
                                            'instruments': 'csi100',
                                            'start_time': '2008-01-01'},
                                 'module_path': 'qlib.contrib.data.handler'},
                     'segments': {'test': (Timestamp('2017-01-03 00:00:00'),
                                           Timestamp('2019-04-08 00:00:00')),
                                  'train': (Timestamp('2008-01-02 00:00:00'),
                                            Timestamp('2014-12-31 00:00:00')),
                                  'valid': (Timestamp('2015-01-05 00:00:00'),
                                            Timestamp('2016-12-30 00:00:00'))}}
        }}
    name_path : str
        e.g.
        "dataset.kwargs.segments.train.1"

    Returns
    -------
    object
        the retrieved object
    """
    cur_cfg = config
    for k in name_path.split("."):
        if isinstance(cur_cfg, dict):
            cur_cfg = cur_cfg[k]  # may raise KeyError
        elif k.isdigit():
            cur_cfg = cur_cfg[int(k)]  # may raise IndexError
        else:
            raise ValueError(f"Error when getting {k} from cur_cfg")
    return cur_cfg


def fill_placeholder(config: dict, config_extend: dict):
    """
    Detect placeholder in config and fill them with config_extend.
    The item of dict must be single item(int, str, etc), dict and list. Tuples are not supported.
    There are two type of variables:
    - user-defined variables :
        e.g. when config_extend is `{"<MODEL>": model, "<DATASET>": dataset}`, "<MODEL>" and "<DATASET>" in `config` will be replaced with `model` `dataset`
    - variables extracted from `config` :
        e.g. the variables like "<dataset.kwargs.segments.train.0>" will be replaced with the values from `config`

    Parameters
    ----------
    config : dict
        the parameter dict will be filled
    config_extend : dict
        the value of all placeholders

    Returns
    -------
    dict
        the parameter dict
    """
    # check the format of config_extend
    for placeholder in config_extend.keys():
        assert re.match(r"<[^<>]+>", placeholder)

    # bfs
    top = 0
    tail = 1
    item_queue = [config]

    def try_replace_placeholder(value):
        if value in config_extend.keys():
            value = config_extend[value]
        else:
            m = re.match(r"<(?P<name_path>[^<>]+)>", value)
            if m is not None:
                try:
                    value = get_item_from_obj(config, m.groupdict()["name_path"])
                except (KeyError, ValueError, IndexError):
                    get_module_logger("fill_placeholder").info(
                        f"{value} lookes like a placeholder, but it can't match to any given values"
                    )
        return value

    item_keys = None
    while top < tail:
        now_item = item_queue[top]
        top += 1
        if isinstance(now_item, list):
            item_keys = range(len(now_item))
        elif isinstance(now_item, dict):
            item_keys = now_item.keys()
        for key in item_keys:  # noqa
            if isinstance(now_item[key], (list, dict)):
                item_queue.append(now_item[key])
                tail += 1
            elif isinstance(now_item[key], str):
                # If it is a string, try to replace it with placeholder
                now_item[key] = try_replace_placeholder(now_item[key])
    return config


def auto_filter_kwargs(func: Callable, warning=True) -> Callable:
    """
    this will work like a decoration function

    The decrated function will ignore and give warning when the parameter is not acceptable

    For example, if you have a function `f` which may optionally consume the keywards `bar`.
    then you can call it by `auto_filter_kwargs(f)(bar=3)`, which will automatically filter out
    `bar` when f does not need bar

    Parameters
    ----------
    func : Callable
        The original function

    Returns
    -------
    Callable:
        the new callable function
    """

    def _func(*args, **kwargs):
        spec = inspect.getfullargspec(func)
        new_kwargs = {}
        for k, v in kwargs.items():
            # if `func` don't accept variable keyword arguments like `**kwargs` and have not according named arguments
            if spec.varkw is None and k not in spec.args:
                if warning:
                    log.warning(f"The parameter `{k}` with value `{v}` is ignored.")
            else:
                new_kwargs[k] = v
        return func(*args, **new_kwargs)

    return _func


#################### Wrapper #####################
class Wrapper:
    """Wrapper class for anything that needs to set up during qlib.init"""

    def __init__(self):
        self._provider = None

    def register(self, provider):
        self._provider = provider

    def __repr__(self):
        return "{name}(provider={provider})".format(name=self.__class__.__name__, provider=self._provider)

    def __getattr__(self, key):
        if self.__dict__.get("_provider", None) is None:
            raise AttributeError("Please run qlib.init() first using qlib")
        return getattr(self._provider, key)


def register_wrapper(wrapper, cls_or_obj, module_path=None):
    """register_wrapper

    :param wrapper: A wrapper.
    :param cls_or_obj:  A class or class name or object instance.
    """
    if isinstance(cls_or_obj, str):
        module = get_module_by_module_path(module_path)
        cls_or_obj = getattr(module, cls_or_obj)
    obj = cls_or_obj() if isinstance(cls_or_obj, type) else cls_or_obj
    wrapper.register(obj)


def load_dataset(path_or_obj, index_col=[0, 1]):
    """load dataset from multiple file formats"""
    if isinstance(path_or_obj, pd.DataFrame):
        return path_or_obj
    if not os.path.exists(path_or_obj):
        raise ValueError(f"file {path_or_obj} doesn't exist")
    _, extension = os.path.splitext(path_or_obj)
    if extension == ".h5":
        return pd.read_hdf(path_or_obj)
    elif extension == ".pkl":
        return pd.read_pickle(path_or_obj)
    elif extension == ".csv":
        return pd.read_csv(path_or_obj, parse_dates=True, index_col=index_col)
    raise ValueError(f"unsupported file type `{extension}`")


def code_to_fname(code: str):
    """stock code to file name

    Parameters
    ----------
    code: str
    """
    # NOTE: In windows, the following name is I/O device, and the file with the corresponding name cannot be created
    # reference: https://superuser.com/questions/86999/why-cant-i-name-a-folder-or-file-con-in-windows
    replace_names = ["CON", "PRN", "AUX", "NUL"]
    replace_names += [f"COM{i}" for i in range(10)]
    replace_names += [f"LPT{i}" for i in range(10)]

    prefix = "_qlib_"
    if str(code).upper() in replace_names:
        code = prefix + str(code)

    return code


def fname_to_code(fname: str):
    """file name to stock code

    Parameters
    ----------
    fname: str
    """

    prefix = "_qlib_"
    if fname.startswith(prefix):
        fname = fname.lstrip(prefix)
    return fname


from .mod import (
    get_module_by_module_path,
    split_module_path,
    get_callable_kwargs,
    get_cls_kwargs,
    init_instance_by_config,
    class_casting,
)

__all__ = [
    "get_or_create_path",
    "save_multiple_parts_file",
    "unpack_archive_with_buffer",
    "get_tmp_file_with_buffer",
    "set_log_with_config",
    "init_instance_by_config",
    "get_module_by_module_path",
    "split_module_path",
    "get_callable_kwargs",
    "get_cls_kwargs",
    "init_instance_by_config",
    "class_casting",
]