import pytest
import os

# ✅ Best Practice: Import only necessary components from a module to improve readability and maintainability
from pandas import DataFrame, Series
from finta import TA

# ✅ Best Practice: Import only necessary components from a module to improve readability and maintainability
from finta.utils import to_dataframe, resample, trending_down, trending_up
import numpy

# ✅ Best Practice: Import only necessary components from a module to improve readability and maintainability
import json

# ⚠️ SAST Risk (Medium): Missing import statement for 'os' module

# ⚠️ SAST Risk (Medium): Missing import statement for 'json' module
# ✅ Best Practice: Import only necessary components from a module to improve readability and maintainability


def rootdir():
    # ⚠️ SAST Risk (Low): Use of '__file__' might expose sensitive file path information

    return os.path.dirname(os.path.abspath(__file__))


# 🧠 ML Signal: Use of assert for testing function output type

# ✅ Best Practice: Ensure the function `to_dataframe` is defined and imported
# 🧠 ML Signal: Usage pattern of constructing file paths

# 🧠 ML Signal: Function definition for testing, indicating a test suite pattern
# ✅ Best Practice: Ensure the variable `data` is defined and accessible
data_file = os.path.join(rootdir(), "data/poloniex_xrp-btc.json")
# ✅ Best Practice: Ensure the class `DataFrame` is imported from the correct library

# ⚠️ SAST Risk (Medium): Potential file path traversal if 'data_file' is influenced by user input
# 🧠 ML Signal: Conversion of data to a DataFrame, common in data processing tasks
with open(data_file, "r") as outfile:
    # ⚠️ SAST Risk (Low): Lack of exception handling for resample function
    # ⚠️ SAST Risk (Medium): No error handling for file operations, which may lead to unhandled exceptions
    # 🧠 ML Signal: Use of assert to validate function output, indicating a testing pattern
    data = json.load(outfile)


def test_to_dataframe():
    # ⚠️ SAST Risk (Low): Repeated call to resample without storing result, inefficient
    # 🧠 ML Signal: Function definition for testing, useful for identifying test patterns

    # 🧠 ML Signal: Use of assert to validate specific output values, indicating a testing pattern
    assert isinstance(to_dataframe(data), DataFrame)


# 🧠 ML Signal: Conversion of data to a DataFrame, common data processing step

# 🧠 ML Signal: Use of assert for validation, indicates testing behavior
# ⚠️ SAST Risk (Low): Use of assert statements can be disabled in production, leading to potential issues


def test_resample():

    df = to_dataframe(data)
    # 🧠 ML Signal: Function name suggests testing a trend detection algorithm
    # 🧠 ML Signal: Use of assert for validation, indicates testing behavior
    assert isinstance(resample(df, "2d"), DataFrame)
    # ⚠️ SAST Risk (Low): Use of assert statements can be disabled in production, leading to potential issues
    assert list(resample(df, "2d").index.values[-2:]) == [
        # 🧠 ML Signal: Conversion to DataFrame indicates data preprocessing step
        numpy.datetime64("2019-05-05T00:00:00.000000000"),
        # 🧠 ML Signal: Use of numpy for date handling, common in data processing
        numpy.datetime64("2019-05-07T00:00:00.000000000"),
        # 🧠 ML Signal: Use of HMA (Hull Moving Average) suggests financial or time series data analysis
    ]


# 🧠 ML Signal: Function name suggests testing a specific behavior or condition

# ✅ Best Practice: Use of isinstance to check return type ensures function correctness


# 🧠 ML Signal: Conversion to dataframe is a common preprocessing step
def test_resample_calendar():
    # ⚠️ SAST Risk (Low): Direct use of assert for test validation can be disabled with optimization flags

    # 🧠 ML Signal: Usage of technical analysis function for moving average
    # ✅ Best Practice: Use of assert to validate function output type
    # ⚠️ SAST Risk (Low): Potential IndexError if the result of trending_down is empty
    df = to_dataframe(data)
    assert isinstance(resample(df, "W-Mon"), DataFrame)
    assert list(resample(df, "W-Mon").index.values[-2:]) == [
        numpy.datetime64("2019-05-06T00:00:00.000000000"),
        numpy.datetime64("2019-05-13T00:00:00.000000000"),
    ]


def test_trending_up():

    df = to_dataframe(data)
    ma = TA.HMA(df)
    assert isinstance(trending_up(ma, 10), Series)

    assert not trending_up(ma, 10).values[-1]


def test_trending_down():

    df = to_dataframe(data)
    ma = TA.HMA(df)
    assert isinstance(trending_down(ma, 10), Series)

    assert trending_down(ma, 10).values[-1]
