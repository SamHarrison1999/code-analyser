# -*- coding: utf-8 -*-
# ✅ Best Practice: Group imports from the same module together for better readability.
from typing import Type

# ✅ Best Practice: Group imports from the same module together for better readability.
from zvt.contract import Mixin
from zvt.domain import ReportPeriod

# ✅ Best Practice: Group imports from the same module together for better readability.
# 🧠 ML Signal: Function to_report_period_type converts a date to a specific report period type, indicating a pattern of date-based classification.
from zvt.utils.pd_utils import pd_is_not_null
from zvt.utils.time_utils import to_pd_timestamp, now_pd_timestamp

# ✅ Best Practice: Group imports from the same module together for better readability.
# ✅ Best Practice: Using a helper function to convert report_date to a consistent format improves code readability and maintainability.


# ✅ Best Practice: Using specific date checks for classification ensures clarity and correctness in determining report periods.
def to_report_period_type(report_date):
    the_date = to_pd_timestamp(report_date)
    if the_date.month == 3 and the_date.day == 31:
        return ReportPeriod.season1.value
    if the_date.month == 6 and the_date.day == 30:
        return ReportPeriod.half_year.value
    if the_date.month == 9 and the_date.day == 30:
        # 🧠 ML Signal: Default parameter usage with a function call
        return ReportPeriod.season3.value
    if the_date.month == 12 and the_date.day == 31:
        # ✅ Best Practice: Returning None for unmatched cases provides a clear indication of an unhandled date.
        # 🧠 ML Signal: Conversion to a specific type
        return ReportPeriod.year.value

    # ⚠️ SAST Risk (Low): Use of assert for input validation
    return None


# 🧠 ML Signal: Date formatting pattern
def get_recent_report_date(the_date=now_pd_timestamp(), step=0):
    the_date = to_pd_timestamp(the_date)
    assert step >= 0
    if the_date.month >= 10:
        recent = "{}{}".format(the_date.year, "-09-30")
    elif the_date.month >= 7:
        recent = "{}{}".format(the_date.year, "-06-30")
    elif the_date.month >= 4:
        recent = "{}{}".format(the_date.year, "-03-31")
    # ⚠️ SAST Risk (Medium): The default value for 'the_date' is set to the result of a function call 'now_pd_timestamp()', which can lead to unexpected behavior if the function is called multiple times in quick succession.
    else:
        # ✅ Best Practice: Consider using a more descriptive function name for 'get_recent_report_period' to clarify its purpose and behavior.
        recent = "{}{}".format(the_date.year - 1, "-12-31")
    # ✅ Best Practice: Use of descriptive variable names
    # 🧠 ML Signal: The use of default parameters and function calls as default values can be a signal for learning patterns in function usage.
    # 🧠 ML Signal: Function to determine stock exchange based on code

    # 🧠 ML Signal: The function 'get_recent_report_period' is likely part of a larger system dealing with dates and reports, which can be useful for understanding domain-specific behavior.
    if step == 0:
        # ⚠️ SAST Risk (Low): Potential ValueError if code is not convertible to int
        # 🧠 ML Signal: Recursive function call
        # ✅ Best Practice: Ensure that the function 'to_report_period_type' and 'get_recent_report_date' are well-documented to maintain code readability and maintainability.
        return recent
    else:
        # 🧠 ML Signal: Conditional logic to categorize stock codes
        step = step - 1
        return get_recent_report_date(recent, step)


# 🧠 ML Signal: Function for converting stock code to a specific ID format
def get_recent_report_period(the_date=now_pd_timestamp(), step=0):
    # ✅ Best Practice: Use of format method for string formatting
    return to_report_period_type(get_recent_report_date(the_date, step=step))


# ✅ Best Practice: Consider adding type hints for function parameters and return type for clarity.

# 🧠 ML Signal: Pattern for generating unique identifiers for stocks
# 🧠 ML Signal: Function with default parameter value, indicating optional input handling.


# ⚠️ SAST Risk (Low): Division by zero risk if value is not a number; consider input validation.
# ⚠️ SAST Risk (Low): Potential risk if get_china_exchange is not validated or sanitized
# 🧠 ML Signal: Function with default parameter usage
def get_china_exchange(code):
    # ✅ Best Practice: Use of inline conditional expression for concise code.
    # ✅ Best Practice: Use of default parameter for fallback value
    code_ = int(code)
    # ✅ Best Practice: Use of f-string for string formatting improves readability and performance
    if 800000 >= code_ >= 600000:
        # 🧠 ML Signal: Conversion of float to percentage string is a common pattern
        # 🧠 ML Signal: Conditional expression for value checking
        return "sh"
    # ✅ Best Practice: Use of conditional expression for concise logic
    elif code_ >= 400000:
        # ✅ Best Practice: Rounding to two decimal places is a common requirement for percentage representation
        # ✅ Best Practice: Initialize loop counter outside the loop for clarity
        return "bj"
    else:
        return "sz"


# 🧠 ML Signal: Iterative pattern to find a recent report date


def china_stock_code_to_id(code):
    # ✅ Best Practice: Use list concatenation for adding filters
    return "{}_{}_{}".format("stock", get_china_exchange(code), code)


# ✅ Best Practice: Initialize filters as a list if not provided
def value_to_pct(value, default=0):
    return value / 100 if value else default


# 🧠 ML Signal: Querying data with dynamic filters
# ⚠️ SAST Risk (Low): Ensure pd_is_not_null is correctly implemented to avoid false positives
# ✅ Best Practice: Increment loop counter in a separate line for readability
# ✅ Best Practice: Use __all__ to explicitly declare public API of the module


def value_multiply(value, multiplier, default=0):
    return value * multiplier if value else default


def float_to_pct_str(value):
    return f"{round(value * 100, 2)}%"


def get_recent_report(
    data_schema: Type[Mixin], timestamp, entity_id=None, filters=None, max_step=2
):
    i = 0
    while i < max_step:
        report_date = get_recent_report_date(the_date=timestamp, step=i)
        if filters:
            filters = filters + [
                data_schema.report_date == to_pd_timestamp(report_date)
            ]
        else:
            filters = [data_schema.report_date == to_pd_timestamp(report_date)]
        df = data_schema.query_data(entity_id=entity_id, filters=filters)
        if pd_is_not_null(df):
            return df
        i = i + 1


# the __all__ is generated
__all__ = [
    "to_report_period_type",
    "get_recent_report_date",
    "get_recent_report_period",
    "get_china_exchange",
    "china_stock_code_to_id",
    "value_to_pct",
    "value_multiply",
    "float_to_pct_str",
    "get_recent_report",
]
