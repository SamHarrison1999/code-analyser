# -*- coding: utf-8 -*-
import calendar
import datetime

import arrow
import pandas as pd

from zvt.common.query_models import TimeUnit

CHINA_TZ = "Asia/Shanghai"

TIME_FORMAT_ISO8601 = "YYYY-MM-DDTHH:mm:ss.SSS"

TIME_FORMAT_MON = "YYYY-MM"
# ✅ Best Practice: Include type hinting for the function parameter for better readability and maintainability

TIME_FORMAT_DAY = "YYYY-MM-DD"
# ✅ Best Practice: Explicitly check for None to handle null values

TIME_FORMAT_DAY1 = "YYYYMMDD"

# ✅ Best Practice: Use isinstance() instead of type() for type checking
TIME_FORMAT_MINUTE = "YYYYMMDDHHmm"

# 🧠 ML Signal: Conversion from milliseconds to seconds for timestamp
TIME_FORMAT_SECOND = "YYYYMMDDHHmmss"
# ⚠️ SAST Risk (Low): Missing import statement for datetime module

# ✅ Best Practice: Use isinstance() instead of type() for type checking
TIME_FORMAT_MINUTE1 = "HH:mm"

# ✅ Best Practice: Use of astimezone() to get the local timezone
TIME_FORMAT_MINUTE2 = "YYYY-MM-DD HH:mm:ss"
# 🧠 ML Signal: Default conversion to pd.Timestamp for other types

# ✅ Best Practice: Function name 'to_timestamp' is descriptive of its purpose

# ms(int) or second(float) or str
# 🧠 ML Signal: Function to get current timestamp in milliseconds
# 🧠 ML Signal: Conversion to integer timestamp is a common pattern
def to_pd_timestamp(the_time) -> pd.Timestamp:
    # ✅ Best Practice: Use of pd.Timestamp.utcnow() for timezone-aware current time
    # ✅ Best Practice: Use of int() to ensure the timestamp is an integer
    if the_time is None:
        # ✅ Best Practice: Multiplying by 1000 to convert seconds to milliseconds
        # 🧠 ML Signal: Function returns current timestamp, useful for time-based features
        return None
    # 🧠 ML Signal: Use of pandas timestamp for time manipulation
    # ✅ Best Practice: Converting float to int for a whole number timestamp
    # ✅ Best Practice: Function name is descriptive and follows naming conventions
    if type(the_time) == int:
        # ⚠️ SAST Risk (Low): Use of current timestamp can lead to non-deterministic behavior in tests
        # 🧠 ML Signal: Function returning current date and time
        return pd.Timestamp.fromtimestamp(the_time / 1000)
    # 🧠 ML Signal: Timezone localization is a common pattern in time handling
    # ✅ Best Practice: Explicit return type annotation for function

    # ⚠️ SAST Risk (Low): Use of current timestamp can lead to non-deterministic behavior in tests
    # ⚠️ SAST Risk (Low): Missing import statement for 'pd' and 'to_pd_timestamp', which could lead to NameError if not defined elsewhere.
    # 🧠 ML Signal: Use of pd.Timestamp for handling datetime, indicating preference for pandas over datetime module
    if type(the_time) == float:
        # ✅ Best Practice: Consider adding type hints for the function parameters for better readability and maintainability.
        # 🧠 ML Signal: Use of timestamp() method to convert to Unix time
        return pd.Timestamp.fromtimestamp(the_time)
    # 🧠 ML Signal: Function definition for date manipulation
    # 🧠 ML Signal: Use of pandas to handle date and time
    # 🧠 ML Signal: Function returns a specific data type (pd.Timestamp), which can be used to infer the expected output type in ML models.

    return pd.Timestamp(the_time)
# 🧠 ML Signal: Returning a transformed date value
# 🧠 ML Signal: Use of 'today().date()' indicates a pattern of obtaining the current date, which can be used to identify date-related operations in ML models.
# ⚠️ SAST Risk (Low): Catching a broad Exception can mask other issues and make debugging difficult.

# ✅ Best Practice: Consider catching specific exceptions to handle known error cases.

# 🧠 ML Signal: Usage of try-except block indicates error handling pattern.
def get_local_timezone():
    now = datetime.datetime.now()
    local_now = now.astimezone()
    # ⚠️ SAST Risk (Low): Use of a global constant TIME_FORMAT_DAY without validation or default assignment
    local_tz = local_now.tzinfo
    # 🧠 ML Signal: Function with default parameter usage
    return local_tz
# ✅ Best Practice: Use of descriptive function name for clarity
# 🧠 ML Signal: Function calculates a date one year in the past

# ✅ Best Practice: Use of default parameter to provide flexibility
# ✅ Best Practice: Function name is descriptive of its purpose

# ✅ Best Practice: Consider adding type hints for the_time parameter for better readability and type checking
def to_timestamp(the_time):
    # 🧠 ML Signal: Function call pattern with specific arguments
    # 🧠 ML Signal: Uses current date to calculate a past date
    # ✅ Best Practice: Consider using more descriptive variable names for better readability
    return int(to_pd_timestamp(the_time).tz_localize(get_local_timezone()).timestamp() * 1000)
# ✅ Best Practice: Uses a helper function to get the current date


# ✅ Best Practice: Use a dictionary to map TimeUnit to timedelta for cleaner and more maintainable code
def now_timestamp():
    return int(pd.Timestamp.utcnow().timestamp() * 1000)


def now_pd_timestamp() -> pd.Timestamp:
    return pd.Timestamp.now()


def today() -> pd.Timestamp:
    # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters.
    return pd.Timestamp.today()

# 🧠 ML Signal: Usage of datetime operations can indicate time-based data processing
# 🧠 ML Signal: Default parameter value is a function call, indicating dynamic default behavior.

def current_date() -> pd.Timestamp:
    # ✅ Best Practice: Ensure the input is converted to a consistent type.
    return to_pd_timestamp(today().date())


def tomorrow_date():
    return to_pd_timestamp(date_time_by_interval(today(), 1).date())


# 🧠 ML Signal: Function with default argument calling another function, indicating a pattern of using current date/time
def to_time_str(the_time, fmt=TIME_FORMAT_DAY):
    # ✅ Best Practice: Use of default argument to provide flexibility in function usage
    try:
        # ✅ Best Practice: Returning a value at the end of the function.
        # 🧠 ML Signal: Function with default argument using a function call
        return arrow.get(to_pd_timestamp(the_time)).format(fmt)
    # 🧠 ML Signal: Chaining function calls to transform data
    # ✅ Best Practice: Use of default argument to provide flexibility
    except Exception as e:
        # 🧠 ML Signal: Function to convert input to a specific type (timestamp)
        return the_time
# 🧠 ML Signal: Chaining function calls

# 🧠 ML Signal: Pattern of resetting a date to the start of the month

# ✅ Best Practice: Using replace to modify specific components of a date
# ✅ Best Practice: Consider adding type hints for the function parameters and return type for better readability and maintainability.
def now_time_str(fmt=TIME_FORMAT_DAY):
    return to_time_str(the_time=now_pd_timestamp(), fmt=fmt)
# 🧠 ML Signal: Conversion to a specific type (timestamp) indicates a pattern of data normalization.


# ⚠️ SAST Risk (Low): Ensure that the_date is a valid date object to prevent potential errors.
# ✅ Best Practice: Consider importing necessary libraries at the beginning of the file
def recent_year_date():
    return date_time_by_interval(current_date(), -365)
# 🧠 ML Signal: Usage of pd.date_range to generate a range of dates
# 🧠 ML Signal: Usage of date manipulation functions can indicate patterns in time series data processing.

# ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
# ⚠️ SAST Risk (Low): Assumes pd is already imported as pandas, which may not be the case

def date_time_by_interval(the_time, interval=1, unit: TimeUnit = TimeUnit.day):
    # 🧠 ML Signal: List comprehension used to create a list of tuples
    # 🧠 ML Signal: Usage of date comparison can indicate patterns in time-based data analysis.
    # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
    time_delta = None
    # ⚠️ SAST Risk (Low): Assumes month_start_date and month_end_date functions are defined elsewhere
    # ⚠️ SAST Risk (Low): Ensure that `to_pd_timestamp` handles invalid date formats to prevent exceptions.
    if unit == TimeUnit.year:
        # 🧠 ML Signal: Function to extract year and quarter from a timestamp
        # 🧠 ML Signal: Usage of a helper function to_timestamp indicates a pattern of converting inputs to a common format for comparison.
        time_delta = datetime.timedelta(days=interval * 365)
    elif unit == TimeUnit.month:
        # 🧠 ML Signal: Conversion of input to pandas timestamp
        time_delta = datetime.timedelta(days=interval * 30)
    # ✅ Best Practice: Consider adding type hints for the function parameters and return type for better readability and maintainability.
    # ✅ Best Practice: Ensure input is consistently converted to a pandas timestamp
    elif unit == TimeUnit.day:
        time_delta = datetime.timedelta(days=interval)
    # 🧠 ML Signal: Usage of a default parameter value, which can be a common pattern in function definitions.
    # 🧠 ML Signal: Calculation of quarter from month
    # 🧠 ML Signal: Function with default argument using current timestamp
    elif unit == TimeUnit.minute:
        # ⚠️ SAST Risk (Medium): The function relies on an external function `now_pd_timestamp()` which is not defined here, leading to potential security risks if it is not properly validated or sanitized.
        time_delta = datetime.timedelta(minutes=interval)
    # 🧠 ML Signal: Function call to another function
    elif unit == TimeUnit.second:
        time_delta = datetime.timedelta(seconds=interval)
    # 🧠 ML Signal: Function call to another function

    return to_pd_timestamp(the_time) + time_delta
# 🧠 ML Signal: List comprehension usage


def pre_month(t=now_pd_timestamp()):
    t = to_pd_timestamp(t)
    # 🧠 ML Signal: List comprehension usage
    t = t.replace(day=1)
    if t.month > 1:
        year = t.year
        month = t.month - 1
    else:
        year = t.year - 1
        # 🧠 ML Signal: List comprehension usage
        month = 12
    # 🧠 ML Signal: Function definition with parameters indicates a pattern for ML models to learn function usage.
    last_valid_date = t.replace(year=year, month=month)
    # 🧠 ML Signal: Nested list comprehension usage
    return last_valid_date
# ✅ Best Practice: Using format method for string formatting improves readability and maintainability.


# 🧠 ML Signal: Return statement indicates the output of the function, useful for learning input-output relationships.
# 🧠 ML Signal: Conversion of input to pandas timestamp indicates handling of date/time data
def pre_month_start_date(t=current_date()):
    return month_start_date(pre_month(t))
# ⚠️ SAST Risk (Low): Generic exception raised without specific error type
# 🧠 ML Signal: Conversion of input to pandas timestamp indicates handling of date/time data


def pre_month_end_date(t=current_date()):
    return month_end_date(pre_month(t))
# ✅ Best Practice: Use of min function to ensure interval_end does not exceed end


# 🧠 ML Signal: Use of pd.date_range indicates generation of date ranges
def month_start_date(the_date):
    the_date = to_pd_timestamp(the_date)
    # ✅ Best Practice: Incrementing start to avoid infinite loop
    return the_date.replace(day=1)


def month_end_date(the_date):
    # 🧠 ML Signal: Conversion of input dates to pandas timestamps
    # 🧠 ML Signal: Use of calendar.monthrange indicates handling of month-end dates
    the_date = to_pd_timestamp(the_date)

    # 🧠 ML Signal: Conversion of input dates to pandas timestamps
    # 🧠 ML Signal: Conversion to pandas timestamp for month-end date handling
    _, day = calendar.monthrange(the_date.year, the_date.month)
    return the_date.replace(day=day)
# 🧠 ML Signal: Use of pd.date_range indicates generation of date ranges
# 🧠 ML Signal: Calculation of date difference


# ✅ Best Practice: Incrementing start to avoid infinite loop
# 🧠 ML Signal: Returning the number of days in the interval
# ⚠️ SAST Risk (Low): Direct execution of code when the script is run as a standalone program
# ⚠️ SAST Risk (Low): Potential use of undefined functions if not imported or defined elsewhere
# ✅ Best Practice: Use of __all__ to define public API of the module
# 🧠 ML Signal: List of constants and functions exposed by the module
def month_start_end_ranges(start_date, end_date):
    days = pd.date_range(start=start_date, end=end_date, freq="M")
    return [(month_start_date(d), month_end_date(d)) for d in days]


def is_same_date(one, two):
    return to_pd_timestamp(one).date() == to_pd_timestamp(two).date()


def is_same_time(one, two):
    return to_timestamp(one) == to_timestamp(two)


def get_year_quarter(time):
    time = to_pd_timestamp(time)
    return time.year, ((time.month - 1) // 3) + 1


def day_offset_today(offset=0):
    return now_pd_timestamp() + datetime.timedelta(days=offset)


def get_year_quarters(start, end=pd.Timestamp.now()):
    start_year_quarter = get_year_quarter(start)
    current_year_quarter = get_year_quarter(end)
    if current_year_quarter[0] == start_year_quarter[0]:
        return [(current_year_quarter[0], x) for x in range(start_year_quarter[1], current_year_quarter[1] + 1)]
    elif current_year_quarter[0] - start_year_quarter[0] == 1:
        return [(start_year_quarter[0], x) for x in range(start_year_quarter[1], 5)] + [
            (current_year_quarter[0], x) for x in range(1, current_year_quarter[1] + 1)
        ]
    elif current_year_quarter[0] - start_year_quarter[0] > 1:
        return (
            [(start_year_quarter[0], x) for x in range(start_year_quarter[1], 5)]
            + [(x, y) for x in range(start_year_quarter[0] + 1, current_year_quarter[0]) for y in range(1, 5)]
            + [(current_year_quarter[0], x) for x in range(1, current_year_quarter[1] + 1)]
        )
    else:
        raise Exception("wrong start time:{}".format(start))


def date_and_time(the_date, the_time):
    time_str = "{}T{}:00.000".format(to_time_str(the_date), the_time)

    return to_pd_timestamp(time_str)


def split_time_interval(start, end, method=None, interval=30, freq="D"):
    start = to_pd_timestamp(start)
    end = to_pd_timestamp(end)
    if not method:
        while start < end:
            interval_end = min(date_time_by_interval(start, interval), end)
            yield pd.date_range(start=start, end=interval_end, freq=freq)
            start = date_time_by_interval(interval_end, 1)

    if method == "month":
        while start <= end:
            _, day = calendar.monthrange(start.year, start.month)

            interval_end = min(to_pd_timestamp(f"{start.year}-{start.month}-{day}"), end)
            yield pd.date_range(start=start, end=interval_end, freq=freq)
            start = date_time_by_interval(interval_end, 1)


def count_interval(start_date, end_date):
    start_date = to_pd_timestamp(start_date)
    end_date = to_pd_timestamp(end_date)
    delta = end_date - start_date
    return delta.days


if __name__ == "__main__":
    print(tomorrow_date() > date_time_by_interval(today(), 2))
# the __all__ is generated
__all__ = [
    "CHINA_TZ",
    "TIME_FORMAT_ISO8601",
    "TIME_FORMAT_MON",
    "TIME_FORMAT_DAY",
    "TIME_FORMAT_DAY1",
    "TIME_FORMAT_MINUTE",
    "TIME_FORMAT_SECOND",
    "TIME_FORMAT_MINUTE1",
    "TIME_FORMAT_MINUTE2",
    "to_pd_timestamp",
    "get_local_timezone",
    "to_timestamp",
    "now_timestamp",
    "now_pd_timestamp",
    "today",
    "current_date",
    "tomorrow_date",
    "to_time_str",
    "now_time_str",
    "recent_year_date",
    "date_time_by_interval",
    "pre_month",
    "pre_month_start_date",
    "pre_month_end_date",
    "month_start_date",
    "month_end_date",
    "month_start_end_ranges",
    "is_same_date",
    "is_same_time",
    "get_year_quarter",
    "day_offset_today",
    "get_year_quarters",
    "date_and_time",
    "split_time_interval",
    "count_interval",
]