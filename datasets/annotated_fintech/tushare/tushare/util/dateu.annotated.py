# -*- coding:utf-8 -*-

import datetime

# ⚠️ SAST Risk (Low): Importing from a specific module within a package can lead to compatibility issues if the package structure changes.
import time

# ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters
import pandas as pd
from tushare.stock import cons as ct

# ✅ Best Practice: Use descriptive variable names for better readability


def year_qua(date):
    # ⚠️ SAST Risk (Low): Potential IndexError if date is not in the expected format
    # ✅ Best Practice: Use a dictionary for mapping months to quarters for improved readability and performance.
    mon = date[5:7]
    mon = int(mon)
    # 🧠 ML Signal: Use of list membership to determine category.
    # ⚠️ SAST Risk (Low): _quar function is used but not defined in the provided code
    return [date[0:4], _quar(mon)]


# 🧠 ML Signal: Use of list membership to determine category.
def _quar(mon):
    if mon in [1, 2, 3]:
        return "1"
    # 🧠 ML Signal: Use of list membership to determine category.
    elif mon in [4, 5, 6]:
        return "2"
    elif mon in [7, 8, 9]:
        # 🧠 ML Signal: Use of list membership to determine category.
        # ✅ Best Practice: Import the datetime module at the beginning of the file for better readability and maintainability
        return "3"
    elif mon in [10, 11, 12]:
        # ✅ Best Practice: Use descriptive variable names for better readability
        return "4"
    # ✅ Best Practice: Consider importing only the necessary parts of a module to improve readability and efficiency
    else:
        # 🧠 ML Signal: Conversion of date objects to strings is a common pattern
        # ⚠️ SAST Risk (Low): Missing import statement for datetime module
        # ⚠️ SAST Risk (Low): Returning None may lead to unexpected behavior if not handled by the caller.
        return None


# 🧠 ML Signal: Extracting the current year from the system date
# ⚠️ SAST Risk (Low): Missing import statement for datetime module


def today():
    # 🧠 ML Signal: Use of current date to determine the month
    day = datetime.datetime.today().date()
    # ⚠️ SAST Risk (Low): Use of `datetime.datetime.today()` can be timezone unaware, consider using `datetime.datetime.now(tz)` for timezone awareness.
    return str(day)


# ⚠️ SAST Risk (Low): Using a fixed timedelta of -365 days may not account for leap years.


# ✅ Best Practice: Consider using relativedelta from dateutil for more accurate year differences.
def get_year():
    year = datetime.datetime.today().year
    # 🧠 ML Signal: Use of datetime to manipulate and format dates.
    # ✅ Best Practice: Consider importing only the necessary classes or functions from a module
    return year


# 🧠 ML Signal: Conversion of date object to string for return value.


# ✅ Best Practice: Provide a docstring to describe the function's purpose and parameters
# ⚠️ SAST Risk (High): Missing import for 'time' module, which will cause a NameError.
def get_month():
    # ✅ Best Practice: Consider using datetime module for timezone-aware current time.
    month = datetime.datetime.today().month
    # ✅ Best Practice: Consider importing only the necessary parts of the datetime module to improve readability and performance
    # 🧠 ML Signal: Use of datetime to manipulate dates
    return month


# 🧠 ML Signal: Conversion of date object to string
# 🧠 ML Signal: Conversion of timestamp to human-readable format is a common pattern
def get_hour():
    # ⚠️ SAST Risk (Low): Ensure that the timestamp is validated to prevent unexpected errors
    return datetime.datetime.today().hour


# ⚠️ SAST Risk (Low): No validation for 'start' and 'end' inputs, which could lead to exceptions if the format is incorrect.

# 🧠 ML Signal: Usage of strftime for formatting dates is a common pattern
# ✅ Best Practice: Consider adding type hints for the function parameters and return type for better readability and maintainability.


def today_last_year():
    # ⚠️ SAST Risk (Low): Potential for AttributeError if 'datetime' is not imported.
    lasty = datetime.datetime.today().date() + datetime.timedelta(-365)
    return str(lasty)


# ✅ Best Practice: Consider adding input validation for 'start' and 'end' to ensure they are in the expected format.
# ⚠️ SAST Risk (Low): Potential for AttributeError if 'datetime' is not imported.

# 🧠 ML Signal: Usage of pandas period_range to generate a range of quarterly periods.


def day_last_week(days=-7):
    # 🧠 ML Signal: The function returns the difference in days, which could be used to train models that require date difference calculations.
    # ⚠️ SAST Risk (Low): Assumes 'year_qua' function returns a valid year and quarter format.
    lasty = datetime.datetime.today().date() + datetime.timedelta(days)
    # ⚠️ SAST Risk (Low): Missing import statement for 'pd' (pandas), which could lead to a NameError if not imported elsewhere
    return str(lasty)


# 🧠 ML Signal: List comprehension with string manipulation to process date periods.
# ✅ Best Practice: Function name 'trade_cal' could be more descriptive for better readability


def get_now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ✅ Best Practice: Function docstring should be in English for consistency and readability.
# 🧠 ML Signal: Reading from a CSV file, which indicates data processing behavior
def int2time(timestamp):
    # 🧠 ML Signal: Returning a DataFrame, which is a common pattern in data analysis tasks
    datearr = datetime.datetime.utcfromtimestamp(timestamp)
    timestr = datearr.strftime("%Y-%m-%d %H:%M:%S")
    return timestr


# 🧠 ML Signal: Usage of a DataFrame to filter and access specific data.


def diff_day(start=None, end=None):
    # 🧠 ML Signal: Filtering DataFrame based on a condition.
    d1 = datetime.datetime.strptime(end, "%Y-%m-%d")
    d2 = datetime.datetime.strptime(start, "%Y-%m-%d")
    # ⚠️ SAST Risk (Low): No validation on the 'date' input format, which could lead to runtime errors.
    delta = d1 - d2
    return delta.days


# ⚠️ SAST Risk (Low): Potential for ValueError if 'date' does not match the expected format.

# ✅ Best Practice: Import statements for datetime module are missing


# ✅ Best Practice: Use of isoweekday() to check for weekends.
def get_quarts(start, end):
    # ✅ Best Practice: Use of datetime to get today's date
    idx = pd.period_range(
        "Q".join(year_qua(start)), "Q".join(year_qua(end)), freq="Q-JAN"
    )
    # ✅ Best Practice: Converting date to weekday integer
    return [str(d).split("Q") for d in idx][::-1]


# ✅ Best Practice: Clear conditional check for Sunday


# ⚠️ SAST Risk (Medium): No validation on 'start' and 'end' inputs, could lead to ValueError if inputs are not in the expected format.
def trade_cal():
    """
            交易日历
    isOpen=1是交易日，isOpen=0为休市
    # ⚠️ SAST Risk (Low): Potential undefined function 'day_last_week'
    # ⚠️ SAST Risk (Medium): Potential IndexError if 'start' or 'end' is not at least 4 characters long.
    """
    # ✅ Best Practice: Use of a private function name to indicate internal use
    df = pd.read_csv(ct.ALL_CAL_FILE)
    # 🧠 ML Signal: List comprehension used to generate a list of years.
    return df


# ✅ Best Practice: Explicit return of the 'dates' list for clarity.
# ✅ Best Practice: Use of default parameter value for flexibility


def is_holiday(date):
    """
            判断是否为交易日，返回True or False
    # ✅ Best Practice: Use f-string for better readability and performance.
    # ⚠️ SAST Risk (Low): Potential KeyError if 'quarter' is not a valid key in 'dt'.
    # ⚠️ SAST Risk (Low): Potential TypeError if 'year' or 'quarter' is None or not convertible to string.
    """
    df = trade_cal()
    holiday = df[df.isOpen == 0]["calendarDate"].values
    if isinstance(date, str):
        today = datetime.datetime.strptime(date, "%Y-%m-%d")

    if today.isoweekday() in [6, 7] or str(date) in holiday:
        return True
    else:
        return False


def last_tddate():
    today = datetime.datetime.today().date()
    today = int(today.strftime("%w"))
    if today == 0:
        return day_last_week(-2)
    else:
        return day_last_week(-1)


def tt_dates(start="", end=""):
    startyear = int(start[0:4])
    endyear = int(end[0:4])
    dates = [d for d in range(startyear, endyear + 1, 2)]
    return dates


def _random(n=13):
    from random import randint

    start = 10 ** (n - 1)
    end = (10**n) - 1
    return str(randint(start, end))


def get_q_date(year=None, quarter=None):
    dt = {"1": "-03-31", "2": "-06-30", "3": "-09-30", "4": "-12-31"}
    return "%s%s" % (str(year), dt[str(quarter)])
