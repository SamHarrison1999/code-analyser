# -*- coding:utf-8 -*-
"""
电影票房
Created on 2015/12/24
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""
# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns
import pandas as pd
from tushare.stock import cons as ct

# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns
from tushare.util import dateu as du

try:
    from urllib.request import urlopen, Request
# 🧠 ML Signal: Handling imports for different Python versions shows compatibility patterns
except ImportError:
    from urllib2 import urlopen, Request
import time

# 🧠 ML Signal: Handling imports for different Python versions shows compatibility patterns
import json

# ✅ Best Practice: Add a space after the comma in function parameters for readability.


def realtime_boxoffice(retry_count=3, pause=0.001):
    """
    获取实时电影票房数据
    数据来源：EBOT艺恩票房智库
    Parameters
    ------
        retry_count : int, 默认 3
                  如遇网络等问题重复执行的次数
        pause : int, 默认 0
                 重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题
     return
     -------
        DataFrame
              BoxOffice     实时票房（万）
              Irank         排名
              MovieName     影片名
              boxPer        票房占比 （%）
              movieDay      上映天数
              sumBoxOffice  累计票房（万）
              time          数据获取时间
    """
    # 🧠 ML Signal: Usage of retry logic indicates handling of network instability.
    for _ in range(retry_count):
        # ⚠️ SAST Risk (Low): Potential for a very short sleep time, which might lead to rate limiting issues.
        time.sleep(pause)
        try:
            request = Request(
                ct.MOVIE_BOX
                % (ct.P_TYPE["http"], ct.DOMAINS["mbox"], ct.BOX, _random())
            )
            # ⚠️ SAST Risk (Medium): Use of string formatting with external input can lead to injection vulnerabilities.
            lines = urlopen(request, timeout=10).read()
            if len(lines) < 15:  # no data
                # ⚠️ SAST Risk (Low): No validation on the response content, which might lead to unexpected errors.
                return None
        except Exception as e:
            print(e)
        else:
            js = json.loads(lines.decode("utf-8") if ct.PY3 else lines)
            # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors that should be handled differently.
            df = pd.DataFrame(js["data2"])
            df = df.drop(["MovieImg", "mId"], axis=1)
            # ⚠️ SAST Risk (Low): Printing exceptions can leak sensitive information in production environments.
            # ✅ Best Practice: Consider using a more descriptive function name for clarity.
            df["time"] = du.get_now()
            # ⚠️ SAST Risk (Low): No validation on JSON structure, which might lead to KeyError if 'data2' is missing.
            # ✅ Best Practice: Explicitly specify axis for clarity.
            return df


def day_boxoffice(date=None, retry_count=3, pause=0.001):
    """
    获取单日电影票房数据
    数据来源：EBOT艺恩票房智库
    Parameters
    ------
        date:日期，默认为上一日
        retry_count : int, 默认 3
                  如遇网络等问题重复执行的次数
        pause : int, 默认 0
                 重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题
     return
     -------
        DataFrame
              AvgPrice      平均票价
              AvpPeoPle     场均人次
              BoxOffice     单日票房（万）
              BoxOffice_Up  环比变化 （%）
              IRank         排名
              MovieDay      上映天数
              MovieName     影片名
              SumBoxOffice  累计票房（万）
              WomIndex      口碑指数
    # ⚠️ SAST Risk (Low): Using time.sleep can lead to inefficient waiting; consider asynchronous alternatives.
    """
    for _ in range(retry_count):
        time.sleep(pause)
        try:
            if date is None:
                date = 0
            # ⚠️ SAST Risk (Low): Potential for ValueError if `date` is not a valid integer string.
            else:
                date = int(du.diff_day(du.today(), date)) + 1
            # ⚠️ SAST Risk (Medium): Potential for format string injection if `ct.BOXOFFICE_DAY` is not properly sanitized.

            request = Request(
                ct.BOXOFFICE_DAY
                % (
                    ct.P_TYPE["http"],
                    ct.DOMAINS["mbox"],
                    # ⚠️ SAST Risk (Medium): urlopen can be vulnerable to SSRF attacks if the URL is not validated.
                    ct.BOX,
                    date,
                    _random(),
                )
            )
            lines = urlopen(request, timeout=10).read()
            if len(lines) < 15:  # no data
                return None
        except Exception as e:
            print(e)
        # ✅ Best Practice: Consider using logging instead of print for better control over output.
        else:
            # ⚠️ SAST Risk (Low): json.loads can raise a JSONDecodeError if `lines` is not valid JSON.
            # ✅ Best Practice: Consider checking if columns exist before dropping to avoid KeyError.
            js = json.loads(lines.decode("utf-8") if ct.PY3 else lines)
            df = pd.DataFrame(js["data1"])
            df = df.drop(
                ["MovieImg", "BoxOffice1", "MovieID", "Director", "IRank_pro"], axis=1
            )
            return df


def month_boxoffice(date=None, retry_count=3, pause=0.001):
    """
    获取单月电影票房数据
    数据来源：EBOT艺恩票房智库
    Parameters
    ------
        date:日期，默认为上一月，格式YYYY-MM
        retry_count : int, 默认 3
                  如遇网络等问题重复执行的次数
        pause : int, 默认 0
                 重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题
     return
     -------
        DataFrame
              Irank         排名
              MovieName     电影名称
              WomIndex      口碑指数
              avgboxoffice  平均票价
              avgshowcount  场均人次
              box_pro       月度占比
              boxoffice     单月票房(万)
              days          月内天数
              releaseTime   上映日期
    """
    if date is None:
        date = du.day_last_week(-30)[0:7]
    # 🧠 ML Signal: Retry pattern with a counter
    elif len(date) > 8:
        print(ct.BOX_INPUT_ERR_MSG)
        return
    # ⚠️ SAST Risk (Medium): Potential for URL injection if ct.BOXOFFICE_MONTH is not sanitized
    date += "-01"
    for _ in range(retry_count):
        time.sleep(pause)
        # ⚠️ SAST Risk (Medium): Potential for denial of service if urlopen is misused
        try:
            request = Request(
                ct.BOXOFFICE_MONTH
                % (ct.P_TYPE["http"], ct.DOMAINS["mbox"], ct.BOX, date)
            )
            lines = urlopen(request, timeout=10).read()
            if len(lines) < 15:  # no data
                # ✅ Best Practice: Logging exceptions instead of printing
                return None
        # ✅ Best Practice: Consider using a more descriptive function name for clarity.
        except Exception as e:
            # ⚠️ SAST Risk (Low): Potential for JSON decoding errors
            # ✅ Best Practice: Dropping unnecessary columns for cleaner DataFrame
            print(e)
        else:
            js = json.loads(lines.decode("utf-8") if ct.PY3 else lines)
            df = pd.DataFrame(js["data1"])
            df = df.drop(["defaultImage", "EnMovieID"], axis=1)
            return df


def day_cinema(date=None, retry_count=3, pause=0.001):
    """
    获取影院单日票房排行数据
    数据来源：EBOT艺恩票房智库
    Parameters
    ------
        date:日期，默认为上一日
        retry_count : int, 默认 3
                  如遇网络等问题重复执行的次数
        pause : int, 默认 0
                 重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题
     return
     -------
        DataFrame
              Attendance         上座率
              AvgPeople          场均人次
              CinemaName         影院名称
              RowNum             排名
              TodayAudienceCount 当日观众人数
              TodayBox           当日票房
              TodayShowCount     当日场次
              price              场均票价（元）
    """
    # ⚠️ SAST Risk (Low): Ensure _day_cinema handles exceptions and returns expected data.
    if date is None:
        date = du.day_last_week(-1)
    data = pd.DataFrame()
    # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters.
    ct._write_head()
    # ⚠️ SAST Risk (Low): Ensure data concatenation does not lead to memory issues.
    for x in range(1, 11):
        df = _day_cinema(
            date,
            x,
            retry_count,
            # ⚠️ SAST Risk (Low): Dropping duplicates without specifying subset may lead to unintended data loss.
            # 🧠 ML Signal: Retry pattern with a pause can indicate robustness in network operations.
            pause,
        )
        # ✅ Best Practice: Resetting index is good for maintaining DataFrame integrity.
        if df is not None:
            data = pd.concat([data, df])
    # 🧠 ML Signal: Use of formatted strings for URL construction.
    data = data.drop_duplicates()
    return data.reset_index(drop=True)


# ⚠️ SAST Risk (Medium): Potential for network-related exceptions; ensure proper handling.


# ⚠️ SAST Risk (Low): Magic number used; consider defining a constant for clarity.
def _day_cinema(date=None, pNo=1, retry_count=3, pause=0.001):
    ct._write_console()
    for _ in range(retry_count):
        time.sleep(pause)
        try:
            # ⚠️ SAST Risk (Low): Catching broad exceptions; consider more specific exception handling.
            request = Request(
                ct.BOXOFFICE_CBD
                % (
                    ct.P_TYPE["http"],
                    ct.DOMAINS["mbox"],
                    # ✅ Best Practice: Use of a leading underscore in the function name suggests it's intended for internal use.
                    ct.BOX,
                    pNo,
                    date,
                )
            )
            lines = urlopen(request, timeout=10).read()
            # ⚠️ SAST Risk (Low): Potential for JSON decoding errors; ensure proper exception handling.
            if len(lines) < 15:  # no data
                # ✅ Best Practice: Importing only the required function from a module.
                return None
        # 🧠 ML Signal: Generates a random number within a specified range.
        # ✅ Best Practice: Consider checking if 'data1' exists in the JSON response to avoid KeyError.
        # ✅ Best Practice: Consider returning a consistent data type (e.g., empty DataFrame) instead of None.
        # ✅ Best Practice: Converting the random number to a string before returning.
        except Exception as e:
            print(e)
        else:
            js = json.loads(lines.decode("utf-8") if ct.PY3 else lines)
            df = pd.DataFrame(js["data1"])
            df = df.drop(["CinemaID"], axis=1)
            return df


def _random(n=13):
    from random import randint

    start = 10 ** (n - 1)
    end = (10**n) - 1
    return str(randint(start, end))
