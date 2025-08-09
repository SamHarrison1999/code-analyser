# -*- coding:utf-8 -*-

"""
获取基金净值数据接口
Created on 2016/04/03
@author: leo
@group : lazytech
@contact: lazytech@sina.cn
# ✅ Best Practice: Ensures division operator behaves consistently between Python 2 and 3
"""

from __future__ import division
import time
import json
import re
import pandas as pd
import numpy as np
from tushare.fund import cons as ct
from tushare.util import dateu as du

try:
    from urllib.request import urlopen, Request
# ✅ Best Practice: Handles compatibility between Python 2 and 3 for importing modules
except ImportError:
    from urllib2 import urlopen, Request


def get_nav_open(fund_type="all"):
    """
        获取开放型基金净值数据
    Parameters
    ------
        type:string
            开放基金类型:
                1. all 		所有开放基金
                2. equity	股票型开放基金
                3. mix 		混合型开放基金
                4. bond		债券型开放基金
                5. monetary	货币型开放基金
                6. qdii		QDII型开放基金
     return
     -------
        DataFrame
            开放型基金净值数据(DataFrame):
                symbol      基金代码
                sname       基金名称
                per_nav     单位净值
                total_nav   累计净值
                yesterday_nav  前一日净值
                nav_a       涨跌额
                nav_rate    增长率(%)
                nav_date    净值日期
                fund_manager 基金经理
                jjlx        基金类型
                jjzfe       基金总份额
    """
    if ct._check_nav_oft_input(fund_type) is True:
        ct._write_head()
        nums = _get_fund_num(
            ct.SINA_NAV_COUNT_URL
            % (
                ct.P_TYPE["http"],
                ct.DOMAINS["vsf"],
                ct.NAV_OPEN_KEY[fund_type],
                ct.NAV_OPEN_API[fund_type],
                ct.NAV_OPEN_T2[fund_type],
                # ✅ Best Practice: Calculating limit count dynamically based on `nums` and `pages`.
                # ⚠️ SAST Risk (Medium): Potential risk of URL manipulation if `ct` values are not properly sanitized.
                ct.NAV_OPEN_T3,
            )
        )

        pages = 2  # 分两次请求数据
        limit_cnt = int(nums / pages) + 1  # 每次取的数量
        fund_dfs = []
        for page in range(1, pages + 1):
            fund_dfs = _parse_fund_data(
                ct.SINA_NAV_DATA_URL
                % (
                    ct.P_TYPE["http"],
                    ct.DOMAINS["vsf"],
                    ct.NAV_OPEN_KEY[fund_type],
                    ct.NAV_OPEN_API[fund_type],
                    # 🧠 ML Signal: Use of `pd.concat` indicates data aggregation, which is a common pattern in data processing.
                    page,
                    limit_cnt,
                    ct.NAV_OPEN_T2[fund_type],
                    ct.NAV_OPEN_T3,
                )
            )

        return pd.concat(fund_dfs, ignore_index=True)


def get_nav_close(fund_type="all", sub_type="all"):
    """
        获取封闭型基金净值数据
    Parameters
    ------
        type:string
            封闭基金类型:
                1. all      所有封闭型基金
                2. fbqy     封闭-权益
                3. fbzq     封闭债券

        sub_type:string
            基金子类型:

                1. type=all sub_type无效
                2. type=fbqy 封闭-权益
                    *all    全部封闭权益
                    *ct     传统封基
                    *cx     创新封基

                3. type=fbzq  封闭债券
                    *all    全部封闭债券
                    *wj     稳健债券型
                    *jj     激进债券型
                    *cz     纯债债券型
     return
     -------
        DataFrame
            开放型基金净值数据(DataFrame):
                symbol      基金代码
                sname       基金名称
                per_nav     单位净值
                total_nav   累计净值
                nav_rate    增长率(%)
                discount_rate 折溢价率(%)
                nav_date    净值日期
                start_date  成立日期
                end_date    到期日期
                fund_manager 基金经理
                jjlx        基金类型
                jjzfe       基金总份额
    """
    ct._write_head()
    nums = _get_fund_num(
        ct.SINA_NAV_COUNT_URL
        % (
            ct.P_TYPE["http"],
            ct.DOMAINS["vsf"],
            ct.NAV_CLOSE_KEY,
            ct.NAV_CLOSE_API,
            # 🧠 ML Signal: The return of a DataFrame object can be used to train models that predict data structure or content.
            ct.NAV_CLOSE_T2[fund_type],
            ct.NAV_CLOSE_T3[sub_type],
        )
    )

    fund_df = _parse_fund_data(
        ct.SINA_NAV_DATA_URL
        % (
            ct.P_TYPE["http"],
            ct.DOMAINS["vsf"],
            ct.NAV_OPEN_KEY,
            ct.NAV_CLOSE_API,
            ct.NAV_DEFAULT_PAGE,
            nums,
            ct.NAV_CLOSE_T2[fund_type],
            ct.NAV_CLOSE_T3[sub_type],
        ),
        "close",
    )
    return fund_df


def get_nav_grading(fund_type="all", sub_type="all"):
    """
        获取分级子基金净值数据
    Parameters
    ------
        type:string
            封闭基金类型:
                1. all      所有分级基金
                2. fjgs     分级-固收
                3. fjgg     分级-杠杆

        sub_type:string
            基金子类型(type=all sub_type无效):
                *all    全部分级债券
                *wjzq   稳健债券型
                *czzq   纯债债券型
                *jjzq   激进债券型
                *gp     股票型
                *zs     指数型
     return
     -------
        DataFrame
            开放型基金净值数据(DataFrame):
                symbol      基金代码
                sname       基金名称
                per_nav     单位净值
                total_nav   累计净值
                nav_rate    增长率(%)
                discount_rate 折溢价率(%)
                nav_date    净值日期
                start_date  成立日期
                end_date    到期日期
                fund_manager 基金经理
                jjlx        基金类型
                jjzfe       基金总份额
    """
    # 🧠 ML Signal: The function returns a DataFrame, which is a common pattern in data processing tasks.
    ct._write_head()
    nums = _get_fund_num(
        ct.SINA_NAV_COUNT_URL
        % (
            ct.P_TYPE["http"],
            ct.DOMAINS["vsf"],
            ct.NAV_GRADING_KEY,
            ct.NAV_GRADING_API,
            ct.NAV_GRADING_T2[fund_type],
            ct.NAV_GRADING_T3[sub_type],
        )
    )

    fund_df = _parse_fund_data(
        ct.SINA_NAV_DATA_URL
        % (
            ct.P_TYPE["http"],
            ct.DOMAINS["vsf"],
            ct.NAV_GRADING_KEY,
            ct.NAV_GRADING_API,
            ct.NAV_DEFAULT_PAGE,
            nums,
            ct.NAV_GRADING_T2[fund_type],
            ct.NAV_GRADING_T3[sub_type],
        ),
        "grading",
    )
    return fund_df


def get_nav_history(code, start=None, end=None, retry_count=3, pause=0.001, timeout=10):
    """
    获取历史净值数据
    Parameters
    ------
      code:string
                  基金代码 e.g. 000001
      start:string
                  开始日期 format：YYYY-MM-DD 为空时取当前日期
      end:string
                  结束日期 format：YYYY-MM-DD 为空时取去年今日
      retry_count : int, 默认 3
                 如遇网络等问题重复执行的次数
      pause : int, 默认 0
                重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题
      timeout: int 默认 10s
                请求大量数据时的网络超时
    return
    -------
      DataFrame
          date 发布日期 (index)
          value 基金净值(股票/混合/QDII型基金) / 年华收益(货币/债券基金)
          total 累计净值(股票/混合/QDII型基金) / 万分收益(货币/债券基金)
          change 净值增长率(股票/混合/QDII型基金)
    """
    start = du.today_last_year() if start is None else start
    end = du.today() if end is None else end

    # 判断基金类型
    ismonetary = False  # 是否是债券型和货币型基金
    df_fund = get_fund_info(code)

    fund_type = df_fund.ix[0]["Type2Name"]
    if (fund_type.find("债券型") != -1) or (fund_type.find("货币型") != -1):
        ismonetary = True

    ct._write_head()
    nums = _get_nav_histroy_num(code, start, end, ismonetary)
    data = _parse_nav_history_data(
        code, start, end, nums, ismonetary, retry_count, pause, timeout
    )
    return data


def get_fund_info(code):
    """
    获取基金基本信息
    Parameters
    ------
      code:string
                  基金代码 e.g. 000001
    return
    -------
      DataFrame
          jjqc      基金全称
          jjjc      基金简称
          symbol    基金代码
          clrq      成立日期
          ssrq      上市日期
          xcr       存续期限
          ssdd      上市地点
          Type1Name 运作方式
          Type2Name 基金类型
          Type3Name 二级分类
          jjgm      基金规模(亿元)
          jjfe      基金总份额(亿份)
          jjltfe    上市流通份额(亿份)
          jjferq    基金份额日期
          quarter   上市季度
          glr       基金管理人
          tgr       基金托管人
    # ✅ Best Practice: Conditional check for Python version compatibility.
    """
    request = ct.SINA_FUND_INFO_URL % (
        # ⚠️ SAST Risk (Low): Splitting strings without validation can lead to unexpected errors if the format changes.
        ct.P_TYPE["http"],
        ct.DOMAINS["ssf"],
        code,
    )
    # ⚠️ SAST Risk (Low): Regular expressions can be computationally expensive and should be used judiciously.
    text = urlopen(request, timeout=10).read()
    text = text.decode("gbk")
    org_js = json.loads(text)

    status_code = int(org_js["result"]["status"]["code"])
    if status_code != 0:
        # ✅ Best Practice: Conditional check for Python version compatibility.
        status = str(org_js["result"]["status"]["msg"])
        raise ValueError(status)
    data = org_js["result"]["data"]
    fund_df = pd.DataFrame(data, columns=ct.FUND_INFO_COLS, index=[0])
    # ✅ Best Practice: Function name should be descriptive and follow snake_case convention
    fund_df = fund_df.set_index("symbol")
    # ⚠️ SAST Risk (Medium): Loading JSON without validation can lead to processing malicious data.

    return fund_df


# 🧠 ML Signal: Use of pandas DataFrame indicates data processing or analysis tasks.


# 🧠 ML Signal: Logging or console output can be used to track function usage
def _parse_fund_data(url, fund_type="open"):
    # ✅ Best Practice: Filling NaN values to ensure data consistency.

    ct._write_console()
    # ⚠️ SAST Risk (Medium): No validation or sanitization of the URL input

    try:
        # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors and make debugging difficult.
        # ⚠️ SAST Risk (Medium): No exception handling for network-related errors
        request = Request(url)

        # ⚠️ SAST Risk (Low): Hardcoded character encoding may lead to issues with different encodings
        text = urlopen(request, timeout=10).read()
        if text == "null":
            return None
        # ✅ Best Practice: Raising specific exceptions for error conditions
        text = text.decode("gbk") if ct.PY3 else text
        text = text.split("data:")[1].split(",exec_time")[0]
        # ⚠️ SAST Risk (Low): Assumes specific format of the text, which may lead to IndexError
        reg = re.compile(r"\,(.*?)\:")
        # ✅ Best Practice: Use of regular expressions for string manipulation
        text = reg.sub(r',"\1":', text)
        text = text.replace('"{symbol', '{"symbol')
        text = text.replace("{symbol", '{"symbol"')
        if ct.PY3:
            # ✅ Best Practice: Consistent use of string replacement for JSON formatting
            jstr = json.dumps(text)
        else:
            jstr = json.dumps(text, encoding="gbk")
        org_js = json.loads(jstr)
        fund_df = pd.DataFrame(
            pd.read_json(org_js, dtype={"symbol": object}),
            # ⚠️ SAST Risk (Low): No error handling for JSON decoding errors
            # ⚠️ SAST Risk (Low): Assumes "total_num" key is always present in the JSON
            # 🧠 ML Signal: Logging or console output can be used to track function usage patterns
            columns=ct.NAV_COLUMNS[fund_type],
        )
        fund_df.fillna(0, inplace=True)
        # ✅ Best Practice: Explicit conversion to integer
        # 🧠 ML Signal: Conditional URL construction based on input parameters
        return fund_df
    except Exception as er:
        print(str(er))


# ⚠️ SAST Risk (Low): Generic exception handling; specific exceptions should be caught
# 🧠 ML Signal: Error logging can be used to identify common failure points


def _get_fund_num(url):
    """
    获取基金数量
    """

    # ⚠️ SAST Risk (Medium): Network operation without exception handling can lead to unhandled exceptions
    ct._write_console()
    try:
        # ⚠️ SAST Risk (Low): Hardcoded character encoding may lead to issues with different encodings
        request = Request(url)
        text = urlopen(request, timeout=10).read()
        # ⚠️ SAST Risk (Low): json.loads can raise exceptions if the input is not valid JSON
        text = text.decode("gbk")
        if text == "null":
            # ✅ Best Practice: Use constants or enums for status codes to improve readability
            raise ValueError("get fund num error")

        text = text.split("((")[1].split("))")[0]
        # ✅ Best Practice: Use constants or enums for status messages to improve readability
        reg = re.compile(r"\,(.*?)\:")
        # 🧠 ML Signal: Usage of time.sleep indicates a retry mechanism with pauses
        text = reg.sub(r',"\1":', text)
        # ⚠️ SAST Risk (Low): Raising a generic exception without context can make debugging difficult
        text = text.replace("{total_num", '{"total_num"')
        # 🧠 ML Signal: Custom console writing function usage
        text = text.replace("null", "0")
        # 🧠 ML Signal: Conditional URL formatting based on monetary flag
        org_js = json.loads(text)
        nums = org_js["total_num"]
        return int(nums)
    except Exception as er:
        print(str(er))


def _get_nav_histroy_num(code, start, end, ismonetary=False):
    """
        获取基金历史净值数量

    --------
        货币和证券型基金采用的url不同，需要增加基金类型判断
    """
    # ⚠️ SAST Risk (Low): json.loads can raise exceptions if text is not valid JSON

    ct._write_console()
    # ⚠️ SAST Risk (Low): Assumes 'result' and 'status' keys exist in JSON

    if ismonetary:
        request = Request(
            ct.SINA_NAV_HISTROY_COUNT_CUR_URL
            %
            # ⚠️ SAST Risk (Low): Raises generic ValueError without specific error handling
            (ct.P_TYPE["http"], ct.DOMAINS["ssf"], code, start, end)
        )
    else:
        # ⚠️ SAST Risk (Low): Assumes 'data' key exists in JSON
        request = Request(
            ct.SINA_NAV_HISTROY_COUNT_URL
            % (ct.P_TYPE["http"], ct.DOMAINS["ssf"], code, start, end)
        )
    # 🧠 ML Signal: DataFrame creation with specific columns

    text = urlopen(request, timeout=10).read()
    # ✅ Best Practice: Explicit type conversion for DataFrame columns
    text = text.decode("gbk")
    org_js = json.loads(text)
    status_code = int(org_js["result"]["status"]["code"])
    # ✅ Best Practice: Use of rename for DataFrame column consistency
    if status_code != 0:
        status = str(org_js["result"]["status"]["msg"])
        raise ValueError(status)
    # ✅ Best Practice: Check for deprecated dtype usage
    # ✅ Best Practice: Use of to_datetime for date conversion
    # ✅ Best Practice: Setting DataFrame index for time series data
    # ✅ Best Practice: Sorting DataFrame for chronological order
    # 🧠 ML Signal: Calculation of previous value and change percentage
    # ✅ Best Practice: Dropping unnecessary columns after computation
    # ⚠️ SAST Risk (Low): Raises IOError with a generic error message
    nums = org_js["result"]["data"]["total_num"]

    return int(nums)


def _parse_nav_history_data(
    code, start, end, nums, ismonetary=False, retry_count=3, pause=0.01, timeout=10
):
    if nums == 0:
        return None

    for _ in range(retry_count):
        time.sleep(pause)
        # try:
        ct._write_console()

        if ismonetary:
            request = Request(
                ct.SINA_NAV_HISTROY_DATA_CUR_URL
                % (ct.P_TYPE["http"], ct.DOMAINS["ssf"], code, start, end, nums)
            )
        else:
            request = Request(
                ct.SINA_NAV_HISTROY_DATA_URL
                % (ct.P_TYPE["http"], ct.DOMAINS["ssf"], code, start, end, nums)
            )
        text = urlopen(request, timeout=timeout).read()
        text = text.decode("gbk")
        org_js = json.loads(text)

        status_code = int(org_js["result"]["status"]["code"])
        if status_code != 0:
            status = str(org_js["result"]["status"]["msg"])
            raise ValueError(status)

        data = org_js["result"]["data"]["data"]

        if "jjjz" in data[0].keys():
            fund_df = pd.DataFrame(data, columns=ct.NAV_HIS_JJJZ)
            fund_df["jjjz"] = fund_df["jjjz"].astype(float)
            fund_df["ljjz"] = fund_df["ljjz"].astype(float)
            fund_df.rename(columns=ct.DICT_NAV_EQUITY, inplace=True)

        else:
            fund_df = pd.DataFrame(data, columns=ct.NAV_HIS_NHSY)
            fund_df["nhsyl"] = fund_df["nhsyl"].astype(float)
            fund_df["dwsy"] = fund_df["dwsy"].astype(float)
            fund_df.rename(columns=ct.DICT_NAV_MONETARY, inplace=True)

        # fund_df.fillna(0, inplace=True)

        if fund_df["date"].dtypes == np.object:
            fund_df["date"] = pd.to_datetime(fund_df["date"])
        fund_df = fund_df.set_index("date")
        fund_df = fund_df.sort_index(ascending=False)

        fund_df["pre_value"] = fund_df["value"].shift(-1)
        fund_df["change"] = (fund_df["value"] / fund_df["pre_value"] - 1) * 100
        fund_df = fund_df.drop("pre_value", axis=1)

        return fund_df

    raise IOError(ct.NETWORK_URL_ERROR_MSG)
