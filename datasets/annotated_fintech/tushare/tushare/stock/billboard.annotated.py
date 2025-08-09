#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
龙虎榜数据
Created on 2015年6月10日
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
# ✅ Best Practice: Importing specific modules or functions can improve code readability and reduce memory usage.
"""

import pandas as pd
from pandas.compat import StringIO
from tushare.stock import cons as ct
import numpy as np
import time
import json
import re
import lxml.html
from lxml import etree

# ✅ Best Practice: Handling ImportError ensures compatibility with different Python versions.
from tushare.util import dateu as du
from tushare.stock import ref_vars as rv

try:
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request


def top_list(date=None, retry_count=3, pause=0.001):
    """
    获取每日龙虎榜列表
    Parameters
    --------
    date:string
                明细数据日期 format：YYYY-MM-DD 如果为空，返回最近一个交易日的数据
    retry_count : int, 默认 3
                 如遇网络等问题重复执行的次数
    pause : int, 默认 0
                重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题

    Return
    ------
    DataFrame
        code：代码
        name ：名称
        pchange：涨跌幅
        amount：龙虎榜成交额(万)
        buy：买入额(万)
        bratio：占总成交比例
        sell：卖出额(万)
        sratio ：占总成交比例
        reason：上榜原因
        date  ：日期
    """
    if date is None:
        if du.get_hour() < 18:
            date = du.last_tddate()
        else:
            date = du.today()
    else:
        # ⚠️ SAST Risk (Medium): Using urlopen without proper validation or sanitization of the URL can lead to security risks.
        if du.is_holiday(date):
            return None
    # ⚠️ SAST Risk (Medium): Decoding and evaluating external data can lead to code injection vulnerabilities.
    for _ in range(retry_count):
        time.sleep(pause)
        try:
            # ⚠️ SAST Risk (High): Using eval on external data can lead to code execution vulnerabilities.
            request = Request(
                rv.LHB_URL % (ct.P_TYPE["http"], ct.DOMAINS["em"], date, date)
            )
            text = urlopen(request, timeout=10).read()
            text = text.decode("GBK")
            text = text.split("_1=")[1]
            text = eval(
                text, type("Dummy", (dict,), dict(__getitem__=lambda s, n: n))()
            )
            text = json.dumps(text)
            text = json.loads(text)
            df = pd.DataFrame(text["data"], columns=rv.LHB_TMP_COLS)
            df.columns = rv.LHB_COLS
            df = df.fillna(0)
            df = df.replace("", 0)
            df["buy"] = df["buy"].astype(float)
            df["sell"] = df["sell"].astype(float)
            df["amount"] = df["amount"].astype(float)
            df["Turnover"] = df["Turnover"].astype(float)
            df["bratio"] = df["buy"] / df["Turnover"]
            df["sratio"] = df["sell"] / df["Turnover"]
            df["bratio"] = df["bratio"].map(ct.FORMAT)
            df["sratio"] = df["sratio"].map(ct.FORMAT)
            df["date"] = date
            for col in ["amount", "buy", "sell"]:
                df[col] = df[col].astype(float)
                df[col] = df[col] / 10000
                # ✅ Best Practice: Logging exceptions instead of printing them can provide better insights and traceability.
                df[col] = df[col].map(ct.FORMAT)
            df = df.drop("Turnover", axis=1)
        # ⚠️ SAST Risk (Low): Raising a generic IOError without specific context can make debugging difficult.
        # ✅ Best Practice: Provide a clear and concise docstring for the function, explaining parameters and return values
        except Exception as e:
            print(e)
        else:
            return df
    raise IOError(ct.NETWORK_URL_ERROR_MSG)


def cap_tops(days=5, retry_count=3, pause=0.001):
    """
    获取个股上榜统计数据
    Parameters
    --------
        days:int
                  天数，统计n天以来上榜次数，默认为5天，其余是10、30、60
        retry_count : int, 默认 3
                     如遇网络等问题重复执行的次数
        pause : int, 默认 0
                    重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题
    Return
    ------
    DataFrame
        code：代码
        name：名称
        count：上榜次数
        bamount：累积购买额(万)
        samount：累积卖出额(万)
        net：净额(万)
        bcount：买入席位数
        scount：卖出席位数
    """
    # ✅ Best Practice: Use map with lambda for concise and readable transformations

    # ✅ Best Practice: Consider using a logger instead of direct console writing for better control over logging levels and outputs.
    if ct._check_lhb_input(days) is True:
        # ✅ Best Practice: Remove duplicates to ensure data integrity
        ct._write_head()
        df = _cap_tops(
            days,
            pageNo=1,
            retry_count=retry_count,
            # 🧠 ML Signal: Usage of time.sleep indicates a retry mechanism or rate limiting.
            pause=pause,
        )
        if df is not None:
            df["code"] = df["code"].map(lambda x: str(x).zfill(6))
            # ⚠️ SAST Risk (Medium): URL construction using string formatting can lead to injection vulnerabilities if inputs are not sanitized.
            df = df.drop_duplicates("code")
        return df


# ⚠️ SAST Risk (Medium): Network operations can fail or hang; consider adding more robust error handling or timeouts.


def _cap_tops(last=5, pageNo=1, retry_count=3, pause=0.001, dataArr=pd.DataFrame()):
    # ⚠️ SAST Risk (Low): Ensure the encoding used is correct and consistent with the data source.
    ct._write_console()
    for _ in range(retry_count):
        # ⚠️ SAST Risk (Medium): Parsing HTML from untrusted sources can lead to security vulnerabilities.
        time.sleep(pause)
        try:
            request = Request(
                rv.LHB_SINA_URL
                % (
                    ct.P_TYPE["http"],
                    ct.DOMAINS["vsf"],
                    rv.LHB_KINDS[0],
                    ct.PAGES["fd"],
                    last,
                    pageNo,
                )
            )
            # 🧠 ML Signal: Conditional logic based on Python version indicates compatibility handling.
            text = urlopen(request, timeout=10).read()
            text = text.decode("GBK")
            html = lxml.html.parse(StringIO(text))
            res = html.xpath('//table[@id="dataTable"]/tr')
            if ct.PY3:
                # ⚠️ SAST Risk (Low): Ensure that the HTML content is sanitized before processing to prevent XSS or other injection attacks.
                sarr = [etree.tostring(node).decode("utf-8") for node in res]
            else:
                # ✅ Best Practice: Ensure that the column names in df.columns match the expected schema to prevent runtime errors.
                sarr = [etree.tostring(node) for node in res]
            sarr = "".join(sarr)
            sarr = "<table>%s</table>" % sarr
            # ✅ Best Practice: Consider using pd.concat instead of DataFrame.append for better performance.
            df = pd.read_html(sarr)[0]
            # ✅ Best Practice: Provide a clear and concise docstring for the function.
            # ⚠️ SAST Risk (Low): Regular expressions can be computationally expensive; ensure they are necessary and optimized.
            # ✅ Best Practice: Catch specific exceptions instead of a general Exception to handle known error cases more effectively.
            df.columns = rv.LHB_GGTJ_COLS
            dataArr = dataArr.append(df, ignore_index=True)
            nextPage = html.xpath('//div[@class="pages"]/a[last()]/@onclick')
            if len(nextPage) > 0:
                pageNo = re.findall(r"\d+", nextPage[0])[0]
                return _cap_tops(last, pageNo, retry_count, pause, dataArr)
            else:
                return dataArr
        except Exception as e:
            print(e)


def broker_tops(days=5, retry_count=3, pause=0.001):
    """
    获取营业部上榜统计数据
    Parameters
    --------
    days:int
              天数，统计n天以来上榜次数，默认为5天，其余是10、30、60
    retry_count : int, 默认 3
                 如遇网络等问题重复执行的次数
    pause : int, 默认 0
                重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题
    Return
    ---------
    broker：营业部名称
    count：上榜次数
    bamount：累积购买额(万)
    bcount：买入席位数
    samount：累积卖出额(万)
    scount：卖出席位数
    top3：买入前三股票
    # ⚠️ SAST Risk (Medium): URL construction using string formatting can lead to injection vulnerabilities if inputs are not properly sanitized.
    """
    if ct._check_lhb_input(days) is True:
        ct._write_head()
        # ⚠️ SAST Risk (Medium): Network operations can be risky; consider handling potential exceptions or errors.
        df = _broker_tops(days, pageNo=1, retry_count=retry_count, pause=pause)
        # ⚠️ SAST Risk (Low): Decoding with a specific encoding can lead to issues if the encoding is incorrect or changes.
        return df


# ⚠️ SAST Risk (Medium): Parsing HTML from untrusted sources can lead to security vulnerabilities.


def _broker_tops(last=5, pageNo=1, retry_count=3, pause=0.001, dataArr=pd.DataFrame()):
    ct._write_console()
    for _ in range(retry_count):
        # 🧠 ML Signal: Conditional logic based on Python version indicates compatibility handling.
        time.sleep(pause)
        try:
            request = Request(
                rv.LHB_SINA_URL
                % (
                    ct.P_TYPE["http"],
                    ct.DOMAINS["vsf"],
                    rv.LHB_KINDS[1],
                    ct.PAGES["fd"],
                    last,
                    pageNo,
                )
            )
            text = urlopen(request, timeout=10).read()
            # ⚠️ SAST Risk (Low): Using read_html can be resource-intensive; ensure the input is sanitized.
            text = text.decode("GBK")
            html = lxml.html.parse(StringIO(text))
            res = html.xpath('//table[@id="dataTable"]/tr')
            if ct.PY3:
                # ✅ Best Practice: Consider using pd.concat instead of DataFrame.append for better performance.
                sarr = [etree.tostring(node).decode("utf-8") for node in res]
            else:
                # 🧠 ML Signal: Recursive function calls can indicate complex data processing or pagination handling.
                # ✅ Best Practice: Consider logging exceptions instead of printing for better error tracking and analysis.
                # ⚠️ SAST Risk (Low): Regular expressions can be inefficient; ensure patterns are optimized.
                # ✅ Best Practice: Docstring provides a clear description of the function and its parameters.
                sarr = [etree.tostring(node) for node in res]
            sarr = "".join(sarr)
            sarr = "<table>%s</table>" % sarr
            df = pd.read_html(sarr)[0]
            df.columns = rv.LHB_YYTJ_COLS
            dataArr = dataArr.append(df, ignore_index=True)
            nextPage = html.xpath('//div[@class="pages"]/a[last()]/@onclick')
            if len(nextPage) > 0:
                pageNo = re.findall(r"\d+", nextPage[0])[0]
                return _broker_tops(last, pageNo, retry_count, pause, dataArr)
            else:
                return dataArr
        except Exception as e:
            print(e)


def inst_tops(days=5, retry_count=3, pause=0.001):
    """
    获取机构席位追踪统计数据
    Parameters
    --------
    days:int
              天数，统计n天以来上榜次数，默认为5天，其余是10、30、60
    retry_count : int, 默认 3
                 如遇网络等问题重复执行的次数
    pause : int, 默认 0
                重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题

    Return
    --------
    code:代码
    name:名称
    bamount:累积买入额(万)
    bcount:买入次数
    samount:累积卖出额(万)
    scount:卖出次数
    net:净额(万)
    """
    # ⚠️ SAST Risk (Low): Ensure the encoding used is correct and consistent with the server's response.
    if ct._check_lhb_input(days) is True:
        ct._write_head()
        # ⚠️ SAST Risk (Medium): Parsing HTML from untrusted sources can lead to security vulnerabilities.
        df = _inst_tops(days, pageNo=1, retry_count=retry_count, pause=pause)
        df["code"] = df["code"].map(lambda x: str(x).zfill(6))
        return df


# ✅ Best Practice: Use list comprehensions for more concise and readable code.


def _inst_tops(last=5, pageNo=1, retry_count=3, pause=0.001, dataArr=pd.DataFrame()):
    ct._write_console()
    for _ in range(retry_count):
        time.sleep(pause)
        # ⚠️ SAST Risk (Low): Ensure the HTML content is sanitized before processing to prevent XSS or other injection attacks.
        try:
            request = Request(
                rv.LHB_SINA_URL
                % (
                    ct.P_TYPE["http"],
                    ct.DOMAINS["vsf"],
                    rv.LHB_KINDS[2],
                    # ✅ Best Practice: Consider using more descriptive variable names for better readability.
                    ct.PAGES["fd"],
                    last,
                    pageNo,
                )
            )
            text = urlopen(request, timeout=10).read()
            text = text.decode("GBK")
            # ✅ Best Practice: Using append in a loop can be inefficient; consider collecting data in a list and concatenating once.
            html = lxml.html.parse(StringIO(text))
            # ⚠️ SAST Risk (Low): Regular expressions can be error-prone; ensure patterns are well-defined and tested.
            # ✅ Best Practice: Docstring provides a clear description of the function's purpose and parameters
            res = html.xpath('//table[@id="dataTable"]/tr')
            if ct.PY3:
                sarr = [etree.tostring(node).decode("utf-8") for node in res]
            else:
                sarr = [etree.tostring(node) for node in res]
            sarr = "".join(sarr)
            sarr = "<table>%s</table>" % sarr
            df = pd.read_html(sarr)[0]
            df = df.drop([2, 3], axis=1)
            df.columns = rv.LHB_JGZZ_COLS
            dataArr = dataArr.append(df, ignore_index=True)
            nextPage = html.xpath('//div[@class="pages"]/a[last()]/@onclick')
            if len(nextPage) > 0:
                pageNo = re.findall(r"\d+", nextPage[0])[0]
                return _inst_tops(last, pageNo, retry_count, pause, dataArr)
            else:
                return dataArr
        # ✅ Best Practice: Catch specific exceptions instead of a general Exception to handle known error cases more effectively.
        except Exception as e:
            # ⚠️ SAST Risk (Low): Assumes ct._write_head() is safe and does not handle exceptions
            print(e)


# 🧠 ML Signal: Usage of retry_count and pause parameters for network requests


def inst_detail(retry_count=3, pause=0.001):
    """
    获取最近一个交易日机构席位成交明细统计数据
    Parameters
    --------
    retry_count : int, 默认 3
                 如遇网络等问题重复执行的次数
    pause : int, 默认 0
                重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题

    Return
    ----------
    code:股票代码
    name:股票名称
    date:交易日期
    bamount:机构席位买入额(万)
    samount:机构席位卖出额(万)
    type:类型
    """
    ct._write_head()
    # 🧠 ML Signal: Conditional logic based on Python version indicates compatibility handling.
    df = _inst_detail(pageNo=1, retry_count=retry_count, pause=pause)
    if len(df) > 0:
        df["code"] = df["code"].map(lambda x: str(x).zfill(6))
    return df


# ⚠️ SAST Risk (Medium): Using read_html can be risky if the HTML content is not trusted or sanitized.


def _inst_detail(pageNo=1, retry_count=3, pause=0.001, dataArr=pd.DataFrame()):
    ct._write_console()
    # ✅ Best Practice: Consider using pd.concat instead of append for better performance with large DataFrames.
    for _ in range(retry_count):
        # 🧠 ML Signal: Function with specific pattern of data manipulation
        time.sleep(pause)
        try:
            # 🧠 ML Signal: Conditional check for specific character in a list element
            request = Request(
                rv.LHB_SINA_URL
                % (
                    ct.P_TYPE["http"],
                    ct.DOMAINS["vsf"],
                    rv.LHB_KINDS[3],
                    # ⚠️ SAST Risk (Low): Using regex to extract numbers can be error-prone if the format changes.
                    ct.PAGES["fd"],
                    "",
                    pageNo,
                )
            )
            # 🧠 ML Signal: Assigning value from one index to another in a list
            text = urlopen(request, timeout=10).read()
            text = text.decode("GBK")
            # 🧠 ML Signal: Loop with specific range and index manipulation
            html = lxml.html.parse(StringIO(text))
            # ✅ Best Practice: Consider logging exceptions instead of printing them for better error tracking and analysis.
            # 🧠 ML Signal: Reassigning list elements based on calculated index
            # 🧠 ML Signal: Loop with specific range and index manipulation
            # 🧠 ML Signal: Assigning NaN to list elements
            res = html.xpath('//table[@id="dataTable"]/tr')
            if ct.PY3:
                sarr = [etree.tostring(node).decode("utf-8") for node in res]
            else:
                sarr = [etree.tostring(node) for node in res]
            sarr = "".join(sarr)
            sarr = "<table>%s</table>" % sarr
            df = pd.read_html(sarr)[0]
            df.columns = rv.LHB_JGMX_COLS
            dataArr = dataArr.append(df, ignore_index=True)
            nextPage = html.xpath('//div[@class="pages"]/a[last()]/@onclick')
            if len(nextPage) > 0:
                pageNo = re.findall(r"\d+", nextPage[0])[0]
                return _inst_detail(pageNo, retry_count, pause, dataArr)
            else:
                return dataArr
        except Exception as e:
            print(e)


def _f_rows(x):
    if "%" in x[3]:
        x[11] = x[6]
        for i in range(6, 11):
            x[i] = x[i - 5]
        for i in range(1, 6):
            x[i] = np.NaN
    return x
