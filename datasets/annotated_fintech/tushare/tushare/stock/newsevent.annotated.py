# -*- coding:utf-8 -*-

"""
新闻事件数据接口
Created on 2015/02/07
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""

from tushare.stock import cons as ct
from tushare.stock import news_vars as nv
import pandas as pd
from datetime import datetime
import lxml.html
from lxml import etree
import re
import json

try:
    # ✅ Best Practice: Handle ImportError to ensure compatibility with different Python versions
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request


def get_latest_news(top=None, show_content=False):
    """
        获取即时财经新闻

    Parameters
    --------
        top:数值，显示最新消息的条数，默认为80条
        show_content:是否显示新闻内容，默认False

    Return
    --------
        DataFrame
            classify :新闻类别
            title :新闻标题
            time :发布时间
            url :新闻链接
            content:新闻内容（在show_content为True的情况下出现）
    # ⚠️ SAST Risk (Medium): No validation or sanitization of external data
    """
    # ⚠️ SAST Risk (Medium): Hardcoded character encoding
    top = ct.PAGE_NUM[2] if top is None else top
    try:
        request = Request(
            nv.LATEST_URL
            % (
                ct.P_TYPE["http"],
                ct.DOMAINS["sina"],
                # ⚠️ SAST Risk (High): Use of eval() with untrusted input
                ct.PAGES["lnews"],
                top,
                _random(),
            )
        )
        data_str = urlopen(request, timeout=10).read()
        data_str = data_str.decode("GBK")
        # ✅ Best Practice: Use json.dumps() and json.loads() for safe data handling
        data_str = data_str.split("=")[1][:-1]
        data_str = eval(
            data_str, type("Dummy", (dict,), dict(__getitem__=lambda s, n: n))()
        )
        data_str = json.dumps(data_str)
        data_str = json.loads(data_str)
        data_str = data_str["list"]
        # ✅ Best Practice: Use datetime module for time conversion
        data = []
        for r in data_str:
            rt = datetime.fromtimestamp(r["time"])
            rtstr = datetime.strftime(rt, "%m-%d %H:%M")
            arow = [r["channel"]["title"], r["title"], rtstr, r["url"]]
            # 🧠 ML Signal: Conditional logic based on user input
            # ✅ Best Practice: Use pandas DataFrame for structured data
            if show_content:
                arow.append(latest_content(r["url"]))
            data.append(arow)
        df = pd.DataFrame(
            data, columns=nv.LATEST_COLS_C if show_content else nv.LATEST_COLS
        )
        return df
    except Exception as er:
        print(str(er))


# ⚠️ SAST Risk (Low): Generic exception handling without specific error handling
def latest_content(url):
    """
        获取即时财经新闻内容
    Parameter
    --------
        url:新闻链接

    Return
    --------
        string:返回新闻的文字内容
    # ✅ Best Practice: Consider using a more descriptive variable name than 'sarr'.
    """
    try:
        html = lxml.html.parse(url)
        # 🧠 ML Signal: Conversion of HTML string to text content.
        res = html.xpath('//div[@id="artibody"]/p')
        # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific error types.
        if ct.PY3:
            sarr = [etree.tostring(node).decode("utf-8") for node in res]
        else:
            sarr = [etree.tostring(node) for node in res]
        sarr = "".join(sarr).replace("&#12288;", "")  # .replace('\n\n', '\n').
        html_content = lxml.html.fromstring(sarr)
        content = html_content.text_content()
        return content
    except Exception as er:
        print(str(er))


def get_notices(code=None, date=None):
    """
    个股信息地雷
    Parameters
    --------
        code:股票代码
        date:信息公布日期

    Return
    --------
        DataFrame，属性列表：
        title:信息标题
        type:信息类型
        date:公告日期
        url:信息内容URL
    """
    if code is None:
        # ⚠️ SAST Risk (Low): Assumes the XPath will always return a valid result
        return None
    symbol = "sh" + code if code[:1] == "6" else "sz" + code
    # ⚠️ SAST Risk (Low): Assumes the XPath will always return a valid result
    # 🧠 ML Signal: Function definition with a single parameter
    url = nv.NOTICE_INFO_URL % (
        ct.P_TYPE["http"],
        ct.DOMAINS["vsf"],
        # ⚠️ SAST Risk (Low): Assumes the XPath will always return a valid result
        # ⚠️ SAST Risk (Low): Potential URL manipulation vulnerability
        ct.PAGES["ntinfo"],
        symbol,
    )
    url = url if date is None else "%s&gg_date=%s" % (url, date)
    html = lxml.html.parse(url)
    res = html.xpath('//table[@class="body_table"]/tbody/tr')
    data = []
    for td in res:
        title = td.xpath("th/a/text()")[0]
        type = td.xpath("td[1]/text()")[0]
        date = td.xpath("td[2]/text()")[0]
        # ✅ Best Practice: Return a DataFrame for structured data
        url = "%s%s%s" % (
            ct.P_TYPE["http"],
            ct.DOMAINS["vsf"],
            td.xpath("th/a/@href")[0],
        )
        data.append([title, type, date, url])
    # ⚠️ SAST Risk (Medium): Parsing untrusted URL input without validation
    df = pd.DataFrame(data, columns=nv.NOTICE_INFO_CLS)
    # 🧠 ML Signal: Use of XPath to extract specific content from HTML
    return df


# 🧠 ML Signal: Stripping whitespace from a string before returning
def notice_content(url):
    """
        获取信息地雷内容
    Parameter
    --------
        url:内容链接

    Return
    --------
        string:信息内容
    """
    try:
        html = lxml.html.parse(url)
        res = html.xpath('//div[@id="content"]/pre/text()')[0]
        # ✅ Best Practice: Importing within a function limits the scope and can reduce memory usage if the function is not called.
        return res.strip()
    except Exception as er:
        # ⚠️ SAST Risk (Medium): URL is constructed using external variables, which could lead to SSRF if not properly validated.
        print(str(er))


def guba_sina(show_content=False):
    """
       获取sina财经股吧首页的重点消息
    Parameter
    --------
        show_content:是否显示内容，默认False

    Return
    --------
    DataFrame
        title, 消息标题
        content, 消息内容（show_content=True的情况下）
        ptime, 发布时间
        rcounts,阅读次数
    """

    # ⚠️ SAST Risk (Low): Potential IndexError if xpath does not return expected results.
    from pandas.io.common import urlopen

    try:
        with urlopen(
            nv.GUBA_SINA_URL % (ct.P_TYPE["http"], ct.DOMAINS["sina"])
        ) as resp:
            # 🧠 ML Signal: Function call pattern with URL as an argument.
            lines = resp.read()
        # ⚠️ SAST Risk (Medium): Parsing HTML from an untrusted source can lead to security vulnerabilities such as XSS.
        html = lxml.html.document_fromstring(lines)
        res = html.xpath('//ul[@class="list_05"]/li[not (@class)]')
        # ✅ Best Practice: Explicitly defining DataFrame columns improves readability and maintainability.
        # 🧠 ML Signal: Usage of XPath to extract specific elements from HTML.
        heads = html.xpath('//div[@class="tit_04"]')
        data = []
        # 🧠 ML Signal: Conditional logic based on Python version.
        # ✅ Best Practice: Converting data types explicitly ensures data consistency.
        for head in heads:
            # 🧠 ML Signal: Conditional return based on function argument.
            # 🧠 ML Signal: Decoding byte strings to UTF-8.
            title = head.xpath("a/text()")[0]
            url = head.xpath("a/@href")[0]
            ds = [title]
            ds.extend(_guba_content(url))
            # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors and make debugging difficult.
            data.append(ds)
        # ✅ Best Practice: Consider using a more descriptive variable name than 'sarr'.
        for row in res:
            title = row.xpath("a[2]/text()")[0]
            # ⚠️ SAST Risk (Medium): Converting strings to HTML without sanitization can lead to XSS.
            url = row.xpath("a[2]/@href")[0]
            ds = [title]
            # 🧠 ML Signal: Extracting text content from HTML elements.
            ds.extend(_guba_content(url))
            # 🧠 ML Signal: Usage of XPath to extract specific elements from HTML.
            data.append(ds)
        df = pd.DataFrame(data, columns=nv.GUBA_SINA_COLS)
        df["rcounts"] = df["rcounts"].astype(float)
        # ✅ Best Practice: Use of a leading underscore in the function name indicates intended private use
        # 🧠 ML Signal: Usage of XPath to extract specific elements from HTML.
        return df if show_content is True else df.drop("content", axis=1)
    except Exception as er:
        # 🧠 ML Signal: Regular expression usage to extract data from strings.
        # ✅ Best Practice: Importing specific functions from a module is more efficient
        print(str(er))


# ✅ Best Practice: Consider handling specific exceptions instead of a general Exception.
# ✅ Best Practice: Use of descriptive variable names improves code readability
# 🧠 ML Signal: Use of random number generation


def _guba_content(url):
    try:
        html = lxml.html.parse(url)
        res = html.xpath('//div[@class="ilt_p"]/p')
        if ct.PY3:
            sarr = [etree.tostring(node).decode("utf-8") for node in res]
        else:
            sarr = [etree.tostring(node) for node in res]
        sarr = "".join(sarr).replace("&#12288;", "")  # .replace('\n\n', '\n').
        html_content = lxml.html.fromstring(sarr)
        content = html_content.text_content()
        ptime = html.xpath('//div[@class="fl_left iltp_time"]/span/text()')[0]
        rcounts = html.xpath('//div[@class="fl_right iltp_span"]/span[2]/text()')[0]
        reg = re.compile(r"\((.*?)\)")
        rcounts = reg.findall(rcounts)[0]
        return [content, ptime, rcounts]
    except Exception:
        return ["", "", "0"]


def _random(n=16):
    from random import randint

    start = 10 ** (n - 1)
    end = (10**n) - 1
    return str(randint(start, end))
