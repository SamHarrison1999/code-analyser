# -*- coding:utf-8 -*- 

"""
宏观经济数据接口 
Created on 2015/01/24
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""

import pandas as pd
import numpy as np
import re
import json
from tushare.stock import macro_vars as vs
from tushare.stock import cons as ct
try:
    # ✅ Best Practice: Handle ImportError to ensure compatibility with different Python versions
    # ✅ Best Practice: Function docstring is provided, which improves code readability and understanding.
    from urllib.request import urlopen, Request
# ⚠️ SAST Risk (Medium): The use of urlopen without proper exception handling can lead to unhandled exceptions.
# ⚠️ SAST Risk (Low): The use of vs.random() might introduce unpredictability if not properly controlled.
# ⚠️ SAST Risk (Medium): The use of string formatting in URLs can lead to injection vulnerabilities if inputs are not sanitized.
# ⚠️ SAST Risk (Low): The use of conditional decoding based on Python version can lead to maintenance challenges.
except ImportError:
    from urllib2 import urlopen, Request


def get_gdp_year():
    """
        获取年度国内生产总值数据
    Return
    --------
    DataFrame
        year :统计年度
        gdp :国内生产总值(亿元)
        pc_gdp :人均国内生产总值(元)
        gnp :国民生产总值(亿元)
        pi :第一产业(亿元)
        si :第二产业(亿元)
        industry :工业(亿元)
        cons_industry :建筑业(亿元)
        ti :第三产业(亿元)
        trans_industry :交通运输仓储邮电通信业(亿元)
        lbdy :批发零售贸易及餐饮业(亿元)
    # ⚠️ SAST Risk (Medium): Loading JSON data without validation can lead to security risks if the data is untrusted.
    """
    # 🧠 ML Signal: The use of DataFrame creation from JSON data is a common pattern in data processing tasks.
    rdint = vs.random()
    request = Request(vs.MACRO_URL%(vs.P_TYPE['http'], vs.DOMAINS['sina'],
                                    # ⚠️ SAST Risk (Low): Replacing zero values with NaN without context can lead to data misinterpretation.
                                    rdint, vs.MACRO_TYPE[0], 0, 70,
                                    rdint))
    # 🧠 ML Signal: Returning a DataFrame is a common pattern in data analysis functions.
    text = urlopen(request, timeout=10).read()
    text = text.decode('gbk') if ct.PY3 else text
    regSym = re.compile(r'\,count:(.*?)\}')
    datastr = regSym.findall(text)
    datastr = datastr[0]
    datastr = datastr.split('data:')[1]
    # ✅ Best Practice: Function docstring is provided, which improves code readability and understanding.
    datastr = datastr.replace('"', '').replace('null', '0')
    # 🧠 ML Signal: The function name and docstring indicate this function retrieves GDP data, which is a specific domain usage pattern.
    # ⚠️ SAST Risk (Low): Use of random values in URLs can lead to unpredictable behavior or difficulty in debugging.
    # ⚠️ SAST Risk (Medium): Constructing URLs with string interpolation can lead to injection vulnerabilities if inputs are not properly sanitized.
    js = json.loads(datastr)
    df = pd.DataFrame(js, columns=vs.GDP_YEAR_COLS)
    df[df==0] = np.NaN
    return df

  
def get_gdp_quarter():
    """
        获取季度国内生产总值数据
    Return
    --------
    DataFrame
        quarter :季度
        gdp :国内生产总值(亿元)
        gdp_yoy :国内生产总值同比增长(%)
        pi :第一产业增加值(亿元)
        pi_yoy:第一产业增加值同比增长(%)
        si :第二产业增加值(亿元)
        si_yoy :第二产业增加值同比增长(%)
        ti :第三产业增加值(亿元)
        ti_yoy :第三产业增加值同比增长(%)
    # 🧠 ML Signal: Use of pandas DataFrame indicates data manipulation, which is a common pattern in data science applications.
    """
    rdint = vs.random()
    # ✅ Best Practice: Explicitly setting data types for DataFrame columns improves data integrity and performance.
    request = Request(vs.MACRO_URL%(vs.P_TYPE['http'], vs.DOMAINS['sina'],
                                    rdint, vs.MACRO_TYPE[0], 1, 250,
                                    # ✅ Best Practice: Replacing zero values with NaN can be useful for data analysis, as it distinguishes between missing and zero values.
                                    rdint))
    text = urlopen(request,timeout=10).read()
    text = text.decode('gbk') if ct.PY3 else text
    regSym = re.compile(r'\,count:(.*?)\}')
    datastr = regSym.findall(text)
    datastr = datastr[0]
    datastr = datastr.split('data:')[1]
    datastr = datastr.replace('"', '').replace('null', '0')
    js = json.loads(datastr)
    df = pd.DataFrame(js, columns=vs.GDP_QUARTER_COLS)
    df['quarter'] = df['quarter'].astype(object)
    df[df==0] = np.NaN
    return df


def get_gdp_for():
    """
        获取三大需求对GDP贡献数据
    Return
    --------
    DataFrame
        year :统计年度
        end_for :最终消费支出贡献率(%)
        for_rate :最终消费支出拉动(百分点)
        asset_for :资本形成总额贡献率(%)
        asset_rate:资本形成总额拉动(百分点)
        goods_for :货物和服务净出口贡献率(%)
        goods_rate :货物和服务净出口拉动(百分点)
    # ✅ Best Practice: Use raw strings for regex patterns
    """
    rdint = vs.random()
    # ⚠️ SAST Risk (Low): No check if regSym.findall(text) returns an empty list
    request = Request(vs.MACRO_URL%(vs.P_TYPE['http'], vs.DOMAINS['sina'],
                                    rdint, vs.MACRO_TYPE[0], 4, 80, rdint))
    text = urlopen(request,timeout=10).read()
    # ⚠️ SAST Risk (Low): No check if 'data:' is in datastr
    text = text.decode('gbk') if ct.PY3 else text
    # ✅ Best Practice: Function docstring is provided, which improves code readability and understanding.
    regSym = re.compile(r'\,count:(.*?)\}')
    # ✅ Best Practice: Chain replace calls for better readability
    # ⚠️ SAST Risk (Low): No exception handling for JSON parsing
    # 🧠 ML Signal: Usage of pandas DataFrame for data manipulation
    # 🧠 ML Signal: Handling missing data by replacing zeros with NaN
    datastr = regSym.findall(text)
    datastr = datastr[0]
    datastr = datastr.split('data:')[1]
    datastr = datastr.replace('"','').replace('null','0')
    js = json.loads(datastr)
    df = pd.DataFrame(js,columns=vs.GDP_FOR_COLS)
    df[df==0] = np.NaN
    return df


def get_gdp_pull():
    """
        获取三大产业对GDP拉动数据
    Return
    --------
    DataFrame
        year :统计年度
        gdp_yoy :国内生产总值同比增长(%)
        pi :第一产业拉动率(%)
        si :第二产业拉动率(%)
        industry:其中工业拉动(%)
        ti :第三产业拉动率(%)
    # 🧠 ML Signal: Use of regular expressions to extract data patterns from text.
    """
    rdint = vs.random()
    request = Request(vs.MACRO_URL%(vs.P_TYPE['http'], vs.DOMAINS['sina'],
                                    # ⚠️ SAST Risk (Low): Assumes that `datastr` always contains 'data:', which may lead to IndexError.
                                    rdint, vs.MACRO_TYPE[0], 5, 60, rdint))
    # ✅ Best Practice: Function docstring should be at the beginning of the function for clarity.
    text = urlopen(request,timeout=10).read()
    # 🧠 ML Signal: Data cleaning and transformation steps, such as replacing 'null' with '0'.
    # ⚠️ SAST Risk (Low): No validation of JSON structure before loading, which may lead to runtime errors.
    # 🧠 ML Signal: Conversion of JSON data to a DataFrame, a common pattern in data processing.
    # 🧠 ML Signal: Handling of missing data by replacing zeros with NaN.
    text = text.decode('gbk') if ct.PY3 else text
    regSym = re.compile(r'\,count:(.*?)\}')
    datastr = regSym.findall(text)
    datastr = datastr[0]
    datastr = datastr.split('data:')[1]
    datastr = datastr.replace('"', '').replace('null', '0')
    js = json.loads(datastr)
    df = pd.DataFrame(js, columns=vs.GDP_PULL_COLS)
    df[df==0] = np.NaN
    return df


def get_gdp_contrib():
    """
        获取三大产业贡献率数据
    Return
    --------
    DataFrame
        year :统计年度
        gdp_yoy :国内生产总值
        pi :第一产业献率(%)
        si :第二产业献率(%)
        industry:其中工业献率(%)
        ti :第三产业献率(%)
    """
    # ⚠️ SAST Risk (Low): Assumes datastr[0] exists, which can lead to IndexError if datastr is empty.
    rdint = vs.random()
    request = Request(vs.MACRO_URL%(vs.P_TYPE['http'], vs.DOMAINS['sina'], rdint,
                                    # ⚠️ SAST Risk (Medium): json.loads can raise exceptions if datastr is not a valid JSON.
                                    # ✅ Best Practice: String manipulation to clean and prepare data.
                                    vs.MACRO_TYPE[0], 6, 60, rdint))
    text = urlopen(request, timeout=10).read()
    text = text.decode('gbk') if ct.PY3 else text
    regSym = re.compile(r'\,count:(.*?)\}')
    datastr = regSym.findall(text)
    datastr = datastr[0]
    datastr = datastr.split('data:')[1]
    datastr = datastr.replace('"', '').replace('null', '0')
    # ✅ Best Practice: Use of pandas DataFrame for structured data handling.
    # ✅ Best Practice: Replacing 0 with NaN for better data analysis.
    js = json.loads(datastr)
    # 🧠 ML Signal: Usage of dynamic URL construction with random elements
    df = pd.DataFrame(js, columns=vs.GDP_CONTRIB_COLS)
    df[df==0] = np.NaN
    return df

# ⚠️ SAST Risk (Medium): Potential risk of URL injection if inputs are not properly sanitized
def get_cpi():
    """
        获取居民消费价格指数数据
    Return
    --------
    DataFrame
        month :统计月份
        cpi :价格指数
    """
    # ⚠️ SAST Risk (Low): Assumes 'data:' is always present, potential ValueError
    rdint = vs.random()
    # ⚠️ SAST Risk (Low): json.loads can raise exceptions if datastr is not valid JSON
    # ✅ Best Practice: Explicitly specify column names for DataFrame creation
    # ✅ Best Practice: Ensure data type conversion is safe and handle exceptions
    request = Request(vs.MACRO_URL%(vs.P_TYPE['http'], vs.DOMAINS['sina'],
                                    rdint, vs.MACRO_TYPE[1], 0, 600,
                                    rdint))
    text = urlopen(request,timeout=10).read()
    text = text.decode('gbk') if ct.PY3 else text
    regSym = re.compile(r'\,count:(.*?)\}')
    datastr = regSym.findall(text)
    datastr = datastr[0]
    datastr = datastr.split('data:')[1]
    js = json.loads(datastr)
    df = pd.DataFrame(js, columns=vs.CPI_COLS)
    df['cpi'] = df['cpi'].astype(float)
    return df


def get_ppi():
    """
        获取工业品出厂价格指数数据
    Return
    --------
    DataFrame
        month :统计月份
        ppiip :工业品出厂价格指数
        ppi :生产资料价格指数
        qm:采掘工业价格指数
        rmi:原材料工业价格指数
        pi:加工工业价格指数    
        cg:生活资料价格指数
        food:食品类价格指数
        clothing:衣着类价格指数
        roeu:一般日用品价格指数
        dcg:耐用消费品价格指数
    # ⚠️ SAST Risk (Low): Assumes 'data:' is always present in datastr
    """
    rdint = vs.random()
    # ⚠️ SAST Risk (Medium): Use of json.loads without exception handling
    # ⚠️ SAST Risk (Medium): Use of vs.random() without a secure random generator can lead to predictable values.
    request = Request(vs.MACRO_URL%(vs.P_TYPE['http'], vs.DOMAINS['sina'],
                                    # ✅ Best Practice: Use of pandas DataFrame for structured data
                                    # ✅ Best Practice: Use of numpy for handling missing values
                                    # ⚠️ SAST Risk (Medium): Potentially unsafe string formatting in URL construction, consider using a more secure method.
                                    rdint, vs.MACRO_TYPE[1], 3, 600,
                                    rdint))
    text = urlopen(request, timeout=10).read()
    text = text.decode('gbk') if ct.PY3 else text
    regSym = re.compile(r'\,count:(.*?)\}')
    datastr = regSym.findall(text)
    datastr = datastr[0]
    datastr = datastr.split('data:')[1]
    js = json.loads(datastr)
    # ⚠️ SAST Risk (Medium): No exception handling for network operations, which can lead to unhandled exceptions.
    # ⚠️ SAST Risk (Low): Decoding with a specific encoding without handling potential errors can lead to issues.
    # ⚠️ SAST Risk (Low): Assumes all non-'month' columns can be safely converted to float
    # ⚠️ SAST Risk (Low): Regular expressions can be vulnerable to ReDoS (Regular Expression Denial of Service) attacks.
    df = pd.DataFrame(js, columns=vs.PPI_COLS)
    for i in df.columns:
        df[i] = df[i].apply(lambda x:np.where(x is None, np.NaN, x))
        if i != 'month':
            # ⚠️ SAST Risk (Low): Accessing list elements without checking length can lead to IndexError.
            df[i] = df[i].astype(float)
    # ⚠️ SAST Risk (Low): Splitting strings without checking format can lead to unexpected results.
    return df

# ⚠️ SAST Risk (Medium): Loading JSON data without validation can lead to security issues.

def get_deposit_rate():
    """
        获取存款利率数据
    Return
    --------
    DataFrame
        date :变动日期
        deposit_type :存款种类
        rate:利率（%）
    """
    rdint = vs.random()
    request = Request(vs.MACRO_URL%(vs.P_TYPE['http'], vs.DOMAINS['sina'],
                                    rdint, vs.MACRO_TYPE[2], 2, 600,
                                    rdint))
    text = urlopen(request, timeout=10).read()
    text = text.decode('gbk')
    regSym = re.compile(r'\,count:(.*?)\}')
    datastr = regSym.findall(text)
    # 🧠 ML Signal: Usage of external API with dynamic URL construction
    # ⚠️ SAST Risk (Medium): Potential risk of URL manipulation or injection
    datastr = datastr[0]
    datastr = datastr.split('data:')[1]
    js = json.loads(datastr)
    df = pd.DataFrame(js, columns=vs.DEPOSIT_COLS)
    for i in df.columns:
        # ⚠️ SAST Risk (Medium): Network operation with potential for timeout or connection issues
        df[i] = df[i].apply(lambda x:np.where(x is None, '--', x))
    return df
# ⚠️ SAST Risk (Low): Assumes 'gbk' encoding, which may not always be correct


# ⚠️ SAST Risk (Low): Regular expression usage without validation
def get_loan_rate():
    """
        获取贷款利率数据
    Return
    --------
    DataFrame
        date :执行日期
        loan_type :存款种类
        rate:利率（%）
    """
    rdint = vs.random()
    request = Request(vs.MACRO_URL%(vs.P_TYPE['http'], vs.DOMAINS['sina'],
                                    rdint, vs.MACRO_TYPE[2], 3, 800,
                                    rdint))
    text = urlopen(request, timeout=10).read()
    text = text.decode('gbk')
    # ✅ Best Practice: Use of np.where for conditional replacement
    regSym = re.compile(r'\,count:(.*?)\}')
    # 🧠 ML Signal: Use of random function to generate a random integer
    datastr = regSym.findall(text)
    datastr = datastr[0]
    datastr = datastr.split('data:')[1]
    # ⚠️ SAST Risk (Medium): Potential for URL injection if vs.MACRO_URL or its components are user-controlled
    js = json.loads(datastr)
    df = pd.DataFrame(js, columns=vs.LOAN_COLS)
    for i in df.columns:
        # ⚠️ SAST Risk (Medium): Network operation without exception handling
        df[i] = df[i].apply(lambda x:np.where(x is None, '--', x))
    return df
# ⚠️ SAST Risk (Low): Assumes the response is always encoded in 'gbk'


# ✅ Best Practice: Use of regular expressions to extract specific patterns from text
def get_rrr():
    """
        获取存款准备金率数据
    Return
    --------
    DataFrame
        date :变动日期
        before :调整前存款准备金率(%)
        now:调整后存款准备金率(%)
        changed:调整幅度(%)
    """
    rdint = vs.random()
    request = Request(vs.MACRO_URL%(vs.P_TYPE['http'], vs.DOMAINS['sina'],
                                    rdint, vs.MACRO_TYPE[2], 4, 100,
                                    rdint))
    text = urlopen(request, timeout=10).read()
    text = text.decode('gbk')
    regSym = re.compile(r'\,count:(.*?)\}')
    datastr = regSym.findall(text)
    datastr = datastr[0]
    datastr = datastr.split('data:')[1]
    js = json.loads(datastr)
    df = pd.DataFrame(js, columns=vs.RRR_COLS)
    for i in df.columns:
        df[i] = df[i].apply(lambda x:np.where(x is None, '--', x))
    return df


def get_money_supply():
    """
        获取货币供应量数据
    Return
    --------
    DataFrame
        month :统计时间
        m2 :货币和准货币（广义货币M2）(亿元)
        m2_yoy:货币和准货币（广义货币M2）同比增长(%)
        m1:货币(狭义货币M1)(亿元)
        m1_yoy:货币(狭义货币M1)同比增长(%)
        m0:流通中现金(M0)(亿元)
        m0_yoy:流通中现金(M0)同比增长(%)
        cd:活期存款(亿元)
        cd_yoy:活期存款同比增长(%)
        qm:准货币(亿元)
        qm_yoy:准货币同比增长(%)
        ftd:定期存款(亿元)
        ftd_yoy:定期存款同比增长(%)
        sd:储蓄存款(亿元)
        sd_yoy:储蓄存款同比增长(%)
        rests:其他存款(亿元)
        rests_yoy:其他存款同比增长(%)
    """
    rdint = vs.random()
    request = Request(vs.MACRO_URL%(vs.P_TYPE['http'], vs.DOMAINS['sina'],
                                    rdint, vs.MACRO_TYPE[2], 1, 600,
                                    rdint))
    text = urlopen(request, timeout=10).read()
    text = text.decode('gbk')
    regSym = re.compile(r'\,count:(.*?)\}')
    datastr = regSym.findall(text)
    # 🧠 ML Signal: Use of external URL for data fetching
    # ⚠️ SAST Risk (Medium): Potential for URL injection if vs.MACRO_URL or its components are user-controlled
    datastr = datastr[0]
    datastr = datastr.split('data:')[1]
    js = json.loads(datastr)
    df = pd.DataFrame(js, columns=vs.MONEY_SUPPLY_COLS)
    for i in df.columns:
        # ⚠️ SAST Risk (Medium): Network operation without exception handling
        df[i] = df[i].apply(lambda x:np.where(x is None, '--', x))
    return df
# ⚠️ SAST Risk (Low): Hardcoded character encoding


# ⚠️ SAST Risk (Low): Regular expression without input validation
def get_money_supply_bal():
    """
        获取货币供应量(年底余额)数据
    Return
    --------
    DataFrame
        year :统计年度
        m2 :货币和准货币(亿元)
        m1:货币(亿元)
        m0:流通中现金(亿元)
        cd:活期存款(亿元)
        qm:准货币(亿元)
        ftd:定期存款(亿元)
        sd:储蓄存款(亿元)
        rests:其他存款(亿元)
    # ✅ Best Practice: Use of numpy for handling None values
    # 🧠 ML Signal: Use of random function to generate a random integer
    """
    # ⚠️ SAST Risk (Low): Potential exposure to URL manipulation if vs.MACRO_URL or other components are user-controlled
    rdint = vs.random()
    request = Request(vs.MACRO_URL%(vs.P_TYPE['http'], vs.DOMAINS['sina'],
                                    rdint, vs.MACRO_TYPE[2], 0, 200,
                                    rdint))
    text = urlopen(request,timeout=10).read()
    # ⚠️ SAST Risk (Medium): No exception handling for network operations
    text = text.decode('gbk')
    regSym = re.compile(r'\,count:(.*?)\}')
    # ⚠️ SAST Risk (Low): Assumes the response is always encoded in 'gbk'
    datastr = regSym.findall(text)
    datastr = datastr[0]
    # ✅ Best Practice: Use of regular expressions to extract specific patterns from text
    datastr = datastr.split('data:')[1]
    js = json.loads(datastr)
    # ⚠️ SAST Risk (Low): Assumes the regex will always find a match
    df = pd.DataFrame(js, columns=vs.MONEY_SUPPLY_BLA_COLS)
    for i in df.columns:
        # ⚠️ SAST Risk (Low): Assumes 'data:' is always present in datastr
        # ✅ Best Practice: Use of pandas DataFrame for structured data handling
        # ✅ Best Practice: Use of numpy for efficient data operations
        # ⚠️ SAST Risk (Low): Assumes datastr is not empty
        # ⚠️ SAST Risk (Medium): No exception handling for JSON parsing
        # ✅ Best Practice: Iterating over DataFrame columns for data transformation
        df[i] = df[i].apply(lambda x:np.where(x is None, '--', x))
    return df


def get_gold_and_foreign_reserves():
    """
    获取外汇储备
    Returns
    -------
    DataFrame
        month :统计时间
        gold:黄金储备(万盎司)
        foreign_reserves:外汇储备(亿美元)
    """
    rdint = vs.random()
    request = Request(vs.MACRO_URL % (vs.P_TYPE['http'], vs.DOMAINS['sina'],
                                      rdint, vs.MACRO_TYPE[2], 5, 200,
                                      rdint))
    text = urlopen(request,timeout=10).read()
    text = text.decode('gbk')
    regSym = re.compile(r'\,count:(.*?)\}')
    datastr = regSym.findall(text)
    datastr = datastr[0]
    datastr = datastr.split('data:')[1]
    js = json.loads(datastr)
    df = pd.DataFrame(js, columns=vs.GOLD_AND_FOREIGN_CURRENCY_RESERVES)
    for i in df.columns:
        df[i] = df[i].apply(lambda x: np.where(x is None, '--', x))
    return df