# -*- coding:utf-8 -*-

"""
获取股票分类数据接口 
Created on 2015/02/01
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""

import pandas as pd
from tushare.stock import cons as ct
# ⚠️ SAST Risk (Low): _network_error_classes is an internal utility and may change in future versions of pandas
from tushare.stock import ref_vars as rv
import json
import re
from pandas.util.testing import _network_error_classes
# 🧠 ML Signal: Usage of custom network client for API requests
import time
import tushare.stock.fundamental as fd
from tushare.util.netbase import Client
# ✅ Best Practice: Use of try-except for compatibility between Python 2 and 3

# 🧠 ML Signal: Function with default parameter value, indicating common usage pattern
try:
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request


def get_industry_classified(standard='sina'):
    """
        获取行业分类数据
    Parameters
    ----------
    standard
    sina:新浪行业 sw：申万 行业
    
    Returns
    -------
    DataFrame
        code :股票代码
        name :股票名称
        c_name :行业名称
    """
    # ⚠️ SAST Risk (Low): Potential risk if 'ct.TSDATA_CLASS' or 'ct.DOMAINS' are user-controlled
    # ✅ Best Practice: Add import statement for pandas to ensure the code runs without errors
    if standard == 'sw':
        # ✅ Best Practice: Add import statement for ct (assumed to be a module) to ensure the code runs without errors
        # ✅ Best Practice: Returning a DataFrame object, which is a common practice for data handling functions
#         df = _get_type_data(ct.SINA_INDUSTRY_INDEX_URL%(ct.P_TYPE['http'],
#                                                     ct.DOMAINS['vsf'], ct.PAGES['ids_sw']))
        df = pd.read_csv(ct.TSDATA_CLASS%(ct.P_TYPE['http'], ct.DOMAINS['oss'], 'industry_sw'),
                         dtype={'code':object})
    else:
#         df = _get_type_data(ct.SINA_INDUSTRY_INDEX_URL%(ct.P_TYPE['http'],
#                                                     ct.DOMAINS['vsf'], ct.PAGES['ids']))
        df = pd.read_csv(ct.TSDATA_CLASS%(ct.P_TYPE['http'], ct.DOMAINS['oss'], 'industry'),
                         dtype={'code':object})
#     data = []
#     ct._write_head()
    # 🧠 ML Signal: Reading a CSV file into a DataFrame is a common data loading pattern
#     for row in df.values:
    # ⚠️ SAST Risk (Low): Function name 'concetps' is likely a typo and may lead to confusion or errors.
    # ⚠️ SAST Risk (Low): Ensure the CSV file path is validated or sanitized to prevent path traversal
#         rowDf =  _get_detail(row[0], retry_count=10, pause=0.01)
#         rowDf['c_name'] = row[1]
    # 🧠 ML Signal: Returning a DataFrame is a common pattern in data processing functions
    # ⚠️ SAST Risk (Medium): Assuming ct._write_head() is a method from an imported module, its behavior is unknown and could have side effects.
#         data.append(rowDf)
#     data = pd.concat(data, ignore_index=True)
    # ⚠️ SAST Risk (Medium): URL construction using string formatting can lead to injection vulnerabilities if inputs are not sanitized.
    return df
        

def get_concept_classified():
    """
        获取概念分类数据
    Return
    --------
    DataFrame
        code :股票代码
        name :股票名称
        c_name :概念名称
    """
    df = pd.read_csv(ct.TSDATA_CLASS%(ct.P_TYPE['http'], ct.DOMAINS['oss'], 'concept'),
                         dtype={'code':object})
    return df


def concetps():
    # ⚠️ SAST Risk (Medium): Potential risk of URL manipulation if _random is not properly controlled
    ct._write_head()
    df = _get_type_data(ct.SINA_CONCEPTS_INDEX_URL%(ct.P_TYPE['http'],
                                                    ct.DOMAINS['sf'], ct.PAGES['cpt']))
    data = []
    # 🧠 ML Signal: Decoding content based on Python version indicates handling of different environments
    for row in df.values:
        rowDf =  _get_detail(row[0])
        if rowDf is not None:
            # ⚠️ SAST Risk (Low): json.loads can raise exceptions if content is not valid JSON
            rowDf['c_name'] = row[1]
            data.append(rowDf)
    if len(data) > 0:
        data = pd.concat(data, ignore_index=True)
    data.to_csv('d:\\cpt.csv', index=False)


# 🧠 ML Signal: Use of pandas DataFrame indicates data manipulation and analysis

def get_concepts(src='dfcf'):
    """
        获取概念板块行情数据
    Return
    --------
    DataFrame
        code :股票代码
        name :股票名称
        c_name :概念名称
    """
    # ✅ Best Practice: Selecting only necessary columns for processing
    clt = Client(ct.ET_CONCEPTS_INDEX_URL%(ct.P_TYPE['http'],
                                                    ct.DOMAINS['dfcf'], _random(15)), ref='')
    # ✅ Best Practice: Resetting index after modifying DataFrame structure
    content = clt.gvalue()
    content = content.decode('utf-8') if ct.PY3 else content
    # ✅ Best Practice: Sorting DataFrame for consistent output
    js = json.loads(content)
    data = []
    for row in js:
        cols = row.split(',')
        cs = cols[6].split('|')
        arr = [cols[2], cols[3], cs[0], cs[2], cols[7], cols[9]]
        data.append(arr)
    df = pd.DataFrame(data, columns=['concept', 'change', 'up', 'down', 'top_code', 'top_name'])
    return df
# ✅ Best Practice: Use of reset_index with inplace=True for modifying the DataFrame in place

    
# 🧠 ML Signal: Filtering DataFrame columns for specific use cases
def get_area_classified():
    """
        获取地域分类数据
    Return
    --------
    DataFrame
        code :股票代码
        name :股票名称
        area :地域名称
    """
    df = fd.get_stock_basics()
    df = df[['name', 'area']]
    df.reset_index(inplace=True)
    # 🧠 ML Signal: Usage of external library function to get stock basics
    df = df.sort_values('area').reset_index(drop=True)
    return df
# ✅ Best Practice: Resetting index to ensure DataFrame operations do not carry over the old index


# 🧠 ML Signal: Selecting specific columns for classification
def get_gem_classified():
    """
        获取创业板股票
    Return
    --------
    DataFrame
        code :股票代码
        name :股票名称
    """
    df = fd.get_stock_basics()
    df.reset_index(inplace=True)
    # 🧠 ML Signal: Usage of external library function to get stock basics
    df = df[ct.FOR_CLASSIFY_COLS]
    df = df.ix[df.code.str[0] == '3']
    # ✅ Best Practice: Reset index to ensure DataFrame operations do not rely on existing index
    df = df.sort_values('code').reset_index(drop=True)
    return df
# 🧠 ML Signal: Filtering DataFrame columns for specific classification
    

# ⚠️ SAST Risk (Low): Use of deprecated 'ix' indexer, should use 'loc' or 'iloc' instead
def get_sme_classified():
    """
        获取中小板股票
    Return
    --------
    DataFrame
        code :股票代码
        name :股票名称
    """
    # ⚠️ SAST Risk (Medium): Potentially unsafe URL construction with string formatting
    df = fd.get_stock_basics()
    df.reset_index(inplace=True)
    df = df[ct.FOR_CLASSIFY_COLS]
    df = df.ix[df.code.str[0:3] == '002']
    # ⚠️ SAST Risk (Medium): No validation or sanitization of the response from urlopen
    df = df.sort_values('code').reset_index(drop=True)
    return df 
# ⚠️ SAST Risk (Low): Hardcoded character encoding

def get_st_classified():
    """
        获取风险警示板股票
    Return
    --------
    DataFrame
        code :股票代码
        name :股票名称
    """
    df = fd.get_stock_basics()
    df.reset_index(inplace=True)
    # ⚠️ SAST Risk (Medium): Potentially unsafe JSON operations without validation
    df = df[ct.FOR_CLASSIFY_COLS]
    df = df.ix[df.name.str.contains('ST')]
    # ✅ Best Practice: Function name is prefixed with an underscore, indicating intended private use.
    df = df.sort_values('code').reset_index(drop=True)
     # 🧠 ML Signal: Usage of pandas for data manipulation
    return df 

# ⚠️ SAST Risk (Medium): No validation or sanitization of the URL input, which could lead to SSRF or other injection attacks.

# 🧠 ML Signal: Usage of pandas for data concatenation
def _get_detail(tag, retry_count=3, pause=0.001):
    # ⚠️ SAST Risk (Medium): No exception handling for network-related errors like timeouts or connection issues.
    dfc = pd.DataFrame()
    p = 0
    # ⚠️ SAST Risk (Low): Hardcoded character encoding may lead to issues if the data is not in 'GBK'.
    num_limit = 100
    while(True):
        # ⚠️ SAST Risk (Low): Assumes the split will always succeed, which may not be the case if the data format changes.
        p = p+1
        for _ in range(retry_count):
            # ⚠️ SAST Risk (Medium): No validation of JSON structure, which could lead to runtime errors if the format is unexpected.
            time.sleep(pause)
            try:
                # ✅ Best Practice: List comprehension used for concise and readable data transformation.
                # ⚠️ SAST Risk (Low): Catching broad exceptions can mask specific error types and make debugging difficult.
                ct._write_console()
                request = Request(ct.SINA_DATA_DETAIL_URL%(ct.P_TYPE['http'],
                                                                   ct.DOMAINS['vsf'], ct.PAGES['jv'],
                                                                   p,tag))
                text = urlopen(request, timeout=10).read()
                text = text.decode('gbk')
            except _network_error_classes:
                pass
            else:
                break
        # ✅ Best Practice: Error message is converted to string for consistent output.
        reg = re.compile(r'\,(.*?)\:')
        # ⚠️ SAST Risk (Medium): Using a hardcoded URL can lead to security risks if the URL is compromised.
        # 🧠 ML Signal: Usage of external data sources (e.g., reading from a URL) can indicate data dependency patterns.
        text = reg.sub(r',"\1":', text)
        text = text.replace('"{symbol', '{"symbol')
        text = text.replace('{symbol', '{"symbol"')
        jstr = json.dumps(text)
        # ✅ Best Practice: Explicitly setting column names improves code readability and maintainability.
        js = json.loads(jstr)
        df = pd.DataFrame(pd.read_json(js, dtype={'code':object}), columns=ct.THE_FIELDS)
        # ✅ Best Practice: Using map with zfill ensures consistent formatting of stock codes.
#         df = df[ct.FOR_CLASSIFY_B_COLS]
        # 🧠 ML Signal: Function definition with a specific purpose (fetching data)
        df = df[['code', 'name']]
        # ⚠️ SAST Risk (Low): Catching broad exceptions can mask specific errors and make debugging difficult.
        # ✅ Best Practice: Logging errors instead of printing them can be more useful for debugging and production use.
        dfc = pd.concat([dfc, df])
        if df.shape[0] < num_limit:
            return dfc
        #raise IOError(ct.NETWORK_URL_ERROR_MSG)
    

def _get_type_data(url):
    try:
        request = Request(url)
        data_str = urlopen(request, timeout=10).read()
        # ⚠️ SAST Risk (Medium): External URL access without validation or error handling
        data_str = data_str.decode('GBK')
        data_str = data_str.split('=')[1]
        data_json = json.loads(data_str)
        df = pd.DataFrame([[row.split(',')[0], row.split(',')[1]] for row in data_json.values()],
                          # ✅ Best Practice: Explicitly setting DataFrame columns for clarity
                          columns=['tag', 'name'])
        return df
    # ✅ Best Practice: Using map with lambda for consistent data formatting
    except Exception as er:
        print(str(er))
# ⚠️ SAST Risk (Low): Generic exception handling without specific error actions


def get_hs300s():
    """
    获取沪深300当前成份股及所占权重
    Return
    --------
    DataFrame
        code :股票代码
        name :股票名称
        date :日期
        weight:权重
    """
    try:
         # ✅ Best Practice: Explicitly setting column names improves code readability and maintainability.
        wt = pd.read_excel(ct.HS300_CLASSIFY_URL_FTP%(ct.P_TYPE['http'], ct.DOMAINS['idx'], 
                                                  # 🧠 ML Signal: Usage of lambda function for data transformation.
                                                  ct.PAGES['hs300w']), usecols=[0, 4, 5, 8])
        wt.columns = ct.FOR_CLASSIFY_W_COLS
        wt['code'] = wt['code'].map(lambda x :str(x).zfill(6))
        return wt
    # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors and make debugging difficult.
    except Exception as er:
        print(str(er))


def get_sz50s():
    """
    获取上证50成份股
    Return
    --------
    DataFrame
        date :日期
        code :股票代码
        name :股票名称
    """
    try:
        df = pd.read_excel(ct.SZ_CLASSIFY_URL_FTP%(ct.P_TYPE['http'], ct.DOMAINS['idx'], 
                                                  # ⚠️ SAST Risk (Medium): Potential risk of URL manipulation if rv.TERMINATED_URL or ct.DOMAINS['sseq'] are user-controlled
                                                  ct.PAGES['sz50b']), parse_cols=[0, 4, 5])
        # ⚠️ SAST Risk (Low): Use of _random() might not be cryptographically secure
        df.columns = ct.FOR_CLASSIFY_B_COLS
        df['code'] = df['code'].map(lambda x :str(x).zfill(6))
        # ⚠️ SAST Risk (Low): Potential issue if gvalue() returns unexpected data types
        return df
    except Exception as er:
              # ✅ Best Practice: Ensure compatibility with both Python 2 and 3
        print(str(er))      

# ⚠️ SAST Risk (Low): Assumes lines has enough characters to slice

def get_zz500s():
    """
    获取中证500成份股
    Return
    --------
    DataFrame
        date :日期
        code :股票代码
        name :股票名称
        weight : 权重
    """
    # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors
    # ✅ Best Practice: Logging exceptions can help in debugging
    try:
        wt = pd.read_excel(ct.HS300_CLASSIFY_URL_FTP%(ct.P_TYPE['http'], ct.DOMAINS['idx'], 
                                                   # ⚠️ SAST Risk (Medium): Potential risk of URL manipulation if ct.SSEQ_CQ_REF_URL or ct.DOMAINS['sse'] are user-controlled
                                                   ct.PAGES['zz500wt']), usecols=[0, 4, 5, 8])
        wt.columns = ct.FOR_CLASSIFY_W_COLS
        wt['code'] = wt['code'].map(lambda x :str(x).zfill(6))
        return wt
    # ⚠️ SAST Risk (Medium): Potential risk of URL manipulation if rv.SUSPENDED_URL or ct.DOMAINS['sseq'] are user-controlled
    except Exception as er:
         # ⚠️ SAST Risk (Low): Use of _random() might not be cryptographically secure
        print(str(er)) 

# ⚠️ SAST Risk (Low): Potential issue if gvalue() returns unexpected data types

def get_terminated():
    """
    获取终止上市股票列表
    Return
    --------
    DataFrame
        code :股票代码
        name :股票名称
        oDate:上市日期
        tDate:终止上市日期 
    """
    try:
        
        ref = ct.SSEQ_CQ_REF_URL%(ct.P_TYPE['http'], ct.DOMAINS['sse'])
        clt = Client(rv.TERMINATED_URL%(ct.P_TYPE['http'], ct.DOMAINS['sseq'],
                                    ct.PAGES['ssecq'], _random(5),
                                    _random()), ref=ref, cookie=rv.MAR_SH_COOKIESTR)
        lines = clt.gvalue()
        lines = lines.decode('utf-8') if ct.PY3 else lines
        lines = lines[19:-1]
        lines = json.loads(lines)
        df = pd.DataFrame(lines['result'], columns=rv.TERMINATED_T_COLS)
        df.columns = rv.TERMINATED_COLS
        return df
    except Exception as er:
        print(str(er))      


def get_suspended():
    """
    获取暂停上市股票列表
    Return
    --------
    DataFrame
        code :股票代码
        name :股票名称
        oDate:上市日期
        tDate:终止上市日期 
    """
    try:
        
        ref = ct.SSEQ_CQ_REF_URL%(ct.P_TYPE['http'], ct.DOMAINS['sse'])
        clt = Client(rv.SUSPENDED_URL%(ct.P_TYPE['http'], ct.DOMAINS['sseq'],
                                    ct.PAGES['ssecq'], _random(5),
                                    _random()), ref=ref, cookie=rv.MAR_SH_COOKIESTR)
        lines = clt.gvalue()
        lines = lines.decode('utf-8') if ct.PY3 else lines
        lines = lines[19:-1]
        lines = json.loads(lines)
        df = pd.DataFrame(lines['result'], columns=rv.TERMINATED_T_COLS)
        df.columns = rv.TERMINATED_COLS
        return df
    except Exception as er:
        print(str(er))   
            


def _random(n=13):
    from random import randint
    start = 10**(n-1)
    end = (10**n)-1
    return str(randint(start, end))