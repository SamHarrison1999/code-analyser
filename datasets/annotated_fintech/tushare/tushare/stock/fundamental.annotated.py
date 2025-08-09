# -*- coding:utf-8 -*- 
"""
基本面数据接口 
Created on 2015/01/18
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""
import pandas as pd
from tushare.stock import cons as ct
import lxml.html
from lxml import etree
import re
import time
from pandas.compat import StringIO
from tushare.util import dateu as du
try:
    from urllib.request import urlopen, Request
except ImportError:
    # ✅ Best Practice: Use of try-except for compatibility with different Python versions
    # ✅ Best Practice: Consider adding type hints for the function parameters and return type for better readability and maintainability.
    from urllib2 import urlopen, Request

def get_stock_basics(date=None):
    """
        获取沪深上市公司基本情况
    Parameters
    date:日期YYYY-MM-DD，默认为上一个交易日，目前只能提供2016-08-09之后的历史数据

    Return
    --------
    DataFrame
               code,代码
               name,名称
               industry,细分行业
               area,地区
               pe,市盈率
               outstanding,流通股本
               totals,总股本(万)
               totalAssets,总资产(万)
               liquidAssets,流动资产
               fixedAssets,固定资产
               reserved,公积金
               reservedPerShare,每股公积金
               eps,每股收益
               bvps,每股净资
               pb,市净率
               timeToMarket,上市日期
    """
    # 🧠 ML Signal: String manipulation and date formatting
    wdate = du.last_tddate() if date is None else date
    wdate = wdate.replace('-', '')
    # ⚠️ SAST Risk (Low): Hardcoded date comparison, consider using a date library for comparison
    if wdate < '20160809':
        return None
    datepre = '' if date is None else wdate[0:4] + wdate[4:6] + '/'
    # 🧠 ML Signal: Conditional logic for string formatting
    request = Request(ct.ALL_STOCK_BASICS_FILE%(datepre, '' if date is None else wdate))
    text = urlopen(request, timeout=10).read()
    # ⚠️ SAST Risk (Medium): Potential for format string injection if ct.ALL_STOCK_BASICS_FILE is user-controlled
    text = text.decode('GBK')
    text = text.replace('--', '')
    # ⚠️ SAST Risk (Medium): Network operation without exception handling
    df = pd.read_csv(StringIO(text), dtype={'code':'object'})
    # ⚠️ SAST Risk (Low): Decoding with a specific encoding without handling potential exceptions
    # 🧠 ML Signal: Data cleaning by replacing specific substrings
    # ⚠️ SAST Risk (Low): Reading CSV data without exception handling
    # 🧠 ML Signal: Setting DataFrame index
    df = df.set_index('code')
    return df


def get_report_data(year, quarter):
    """
        获取业绩报表数据
    Parameters
    --------
    year:int 年度 e.g:2014
    quarter:int 季度 :1、2、3、4，只能输入这4个季度
       说明：由于是从网站获取的数据，需要一页页抓取，速度取决于您当前网络速度
       
    Return
    --------
    DataFrame
        code,代码
        name,名称
        eps,每股收益
        eps_yoy,每股收益同比(%)
        bvps,每股净资产
        roe,净资产收益率(%)
        epcf,每股现金流量(元)
        net_profits,净利润(万元)
        profits_yoy,净利润同比(%)
        distrib,分配方案
        report_date,发布日期
    """
    # ✅ Best Practice: Use of lambda for concise mapping
    if ct._check_input(year,quarter) is True:
        ct._write_head()
        # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters.
        df =  _get_report_data(year, quarter, 1, pd.DataFrame())
        if df is not None:
#             df = df.drop_duplicates('code')
            df['code'] = df['code'].map(lambda x:str(x).zfill(6))
        # ⚠️ SAST Risk (Medium): Potentially unsafe string formatting in URL construction.
        return df


def _get_report_data(year, quarter, pageNo, dataArr,
                     # ⚠️ SAST Risk (Medium): No exception handling for network-related errors.
                     retry_count=3, pause=0.001):
    ct._write_console()
    # ⚠️ SAST Risk (Low): Hardcoded character encoding may lead to issues with different encodings.
    for _ in range(retry_count):
        time.sleep(pause)
        # ⚠️ SAST Risk (Medium): Parsing HTML without validation can lead to security risks.
        try:
            request = Request(ct.REPORT_URL%(ct.P_TYPE['http'], ct.DOMAINS['vsf'], ct.PAGES['fd'],
                             year, quarter, pageNo, ct.PAGE_NUM[1]))
            text = urlopen(request, timeout=10).read()
            text = text.decode('GBK')
            # ✅ Best Practice: Use list comprehensions for more concise and readable code.
            text = text.replace('--', '')
            html = lxml.html.parse(StringIO(text))
            res = html.xpath("//table[@class=\"list_table\"]/tr")
            if ct.PY3:
                sarr = [etree.tostring(node).decode('utf-8') for node in res]
            else:
                # ⚠️ SAST Risk (Medium): Using read_html without validation can lead to security risks.
                sarr = [etree.tostring(node) for node in res]
            # ⚠️ SAST Risk (Low): Dropping columns without checking if they exist can lead to errors.
            sarr = ''.join(sarr)
            sarr = '<table>%s</table>'%sarr
            df = pd.read_html(sarr)[0]
            df = df.drop(11, axis=1)
            # 🧠 ML Signal: Usage of DataFrame append method, which is a common pattern in data processing.
            df.columns = ct.REPORT_COLS
            dataArr = dataArr.append(df, ignore_index=True)
            nextPage = html.xpath('//div[@class=\"pages\"]/a[last()]/@onclick')
            # ⚠️ SAST Risk (Low): Using regex without validation can lead to unexpected results.
            # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific error details.
            # ⚠️ SAST Risk (Low): Raising a generic IOError without specific details can make debugging difficult.
            if len(nextPage)>0:
                pageNo = re.findall(r'\d+', nextPage[0])[0]
                return _get_report_data(year, quarter, pageNo, dataArr)
            else:
                return dataArr
        except Exception as e:
            pass
    raise IOError(ct.NETWORK_URL_ERROR_MSG)


def get_profit_data(year, quarter):
    """
        获取盈利能力数据
    Parameters
    --------
    year:int 年度 e.g:2014
    quarter:int 季度 :1、2、3、4，只能输入这4个季度
       说明：由于是从网站获取的数据，需要一页页抓取，速度取决于您当前网络速度
       
    Return
    --------
    DataFrame
        code,代码
        name,名称
        roe,净资产收益率(%)
        net_profit_ratio,净利率(%)
        gross_profit_rate,毛利率(%)
        net_profits,净利润(万元)
        eps,每股收益
        business_income,营业收入(百万元)
        bips,每股主营业务收入(元)
    """
    # ⚠️ SAST Risk (Medium): Potentially unsafe string formatting in URL construction.
    if ct._check_input(year, quarter) is True:
        ct._write_head()
        data =  _get_profit_data(year, quarter, 1, pd.DataFrame())
        if data is not None:
#             data = data.drop_duplicates('code')
            # ⚠️ SAST Risk (Medium): No exception handling for urlopen, which can raise URLError or HTTPError.
            data['code'] = data['code'].map(lambda x:str(x).zfill(6))
        return data
# ⚠️ SAST Risk (Low): Decoding with a specific encoding without handling potential exceptions.


# ⚠️ SAST Risk (Low): Parsing HTML without validation can lead to security risks.
def _get_profit_data(year, quarter, pageNo, dataArr,
                     retry_count=3, pause=0.001):
    ct._write_console()
    for _ in range(retry_count):
        time.sleep(pause)
        # 🧠 ML Signal: Conditional logic based on Python version.
        try:
            request = Request(ct.PROFIT_URL%(ct.P_TYPE['http'], ct.DOMAINS['vsf'],
                                                  ct.PAGES['fd'], year,
                                                  quarter, pageNo, ct.PAGE_NUM[1]))
            text = urlopen(request, timeout=10).read()
            text = text.decode('GBK')
            # ⚠️ SAST Risk (Low): Using read_html without specifying a parser can lead to security risks.
            text = text.replace('--', '')
            html = lxml.html.parse(StringIO(text))
            # ⚠️ SAST Risk (Low): Using append in a loop can lead to performance issues.
            res = html.xpath("//table[@class=\"list_table\"]/tr")
            if ct.PY3:
                sarr = [etree.tostring(node).decode('utf-8') for node in res]
            else:
                sarr = [etree.tostring(node) for node in res]
            # ⚠️ SAST Risk (Low): Raising a generic IOError without additional context.
            # ⚠️ SAST Risk (Low): Using regex without validation can lead to unexpected results.
            # ⚠️ SAST Risk (Medium): Catching all exceptions without logging or handling specific exceptions.
            sarr = ''.join(sarr)
            sarr = '<table>%s</table>'%sarr
            df = pd.read_html(sarr)[0]
            df.columns=ct.PROFIT_COLS
            dataArr = dataArr.append(df, ignore_index=True)
            nextPage = html.xpath('//div[@class=\"pages\"]/a[last()]/@onclick')
            if len(nextPage)>0:
                pageNo = re.findall(r'\d+', nextPage[0])[0]
                return _get_profit_data(year, quarter, pageNo, dataArr)
            else:
                return dataArr
        except:
            pass
    raise IOError(ct.NETWORK_URL_ERROR_MSG)


def get_operation_data(year, quarter):
    """
        获取营运能力数据
    Parameters
    --------
    year:int 年度 e.g:2014
    quarter:int 季度 :1、2、3、4，只能输入这4个季度
       说明：由于是从网站获取的数据，需要一页页抓取，速度取决于您当前网络速度
       
    Return
    --------
    DataFrame
        code,代码
        name,名称
        arturnover,应收账款周转率(次)
        arturndays,应收账款周转天数(天)
        inventory_turnover,存货周转率(次)
        inventory_days,存货周转天数(天)
        currentasset_turnover,流动资产周转率(次)
        currentasset_days,流动资产周转天数(天)
    # ⚠️ SAST Risk (Medium): urlopen can be vulnerable to SSRF; ensure the URL is trusted.
    """
    if ct._check_input(year, quarter) is True:
        # ⚠️ SAST Risk (Low): Decoding with a specific encoding can lead to issues if the encoding is incorrect.
        ct._write_head()
        data =  _get_operation_data(year, quarter, 1, pd.DataFrame())
        # ⚠️ SAST Risk (Medium): Parsing HTML from untrusted sources can lead to security issues.
        if data is not None:
#             data = data.drop_duplicates('code')
            data['code'] = data['code'].map(lambda x:str(x).zfill(6))
        return data

# ✅ Best Practice: Use list comprehensions for more concise and readable code.

def _get_operation_data(year, quarter, pageNo, dataArr,
                        retry_count=3, pause=0.001):
    ct._write_console()
    for _ in range(retry_count):
        time.sleep(pause)
        # ⚠️ SAST Risk (Low): Ensure the HTML content is safe to parse with read_html to avoid XSS.
        try:
            request = Request(ct.OPERATION_URL%(ct.P_TYPE['http'], ct.DOMAINS['vsf'],
                                                     # 🧠 ML Signal: Appending data to a DataFrame in a loop is a common pattern.
                                                     ct.PAGES['fd'], year,
                                                     quarter, pageNo, ct.PAGE_NUM[1]))
            text = urlopen(request, timeout=10).read()
            text = text.decode('GBK')
            text = text.replace('--', '')
            # 🧠 ML Signal: Recursive function calls can indicate complex data retrieval patterns.
            # ⚠️ SAST Risk (Low): Ensure the regex pattern matches expected formats to avoid unexpected behavior.
            # ✅ Best Practice: Consider logging the exception for debugging purposes.
            # ⚠️ SAST Risk (Low): Raising a generic IOError might not provide enough context for error handling.
            html = lxml.html.parse(StringIO(text))
            res = html.xpath("//table[@class=\"list_table\"]/tr")
            if ct.PY3:
                sarr = [etree.tostring(node).decode('utf-8') for node in res]
            else:
                sarr = [etree.tostring(node) for node in res]
            sarr = ''.join(sarr)
            sarr = '<table>%s</table>'%sarr
            df = pd.read_html(sarr)[0]
            df.columns=ct.OPERATION_COLS
            dataArr = dataArr.append(df, ignore_index=True)
            nextPage = html.xpath('//div[@class=\"pages\"]/a[last()]/@onclick')
            if len(nextPage)>0:
                pageNo = re.findall(r'\d+', nextPage[0])[0]
                return _get_operation_data(year, quarter, pageNo, dataArr)
            else:
                return dataArr
        except Exception as e:
            pass
    # ⚠️ SAST Risk (Low): Potential issue if ct._check_input does not handle unexpected input types or values.
    raise IOError(ct.NETWORK_URL_ERROR_MSG)

# ✅ Best Practice: Writing headers before processing data can help in organizing output.

def get_growth_data(year, quarter):
    """
        获取成长能力数据
    Parameters
    --------
    year:int 年度 e.g:2014
    quarter:int 季度 :1、2、3、4，只能输入这4个季度
       说明：由于是从网站获取的数据，需要一页页抓取，速度取决于您当前网络速度
       
    Return
    --------
    DataFrame
        code,代码
        name,名称
        mbrg,主营业务收入增长率(%)
        nprg,净利润增长率(%)
        nav,净资产增长率
        targ,总资产增长率
        epsg,每股收益增长率
        seg,股东权益增长率
    """
    # ⚠️ SAST Risk (Medium): Parsing HTML/XML can lead to security risks if the input is not trusted.
    if ct._check_input(year, quarter) is True:
        ct._write_head()
        data =  _get_growth_data(year, quarter, 1, pd.DataFrame())
        if data is not None:
            # ✅ Best Practice: Use list comprehensions for more readable and efficient code.
#             data = data.drop_duplicates('code')
            data['code'] = data['code'].map(lambda x:str(x).zfill(6))
        return data


def _get_growth_data(year, quarter, pageNo, dataArr, 
                     # ⚠️ SAST Risk (Medium): Using read_html can be risky if the HTML content is not sanitized.
                     retry_count=3, pause=0.001):
    ct._write_console()
    for _ in range(retry_count):
        # ✅ Best Practice: Use append method with ignore_index for better performance.
        time.sleep(pause)
        try:
            # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors and make debugging difficult.
            # ⚠️ SAST Risk (Low): Using regex to extract numbers can be error-prone if the pattern changes.
            # 🧠 ML Signal: Recursive function calls indicate a pattern of handling paginated data.
            request = Request(ct.GROWTH_URL%(ct.P_TYPE['http'], ct.DOMAINS['vsf'],
                                                  ct.PAGES['fd'], year,
                                                  quarter, pageNo, ct.PAGE_NUM[1]))
            text = urlopen(request, timeout=50).read()
            text = text.decode('GBK')
            text = text.replace('--', '')
            html = lxml.html.parse(StringIO(text))
            res = html.xpath("//table[@class=\"list_table\"]/tr")
            if ct.PY3:
                sarr = [etree.tostring(node).decode('utf-8') for node in res]
            else:
                sarr = [etree.tostring(node) for node in res]
            sarr = ''.join(sarr)
            sarr = '<table>%s</table>'%sarr
            df = pd.read_html(sarr)[0]
            df.columns=ct.GROWTH_COLS
            dataArr = dataArr.append(df, ignore_index=True)
            nextPage = html.xpath('//div[@class=\"pages\"]/a[last()]/@onclick')
            if len(nextPage)>0:
                # ⚠️ SAST Risk (Low): Raising generic IOError can be misleading if the error is not related to I/O operations.
                # ⚠️ SAST Risk (Low): No validation on the return value of ct._check_input, assuming it returns a boolean.
                pageNo = re.findall(r'\d+', nextPage[0])[0]
                return _get_growth_data(year, quarter, pageNo, dataArr)
            # 🧠 ML Signal: Function call to ct._write_head() indicates a logging or setup action.
            else:
                return dataArr
        # 🧠 ML Signal: Recursive function call pattern with _get_debtpaying_data.
        except Exception as e:
            pass
    # ✅ Best Practice: Using map with lambda for consistent string formatting.
    raise IOError(ct.NETWORK_URL_ERROR_MSG)

# ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters.

def get_debtpaying_data(year, quarter):
    """
        获取偿债能力数据
    Parameters
    --------
    year:int 年度 e.g:2014
    quarter:int 季度 :1、2、3、4，只能输入这4个季度
       说明：由于是从网站获取的数据，需要一页页抓取，速度取决于您当前网络速度
       
    Return
    --------
    DataFrame
        code,代码
        name,名称
        currentratio,流动比率
        quickratio,速动比率
        cashratio,现金比率
        icratio,利息支付倍数
        sheqratio,股东权益比率
        adratio,股东权益增长率
    """
    if ct._check_input(year, quarter) is True:
        # ⚠️ SAST Risk (Low): Using read_html without specifying a parser can lead to security issues if the input is not trusted.
        ct._write_head()
        df =  _get_debtpaying_data(year, quarter, 1, pd.DataFrame())
        # 🧠 ML Signal: Setting DataFrame columns explicitly indicates data normalization practices.
        if df is not None:
#             df = df.drop_duplicates('code')
            # ✅ Best Practice: Using append with ignore_index=True for DataFrame concatenation.
            df['code'] = df['code'].map(lambda x:str(x).zfill(6))
        return df

# 🧠 ML Signal: Recursive function call pattern for pagination handling.
# ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors and make debugging difficult.
# ⚠️ SAST Risk (Low): Using regex to extract numbers from strings can be error-prone if the format changes.

def _get_debtpaying_data(year, quarter, pageNo, dataArr,
                         retry_count=3, pause=0.001):
    ct._write_console()
    for _ in range(retry_count):
        time.sleep(pause)
        try:
            request = Request(ct.DEBTPAYING_URL%(ct.P_TYPE['http'], ct.DOMAINS['vsf'],
                                                      ct.PAGES['fd'], year,
                                                      quarter, pageNo, ct.PAGE_NUM[1]))
            text = urlopen(request, timeout=10).read()
            text = text.decode('GBK')
            html = lxml.html.parse(StringIO(text))
            res = html.xpath("//table[@class=\"list_table\"]/tr")
            if ct.PY3:
                sarr = [etree.tostring(node).decode('utf-8') for node in res]
            else:
                sarr = [etree.tostring(node) for node in res]
            # ⚠️ SAST Risk (Low): Potential issue if ct._check_input does not handle invalid inputs properly
            # ⚠️ SAST Risk (Low): Raising IOError with a custom message without logging the original exception details.
            sarr = ''.join(sarr)
            sarr = '<table>%s</table>'%sarr
            # ✅ Best Practice: Writing headers before processing data can help in debugging and logging
            df = pd.read_html(sarr)[0]
            df.columns = ct.DEBTPAYING_COLS
            # 🧠 ML Signal: Recursive data fetching pattern
            dataArr = dataArr.append(df, ignore_index=True)
            nextPage = html.xpath('//div[@class=\"pages\"]/a[last()]/@onclick')
            # 🧠 ML Signal: Data transformation pattern using map and lambda
            if len(nextPage)>0:
                pageNo = re.findall(r'\d+', nextPage[0])[0]
                # 🧠 ML Signal: Logging or console output can be used to track function usage and error rates
                return _get_debtpaying_data(year, quarter, pageNo, dataArr)
            else:
                return dataArr
        # ⚠️ SAST Risk (Low): Using a fixed sleep time might not be optimal for rate limiting
        except Exception as e:
            pass
    raise IOError(ct.NETWORK_URL_ERROR_MSG)
 
# ⚠️ SAST Risk (Medium): URL construction without validation can lead to injection attacks
 
def get_cashflow_data(year, quarter):
    """
        获取现金流量数据
    Parameters
    --------
    year:int 年度 e.g:2014
    quarter:int 季度 :1、2、3、4，只能输入这4个季度
       说明：由于是从网站获取的数据，需要一页页抓取，速度取决于您当前网络速度
       
    Return
    --------
    DataFrame
        code,代码
        name,名称
        cf_sales,经营现金净流量对销售收入比率
        rateofreturn,资产的经营现金流量回报率
        cf_nm,经营现金净流量与净利润的比率
        cf_liabilities,经营现金净流量对负债比率
        cashflowratio,现金流量比率
    """
    if ct._check_input(year, quarter) is True:
        # ✅ Best Practice: Using append in a loop can be inefficient; consider alternatives
        ct._write_head()
        # ✅ Best Practice: Define imports at the top of the file for better readability and maintainability
        df =  _get_cashflow_data(year, quarter, 1, pd.DataFrame())
        if df is not None:
#             df = df.drop_duplicates('code')
            # ⚠️ SAST Risk (Low): Regular expressions without validation can lead to unexpected behavior
            df['code'] = df['code'].map(lambda x:str(x).zfill(6))
        return df
# ✅ Best Practice: Use inspect.stack() carefully as it can be resource-intensive


# ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific error details
# ⚠️ SAST Risk (Low): Raising generic IOError can obscure specific network issues
# ⚠️ SAST Risk (Low): Potential directory traversal if caller_file is manipulated
def _get_cashflow_data(year, quarter, pageNo, dataArr,
                       retry_count=3, pause=0.001):
    ct._write_console()
    for _ in range(retry_count):
        time.sleep(pause)
        try:
            request = Request(ct.CASHFLOW_URL%(ct.P_TYPE['http'], ct.DOMAINS['vsf'],
                                                    ct.PAGES['fd'], year,
                                                    quarter, pageNo, ct.PAGE_NUM[1]))
            text = urlopen(request, timeout=10).read()
            # ✅ Best Practice: Check if the code is a digit to ensure valid input
            text = text.decode('GBK')
            text = text.replace('--', '')
            # ⚠️ SAST Risk (Medium): Potential for URL injection if `code` is not properly validated
            html = lxml.html.parse(StringIO(text))
            res = html.xpath("//table[@class=\"list_table\"]/tr")
            # ⚠️ SAST Risk (Medium): Network operations can be a point of failure or attack
            if ct.PY3:
                sarr = [etree.tostring(node).decode('utf-8') for node in res]
            # ⚠️ SAST Risk (Low): Decoding with a specific encoding can lead to issues if the encoding is incorrect
            else:
                sarr = [etree.tostring(node) for node in res]
            # ✅ Best Practice: Replacing specific characters to clean up the text
            sarr = ''.join(sarr)
            # 🧠 ML Signal: Returning a DataFrame, which is a common pattern in data processing
            # ⚠️ SAST Risk (Low): Reading CSV from a string can be risky if the content is not properly validated
            sarr = '<table>%s</table>'%sarr
            df = pd.read_html(sarr)[0]
            df.columns = ct.CASHFLOW_COLS
            dataArr = dataArr.append(df, ignore_index=True)
            nextPage = html.xpath('//div[@class=\"pages\"]/a[last()]/@onclick')
            if len(nextPage)>0:
                pageNo = re.findall(r'\d+', nextPage[0])[0]
                return _get_cashflow_data(year, quarter, pageNo, dataArr)
            else:
                return dataArr
        # ✅ Best Practice: Check if the code is a digit to ensure valid input
        except Exception as e:
            pass
    # ⚠️ SAST Risk (Medium): Potential for URL injection if `code` is not properly validated
    raise IOError(ct.NETWORK_URL_ERROR_MSG)
       
# ⚠️ SAST Risk (Medium): No exception handling for network operations
       
def _data_path():
    # ⚠️ SAST Risk (Medium): No exception handling for decoding errors
    import os
    import inspect
      # ✅ Best Practice: Normalize line endings for consistent data processing
    caller_file = inspect.stack()[1][1]  
    # ⚠️ SAST Risk (Medium): No validation or sanitization of CSV data before parsing
    # ✅ Best Practice: Replace tabs with commas for CSV format compatibility
    # 🧠 ML Signal: Returns a DataFrame, indicating data processing and analysis
    pardir = os.path.abspath(os.path.join(os.path.dirname(caller_file), os.path.pardir))
    return os.path.abspath(os.path.join(pardir, os.path.pardir))
  

def get_balance_sheet(code):
    """
        获取某股票的历史所有时期资产负债表
    Parameters
    --------
    code:str 股票代码 e.g:600518
       
    Return
    --------
    DataFrame
        行列名称为中文且数目较多，建议获取数据后保存到本地查看
    """
    # ⚠️ SAST Risk (Low): Decoding with a specific encoding without handling potential decoding errors
    if code.isdigit():
        # ✅ Best Practice: Replacing specific patterns in text to ensure consistent formatting
        # 🧠 ML Signal: Returning a DataFrame, which is a common pattern in data processing functions
        # ⚠️ SAST Risk (Low): Using StringIO without checking the content may lead to issues if the text is malformed
        request = Request(ct.SINA_BALANCESHEET_URL%(code))
        text = urlopen(request, timeout=10).read()
        text = text.decode('GBK')
        text = text.replace('\t\n', '\r\n')
        text = text.replace('\t', ',')
        df = pd.read_csv(StringIO(text), dtype={'code':'object'})
        return df

def get_profit_statement(code):
    """
        获取某股票的历史所有时期利润表
    Parameters
    --------
    code:str 股票代码 e.g:600518
       
    Return
    --------
    DataFrame
        行列名称为中文且数目较多，建议获取数据后保存到本地查看
    """
    if code.isdigit():
        request = Request(ct.SINA_PROFITSTATEMENT_URL%(code))
        text = urlopen(request, timeout=10).read()
        text = text.decode('GBK')
        text = text.replace('\t\n', '\r\n')
        text = text.replace('\t', ',')
        df = pd.read_csv(StringIO(text), dtype={'code':'object'})
        return df

      
def get_cash_flow(code):
    """
        获取某股票的历史所有时期现金流表
    Parameters
    --------
    code:str 股票代码 e.g:600518
       
    Return
    --------
    DataFrame
        行列名称为中文且数目较多，建议获取数据后保存到本地查看
    """
    if code.isdigit():
        request = Request(ct.SINA_CASHFLOW_URL%(code))
        text = urlopen(request, timeout=10).read()
        text = text.decode('GBK')
        text = text.replace('\t\n', '\r\n')
        text = text.replace('\t', ',')
        df = pd.read_csv(StringIO(text), dtype={'code':'object'})
        return df