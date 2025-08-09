# -*- coding:utf-8 -*- 
"""
投资参考数据接口 
Created on 2015/03/21
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""
# ✅ Best Practice: Group imports into standard library, third-party, and local sections
from __future__ import division
from tushare.stock import cons as ct
from tushare.stock import ref_vars as rv
# ✅ Best Practice: Group imports into standard library, third-party, and local sections
import pandas as pd
import numpy as np
import time
import lxml.html
from lxml import etree
import re
import json
from pandas.compat import StringIO
from tushare.util import dateu as du
# ✅ Best Practice: Group imports into standard library, third-party, and local sections
from tushare.util.netbase import Client
try:
    from urllib.request import urlopen, Request
except ImportError:
    # ✅ Best Practice: Handle both Python 2 and 3 imports for compatibility
    from urllib2 import urlopen, Request

 # ✅ Best Practice: Handle both Python 2 and 3 imports for compatibility

def profit_data(year=2017, top=25, 
              retry_count=3, pause=0.001):
    """
    获取分配预案数据
    Parameters
    --------
    year:年份
    top:取最新n条数据，默认取最近公布的25条
    retry_count : int, 默认 3
                 如遇网络等问题重复执行的次数 
      pause : int, 默认 0
                重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题
    
    returns
    -------
    DataFrame
    code:股票代码
    name:股票名称
    year:分配年份
    report_date:公布日期
    divi:分红金额（每10股）
    shares:转增和送股数（每10股）
    """
    # 🧠 ML Signal: Custom function call with retry and pause parameters
    
    if top == 'all':
        ct._write_head()
        # 🧠 ML Signal: Appending data to a DataFrame in a loop
        df, pages = _dist_cotent(year, 0, retry_count, pause)
        for idx in range(1,int(pages)):
            df = df.append(_dist_cotent(year, idx, retry_count,
                                        pause), ignore_index=True)
        return df
    # 🧠 ML Signal: Conditional logic based on the 'top' parameter
    elif top <= 25:
        df, pages = _dist_cotent(year, 0, retry_count, pause)
        # 🧠 ML Signal: Returning a subset of data using DataFrame.head()
        return df.head(top)
    else:
        if isinstance(top, int):
            # ✅ Best Practice: Check type of 'top' before using it in calculations
            ct._write_head()
            allPages = top/25+1 if top%25>0 else top/25
            df, pages = _dist_cotent(year, 0, retry_count, pause)
            # ✅ Best Practice: Use integer division for clarity
            if int(allPages) < int(pages):
                pages = allPages
            for idx in range(1, int(pages)):
                df = df.append(_dist_cotent(year, idx, retry_count,
                                            # ⚠️ SAST Risk (Medium): Use of `re.compile` with user input can lead to ReDoS (Regular Expression Denial of Service) if `x` is not properly sanitized.
                                            pause), ignore_index=True)
            return df.head(top)
        # ✅ Best Practice: Use raw strings for regex patterns to avoid issues with escape sequences.
        else:
            # 🧠 ML Signal: Pattern matching and extraction from strings.
            print(ct.TOP_PARAS_MSG)
    

# ✅ Best Practice: Use of ternary operator for concise conditional return.
# ⚠️ SAST Risk (Low): Using print statements for error messages can expose sensitive information
def _fun_divi(x):
    if ct.PY3:
        reg = re.compile(r'分红(.*?)元', re.UNICODE)
        # ✅ Best Practice: Check for type before processing to avoid errors.
        res = reg.findall(x)
         # ✅ Best Practice: Use of `unicode` ensures compatibility with non-ASCII characters in Python 2.
        return 0 if len(res)<1 else float(res[0]) 
    else:
        if isinstance(x, unicode):
            s1 = unicode('分红','utf-8')
            # ⚠️ SAST Risk (Medium): Use of `re.compile` with user input can lead to ReDoS (Regular Expression Denial of Service) if `x` is not properly sanitized.
            # 🧠 ML Signal: Conditional logic based on Python version
            s2 = unicode('元','utf-8')
            reg = re.compile(r'%s(.*?)%s'%(s1, s2), re.UNICODE)
            # 🧠 ML Signal: Regular expression usage pattern
            # 🧠 ML Signal: Pattern matching and extraction from strings.
            res = reg.findall(x)
            return 0 if len(res)<1 else float(res[0])
        # ✅ Best Practice: Use of ternary operator for concise conditional return.
        # 🧠 ML Signal: Regular expression usage pattern
        else:
            return 0
# 🧠 ML Signal: Regular expression findall usage

# 🧠 ML Signal: Regular expression findall usage
# ✅ Best Practice: Return a default value when input is not as expected.

def _fun_into(x):
    if ct.PY3:
            # ✅ Best Practice: Use of ternary operator for concise conditional assignment
            reg1 = re.compile(r'转增(.*?)股', re.UNICODE)
            reg2 = re.compile(r'送股(.*?)股', re.UNICODE)
            # ✅ Best Practice: Use of ternary operator for concise conditional assignment
            res1 = reg1.findall(x)
            res2 = reg2.findall(x)
            res1 = 0 if len(res1)<1 else float(res1[0])
            res2 = 0 if len(res2)<1 else float(res2[0])
            # ⚠️ SAST Risk (Low): Potential issue with handling non-unicode strings in Python 2
            return res1 + res2
    else:
        # 🧠 ML Signal: Use of unicode function in Python 2
        if isinstance(x, unicode):
            # 🧠 ML Signal: Use of unicode function in Python 2
            s1 = unicode('转增','utf-8')
            s2 = unicode('送股','utf-8')
            s3 = unicode('股','utf-8')
            # ✅ Best Practice: Function name is misspelled, should be _dist_content
            # 🧠 ML Signal: Use of unicode function in Python 2
            reg1 = re.compile(r'%s(.*?)%s'%(s1, s3), re.UNICODE)
            reg2 = re.compile(r'%s(.*?)%s'%(s2, s3), re.UNICODE)
            # 🧠 ML Signal: Regular expression usage pattern
            # 🧠 ML Signal: Loop with retry pattern
            res1 = reg1.findall(x)
            res2 = reg2.findall(x)
            # 🧠 ML Signal: Regular expression usage pattern
            # ⚠️ SAST Risk (Low): Use of time.sleep can lead to performance issues in async environments
            res1 = 0 if len(res1)<1 else float(res1[0])
            res2 = 0 if len(res2)<1 else float(res2[0])
            # 🧠 ML Signal: Regular expression findall usage
            return res1 + res2
        else:
            # 🧠 ML Signal: Regular expression findall usage
            # 🧠 ML Signal: Conditional logging based on page number
            return 0
    
# ✅ Best Practice: Use of ternary operator for concise conditional assignment
# ⚠️ SAST Risk (Medium): Potential for URL injection if inputs are not validated
    
def _dist_cotent(year, pageNo, retry_count, pause):
    # ✅ Best Practice: Use of ternary operator for concise conditional assignment
    for _ in range(retry_count):
        # 🧠 ML Signal: Use of XPath for HTML parsing
        time.sleep(pause)
        try:
            if pageNo > 0:
                # 🧠 ML Signal: Conditional logic based on Python version
                ct._write_console()
            html = lxml.html.parse(rv.DP_163_URL%(ct.P_TYPE['http'], ct.DOMAINS['163'],
                     ct.PAGES['163dp'], year, pageNo))  
            res = html.xpath('//div[@class=\"fn_rp_list\"]/table')
            if ct.PY3:
                # ⚠️ SAST Risk (Low): Potential for HTML injection if sarr is not sanitized
                sarr = [etree.tostring(node).decode('utf-8') for node in res]
            else:
                sarr = [etree.tostring(node) for node in res]
            # ✅ Best Practice: Explicitly setting DataFrame columns
            sarr = ''.join(sarr)
            df = pd.read_html(sarr, skiprows=[0])[0]
            # 🧠 ML Signal: Mapping functions to DataFrame columns
            df = df.drop(df.columns[0], axis=1)
            df.columns = rv.DP_163_COLS
            df['divi'] = df['plan'].map(_fun_divi)
            df['shares'] = df['plan'].map(_fun_into)
            df = df.drop('plan', axis=1)
            # 🧠 ML Signal: Zero-padding numeric codes
            df['code'] = df['code'].astype(object)
            df['code'] = df['code'].map(lambda x : str(x).zfill(6))
            pages = []
            if pageNo == 0:
                # 🧠 ML Signal: Conditional logic for pagination
                page = html.xpath('//div[@class=\"mod_pages\"]/a')
                if len(page)>1:
                    # ⚠️ SAST Risk (Low): Catching broad exceptions can hide issues
                    # ✅ Best Practice: Returning tuple for consistent return type
                    asr = page[len(page)-2]
                    pages = asr.xpath('text()')
        except Exception as e:
            print(e)
        else:
            if pageNo == 0:
                return df, pages[0] if len(pages)>0 else 0
            else:
                return df
    raise IOError(ct.NETWORK_URL_ERROR_MSG)    


def profit_divis():
        '''
                        获取分送送股数据
            -------
            Return:DataFrame
                code:代码    
                name:证券简称    
                year:分配年度    
                bshares:送股  
                incshares:转增股
                totals:送转总数 
                cash:派现   
                plandate:预案公布日    
                regdate:股权登记日    
                exdate:除权除息日    
                eventproc:事件进程 ,预案或实施
                anndate:公告日期
                
    '''
        ct._write_head()
        # 🧠 ML Signal: Pattern of replacing specific substrings in HTML content
        p = 'cfidata.aspx?sortfd=&sortway=&curpage=1&fr=content&ndk=A0A1934A1939A1957A1966A1983&xztj=&mystock='
        df =  _profit_divis(1, pd.DataFrame(), p)
        # ⚠️ SAST Risk (Low): Potentially unsafe HTML content parsing without validation
        df = df.drop([3], axis=1)
        df.columns = ct.PROFIT_DIVIS
        # 🧠 ML Signal: Pattern of appending data to a DataFrame
        df['code'] = df['code'].map(lambda x: str(x).zfill(6))
        return df
# ⚠️ SAST Risk (Medium): Potentially unsafe XPath expression without validation


# 🧠 ML Signal: Recursive function pattern
def _profit_divis(pageNo, dataArr, nextPage):
        ct._write_console()
        # 🧠 ML Signal: Recursive function pattern
        html = lxml.html.parse('%sdata.cfi.cn/%s'%(ct.P_TYPE['http'], nextPage))
        res = html.xpath("//table[@class=\"table_data\"]/tr")
        if ct.PY3:
            sarr = [etree.tostring(node).decode('utf-8') for node in res]
        else:
            sarr = [etree.tostring(node) for node in res]
        sarr = ''.join(sarr)
        sarr = sarr.replace('--', '0')
        sarr = '<table>%s</table>'%sarr
        df = pd.read_html(sarr, skiprows=[0])[0]
        dataArr = dataArr.append(df, ignore_index=True)
        nextPage = html.xpath('//div[@id=\"content\"]/div[2]/a[last()]/@href')[0]
        np = nextPage.split('&')[2].split('=')[1]
        if pageNo < int(np):
            return _profit_divis(int(np), dataArr, nextPage)
        else:
            return dataArr


# ⚠️ SAST Risk (Low): No validation on the return value of _check_input, assuming it returns a boolean
def forecast_data(year, quarter):
    """
        获取业绩预告数据
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
        type,业绩变动类型【预增、预亏等】
        report_date,发布日期
        pre_eps,上年同期每股收益
        range,业绩变动范围
        
    """
    # ✅ Best Practice: Use list comprehensions for more concise and readable code.
    if ct._check_input(year, quarter) is True:
        ct._write_head()
        data =  _get_forecast_data(year, quarter, 1, pd.DataFrame())
        df = pd.DataFrame(data, columns=ct.FORECAST_COLS)
        df['code'] = df['code'].map(lambda x: str(x).zfill(6))
        # ✅ Best Practice: Consider using a more descriptive variable name than 'sarr' for clarity.
        return df


# ⚠️ SAST Risk (Low): Using read_html on potentially untrusted HTML content can lead to security issues.
def _get_forecast_data(year, quarter, pageNo, dataArr):
    # ✅ Best Practice: Dropping columns by index can be error-prone; consider using column names instead.
    ct._write_console()
    try:
        gparser = etree.HTMLParser(encoding='GBK')
        html = lxml.html.parse(ct.FORECAST_URL%(ct.P_TYPE['http'], ct.DOMAINS['vsf'], 
                                                # 🧠 ML Signal: Appending data to a DataFrame, indicating data aggregation behavior.
                                                ct.PAGES['fd'], year, quarter, pageNo,
                                                # 🧠 ML Signal: Recursive function call pattern, indicating iterative data fetching.
                                                # ✅ Best Practice: Consider using type hints for function parameters and return type for better readability and maintainability.
                                                ct.PAGE_NUM[1]),
                               parser=gparser)
        # ✅ Best Practice: Consider logging exceptions instead of printing them for better error tracking and analysis.
        res = html.xpath("//table[@class=\"list_table\"]/tr")
        if ct.PY3:
            sarr = [etree.tostring(node).decode('utf-8') for node in res]
        else:
            sarr = [etree.tostring(node) for node in res]
        sarr = ''.join(sarr)
        sarr = sarr.replace('--', '0')
        sarr = '<table>%s</table>'%sarr
        df = pd.read_html(sarr)[0]
        df = df.drop([4, 5, 8], axis=1)
        df.columns = ct.FORECAST_COLS
        dataArr = dataArr.append(df, ignore_index=True)
        nextPage = html.xpath('//div[@class=\"pages\"]/a[last()]/@onclick')
        if len(nextPage)>0:
            pageNo = re.findall(r'\d+',nextPage[0])[0]
            return _get_forecast_data(year, quarter, pageNo, dataArr)
        else:
            return dataArr
    except Exception as e:
            print(e)
 # ✅ Best Practice: Use a more descriptive variable name instead of 'du' for better readability.
    

 # ✅ Best Practice: Use a more descriptive variable name instead of 'du' for better readability.
def xsg_data(year=None, month=None, 
            retry_count=3, pause=0.001):
    """
    获取限售股解禁数据
    Parameters
    --------
    year:年份,默认为当前年
    month:解禁月份，默认为当前月
    retry_count : int, 默认 3
                 如遇网络等问题重复执行的次数 
    pause : int, 默认 0
                重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题
    
    Return
    ------
    DataFrame
    code:股票代码
    name:名称
    date:解禁日期
    count:解禁数量（万股）
    ratio:占总盘比率
    """
    year = du.get_year() if year is None else year
    # ✅ Best Practice: Use more descriptive column indices or names for better readability.
    month = du.get_month() if month is None else month
    for _ in range(retry_count):
        time.sleep(pause)
        # ⚠️ SAST Risk (Low): Potential for ValueError if conversion fails.
        # ⚠️ SAST Risk (Medium): Raising a generic IOError without specific context can be misleading.
        # ✅ Best Practice: Ensure that rv.XSG_COLS matches the expected DataFrame structure.
        try:
            request = Request(rv.XSG_URL%(ct.P_TYPE['http'], ct.DOMAINS['em'],
                                     ct.PAGES['emxsg'], year, month))
            lines = urlopen(request, timeout = 10).read()
            lines = lines.decode('utf-8') if ct.PY3 else lines
        except Exception as e:
            print(e)
        else:
            da = lines[3:len(lines)-3]
            list =  []
            for row in da.split('","'):
                list.append([data for data in row.split(',')])
            df = pd.DataFrame(list)
            df = df[[1, 3, 4, 5, 6]]
            for col in [5, 6]:
                df[col] = df[col].astype(float)
            df[5] = df[5]/10000
            df[6] = df[6]*100
            df[5] = df[5].map(ct.FORMAT)
            df[6] = df[6].map(ct.FORMAT)
            df.columns = rv.XSG_COLS
            return df
    raise IOError(ct.NETWORK_URL_ERROR_MSG)   
# ⚠️ SAST Risk (Low): Potential KeyError if 'quarter' is not a valid key in rv.QUARTS_DIC


def fund_holdings(year, quarter,
                  # ⚠️ SAST Risk (Low): Possible TypeError if 'year' is not an integer
                  retry_count=3, pause=0.001):
    """
    获取基金持股数据
    Parameters
    --------
    year:年份e.g 2014
    quarter:季度（只能输入1，2，3，4这个四个数字）
    retry_count : int, 默认 3
                 如遇网络等问题重复执行的次数 
    pause : int, 默认 0
                重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题
    
    Return
    ------
    DataFrame
    code:股票代码
    name:名称
    date:报告日期
    nums:基金家数
    nlast:与上期相比（增加或减少了）
    count:基金持股数（万股）
    clast:与上期相比
    amount:基金持股市值
    ratio:占流通盘比率
    # ⚠️ SAST Risk (Medium): No exception handling for network operations
    """
    start,end = rv.QUARTS_DIC[str(quarter)]
    # ✅ Best Practice: Compatibility handling for Python 3
    if quarter == 1:
        start = start % str(year-1)
        # ✅ Best Practice: Replacing '--' with '0' might not be universally correct
        end = end%year
    else:
        # ⚠️ SAST Risk (Medium): No validation of JSON structure
        start, end = start%year, end%year
    ct._write_head()
    df, pages = _holding_cotent(start, end, 0, retry_count, pause)
    # 🧠 ML Signal: Data transformation and cleaning pattern
    for idx in range(1, pages):
        df = df.append(_holding_cotent(start, end, idx, retry_count, pause),
                  # ✅ Best Practice: Dropping columns without checking if they exist
                  ignore_index=True)
    return df
# 🧠 ML Signal: Data type conversion pattern


def _holding_cotent(start, end, pageNo, retry_count, pause):
    # 🧠 ML Signal: Data scaling pattern
    for _ in range(retry_count):
        time.sleep(pause)
        if pageNo>0:
                ct._write_console()
        try:
            request = Request(rv.FUND_HOLDS_URL%(ct.P_TYPE['http'], ct.DOMAINS['163'],
                     # 🧠 ML Signal: Data formatting pattern
                     ct.PAGES['163fh'], ct.PAGES['163fh'],
                     # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
                     pageNo, start, end, _random(5)))
            # ✅ Best Practice: Renaming columns without checking if they exist
            # ⚠️ SAST Risk (Low): Generic exception handling
            lines = urlopen(request, timeout = 10).read()
            lines = lines.decode('utf-8') if ct.PY3 else lines
            lines = lines.replace('--', '0')
            lines = json.loads(lines)
            data = lines['list']
            df = pd.DataFrame(data)
            df = df.drop(['CODE', 'ESYMBOL', 'EXCHANGE', 'NAME', 'RN', 'SHANGQIGUSHU',
                              'SHANGQISHIZHI', 'SHANGQISHULIANG'], axis=1)
            for col in ['GUSHU', 'GUSHUBIJIAO', 'SHIZHI', 'SCSTC27']:
                df[col] = df[col].astype(float)
            df['SCSTC27'] = df['SCSTC27']*100
            df['GUSHU'] = df['GUSHU']/10000
            df['GUSHUBIJIAO'] = df['GUSHUBIJIAO']/10000
            df['SHIZHI'] = df['SHIZHI']/10000
            df['GUSHU'] = df['GUSHU'].map(ct.FORMAT)
            df['GUSHUBIJIAO'] = df['GUSHUBIJIAO'].map(ct.FORMAT)
            df['SHIZHI'] = df['SHIZHI'].map(ct.FORMAT)
            df['SCSTC27'] = df['SCSTC27'].map(ct.FORMAT)
            df.columns = rv.FUND_HOLDS_COLS
            df = df[['code', 'name', 'date', 'nums', 'nlast', 'count', 
                         'clast', 'amount', 'ratio']]
        except Exception as e:
            print(e)
        else:
            # ⚠️ SAST Risk (Low): Raising a generic IOError
            if pageNo == 0:
                # ⚠️ SAST Risk (Low): Ensure that 'pd' and 'ct' are properly imported and validated to avoid potential NameError.
                return df, int(lines['pagecount'])
            # ⚠️ SAST Risk (Low): Ensure that 'ct' is properly imported and validated to avoid potential NameError.
            else:
                return df
    raise IOError(ct.NETWORK_URL_ERROR_MSG)    
# ⚠️ SAST Risk (Low): Ensure that '_newstocks' is properly imported and validated to avoid potential NameError.
    
# 🧠 ML Signal: Retry pattern with a fixed number of attempts

def new_stocks(retry_count=3, pause=0.001):
    """
    获取新股上市数据
    Parameters
    --------
    retry_count : int, 默认 3
                 如遇网络等问题重复执行的次数 
    pause : int, 默认 0
                重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题
    
    Return
    ------
    DataFrame
    code:股票代码
    xcode:申购代码
    name:名称
    ipo_date:上网发行日期
    issue_date:上市日期
    amount:发行数量(万股)
    markets:上网发行数量(万股)
    price:发行价格(元)
    pe:发行市盈率
    limit:个人申购上限(万股)
    funds：募集资金(亿元)
    ballot:网上中签率(%)
    """
    # ✅ Best Practice: Use of lambda for concise mapping
    data = pd.DataFrame()
    ct._write_head()
    df = _newstocks(data, 1, retry_count,
                    pause)
    return df
# ✅ Best Practice: Use of ternary operator for concise conditional
# ✅ Best Practice: Consider specifying the data type for the parameters in the docstring for clarity.

# 🧠 ML Signal: Recursive function call pattern
# ⚠️ SAST Risk (Low): Potential for large memory usage with data.append
# ✅ Best Practice: Exception handling with logging

def _newstocks(data, pageNo, retry_count, pause):
    for _ in range(retry_count):
        time.sleep(pause)
        ct._write_console()
        try:
            html = lxml.html.parse(rv.NEW_STOCKS_URL%(ct.P_TYPE['http'],ct.DOMAINS['vsf'],
                         ct.PAGES['newstock'], pageNo))
            res = html.xpath('//table[@id=\"NewStockTable\"]/tr')
            if len(res) == 0:
                return data
            if ct.PY3:
                sarr = [etree.tostring(node).decode('utf-8') for node in res]
            else:
                sarr = [etree.tostring(node) for node in res]
            sarr = ''.join(sarr)
            sarr = sarr.replace('<font color="red">*</font>', '')
            sarr = '<table>%s</table>'%sarr
            df = pd.read_html(StringIO(sarr), skiprows=[0, 1])[0]
            df = df.drop([df.columns[idx] for idx in [12, 13, 14]], axis=1)
            df.columns = rv.NEW_STOCKS_COLS
            df['code'] = df['code'].map(lambda x : str(x).zfill(6))
            df['xcode'] = df['xcode'].map(lambda x : str(x).zfill(6))
            res = html.xpath('//table[@class=\"table2\"]/tr[1]/td[1]/a/text()')
            tag = '下一页' if ct.PY3 else unicode('下一页', 'utf-8')
            hasNext = True if tag in res else False 
            data = data.append(df, ignore_index=True)
            # ✅ Best Practice: Initialize the DataFrame outside of the loop to avoid reinitialization.
            pageNo += 1
            if hasNext:
                data = _newstocks(data, pageNo, retry_count, pause)
        except Exception as ex:
            # 🧠 ML Signal: Usage of retry_count and pause parameters indicates handling of network issues.
            print(ex)
        else:
            return data 

# 🧠 ML Signal: Looping with a fixed range suggests a pattern for data pagination.
# 🧠 ML Signal: Usage of retry_count and pause parameters indicates handling of network issues.

def new_cbonds(default=1, retry_count=3, pause=0.001):
    """
    获取可转债申购列表
    Parameters
    --------
    retry_count : int, 默认 3
                 如遇网络等问题重复执行的次数 
    pause : int, 默认 0
                重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题
    
    Return
    ------
    DataFrame
    bcode:债券代码
    bname:债券名称
    scode:股票代码
    sname:股票名称
    xcode:申购代码
    amount:发行总数(亿元)
    marketprice:最新市场价格
    convprice:转股价格
    firstdayprice:首日收盘价
    ipo_date:上网发行日期
    issue_date:上市日期
    ballot:中签率(%)
    return：打新收益率(%)
    perreturn:每中一股收益（万元）
    
    """
    data = pd.DataFrame()
    if default == 1:
        # ✅ Best Practice: Dropping unnecessary columns for data cleanliness
        data = _newcbonds(1, retry_count,
                    # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
                    pause)
    # ✅ Best Practice: Use of map and lambda for consistent data formatting
    # ⚠️ SAST Risk (Low): Generic exception handling, potential to miss specific errors
    # ✅ Best Practice: Explicitly setting DataFrame columns for clarity
    else:
        for page in range(1, 50):
            df = _newcbonds(page, retry_count,
                    pause)
            if df is not None:
                data = data.append(df, ignore_index=True)
            else:
                break
    return data


def _newcbonds(pageNo, retry_count, pause):
    for _ in range(retry_count):
        time.sleep(pause)
        if pageNo != 1:
            ct._write_console()
        try:
            html = lxml.html.parse(rv.NEW_CBONDS_URL%(ct.P_TYPE['http'],ct.DOMAINS['sstar'],
                         pageNo))
            res = html.xpath('//table/tr')
            if len(res) == 0:
                return None
            if ct.PY3:
                sarr = [etree.tostring(node).decode('utf-8') for node in res]
            # 🧠 ML Signal: Use of default values for function parameters.
            else:
                sarr = [etree.tostring(node) for node in res]
            sarr = ''.join(sarr)
            # ⚠️ SAST Risk (Low): Potential risk if `du.diff_day` does not handle invalid date formats properly.
            sarr = '<table>%s</table>'%sarr
            df = pd.read_html(StringIO(sarr), skiprows=[0])
            if len(df) < 1:
                # 🧠 ML Signal: String manipulation to remove characters.
                return None
            df = df[0]
            df = df.drop([df.columns[14], df.columns[15]], axis=1)
            # 🧠 ML Signal: Use of pandas DataFrame, common in data processing tasks.
            df.columns = rv.NEW_CBONDS_COLS
            # 🧠 ML Signal: Function call to write headers, indicating logging or output preparation.
            # 🧠 ML Signal: Use of retry logic in function parameters.
            df['scode'] = df['scode'].map(lambda x: str(x).zfill(6))
            df['xcode'] = df['xcode'].map(lambda x: str(x).zfill(6))
        except Exception as ex:
            print(ex)
        # 🧠 ML Signal: Loop with retry pattern
        else:
            return df 
# ⚠️ SAST Risk (Low): Potential for high-frequency requests due to low pause


# 🧠 ML Signal: Console writing operation

def sh_margins(start=None, end=None, retry_count=3, pause=0.001):
    """
    获取沪市融资融券数据列表
    Parameters
    --------
    start:string
                  开始日期 format：YYYY-MM-DD 为空时取去年今日
    end:string
                  结束日期 format：YYYY-MM-DD 为空时取当前日期
    retry_count : int, 默认 3
                 如遇网络等问题重复执行的次数 
    pause : int, 默认 0
                重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题
    
    Return
    ------
    DataFrame
    opDate:信用交易日期
    rzye:本日融资余额(元)
    rzmre: 本日融资买入额(元)
    rqyl: 本日融券余量
    rqylje: 本日融券余量金额(元)
    rqmcl: 本日融券卖出量
    rzrqjyzl:本日融资融券余额(元)
    # 🧠 ML Signal: Data slicing pattern
    """
    start = du.today_last_year() if start is None else start
    end = du.today() if end is None else end
    if du.diff_day(start, end) < 0:
        # ⚠️ SAST Risk (Low): JSON parsing without validation
        # 🧠 ML Signal: Calculation of page count
        return None
    start, end = start.replace('-', ''), end.replace('-', '')
    data = pd.DataFrame()
    # 🧠 ML Signal: Calculation of data pages
    ct._write_head()
    # 🧠 ML Signal: DataFrame creation pattern
    df = _sh_hz(data, start=start, end=end,
                retry_count=retry_count,
                pause=pause)
    # 🧠 ML Signal: Date formatting pattern
    # 🧠 ML Signal: Data appending pattern
    # 🧠 ML Signal: Recursive function call pattern
    # ⚠️ SAST Risk (Low): Generic exception handling
    # ⚠️ SAST Risk (Low): Raising a generic IOError
    return df


def _sh_hz(data, start=None, end=None, 
           pageNo='', beginPage='',
           endPage='',
           retry_count=3, pause=0.001):
    for _ in range(retry_count):
        time.sleep(pause)
        ct._write_console()
        try:
            tail = rv.MAR_SH_HZ_TAIL_URL%(pageNo,
                                    beginPage, endPage)
            if pageNo == '':
                pageNo = 6
                tail = ''
            else:
                pageNo += 5
            beginPage = pageNo
            endPage = pageNo + 4
            url = rv.MAR_SH_HZ_URL%(ct.P_TYPE['http'], ct.DOMAINS['sseq'],
                                    ct.PAGES['qmd'], _random(5),
                                    start, end, tail,
                                    _random())
            ref = rv.MAR_SH_HZ_REF_URL%(ct.P_TYPE['http'], ct.DOMAINS['sse'])
            clt = Client(url, ref=ref, cookie=rv.MAR_SH_COOKIESTR)
            lines = clt.gvalue()
            lines = lines.decode('utf-8') if ct.PY3 else lines
            lines = lines[19:-1]
            # ✅ Best Practice: Use of inline if-else for concise date formatting
            lines = json.loads(lines)
            pagecount = int(lines['pageHelp'].get('pageCount'))
            # ✅ Best Practice: Use of inline if-else for concise date formatting
            datapage = int(pagecount/5+1 if pagecount%5>0 else pagecount/5)
            df = pd.DataFrame(lines['result'], columns=rv.MAR_SH_HZ_COLS)
            # ✅ Best Practice: Use of inline if-else for concise date formatting
            df['opDate'] = df['opDate'].map(lambda x: '%s-%s-%s'%(x[0:4], x[4:6], x[6:8]))
            data = data.append(df, ignore_index=True)
            # ✅ Best Practice: Clear logic to prioritize date over start and end
            if beginPage < datapage*5:
                 # ✅ Best Practice: Initialize an empty DataFrame for data collection
                data = _sh_hz(data, start=start, end=end, pageNo=pageNo, 
                       beginPage=beginPage, endPage=endPage, 
                       retry_count=retry_count, pause=pause)
        except Exception as e:
            # ⚠️ SAST Risk (Low): Potential issue if ct._write_head() modifies global state
            print(e)
        # 🧠 ML Signal: Function call with multiple parameters, useful for learning API usage patterns
        else:
            return data
    raise IOError(ct.NETWORK_URL_ERROR_MSG)


 # ✅ Best Practice: Use of retry mechanism to handle transient errors
def sh_margin_details(date='', symbol='', 
                      # ✅ Best Practice: Return the result of the function for further processing
                      start='', end='',
                      # ⚠️ SAST Risk (Low): Potential for high-frequency requests due to low pause value
                      retry_count=3, pause=0.001):
    """
    获取沪市融资融券明细列表
    Parameters
    --------
    date:string
                明细数据日期 format：YYYY-MM-DD 默认为空''
    symbol：string
                标的代码，6位数字e.g.600848，默认为空  
    start:string
                  开始日期 format：YYYY-MM-DD 默认为空''
    end:string
                  结束日期 format：YYYY-MM-DD 默认为空''
    retry_count : int, 默认 3
                 如遇网络等问题重复执行的次数 
    pause : int, 默认 0
                重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题
    
    Return
    ------
    DataFrame
    opDate:信用交易日期
    stockCode:标的证券代码
    securityAbbr:标的证券简称
    rzye:本日融资余额(元)
    rzmre: 本日融资买入额(元)
    rzche:本日融资偿还额(元)
    rqyl: 本日融券余量
    rqmcl: 本日融券卖出量
    rqchl: 本日融券偿还量
    """
    date = date if date == '' else date.replace('-', '')
    start = start if start == '' else start.replace('-', '')
    end = end if end == '' else end.replace('-', '')
    # ✅ Best Practice: Use of pandas for data manipulation
    if (start != '') & (end != ''):
        date = ''
    data = pd.DataFrame()
    ct._write_head()
    # 🧠 ML Signal: Appending data to a DataFrame, common in data processing tasks
    # ✅ Best Practice: Consider importing necessary modules at the beginning of the file for clarity.
    df = _sh_mx(data, date=date, start=start,
                # 🧠 ML Signal: Recursive function call pattern
                # ⚠️ SAST Risk (Low): Generic exception handling, may hide specific errors
                # ⚠️ SAST Risk (Medium): Raising IOError with a potentially user-facing error message
                end=end, symbol=symbol,
                retry_count=retry_count,
                pause=pause)
    return df


def _sh_mx(data, date='', start='', end='', 
           symbol='',
           pageNo='', beginPage='',
           endPage='',
           retry_count=3, pause=0.001):
    for _ in range(retry_count):
        time.sleep(pause)
        ct._write_console()
        try:
            tail = '&pageHelp.pageNo=%s&pageHelp.beginPage=%s&pageHelp.endPage=%s'%(pageNo,
                                    beginPage, endPage)
            if pageNo == '':
                pageNo = 6
                tail = ''
            else:
                pageNo += 5
            beginPage = pageNo
            endPage = pageNo + 4
            # ✅ Best Practice: Initialize variables before use to avoid potential reference errors.
            ref = rv.MAR_SH_HZ_REF_URL%(ct.P_TYPE['http'], ct.DOMAINS['sse'])
            clt = Client(rv.MAR_SH_MX_URL%(ct.P_TYPE['http'], ct.DOMAINS['sseq'],
                                    ct.PAGES['qmd'], _random(5), date, 
                                    # ✅ Best Practice: Use descriptive function names for better readability.
                                    symbol, start, end, tail,
                                    _random()), ref=ref, cookie=rv.MAR_SH_COOKIESTR)
            lines = clt.gvalue()
            lines = lines.decode('utf-8') if ct.PY3 else lines
            # ✅ Best Practice: Use logging instead of print statements for better control over output.
            lines = lines[19:-1]
            lines = json.loads(lines)
            pagecount = int(lines['pageHelp'].get('pageCount'))
            datapage = int(pagecount/5+1 if pagecount%5>0 else pagecount/5)
            # ✅ Best Practice: Use descriptive variable names for better readability.
            if pagecount == 0:
                return data
            if pageNo == 6:
                ct._write_tips(lines['pageHelp'].get('total'))
            df = pd.DataFrame(lines['result'], columns=rv.MAR_SH_MX_COLS)
            df['opDate'] = df['opDate'].map(lambda x: '%s-%s-%s'%(x[0:4], x[4:6], x[6:8]))
            data = data.append(df, ignore_index=True)
            # ⚠️ SAST Risk (Low): Using append in a loop can be inefficient; consider using pd.concat instead.
            if beginPage < datapage*5:
                 # 🧠 ML Signal: Retry logic with a counter and pause can indicate robustness in network operations.
                data = _sh_mx(data, start=start, end=end, pageNo=pageNo, 
                       beginPage=beginPage, endPage=endPage, 
                       # ⚠️ SAST Risk (Medium): Catching broad exceptions can hide errors; specify exception types.
                       # ⚠️ SAST Risk (Low): Using time.sleep can lead to inefficient waiting; consider async alternatives.
                       retry_count=retry_count, pause=pause)
        except Exception as e:
            # ⚠️ SAST Risk (Low): Direct console writing can be a debugging leftover; ensure it's intended for production.
            print(e)
        else:
            return data
    # ⚠️ SAST Risk (Medium): URL construction with string interpolation can lead to injection vulnerabilities.
    raise IOError(ct.NETWORK_URL_ERROR_MSG)


# ⚠️ SAST Risk (Medium): No validation or sanitization of the response data.
def sz_margins(start=None, end=None, retry_count=3, pause=0.001):
    """
    获取深市融资融券数据列表
    Parameters
    --------
    start:string
                  开始日期 format：YYYY-MM-DD 默认为上一周的今天
    end:string
                  结束日期 format：YYYY-MM-DD 默认为今日
    retry_count : int, 默认 3
                 如遇网络等问题重复执行的次数 
    pause : int, 默认 0
                重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题
    
    Return
    ------
    DataFrame
    opDate:信用交易日期(index)
    rzmre: 融资买入额(元)
    rzye:融资余额(元)
    rqmcl: 融券卖出量
    rqyl: 融券余量
    rqye: 融券余量(元)
    rzrqye:融资融券余额(元)
    """
    data = pd.DataFrame()
    if start is None and end is None:
        end = du.today()
        start = du.day_last_week()
    if start is None or end is None:
        ct._write_msg(rv.MAR_SZ_HZ_MSG2)
        # ✅ Best Practice: Use of retry logic to handle transient network issues
        return None
    try:
        # ✅ Best Practice: Pause between retries to avoid overwhelming the server
        date_range = pd.date_range(start=start, end=end, freq='B')
        if len(date_range)>261:
            ct._write_msg(rv.MAR_SZ_HZ_MSG)
        # ⚠️ SAST Risk (Medium): Potentially unsafe string formatting for URLs
        else:
            ct._write_head()
            for date in date_range:
                # ⚠️ SAST Risk (Medium): No validation or sanitization of the URL
                data = data.append(_sz_hz(str(date.date()), retry_count, pause) )
    # ⚠️ SAST Risk (Medium): No exception handling for network errors
    except:
        ct._write_msg(ct.DATA_INPUT_ERROR_MSG)
    # 🧠 ML Signal: Checking response length to determine if data is valid
    else:
        return data
# ⚠️ SAST Risk (Low): Assumes the HTML structure is consistent
        

def _sz_hz(date='', retry_count=3, pause=0.001):
    # 🧠 ML Signal: Mapping function to format stock codes
    for _ in range(retry_count):
        time.sleep(pause)
        ct._write_console()
        # 🧠 ML Signal: Adding a new column with a constant value
        # ✅ Best Practice: Check for None to handle default parameter values
        try:
            request = Request(rv.MAR_SZ_HZ_URL%(ct.P_TYPE['http'], ct.DOMAINS['szse'],
                                    ct.PAGES['szsefc'], date))
            # ⚠️ SAST Risk (Low): Generic exception handling without specific error actions
            lines = urlopen(request, timeout = 10).read()
            # 🧠 ML Signal: Conversion of code to a specific symbol format
            if len(lines) <= 200:
                return pd.DataFrame()
            # 🧠 ML Signal: Conditional assignment based on gdtype value
            df = pd.read_html(lines, skiprows=[0])[0]
            # ⚠️ SAST Risk (Low): Raises a generic IOError without specific context
            df.columns = rv.MAR_SZ_HZ_COLS
            df['opDate'] = date
        # 🧠 ML Signal: Conditional logic based on year and quarter parameters
        except Exception as e:
            print(e)
        # 🧠 ML Signal: Retry pattern with a specified number of attempts
        else:
            return df
    raise IOError(ct.NETWORK_URL_ERROR_MSG)
# ⚠️ SAST Risk (Low): Use of time.sleep can lead to performance issues


def sz_margin_details(date='', retry_count=3, pause=0.001):
    """
    获取深市融资融券明细列表
    Parameters
    --------
    date:string
                明细数据日期 format：YYYY-MM-DD 默认为空''
    retry_count : int, 默认 3
                 如遇网络等问题重复执行的次数 
    pause : int, 默认 0
                重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题
    
    Return
    ------
    DataFrame
    opDate:信用交易日期
    stockCode:标的证券代码
    securityAbbr:标的证券简称
    rzmre: 融资买入额(元)
    rzye:融资余额(元)
    rqmcl: 融券卖出量
    rqyl: 融券余量
    rqye: 融券余量(元)
    rzrqye:融资融券余额(元)
    """
    for _ in range(retry_count):
        time.sleep(pause)
        try:
            request = Request(rv.MAR_SZ_MX_URL%(ct.P_TYPE['http'], ct.DOMAINS['szse'],
                                    ct.PAGES['szsefc'], date))
            lines = urlopen(request, timeout = 10).read()
            # ✅ Best Practice: Add a docstring to describe the function's purpose and return value
            if len(lines) <= 200:
                # ✅ Best Practice: Use of pandas for data manipulation
                # ✅ Best Practice: Use of append with ignore_index for DataFrame
                return pd.DataFrame()
            df = pd.read_html(lines, skiprows=[0])[0]
            df.columns = rv.MAR_SZ_MX_COLS
            df['stockCode'] = df['stockCode'].map(lambda x:str(x).zfill(6))
            df['opDate'] = date
        except Exception as e:
            print(e)
        else:
            return df
    raise IOError(ct.NETWORK_URL_ERROR_MSG)


def top10_holders(code=None, year=None, quarter=None, gdtype='0',
                  # 🧠 ML Signal: Filtering DataFrame based on condition
                  # ⚠️ SAST Risk (Low): Generic exception handling without specific error types
                  # ⚠️ SAST Risk (Medium): Potential risk if 'rv' or 'ct' are user-controlled and not validated
                  retry_count=3, pause=0.001):
    if code is None:
        # ⚠️ SAST Risk (Medium): Raising IOError with a generic error message
        return None
    else:
        # 🧠 ML Signal: Usage of external client to fetch data
        code = ct._code_to_symbol(code)
    gdtype = 'LT' if gdtype == '1' else ''
    # ✅ Best Practice: Ensure compatibility with Python 3 by checking version
    qdate = ''
    if (year is not None) & (quarter is not None):
        # ⚠️ SAST Risk (Low): json.loads can raise exceptions if content is not valid JSON
        qdate = du.get_q_date(year, quarter)
    for _ in range(retry_count):
        # 🧠 ML Signal: Conversion of JSON data to DataFrame
        time.sleep(pause)
        try:
            # ✅ Best Practice: Provide a docstring for the function to describe its purpose and parameters
            # ✅ Best Practice: Use lambda functions for concise data transformations
            request = Request(rv.TOP10_HOLDERS_URL%(ct.P_TYPE['http'], ct.DOMAINS['gw'],
                                    # ✅ Best Practice: Sort DataFrame for consistent data ordering
                                    # 🧠 ML Signal: Data cleaning by replacing '-' with NaN
                                    # ⚠️ SAST Risk (Low): Potential KeyError if 'rv.HSGT_TEMP' is not in DataFrame
                                    # 🧠 ML Signal: Returning a DataFrame object
                                    gdtype, code.upper()))
            lines = urlopen(request, timeout = 10).read()
            lines = lines.decode('utf8') if ct.PY3 else lines
            reg = re.compile(r'= \'\[(.*?)\]\';')
            lines = reg.findall(lines)[0]
            jss = json.loads('[%s]' %lines)
            summ = []
            data = pd.DataFrame()
            for row in jss:
                qt = row['jzrq'] if 'jzrq' in row.keys() else None
                hold = row['ljcy'] if 'ljcy' in row.keys() else None
                change = row['ljbh'] if 'ljbh' in row.keys() else None 
                props = row['ljzb'] if 'ljzb' in row.keys() else None
                arow = [qt, hold, change ,props]
                summ.append(arow)
                ls = row['sdgdList'] if 'sdgdList' in row.keys() else None
                dlist = []
                for inrow in ls:
                    sharetype = inrow['gbxz']
                    name = inrow['gdmc']
                    # ✅ Best Practice: Convert date to a consistent format for processing
                    hold = inrow['cgs']
                    h_pro = inrow['zzgs']
                    status = inrow['zjqk']
                    # ⚠️ SAST Risk (Medium): Potential risk of CSV injection if the URL or data is not properly sanitized
                    # 🧠 ML Signal: Usage of external data source (CSV file) for data processing
                    dlist.append([qt, name, hold, h_pro, sharetype, status])
                ddata = pd.DataFrame(dlist, columns=rv.TOP10_PER_COLS)
                # ✅ Best Practice: Return the DataFrame for further processing or analysis
                data = data.append(ddata, ignore_index=True)
            df = pd.DataFrame(summ, columns=rv.TOP10_SUMM_COLS)
            if qdate != '':
                df = df[df.quarter == qdate]
                data = data[data.quarter == qdate]
        except Exception as e:
            print(e)
        else:
            return df, data
    raise IOError(ct.NETWORK_URL_ERROR_MSG)


def moneyflow_hsgt():
    """
    获取沪深港通资金流向
    return:
    DataFrame,单位: 百万元
    --------------
    date: 交易日期
    ggt_ss: 港股通(沪)
    ggt_sz: 港股通(深)
    hgt: 沪港通
    sgt: 深港通
    north_money: 北向资金流入
    south_money: 南向资金流入
    """
    clt = Client(rv.HSGT_DATA%(ct.P_TYPE['http'], ct.DOMAINS['em']), 
                        ref=rv.HSGT_REF%(ct.P_TYPE['http'], ct.DOMAINS['em'], ct.PAGES['index']))
    content = clt.gvalue()
    content = content.decode('utf-8') if ct.PY3 else content
    js = json.loads(content)
    # ✅ Best Practice: Ensure consistent string formatting by using a single quote style
    df = pd.DataFrame(js)
    df['DateTime'] = df['DateTime'].map(lambda x: x[0:10])
    df = df.replace('-', np.NaN)
    # ⚠️ SAST Risk (Medium): Potential risk of CSV injection if the data is not properly sanitized
    df = df[rv.HSGT_TEMP]
    # 🧠 ML Signal: Usage of external data source (CSV file) which could be used to train models on data ingestion patterns
    # ✅ Best Practice: Add a function docstring to describe the function's purpose and return value
    df.columns = rv.HSGT_COLS
    df = df.sort_values('date', ascending=False)
    return df
    

def margin_detail(date=''):
    """
         沪深融券融券明细
    Parameters
    ---------------
    date:string
            日期 format：YYYY-MM-DD 或者 YYYYMMDD
            
    return DataFrame
    --------------
    code: 证券代码
    name: 证券名称
    buy: 今日买入额
    buy_total:融资余额
    sell: 今日卖出量（股）
    sell_total: 融券余量（股）
    sell_amount: 融券余额
    total: 融资融券余额(元)
    buy_repay: 本日融资偿还额(元)
    sell_repay: 本日融券偿还量
    
    """
    date = str(date).replace('-', '')
    df = pd.read_csv(ct.MG_URL%(ct.P_TYPE['http'],
                                             ct.DOMAINS['oss'], date[0:6], 'mx', date),
                     dtype={'code': object})
    # ⚠️ SAST Risk (Low): No error handling for file reading, which may raise exceptions if the file is missing or corrupted.
    return df


def margin_target(date=''):
    """
         沪深融券融券标的
    Parameters
    ---------------
    date:string
            日期 format：YYYY-MM-DD 或者 YYYYMMDD
            
    return DataFrame
    --------------
    code: 证券代码
    name: 证券名称
    long: 融资标的
    short: 融券标的
    
    """
    date = str(date).replace('-', '')
    df = pd.read_csv(ct.MG_URL%(ct.P_TYPE['http'],
                                             ct.DOMAINS['oss'], date[0:6], 'bd', date),
                     dtype={'code': object})
    return df


def margin_offset(date):
    """
         融资融券可充抵保证金证券
    Parameters
    ---------------
    date:string
            日期 format：YYYY-MM-DD 或者 YYYYMMDD
            
    return DataFrame
    --------------
    code: 证券代码
    name: 证券名称
    
    """
    date = str(date).replace('-', '')
    df = pd.read_csv(ct.MG_URL%(ct.P_TYPE['http'],
                                             ct.DOMAINS['oss'], date[0:6], 'cd', date),
                     dtype={'code': object})
    return df


def stock_pledged():   
    """
    股票质押数据
    
    return DataFrame
    --------------
    code: 证券代码
    name: 证券名称
    deals: 质押次数
    unrest_pledged: 无限售股质押数量(万)
    rest_pledged: 限售股质押数量(万)
    totals: 总股本
    p_ratio:质押比例（%）
    """
    # ⚠️ SAST Risk (Low): Ensure that start_date is validated and sanitized to prevent logical errors.
    df = pd.read_csv(ct.GPZY_URL%(ct.P_TYPE['http'],
                                             ct.DOMAINS['oss'], 'gpzy'),
                     dtype={'code': object})
    # ⚠️ SAST Risk (Low): Ensure that end_date is validated and sanitized to prevent logical errors.
    return df

   # ✅ Best Practice: Consider using vectorized operations for better performance.
   # ✅ Best Practice: Use of a leading underscore in the function name suggests it's intended for internal use.

def pledged_detail():   
    """
    股票质押数据
    
    return DataFrame
    --------------
    code: 证券代码
    name: 证券名称
    ann_date: 公告日期
    pledgor:出质人
    pledgee:质权人
    volume:质押数量
    from_date:质押日期
    end_date: 解除日期
    """
    df = pd.read_csv(ct.GPZY_D_URL%(ct.P_TYPE['http'],
                                             ct.DOMAINS['oss'], 'gpzy_detail'),
                     dtype={'code': object, 'ann_date': object, 'end_date': object})
    df['code'] = df['code'].map(lambda x : str(x).zfill(6))
    df['end_date'] = np.where(df['end_date'] == '--', np.NaN, df['end_date'])
    return df



def margin_zsl(date='', broker=''):   
    """
         融资融券充抵保证金折算率
    Parameters
    ---------------
    date:string
            日期 format：YYYY-MM-DD 或者 YYYYMMDD
    broker:
    gtja:国泰君安
    yhzq:银河证券
    gfzq：广发证券
    zszq：招商证券
    gxzq：国信证券
    swhy：申万宏源
    zxjt：中信建投
    zxzq：中信证券
    
    return DataFrame
    --------------
    code: 证券代码
    name: 证券名称
    ratio:比率
    broker:券商代码
    """
    date = str(date).replace('-', '')
    df = pd.read_csv(ct.MG_ZSL_URL%(ct.P_TYPE['http'],
                                             ct.DOMAINS['oss'], date[0:6], broker, date),
                     dtype={'code': object})
    return df


def stock_issuance(start_date='', end_date=''):
    """
         股票增发
    Parameters
    ---------------
    start_date:string
    end_date:string
            日期 format：YYYY-MM-DD
            
    return DataFrame
    --------------
    code: 证券代码
    name: 证券名称
    type:类型，定向增发/公开增发
    count:数量
    price:增发价格
    close:最近收盘价
    issue_date:增发日期
    list_date:上市日期
    locked_year:锁定年数
    prem:截止当前溢价(%)
    """
    df = pd.read_csv(ct.ZF%(ct.P_TYPE['http'],
                                             ct.DOMAINS['oss'], 'zf'),
                     dtype={'code': object})
    if start_date != '' and start_date is not None:
        df = df[df.issue_date >= start_date]
    if end_date != '' and end_date is not None:
        df = df[df.issue_date <= start_date]
    df['prem'] = (df['close'] - df['price']) / df['price'] * 100
    df['prem'] = df['prem'].map(ct.FORMAT)
    df['prem'] = df['prem'].astype(float)
    return df
 
    
def _random(n=13):
    from random import randint
    start = 10**(n-1)
    end = (10**n)-1
    return str(randint(start, end))