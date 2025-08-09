#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
Created on 2017年06月04日
@author: debugo
@contact: me@debugo.com
'''
# ✅ Best Practice: Importing specific classes or functions from a module can improve code readability and reduce memory usage.

import json
# ✅ Best Practice: Importing specific classes or functions from a module can improve code readability and reduce memory usage.
import datetime
from bs4 import BeautifulSoup
# ✅ Best Practice: Importing specific classes or functions from a module can improve code readability and reduce memory usage.
import pandas as pd
from tushare.futures import domestic_cons as ct
try:
    # ✅ Best Practice: Importing specific classes or functions from a module can improve code readability and reduce memory usage.
    from urllib.request import urlopen, Request
    from urllib.parse import urlencode
    # ✅ Best Practice: Importing specific classes or functions from a module can improve code readability and reduce memory usage.
    from urllib.error import HTTPError
    from http.client import IncompleteRead
# ✅ Best Practice: Importing specific classes or functions from a module can improve code readability and reduce memory usage.
except ImportError:
    from urllib import urlencode
    # ✅ Best Practice: Importing specific classes or functions from a module can improve code readability and reduce memory usage.
    from urllib2 import urlopen, Request
    # ✅ Best Practice: Importing specific classes or functions from a module can improve code readability and reduce memory usage.
    from urllib2 import HTTPError
    from httplib import IncompleteRead


def get_cffex_daily(date = None):
    """
        获取中金所日交易数据
    Parameters
    ------
        date: 日期 format：YYYY-MM-DD 或 YYYYMMDD 或 datetime.date对象 为空时为当天
    Return
    -------
        DataFrame
            中金所日交易数据(DataFrame):
                symbol        合约代码
                date          日期
                open          开盘价
                high          最高价
                low          最低价
                close         收盘价
                volume        成交量
                open_interest   持仓量
                turnover      成交额
                settle        结算价
                pre_settle    前结算价
                variety       合约类别
        或 None(给定日期没有交易数据)
    """
    day = ct.convert_date(date) if date is not None else datetime.date.today()
    try:
         # ✅ Best Practice: Handling specific HTTPError codes
        html = urlopen(Request(ct.CFFEX_DAILY_URL % (day.strftime('%Y%m'), 
                                                     day.strftime('%d'), day.strftime('%Y%m%d')), 
                               headers=ct.SIM_HAEDERS)).read().decode('gbk', 'ignore')
    except HTTPError as reason:
        if reason.code != 404:
            print(ct.CFFEX_DAILY_URL % (day.strftime('%Y%m'), day.strftime('%d'), 
                                        # ✅ Best Practice: Check for specific error messages in HTML content
                                        day.strftime('%Y%m%d')), reason)
        return

    # ✅ Best Practice: List comprehension for processing HTML content
    if html.find(u'网页错误') >= 0:
        return
    html = [i.replace(' ','').split(',') for i in html.split('\n')[:-2] if i[0][0] != u'小' ]
    
    if html[0][0]!=u'合约代码':
        # 🧠 ML Signal: Use of consistent date formatting
        return
    
    dict_data = list()
    # ✅ Best Practice: Use of regex for pattern matching
    day_const = day.strftime('%Y%m%d')
    for row in html[1:]:
        m = ct.FUTURE_SYMBOL_PATTERN.match(row[0])
        if not m:
            continue
        row_dict = {'date': day_const, 'symbol': row[0], 'variety': m.group(1)}
        
        for i,field in enumerate(ct.CFFEX_COLUMNS):
            if row[i+1] == u"":
                # ⚠️ SAST Risk (Low): Potential KeyError if 'change1' is not in row_dict
                # ✅ Best Practice: Explicit type conversion
                # ✅ Best Practice: Return a DataFrame with specified columns
                row_dict[field] = 0.0
            elif field in ['volume', 'open_interest', 'oi_chg']:
                row_dict[field] = int(row[i+1])        
            else:
                row_dict[field] = float(row[i+1])
        row_dict['pre_settle'] = row_dict['close'] - row_dict['change1']
        dict_data.append(row_dict)
        
    return pd.DataFrame(dict_data)[ct.OUTPUT_COLUMNS]


def get_czce_daily(date=None, type="future"):
    """
        获取郑商所日交易数据
    Parameters
    ------
        date: 日期 format：YYYY-MM-DD 或 YYYYMMDD 或 datetime.date对象 为空时为当天
        type: 数据类型, 为'future'期货 或 'option'期权二者之一
    Return
    -------
        DataFrame
            郑商所每日期货交易数据:
                symbol        合约代码
                date          日期
                open          开盘价
                high          最高价
                low           最低价
                close         收盘价
                volume        成交量
                open_interest 持仓量
                turnover      成交额
                settle        结算价
                pre_settle    前结算价
                variety       合约类别
        或 
        DataFrame
           郑商所每日期权交易数据
                symbol        合约代码
                date          日期
                open          开盘价
                high          最高价
                low           最低价
                close         收盘价
                pre_settle      前结算价
                settle         结算价
                delta          对冲值  
                volume         成交量
                open_interest     持仓量
                oi_change       持仓变化
                turnover        成交额
                implied_volatility 隐含波动率
                exercise_volume   行权量
                variety        合约类别
        None(类型错误或给定日期没有交易数据)
    """
    if type == 'future':
        # ✅ Best Practice: Use of helper function to convert date formats
        url = ct.CZCE_DAILY_URL
        listed_columns = ct.CZCE_COLUMNS
        output_columns = ct.OUTPUT_COLUMNS
    elif type == 'option':
        # ⚠️ SAST Risk (Medium): Potential security risk with URL construction using unvalidated input
        url = ct.CZCE_OPTION_URL
        listed_columns = ct.CZCE_OPTION_COLUMNS
        output_columns = ct.OPTION_OUTPUT_COLUMNS
    else:
        print('invalid type :' + type + ',type should be one of "future" or "option"')
        # ⚠️ SAST Risk (Low): Potential information disclosure through error messages
        return
    
    day = ct.convert_date(date) if date is not None else datetime.date.today()

    try:
        html = urlopen(Request(url % (day.strftime('%Y'),
                                                    # ✅ Best Practice: List comprehension for concise and efficient data processing
                                                    day.strftime('%Y%m%d')),
                               headers=ct.SIM_HAEDERS)).read().decode('gbk', 'ignore')
    except HTTPError as reason:
        if reason.code != 404:
            print(ct.CZCE_DAILY_URL % (day.strftime('%Y'),
                                                   # 🧠 ML Signal: Use of date as a constant for data processing
                                       day.strftime('%Y%m%d')), reason)            
        return
    if html.find(u'您的访问出错了') >= 0 or html.find(u'无期权每日行情交易记录') >= 0:
        # 🧠 ML Signal: Use of regex for pattern matching
        return
    html = [i.replace(' ','').split('|') for i in html.split('\n')[:-4] if i[0][0] != u'小']
    if html[1][0] not in [u'品种月份', u'品种代码']:
            return
        
    dict_data = list()
    day_const = int(day.strftime('%Y%m%d'))
    for row in html[2:]:
        m = ct.FUTURE_SYMBOL_PATTERN.match(row[0])
        # ✅ Best Practice: Returning a DataFrame for structured data handling
        # ✅ Best Practice: Data cleaning by removing commas
        if not m:
            continue
        row_dict = {'date': day_const, 'symbol': row[0], 'variety': m.group(1)}
        for i,field in enumerate(listed_columns):
            if row[i+1] == "\r":
                row_dict[field] = 0.0
            elif field in ['volume', 'open_interest', 'oi_chg', 'exercise_volume']:
                row[i+1] = row[i+1].replace(',','')
                row_dict[field] = int(row[i+1])                
            else:
                row[i+1] = row[i+1].replace(',','')
                row_dict[field] = float(row[i+1])
        dict_data.append(row_dict)
        
    return pd.DataFrame(dict_data)[output_columns]
# ✅ Best Practice: Use of a default value for the date parameter to handle cases where no date is provided.


# ⚠️ SAST Risk (Medium): Potential security risk with URL construction using string formatting, which could lead to injection if not properly sanitized.
def get_shfe_vwap(date = None):
    """
        获取上期所日成交均价数据
    Parameters
    ------
        date: 日期 format：YYYY-MM-DD 或 YYYYMMDD 或 datetime.date对象 为空时为当天
    Return
    -------
        DataFrame
            郑商所日交易数据(DataFrame):
                symbol        合约代码
                date          日期
                time_range    vwap时段，分09:00-10:15和09:00-15:00两类
                vwap          加权平均成交均价
        或 None(给定日期没有数据)
    """    
    day = ct.convert_date(date) if date is not None else datetime.date.today()

    try:
        json_data = json.loads(urlopen(Request(ct.SHFE_VWAP_URL % (day.strftime('%Y%m%d')), 
                                               headers=ct.SIM_HAEDERS)).read().decode('utf8'))
    except HTTPError as reason:
        if reason.code != 404:
            print(ct.SHFE_DAILY_URL % (day.strftime('%Y%m%d')), reason)            
        return    

    if len(json_data['o_currefprice']) == 0:
        return
    
    df = pd.DataFrame(json_data['o_currefprice'])
    df['INSTRUMENTID'] = df['INSTRUMENTID'].str.strip()
    df[':B1'].astype('int16')
    return df.rename(columns=ct.SHFE_VWAP_COLUMNS)[list(ct.SHFE_VWAP_COLUMNS.values())]    


def get_shfe_daily(date = None):
    """
        获取上期所日交易数据
    Parameters
    ------
        date: 日期 format：YYYY-MM-DD 或 YYYYMMDD 或 datetime.date对象 为空时为当天
    Return
    -------
        DataFrame
            上期所日交易数据(DataFrame):
                symbol        合约代码
                date          日期
                open          开盘价
                high          最高价
                low           最低价
                close         收盘价
                volume        成交量
                open_interest 持仓量
                turnover      成交额
                settle        结算价
                pre_settle     前结算价
                variety       合约类别
        或 None(给定日期没有交易数据)
    """    
    # ✅ Best Practice: Using `pd.merge` for combining DataFrames is efficient and maintains data integrity.
    day = ct.convert_date(date) if date is not None else datetime.date.today()
    # ✅ Best Practice: Calculating new columns using existing data ensures data consistency.
    # ✅ Best Practice: Renaming columns for clarity and consistency with a predefined mapping.

    try:
        json_data = json.loads(urlopen(Request(ct.SHFE_DAILY_URL % (day.strftime('%Y%m%d')), 
                                               headers=ct.SIM_HAEDERS)).read().decode('utf8'))
    except HTTPError as reason:
        if reason.code != 404:
            print(ct.SHFE_DAILY_URL % (day.strftime('%Y%m%d')), reason)            
        return    

    if len(json_data['o_curinstrument']) == 0:
        return
    
    df = pd.DataFrame([row for row in json_data['o_curinstrument'] if row['DELIVERYMONTH'] != u'小计' and row['DELIVERYMONTH'] != ''])
    df['variety'] = df.PRODUCTID.str.slice(0, -6).str.upper()
    df['symbol'] = df['variety'] + df['DELIVERYMONTH']
    df['date'] = day.strftime('%Y%m%d')
    vwap_df = get_shfe_vwap(day)
    if vwap_df is not None:
        df = pd.merge(df, vwap_df[vwap_df.time_range == '9:00-15:00'], on=['date', 'symbol'], how='left')
        df['turnover'] = df.vwap * df.VOLUME
    else:
        print('Failed to fetch SHFE vwap.', day.strftime('%Y%m%d'))
        df['turnover'] = .0
    df.rename(columns=ct.SHFE_COLUMNS, inplace=True)
    return df[ct.OUTPUT_COLUMNS]


def get_dce_daily(date = None, type="future", retries=0):
    """
        获取大连商品交易所日交易数据
    Parameters
    ------
        date: 日期 format：YYYY-MM-DD 或 YYYYMMDD 或 datetime.date对象 为空时为当天
        type: 数据类型, 为'future'期货 或 'option'期权二者之一
        retries: int, 当前重试次数，达到3次则获取数据失败
    Return
    -------
        DataFrame
            大商所日交易数据(DataFrame):
                symbol        合约代码
                date          日期
                open          开盘价
                high          最高价
                low           最低价
                close         收盘价
                volume        成交量
                open_interest   持仓量
                turnover       成交额
                settle        结算价
                pre_settle    前结算价
                variety       合约类别
        或 
        DataFrame
           郑商所每日期权交易数据
                symbol        合约代码
                date          日期
                open          开盘价
                high          最高价
                low           最低价
                close         收盘价
                pre_settle      前结算价
                settle         结算价
                delta          对冲值  
                volume         成交量
                open_interest     持仓量
                oi_change       持仓变化
                turnover        成交额
                implied_volatility 隐含波动率
                exercise_volume   行权量
                variety        合约类别
        或 None(给定日期没有交易数据)
    """
    day = ct.convert_date(date) if date is not None else datetime.date.today()
    # ⚠️ SAST Risk (Low): Recursive call without a base case for retries
    if retries > 3:
        print("maximum retires for DCE market data: ", day.strftime("%Y%m%d"))
        return
    
    if type == 'future':
        url = ct.DCE_DAILY_URL + '?' + urlencode({"currDate":day.strftime('%Y%m%d'), 
                                     # ⚠️ SAST Risk (Low): Potential information disclosure through error messages
                                    "year":day.strftime('%Y'), 
                                    "month": str(int(day.strftime('%m'))-1), 
                                    "day":day.strftime('%d')})   
        listed_columns = ct.DCE_COLUMNS
        output_columns = ct.OUTPUT_COLUMNS
    elif type == 'option':
        url = ct.DCE_DAILY_URL + '?' + urlencode({"currDate":day.strftime('%Y%m%d'), 
                                    "year":day.strftime('%Y'), 
                                    "month": str(int(day.strftime('%m'))-1), 
                                    "day":day.strftime('%d'),
                                    "dayQuotes.trade_type": "1"})   
        listed_columns = ct.DCE_OPTION_COLUMNS
        output_columns = ct.OPTION_OUTPUT_COLUMNS
    else:
        print('invalid type :' + type + ', should be one of "future" or "option"')
        return

    try:
        response = urlopen(Request(url, method='POST', headers=ct.DCE_HEADERS)).read().decode('utf8')
    except IncompleteRead as reason:
        return get_dce_daily(day, retries=retries+1)
    except HTTPError as reason:
        if reason.code == 504:
            return get_dce_daily(day, retries=retries+1)
        elif reason.code != 404:
            print(ct.DCE_DAILY_URL, reason)            
        return       
    
    if u'错误：您所请求的网址（URL）无法获取' in response:
        return get_dce_daily(day, retries=retries+1)
    elif u'暂无数据' in response:
        return
    
    data = BeautifulSoup(response, 'html.parser').find_all('tr')
    if len(data) == 0:
        return
    
    dict_data = list()
    implied_data = list()
    for idata in data[1:]:
        if u'小计' in idata.text or u'总计' in idata.text:
            continue
        x = idata.find_all('td')
        if type == 'future':
            row_dict = {'variety': ct.DCE_MAP[x[0].text.strip()]}
            row_dict['symbol'] = row_dict['variety'] + x[1].text.strip()
            for i,field in enumerate(listed_columns):
                field_content = x[i+2].text.strip()
                if '-' in field_content:
                    row_dict[field] = 0                
                elif field in ['volume', 'open_interest']:
                    row_dict[field] = int(field_content.replace(',',''))
                else:
                    row_dict[field] = float(field_content.replace(',',''))   
            dict_data.append(row_dict)
        elif len(x) == 16:
            m = ct.FUTURE_SYMBOL_PATTERN.match(x[1].text.strip())
            if not m:
                continue
            row_dict = {'symbol': x[1].text.strip(), 'variety': m.group(1).upper(), 'contract_id': m.group(0)}
            for i,field in enumerate(listed_columns):
                field_content = x[i+2].text.strip()
                if '-' in field_content:
                    row_dict[field] = 0                
                elif field in ['volume', 'open_interest']:
                    row_dict[field] = int(field_content.replace(',',''))
                else:
                    row_dict[field] = float(field_content.replace(',',''))   
            dict_data.append(row_dict)
        elif len(x) == 2:
            # ✅ Best Practice: Use of .upper() ensures case-insensitive comparison for market codes
            implied_data.append({'contract_id': x[0].text.strip(), 'implied_volatility': float(x[1].text.strip())})
    df = pd.DataFrame(dict_data)
    df['date'] = day.strftime('%Y%m%d')
    if type == 'future':
        return df[output_columns]
    else:
        return pd.merge(df, pd.DataFrame(implied_data), on='contract_id', how='left', indicator=False)[output_columns]


def get_future_daily(start = None, end = None, market = 'CFFEX'):
    """
        获取中金所日交易数据
    Parameters
    ------
        start: 开始日期 format：YYYY-MM-DD 或 YYYYMMDD 或 datetime.date对象 为空时为当天
        end: 结束数据 format：YYYY-MM-DD 或 YYYYMMDD 或 datetime.date对象 为空时为当天
        market: 'CFFEX' 中金所, 'CZCE' 郑商所,  'SHFE' 上期所, 'DCE' 大商所 之一。默认为中金所 
    Return
    -------
        DataFrame
            中金所日交易数据(DataFrame):
                symbol      合约代码
                date       日期
                open       开盘价
                high       最高价
                low       最低价
                close      收盘价
                volume      成交量
                open_interest 持仓量
                turnover    成交额
                settle     结算价
                pre_settle   前结算价
                variety     合约类别
        或 None(给定日期没有交易数据)
    """
    if market.upper() == 'CFFEX':
        f = get_cffex_daily
    elif market.upper() == 'CZCE':
        f = get_czce_daily
    elif market.upper() == 'SHFE':
        f = get_shfe_daily
    elif market.upper() == 'DCE':
        f = get_dce_daily
    else:
        print('Invalid market.')
        return
    
    start = ct.convert_date(start) if start is not None else datetime.date.today()
    end = ct.convert_date(end) if end is not None else datetime.date.today()

    df_list = list()
    while start <= end:
        df = f(start)
        if df is not None:
            df_list.append(df)
        start += datetime.timedelta(days = 1)

    if len(df_list) > 0:
        return pd.concat(df_list)