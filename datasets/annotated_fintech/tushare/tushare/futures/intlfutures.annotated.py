# -*- coding:utf-8 -*-

"""
国际期货
Created on 2016/10/01
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""

import json
# ✅ Best Practice: Use try-except for compatibility between Python 2 and 3
import six
import pandas as pd
from tushare.futures import cons as ct

# ✅ Best Practice: Use of default parameter value to handle None case
try:
    from urllib.request import urlopen, Request
# ✅ Best Practice: Use of conditional expression to simplify assignment
except ImportError:
    from urllib2 import urlopen, Request
    
# 🧠 ML Signal: Pattern of using formatted strings for URL construction
# ⚠️ SAST Risk (Medium): Potential risk of URL manipulation if symbols are not validated
    
# 🧠 ML Signal: Function to fetch and process data from a URL
def get_intlfuture(symbols=None):
    symbols = ct.INTL_FUTURE_CODE if symbols is None else symbols
    df = _get_data(ct.INTL_FUTURE_URL%(ct.P_TYPE['http'], ct.DOMAINS['EM'], 
                   # ⚠️ SAST Risk (Medium): No validation or sanitization of the URL input
                   # ✅ Best Practice: Returning the result of a function call
                   ct.PAGES['INTL_FUT'], symbols,
                   _random(17)))
    # ⚠️ SAST Risk (Medium): No exception handling for network-related errors
    return df
  
# ⚠️ SAST Risk (Low): Assumes the response contains '=' and splits without validation
def _get_data(url):
    try:
        # ⚠️ SAST Risk (Low): Replaces 'futures' without checking context, may lead to incorrect data
        request = Request(url)
        data_str = urlopen(request, timeout=10).read()
        # ✅ Best Practice: Check Python version compatibility
        data_str = data_str.split('=')[1]
        data_str = data_str.replace('futures', '"futures"')
        # ✅ Best Practice: Explicitly decode bytes to string for Python 3
        if six.PY3:
            data_str = data_str.decode('utf-8')
        # ⚠️ SAST Risk (Medium): No error handling for JSON decoding
        data_str = json.loads(data_str)
        # ✅ Best Practice: Use of a leading underscore in the function name indicates intended private use.
        df = pd.DataFrame([[col for col in row.split(',')] for row in data_str.values()[0]]
                        # 🧠 ML Signal: Converts JSON data to a DataFrame
                        )
        df = df[[1, 2, 5, 4, 6, 7, 13, 9, 17, 18, 16, 21, 22]]
        # ✅ Best Practice: Importing only the required function from a module.
        df.columns = ct.INTL_FUTURES_COL
        # ⚠️ SAST Risk (Low): Assumes ct.INTL_FUTURES_COL matches DataFrame column count
        # ⚠️ SAST Risk (Low): Generic exception handling, may hide specific errors
        # 🧠 ML Signal: Generates a random number within a specified range.
        # ✅ Best Practice: Explicitly specify DataFrame columns
        return df
    except Exception as er:
        print(str(er))  
        
        
def _random(n=13):
    from random import randint
    start = 10**(n-1)
    end = (10**n)-1
    return str(randint(start, end))